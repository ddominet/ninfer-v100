#pragma once

// Tensor-core decode attention over the paged INT8-G64 KV cache (sm_70), T <= 8 tokens.
//
// One CTA = one KV head x one key split; one warp per 8-row tile of (token, q-head) rows, so
// T tokens of the 24q/4kv geometry need ceil(6T/8) warps. The cache's natural [key][dim] byte
// order already is the mma.m8n8k4 B-fragment order:
//   QK^T (row.col, quadpair-split-N): a lane owns one of 32 keys and consumes 16 contiguous
//         dims per 16-byte load.
//   P.V  (row.row): a lane owns one of four keys and a 16-dim block; the eight lanes sharing a
//         key cover 128 contiguous bytes, so each load instruction reads one full line per key.
// A single tile (one token) takes K and V global -> registers -> dequant -> mma with no shared
// staging and no barrier. Two or more tiles share one staged FP16 copy per 32-key tile instead of
// fetching the cache once per warp. Q is rotated and scaled once per launch by a small pre-kernel.
//
// Output is the partial_acc / partial_m / partial_l format of the split-reduce pipeline, with a
// stale running maximum (P = 2^(s - m) <= 2^8), which the reduce treats like any other m.

#include "core/device.h"
#include "ops/common/volta_mma.cuh"
#include "ops/kernel/paged_kv_address.cuh"
#include "ops/kv_cache/hadamard_d256.cuh"
#include "ops/kv_cache/int8_g64_codec.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cmath>
#include <cstdint>

namespace ninfer::ops {

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ == 700

inline constexpr int kDirectKeys    = 32;
inline constexpr int kDirectQStride = kCausalHeadDim + 8; // halves; rows step 4 banks
inline constexpr int kDirectPStride = kDirectKeys + 8;

constexpr int causal_attention_direct_i8_smem_bytes(int tiles) {
    return tiles * (8 * kDirectQStride * 2 + 8 * kDirectPStride * 2);
}

__device__ __forceinline__ void direct_mma_row_row(float (&d)[8], unsigned a0, unsigned a1,
                                                   unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m8n8k4.row.row.f32.f16.f16.f32 "
                 "{%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9}, {%10, %11}, "
                 "{%0, %1, %2, %3, %4, %5, %6, %7};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]),
                   "+f"(d[6]), "+f"(d[7])
                 : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
}

// 16 signed codes -> 16 halves (code * scale), exact: 0x64XX is 1024 + XX, so subtracting 1152
// recovers the signed code before the single rounding multiply.
__device__ __forceinline__ void direct_dequant16(uint4 codes, half scale, uint4& lo, uint4& hi) {
    const half2 bias    = __half2half2(__ushort_as_half(0x6480));
    const half2 s2      = __half2half2(scale);
    const unsigned w[4] = {codes.x ^ 0x80808080u, codes.y ^ 0x80808080u, codes.z ^ 0x80808080u,
                           codes.w ^ 0x80808080u};
    unsigned out[8];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const unsigned p0 = __byte_perm(w[i], 0x64646464u, 0x4140);
        const unsigned p1 = __byte_perm(w[i], 0x64646464u, 0x4342);
        const half2 h0 = __hmul2(__hsub2(*reinterpret_cast<const half2*>(&p0), bias), s2);
        const half2 h1 = __hmul2(__hsub2(*reinterpret_cast<const half2*>(&p1), bias), s2);
        out[2 * i]     = *reinterpret_cast<const unsigned*>(&h0);
        out[2 * i + 1] = *reinterpret_cast<const unsigned*>(&h1);
    }
    lo = make_uint4(out[0], out[1], out[2], out[3]);
    hi = make_uint4(out[4], out[5], out[6], out[7]);
}

__device__ __forceinline__ float direct_ex2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ float direct_max3(float v) {
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 2));
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 4));
    return fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 8));
}

__device__ __forceinline__ float direct_sum3(float v) {
    v += __shfl_xor_sync(0xffffffffu, v, 2);
    v += __shfl_xor_sync(0xffffffffu, v, 4);
    return v + __shfl_xor_sync(0xffffffffu, v, 8);
}

// Hadamard-rotated, pre-scaled FP16 queries, [batch][token][q head][256]. The partial kernels need them
// in every CTA; rotating once here replaces a ~4000-instruction prologue per row tile and CTA.
template <typename Geometry>
__global__ void __launch_bounds__(32) causal_attention_direct_prepare_q_kernel(
    const __nv_bfloat16* __restrict__ q, half* __restrict__ out, int tokens,
    std::int32_t full_width, std::int32_t column_begin, float scale_log2) {
    const int lane  = static_cast<int>(threadIdx.x);
    const int row   = static_cast<int>(blockIdx.x);
    const int qh    = row % Geometry::QHeads;
    const int token = (row / Geometry::QHeads) % tokens;
    const int batch = (row / Geometry::QHeads) / tokens;
    q += static_cast<std::int64_t>(kCausalHeadDim) * Geometry::QHeads *
         (column_begin + static_cast<std::int64_t>(batch) * full_width);
    float vals[8];
#pragma unroll
    for (int p = 0; p < 8; ++p) {
        vals[p] = __bfloat162float(q[causal_q_index<Geometry>(qh, lane + 32 * p, token)]);
    }
    normalized_hadamard_d256_inplace(vals, lane);
    half* dst = out + static_cast<std::int64_t>(row) * kCausalHeadDim;
#pragma unroll
    for (int p = 0; p < 8; ++p) { dst[lane + 32 * p] = __float2half_rn(vals[p] * scale_log2); }
}

constexpr std::size_t causal_attention_direct_prepared_q_bytes(int q_heads, int tokens,
                                                              int batch_size) {
    return static_cast<std::size_t>(batch_size) * tokens * q_heads * kCausalHeadDim *
           sizeof(half);
}

template <typename Geometry>
void causal_attention_direct_prepare_q(const __nv_bfloat16* q, half* out, int tokens,
                                       int batch_size, std::int32_t full_width,
                                       std::int32_t column_begin, float scale,
                                       cudaStream_t stream) {
    causal_attention_direct_prepare_q_kernel<Geometry>
        <<<batch_size * tokens * Geometry::QHeads, 32, 0, stream>>>(
            q, out, tokens, full_width, column_begin, scale * 1.4426950408889634f);
}

// Fills this warp's `count` Q rows of shared memory, from the prepared plane when there is one.
template <typename Geometry>
__device__ __forceinline__ void direct_fill_q_rows(const __nv_bfloat16* q, const half* q_prepared,
                                                   half* q_s, int kvh, int row0, int count,
                                                   int rows, int lane, float scale_log2) {
    constexpr int G = Geometry::GroupSize;
    if (q_prepared != nullptr) {
        for (int rr = 0; rr < count; ++rr) {
            const int row = row0 + rr;
            uint4 value   = make_uint4(0, 0, 0, 0);
            if (row < rows) {
                value = __ldg(reinterpret_cast<const uint4*>(
                    q_prepared +
                    (static_cast<std::int64_t>(row / G) * Geometry::QHeads + kvh * G + row % G) *
                        kCausalHeadDim +
                    lane * 8));
            }
            *reinterpret_cast<uint4*>(q_s + row * kDirectQStride + lane * 8) = value;
        }
    } else {
        for (int rr = 0; rr < count; ++rr) {
            const int row = row0 + rr;
            float vals[8];
            if (row < rows) {
                const int qh = kvh * G + row % G;
#pragma unroll
                for (int p = 0; p < 8; ++p) {
                    vals[p] = __bfloat162float(
                        q[causal_q_index<Geometry>(qh, lane + 32 * p, row / G)]);
                }
                normalized_hadamard_d256_inplace(vals, lane);
            } else {
#pragma unroll
                for (int p = 0; p < 8; ++p) { vals[p] = 0.0f; }
            }
            half* dst = q_s + row * kDirectQStride;
#pragma unroll
            for (int p = 0; p < 8; ++p) {
                dst[lane + 32 * p] = __float2half_rn(vals[p] * scale_log2);
            }
        }
    }
    __syncwarp();
}

// Online softmax over one 8 x 32 score tile in the log2 domain with a stale running maximum: the
// maximum only moves when a score beats it by 2^8, so P = 2^(s - m) stays a finite half and the
// accumulators are rescaled rarely. On return s holds P. Lane layout is the m8n8k4 C fragment:
// s[i] is row (i & 2) | c_row_lo, key key0 + (i & 1) + 16 * (i >> 2).
__device__ __forceinline__ void direct_softmax_step(float (&s)[8], float (&o)[8][8], float& m_lo,
                                                    float& m_hi, float& l_lo, float& l_hi,
                                                    bool masked, int key0, int pos_lo, int pos_hi) {
    constexpr float kNegInf = -INFINITY;
    constexpr unsigned Full = 0xffffffffu;
    if (masked) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int key     = key0 + (i & 1) + ((i >> 2) << 4);
            const int row_pos = (i & 2) ? pos_hi : pos_lo;
            s[i]              = key <= row_pos ? s[i] : kNegInf;
        }
    }
    const float mx_lo = fmaxf(fmaxf(s[0], s[1]), fmaxf(s[4], s[5]));
    const float mx_hi = fmaxf(fmaxf(s[2], s[3]), fmaxf(s[6], s[7]));
    if (__any_sync(Full, mx_lo > m_lo + 8.0f || mx_hi > m_hi + 8.0f)) {
        const float new_lo   = fmaxf(m_lo, direct_max3(mx_lo));
        const float new_hi   = fmaxf(m_hi, direct_max3(mx_hi));
        const float alpha_lo = new_lo == kNegInf ? 1.0f : direct_ex2(m_lo - new_lo);
        const float alpha_hi = new_hi == kNegInf ? 1.0f : direct_ex2(m_hi - new_hi);
        l_lo *= alpha_lo;
        l_hi *= alpha_hi;
#pragma unroll
        for (int cb = 0; cb < 8; ++cb) {
#pragma unroll
            for (int i = 0; i < 8; ++i) { o[cb][i] *= (i & 2) ? alpha_hi : alpha_lo; }
        }
        m_lo = new_lo;
        m_hi = new_hi;
    }
    const float base_lo = m_lo == kNegInf ? 0.0f : m_lo;
    const float base_hi = m_hi == kNegInf ? 0.0f : m_hi;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        s[i] = direct_ex2(s[i] - ((i & 2) ? base_hi : base_lo));
        if (i & 2) {
            l_hi += s[i];
        } else {
            l_lo += s[i];
        }
    }
}

// Writes this warp's 8-row tile (rows row0..row0+7) in the split-reduce partial format. l_lo and
// l_hi are already reduced over the lanes that share a row.
template <typename Geometry>
__device__ __forceinline__ void direct_write_partials(const float (&o)[8][8], float m_lo,
                                                      float m_hi, float l_lo, float l_hi, int kvh,
                                                      int split, int row0, int rows, int tokens,
                                                      int lane, float* partial_acc,
                                                      float* partial_m, float* partial_l) {
    constexpr int G    = Geometry::GroupSize;
    const int qp       = (lane >> 2) & 3;
    const int c_row_lo = (((lane >> 4) & 1) << 2) | (lane & 1);
#pragma unroll
    for (int i = 0; i < 8; i += 2) {
        const int row = row0 + ((i & 2) | c_row_lo);
        if (row >= rows) { continue; }
        const int token = row / G;
        const int qh    = kvh * G + row % G;
#pragma unroll
        for (int cb = 0; cb < 8; ++cb) {
            const int dim = (cb >> 2) * 128 + (qp + 4 * (i >> 2)) * 16 + (cb & 3) * 4 +
                            (((lane >> 1) & 1) << 1);
            *reinterpret_cast<float2*>(
                partial_acc + causal_partial_acc_index<Geometry>(qh, dim, token, split, tokens)) =
                make_float2(o[cb][i], o[cb][i + 1]);
        }
    }
    if ((lane & 14) == 0) {
        constexpr float kLn2 = 0.6931471805599453f;
        const int row_lo     = row0 + c_row_lo;
        const int row_hi     = row_lo + 2;
        if (row_lo < rows) {
            const auto idx = causal_partial_stat_index<Geometry>(kvh * G + row_lo % G, row_lo / G,
                                                                 split, tokens);
            partial_m[idx] = l_lo > 0.0f ? m_lo * kLn2 : -1e30f;
            partial_l[idx] = l_lo;
        }
        if (row_hi < rows) {
            const auto idx = causal_partial_stat_index<Geometry>(kvh * G + row_hi % G, row_hi / G,
                                                                 split, tokens);
            partial_m[idx] = l_hi > 0.0f ? m_hi * kLn2 : -1e30f;
            partial_l[idx] = l_hi;
        }
    }
}

// Split s covers keys [s * chunk, (s + 1) * chunk) with chunk rounded up to a multiple of 32 so a
// key tile never crosses a 64-token page; the active split count follows
// causal_small_t_active_splits, which the reduce kernel also uses. Every row of token t sees keys
// <= pos[t]; the new tokens must already be appended.
//
// One token (a single 8-row tile): one warp per CTA, K and V straight from global memory.
template <typename Geometry>
__global__ void __launch_bounds__(32) causal_attention_direct_i8_kernel(
    const __nv_bfloat16* __restrict__ q, const half* __restrict__ q_prepared,
    const std::int32_t* __restrict__ pos, const std::int8_t* __restrict__ k_pages,
    const std::int8_t* __restrict__ v_pages,
    const __half* __restrict__ k_scale, const __half* __restrict__ v_scale,
    const std::int32_t* __restrict__ block_tables, const std::int32_t* __restrict__ valid_columns,
    const std::int32_t* __restrict__ table_rows, std::int32_t table_stride, int tokens,
    std::int32_t full_width, std::int32_t column_begin, std::int32_t logical_capacity,
    float scale_log2,
    float* __restrict__ partial_acc, float* __restrict__ partial_m,
    float* __restrict__ partial_l) {
    constexpr int D         = kCausalHeadDim;
    constexpr int G         = Geometry::GroupSize;
    constexpr int KVHeads   = Geometry::KVHeads;
    constexpr float kNegInf = -INFINITY;
    constexpr unsigned Full = 0xffffffffu;
    static_assert(D == 256 && kKVCacheInt8Group == 64);

    // Q rows (read-only after setup) and the P transposition.
    extern __shared__ __align__(16) unsigned char direct_smem[];
    half* q_s = reinterpret_cast<half*>(direct_smem);
    half* p_w = q_s + 8 * kDirectQStride;

    const int kvh   = static_cast<int>(blockIdx.x);
    const int split = static_cast<int>(blockIdx.y);
    const int lane  = static_cast<int>(threadIdx.x);
    const int rows  = tokens * G;

    // grid.z is the stream; each has its own columns, block-table row and partial planes.
    const int batch                = static_cast<int>(blockIdx.z);
    const std::int64_t column_base = column_begin + static_cast<std::int64_t>(batch) * full_width;
    q += static_cast<std::int64_t>(D) * Geometry::QHeads * column_base;
    pos += column_base;
    if (q_prepared != nullptr) {
        q_prepared += static_cast<std::int64_t>(batch) * tokens * Geometry::QHeads * D;
    }
    {
        const std::int64_t stat_row =
            static_cast<std::int64_t>(batch) * Geometry::QHeads * tokens * gridDim.y;
        partial_acc += stat_row * D;
        partial_m += stat_row;
        partial_l += stat_row;
    }
    int valid_tokens = tokens;
    if (valid_columns != nullptr) {
        const int remaining = valid_columns[batch] - column_begin;
        valid_tokens        = remaining <= 0 ? 0 : (remaining < tokens ? remaining : tokens);
    }
    const std::int32_t* block_table =
        block_tables +
        static_cast<std::int64_t>(table_rows == nullptr ? 0 : table_rows[batch]) * table_stride;
    int window = 0;
    if (valid_tokens > 0) {
        const int first_pos = pos[0];
        const int last_pos  = pos[valid_tokens - 1];
        if (first_pos >= 0 && last_pos >= 0 && last_pos < logical_capacity) {
            window = last_pos + 1;
        }
    }
    int begin = 0;
    int end   = 0;
    if (window > 0) {
        const int active =
            causal_small_t_active_splits<Geometry, true>(window, static_cast<int>(gridDim.y), tokens);
        if (split >= active) { return; }
        const int chunk = (((window + active - 1) / active) + kDirectKeys - 1) & ~(kDirectKeys - 1);
        begin           = split * chunk;
        end             = begin + chunk < window ? begin + chunk : window;
    }

    const int r        = (lane & 3) + ((lane & 16) ? 4 : 0);
    const int qp       = (lane >> 2) & 3;
    const int hi       = (lane >> 4) & 1;
    const int c_row_lo = (hi << 2) | (lane & 1);
    const int row_lo   = c_row_lo;
    const int row_hi   = row_lo + 2;
    const int pos_lo   = row_lo < valid_tokens * G ? pos[row_lo / G] : -1;
    const int pos_hi   = row_hi < valid_tokens * G ? pos[row_hi / G] : -1;

    float o[8][8];
#pragma unroll
    for (int cb = 0; cb < 8; ++cb) {
#pragma unroll
        for (int i = 0; i < 8; ++i) { o[cb][i] = 0.0f; }
    }
    float m_lo = kNegInf, m_hi = kNegInf, l_lo = 0.0f, l_hi = 0.0f;

    if (begin < end) {
        direct_fill_q_rows<Geometry>(q, q_prepared, q_s, kvh, 0, 8, rows, lane, scale_log2);

        int warp_min = lane < 8 ? (lane < valid_tokens * G ? pos[lane / G] : -1) : 0x7fffffff;
        int max_pos  = lane < valid_tokens ? pos[lane] : -1;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            warp_min = min(warp_min, __shfl_xor_sync(Full, warp_min, off));
            max_pos  = max(max_pos, __shfl_xor_sync(Full, max_pos, off));
        }

        const int b_sel  = qp * 4 + (lane & 3) + hi * 16;
        const int c_sel  = qp * 4 + (((lane >> 1) & 1) << 1);
        const int v_dim  = (qp + 4 * hi) * 16;
        const half* qrow = q_s + r * kDirectQStride;
        const half* prow = p_w + r * kDirectPStride;

        for (int k0 = begin; k0 < end; k0 += kDirectKeys) {
            const int page = block_table[k0 >> kPagedKVPageShift];
            const int off0 = k0 & kPagedKVPageMask;

            // ---- QK^T: this lane's key, sixteen 16-dim chunks.
            const int k_off = off0 + b_sel;
            const std::int8_t* kptr =
                k_pages + paged_kv_element_offset<D, KVHeads>(page, kvh, k_off, 0);
            const uint2 k_sc        = *reinterpret_cast<const uint2*>(
                k_scale + kv_cache_int8_quant_scale_index<Geometry>(page, kvh, 0, k_off));
            float s[8]  = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
            float s2[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
            for (int g = 0; g < 4; ++g) {
                uint4 codes[4];
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    codes[j] = __ldg(reinterpret_cast<const uint4*>(kptr + (g * 4 + j) * 16));
                }
                const half sc = reinterpret_cast<const half*>(&k_sc)[g];
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int c = g * 4 + j;
                    uint4 lo, hi16;
                    direct_dequant16(codes[j], sc, lo, hi16);
                    const uint4 a0 = *reinterpret_cast<const uint4*>(qrow + c * 16);
                    const uint4 a1 = *reinterpret_cast<const uint4*>(qrow + c * 16 + 8);
                    volta_mma_qp_n(s, a0.x, a0.y, lo.x, lo.y);
                    volta_mma_qp_n(s2, a0.z, a0.w, lo.z, lo.w);
                    volta_mma_qp_n(s, a1.x, a1.y, hi16.x, hi16.y);
                    volta_mma_qp_n(s2, a1.z, a1.w, hi16.z, hi16.w);
                }
            }
#pragma unroll
            for (int i = 0; i < 8; ++i) { s[i] += s2[i]; }

            const bool masked = k0 + kDirectKeys - 1 > warp_min;
            direct_softmax_step(s, o, m_lo, m_hi, l_lo, l_hi, masked, k0 + c_sel, pos_lo, pos_hi);
#pragma unroll
            for (int i = 0; i < 8; i += 2) {
                const int row = (i & 2) | c_row_lo;
                const int col = c_sel + ((i >> 2) << 4);
                *reinterpret_cast<half2*>(p_w + row * kDirectPStride + col) =
                    __floats2half2_rn(s[i], s[i + 1]);
            }
            __syncwarp();
            uint4 pa[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) { pa[j] = *reinterpret_cast<const uint4*>(prow + j * 8); }
            __syncwarp();

            // ---- P.V: four keys per mma slice, this lane's 16-dim block in both 128-dim halves.
#pragma unroll
            for (int pair = 0; pair < 4; ++pair) {
                uint4 vc[2][2];
                uint2 v_sc[2];
#pragma unroll
                for (int u = 0; u < 2; ++u) {
                    const int v_off = off0 + (pair * 2 + u) * 4 + (lane & 3);
                    const std::int8_t* vptr =
                        v_pages + paged_kv_element_offset<D, KVHeads>(page, kvh, v_off, v_dim);
                    vc[u][0] = __ldg(reinterpret_cast<const uint4*>(vptr));
                    vc[u][1] = __ldg(reinterpret_cast<const uint4*>(vptr + 128));
                    v_sc[u]  = *reinterpret_cast<const uint2*>(
                        v_scale + kv_cache_int8_quant_scale_index<Geometry>(page, kvh, 0, v_off));
                }
#pragma unroll
                for (int u = 0; u < 2; ++u) {
                    const int ks      = pair * 2 + u;
                    const bool live   = k0 + ks * 4 + (lane & 3) <= max_pos;
                    const unsigned a0 = (ks & 1) ? pa[ks >> 1].z : pa[ks >> 1].x;
                    const unsigned a1 = (ks & 1) ? pa[ks >> 1].w : pa[ks >> 1].y;
#pragma unroll
                    for (int cp = 0; cp < 2; ++cp) {
                        // Unwritten slots of the last page may hold any scale bits; keep them out
                        // of the accumulators even though their P is zero.
                        uint4 lo = make_uint4(0, 0, 0, 0), hi16 = lo;
                        // Branchless: a per-lane branch here costs ~22% of the kernel.
                        const half raw_sc = reinterpret_cast<const half*>(&v_sc[u])[cp * 2 + hi];
                        direct_dequant16(vc[u][cp], live ? raw_sc : __ushort_as_half(0), lo, hi16);
                        direct_mma_row_row(o[cp * 4 + 0], a0, a1, lo.x, lo.y);
                        direct_mma_row_row(o[cp * 4 + 1], a0, a1, lo.z, lo.w);
                        direct_mma_row_row(o[cp * 4 + 2], a0, a1, hi16.x, hi16.y);
                        direct_mma_row_row(o[cp * 4 + 3], a0, a1, hi16.z, hi16.w);
                    }
                }
            }
        }
        l_lo = direct_sum3(l_lo);
        l_hi = direct_sum3(l_hi);
    }

    direct_write_partials<Geometry>(o, m_lo, m_hi, l_lo, l_hi, kvh, split, 0, rows, tokens, lane,
                                    partial_acc, partial_m, partial_l);
}

// Two or more row tiles. Register-direct warps would each fetch and dequantize the same K and V,
// which at T=4..8 moves 3-6x the cache through HBM and is bound by that redundant traffic (T=8 at
// 45K: 158 GB/s of cache for ~950 GB/s of traffic). Here the CTA's warps dequantize each 32-key
// tile once into one shared FP16 plane (K, then V in the same 16 KB), and every row tile takes
// its mma B fragments from it.
inline constexpr int kStagedKeyStride = kCausalHeadDim + 8; // halves; keys step 4 banks

constexpr int causal_attention_staged_i8_smem_bytes(int tiles) {
    return causal_attention_direct_i8_smem_bytes(tiles) + kDirectKeys * kStagedKeyStride * 2;
}

// Five and six row tiles (6-8 tokens) are bounded to two CTAs per SM. Unbounded, the compiler
// spends 215-242 registers a thread, so only one 6-warp CTA is resident and the barrier-separated
// phases have nothing to hide behind; at two CTAs (at most 170 registers, no spill) the 8-token
// kernel measured 11-28% faster from 8K to 172K keys (305 -> 220 us at 24K, 829 -> 674 at 90K).
// Two CTAs of the six-tile form use 92 KB of shared memory, inside the 96 KB carve-out.
template <int Warps>
inline constexpr int kStagedMinBlocks = Warps >= 5 ? 2 : 1;

#define __syncthreads() __syncthreads()
template <typename Geometry, int Warps>
__global__ void __launch_bounds__(Warps * 32, kStagedMinBlocks<Warps>) causal_attention_staged_i8_kernel(
    const __nv_bfloat16* __restrict__ q, const half* __restrict__ q_prepared,
    const std::int32_t* __restrict__ pos, const std::int8_t* __restrict__ k_pages,
    const std::int8_t* __restrict__ v_pages,
    const __half* __restrict__ k_scale, const __half* __restrict__ v_scale,
    const std::int32_t* __restrict__ block_tables, const std::int32_t* __restrict__ valid_columns,
    const std::int32_t* __restrict__ table_rows, std::int32_t table_stride, int tokens,
    std::int32_t full_width, std::int32_t column_begin, std::int32_t logical_capacity,
    float scale_log2,
    float* __restrict__ partial_acc, float* __restrict__ partial_m,
    float* __restrict__ partial_l) {
    constexpr int D         = kCausalHeadDim;
    constexpr int G         = Geometry::GroupSize;
    constexpr int KVHeads   = Geometry::KVHeads;
    constexpr float kNegInf = -INFINITY;
    constexpr unsigned Full = 0xffffffffu;
    static_assert(D == 256 && kKVCacheInt8Group == 64);

    extern __shared__ __align__(16) unsigned char direct_smem[];
    half* q_s  = reinterpret_cast<half*>(direct_smem);
    half* p_s  = q_s + Warps * 8 * kDirectQStride;
    half* kv_s = p_s + Warps * 8 * kDirectPStride;

    const int kvh   = static_cast<int>(blockIdx.x);
    const int split = static_cast<int>(blockIdx.y);
    const int tid   = static_cast<int>(threadIdx.x);
    const int lane  = tid & 31;
    const int warp  = tid >> 5;
    const int rows  = tokens * G;
    const int row0  = warp * 8;

    // grid.z is the stream; each has its own columns, block-table row and partial planes.
    const int batch                = static_cast<int>(blockIdx.z);
    const std::int64_t column_base = column_begin + static_cast<std::int64_t>(batch) * full_width;
    q += static_cast<std::int64_t>(D) * Geometry::QHeads * column_base;
    pos += column_base;
    if (q_prepared != nullptr) {
        q_prepared += static_cast<std::int64_t>(batch) * tokens * Geometry::QHeads * D;
    }
    {
        const std::int64_t stat_row =
            static_cast<std::int64_t>(batch) * Geometry::QHeads * tokens * gridDim.y;
        partial_acc += stat_row * D;
        partial_m += stat_row;
        partial_l += stat_row;
    }
    int valid_tokens = tokens;
    if (valid_columns != nullptr) {
        const int remaining = valid_columns[batch] - column_begin;
        valid_tokens        = remaining <= 0 ? 0 : (remaining < tokens ? remaining : tokens);
    }
    const std::int32_t* block_table =
        block_tables +
        static_cast<std::int64_t>(table_rows == nullptr ? 0 : table_rows[batch]) * table_stride;
    int window = 0;
    if (valid_tokens > 0) {
        const int first_pos = pos[0];
        const int last_pos  = pos[valid_tokens - 1];
        if (first_pos >= 0 && last_pos >= 0 && last_pos < logical_capacity) {
            window = last_pos + 1;
        }
    }
    int begin = 0;
    int end   = 0;
    if (window > 0) {
        const int active =
            causal_small_t_active_splits<Geometry, true>(window, static_cast<int>(gridDim.y), tokens);
        if (split >= active) { return; }
        const int chunk = (((window + active - 1) / active) + kDirectKeys - 1) & ~(kDirectKeys - 1);
        begin           = split * chunk;
        end             = begin + chunk < window ? begin + chunk : window;
    }

    const int r        = (lane & 3) + ((lane & 16) ? 4 : 0);
    const int qp       = (lane >> 2) & 3;
    const int hi       = (lane >> 4) & 1;
    const int c_row_lo = (hi << 2) | (lane & 1);
    const int row_lo   = row0 + c_row_lo;
    const int row_hi   = row_lo + 2;
    const int pos_lo   = row_lo < valid_tokens * G ? pos[row_lo / G] : -1;
    const int pos_hi   = row_hi < valid_tokens * G ? pos[row_hi / G] : -1;

    float o[8][8];
#pragma unroll
    for (int cb = 0; cb < 8; ++cb) {
#pragma unroll
        for (int i = 0; i < 8; ++i) { o[cb][i] = 0.0f; }
    }
    float m_lo = kNegInf, m_hi = kNegInf, l_lo = 0.0f, l_hi = 0.0f;

    if (begin < end) {
        direct_fill_q_rows<Geometry>(q, q_prepared, q_s, kvh, row0, 8, rows, lane, scale_log2);

        int warp_min = lane < 8 ? (row0 + lane < valid_tokens * G ? pos[(row0 + lane) / G] : -1)
                                : 0x7fffffff;
        int max_pos  = lane < valid_tokens ? pos[lane] : -1;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            warp_min = min(warp_min, __shfl_xor_sync(Full, warp_min, off));
            max_pos  = max(max_pos, __shfl_xor_sync(Full, max_pos, off));
        }

        const int b_sel  = qp * 4 + (lane & 3) + hi * 16;
        const int c_sel  = qp * 4 + (((lane >> 1) & 1) << 1);
        const int v_dim  = (qp + 4 * hi) * 16;
        const half* qrow = q_s + (row0 + r) * kDirectQStride;
        half* p_w        = p_s + row0 * kDirectPStride;
        const half* prow = p_w + r * kDirectPStride;
        const half* krow = kv_s + b_sel * kStagedKeyStride;

        // The 32 x 256 code tile is 512 16-byte chunks, dealt round-robin to the CTA's threads so a
        // warp's step covers two whole keys (four 128-byte lines). Loads are issued a phase ahead
        // of their dequantization: V's while QK^T runs, the next tile's K while P.V runs, so the
        // HBM round trip never sits on the barrier path.
        constexpr int kChunks = kDirectKeys * 16;
        constexpr int kMine   = (kChunks + Warps * 32 - 1) / (Warps * 32);
        uint4 codes[kMine];
        half code_scale[kMine];
        auto fetch = [&](const std::int8_t* pages, const __half* scales, int page, int off0) {
#pragma unroll
            for (int j = 0; j < kMine; ++j) {
                const int c     = min(tid + j * Warps * 32, kChunks - 1);
                const int key   = c >> 4;
                const int piece = c & 15;
                codes[j] = __ldg(reinterpret_cast<const uint4*>(
                    pages + paged_kv_element_offset<D, KVHeads>(page, kvh, off0 + key, piece * 16)));
                code_scale[j] = scales[kv_cache_int8_quant_scale_index<Geometry>(
                    page, kvh, piece >> 2, off0 + key)];
            }
        };
        auto commit = [&](int k0, bool mask_dead) {
#pragma unroll
            for (int j = 0; j < kMine; ++j) {
                const int c     = min(tid + j * Warps * 32, kChunks - 1);
                const int key   = c >> 4;
                const int piece = c & 15;
                half sc         = code_scale[j];
                // Unwritten slots of the last page may hold any scale bits.
                if (mask_dead) { sc = k0 + key <= max_pos ? sc : __ushort_as_half(0); }
                uint4 lo, hi16;
                direct_dequant16(codes[j], sc, lo, hi16);
                half* dst = kv_s + key * kStagedKeyStride + piece * 16;
                *reinterpret_cast<uint4*>(dst)     = lo;
                *reinterpret_cast<uint4*>(dst + 8) = hi16;
            }
        };

        fetch(k_pages, k_scale, block_table[begin >> kPagedKVPageShift], begin & kPagedKVPageMask);
        for (int k0 = begin; k0 < end; k0 += kDirectKeys) {
            const int page = block_table[k0 >> kPagedKVPageShift];
            const int off0 = k0 & kPagedKVPageMask;

            commit(k0, false);
            __syncthreads();
            fetch(v_pages, v_scale, page, off0);

            // ---- QK^T: this lane's key, sixteen 16-dim chunks.
            float s[8]  = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
            float s2[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
            for (int c = 0; c < 16; ++c) {
                const uint4 lo   = *reinterpret_cast<const uint4*>(krow + c * 16);
                const uint4 hi16 = *reinterpret_cast<const uint4*>(krow + c * 16 + 8);
                const uint4 a0   = *reinterpret_cast<const uint4*>(qrow + c * 16);
                const uint4 a1   = *reinterpret_cast<const uint4*>(qrow + c * 16 + 8);
                volta_mma_qp_n(s, a0.x, a0.y, lo.x, lo.y);
                volta_mma_qp_n(s2, a0.z, a0.w, lo.z, lo.w);
                volta_mma_qp_n(s, a1.x, a1.y, hi16.x, hi16.y);
                volta_mma_qp_n(s2, a1.z, a1.w, hi16.z, hi16.w);
            }
#pragma unroll
            for (int i = 0; i < 8; ++i) { s[i] += s2[i]; }

            const bool masked = k0 + kDirectKeys - 1 > warp_min;
            direct_softmax_step(s, o, m_lo, m_hi, l_lo, l_hi, masked, k0 + c_sel, pos_lo, pos_hi);
#pragma unroll
            for (int i = 0; i < 8; i += 2) {
                const int row = (i & 2) | c_row_lo;
                const int col = c_sel + ((i >> 2) << 4);
                *reinterpret_cast<half2*>(p_w + row * kDirectPStride + col) =
                    __floats2half2_rn(s[i], s[i + 1]);
            }
            __syncthreads(); // every warp is done with K; also orders this warp's P stores
            uint4 pa[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) { pa[j] = *reinterpret_cast<const uint4*>(prow + j * 8); }

            commit(k0, true);
            __syncthreads();
            if (k0 + kDirectKeys < end) {
                const int next = k0 + kDirectKeys;
                fetch(k_pages, k_scale, block_table[next >> kPagedKVPageShift],
                      next & kPagedKVPageMask);
            }

            // ---- P.V: four keys per mma slice, this lane's 16-dim block in both 128-dim halves.
#pragma unroll
            for (int ks = 0; ks < 8; ++ks) {
                const half* vrow  = kv_s + (ks * 4 + (lane & 3)) * kStagedKeyStride + v_dim;
                const unsigned a0 = (ks & 1) ? pa[ks >> 1].z : pa[ks >> 1].x;
                const unsigned a1 = (ks & 1) ? pa[ks >> 1].w : pa[ks >> 1].y;
#pragma unroll
                for (int cp = 0; cp < 2; ++cp) {
                    const uint4 lo   = *reinterpret_cast<const uint4*>(vrow + cp * 128);
                    const uint4 hi16 = *reinterpret_cast<const uint4*>(vrow + cp * 128 + 8);
                    direct_mma_row_row(o[cp * 4 + 0], a0, a1, lo.x, lo.y);
                    direct_mma_row_row(o[cp * 4 + 1], a0, a1, lo.z, lo.w);
                    direct_mma_row_row(o[cp * 4 + 2], a0, a1, hi16.x, hi16.y);
                    direct_mma_row_row(o[cp * 4 + 3], a0, a1, hi16.z, hi16.w);
                }
            }
            __syncthreads(); // every warp is done with V
        }
        l_lo = direct_sum3(l_lo);
        l_hi = direct_sum3(l_hi);
    }

    direct_write_partials<Geometry>(o, m_lo, m_hi, l_lo, l_hi, kvh, split, row0, rows, tokens,
                                    lane, partial_acc, partial_m, partial_l);
}

constexpr int kDirectMaxTiles = 6;

// Call once before the first launch (the kernels use dynamic shared memory).
template <typename Geometry>
void causal_attention_direct_i8_configure() {
    CUDA_CHECK(cudaFuncSetAttribute(causal_attention_direct_i8_kernel<Geometry>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    causal_attention_direct_i8_smem_bytes(1)));
#define NINFER_STAGED_CONFIGURE(TILES)                                                           \
    CUDA_CHECK(cudaFuncSetAttribute(causal_attention_staged_i8_kernel<Geometry, TILES>,          \
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,                 \
                                    causal_attention_staged_i8_smem_bytes(TILES)));              \
    if (kStagedMinBlocks<TILES> > 1) {                                                           \
        CUDA_CHECK(cudaFuncSetAttribute(causal_attention_staged_i8_kernel<Geometry, TILES>,      \
                                        cudaFuncAttributePreferredSharedMemoryCarveout, 100));   \
    }
    NINFER_STAGED_CONFIGURE(2) NINFER_STAGED_CONFIGURE(3) NINFER_STAGED_CONFIGURE(4)
    NINFER_STAGED_CONFIGURE(5) NINFER_STAGED_CONFIGURE(6)
#undef NINFER_STAGED_CONFIGURE
}

// grid = (KV heads, splits, streams); tokens <= kCausalDirectMaxTokens per stream; q_prepared is
// the causal_attention_direct_prepare_q plane or nullptr.
template <typename Geometry>
void causal_attention_direct_i8_launch(dim3 grid, const __nv_bfloat16* q, const half* q_prepared,
                                       const std::int32_t* pos, const std::int8_t* k_pages,
                                       const std::int8_t* v_pages, const __half* k_scale,
                                       const __half* v_scale, const std::int32_t* block_tables,
                                       const std::int32_t* valid_columns,
                                       const std::int32_t* table_rows, std::int32_t table_stride,
                                       int tokens, std::int32_t full_width,
                                       std::int32_t column_begin,
                                       std::int32_t logical_capacity, float scale,
                                       float* partial_acc, float* partial_m, float* partial_l,
                                       cudaStream_t stream) {
    static_assert(kCausalDirectMaxTokens * Geometry::GroupSize <= kDirectMaxTiles * 8);
    const float scale_log2 = scale * 1.4426950408889634f;
    const int tiles        = (tokens * Geometry::GroupSize + 7) / 8;
    if (tiles == 1) {
        causal_attention_direct_i8_kernel<Geometry>
            <<<grid, 32, causal_attention_direct_i8_smem_bytes(1), stream>>>(
                q, q_prepared, pos, k_pages, v_pages, k_scale, v_scale, block_tables,
                valid_columns, table_rows, table_stride, tokens, full_width, column_begin,
                logical_capacity, scale_log2, partial_acc, partial_m, partial_l);
        return;
    }
#define NINFER_STAGED_LAUNCH(TILES)                                                              \
    if (tiles == TILES) {                                                                        \
        causal_attention_staged_i8_kernel<Geometry, TILES>                                       \
            <<<grid, TILES * 32, causal_attention_staged_i8_smem_bytes(TILES), stream>>>(        \
                q, q_prepared, pos, k_pages, v_pages, k_scale, v_scale, block_tables,            \
                valid_columns, table_rows, table_stride, tokens, full_width, column_begin,       \
                logical_capacity, scale_log2, partial_acc, partial_m, partial_l);                \
    }
    NINFER_STAGED_LAUNCH(2) NINFER_STAGED_LAUNCH(3) NINFER_STAGED_LAUNCH(4)
    NINFER_STAGED_LAUNCH(5) NINFER_STAGED_LAUNCH(6)
#undef NINFER_STAGED_LAUNCH
}

#endif // sm_70

} // namespace ninfer::ops
