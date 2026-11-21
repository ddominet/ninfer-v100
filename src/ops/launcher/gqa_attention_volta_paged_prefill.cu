// Fused Volta prefill attention over the paged INT8-G64 KV cache.
//
// Measured limits at 2048 tokens x 43K keys (54 ms, ~42 TFLOPS), by ablation on a V100:
//   - every mma removed: 47.6 ms. The tensor cores are not the bound.
//   - QK^T loop removed: 27.5 ms; PV loop removed: 38.0 ms; staging removed: 43.2 ms.
//   - QK^T is co-bound by (a) shared-memory loads (one K load of 32 distinct key rows per two
//     mma) and (b) mma dependency latency: each S accumulator is a chain of 32 dependent mma per
//     32-key tile, ~140 cycles per step.
//   - A split-role variant (three QK warps owning 32 rows each in the plain m32n8k4 mapping, one Q
//     load per eight mma, P handed to the PV warps through shared memory) fixes (a) but was
//     57 ms with two chains per key fragment and 88 ms with one: it only moves the bound to (b).
//     More chains need registers the PV accumulators (64 per thread, live across tiles) already
//     hold, more QK warps need a partial-S exchange that does not fit in the 96 KiB of shared
//     memory, and FP16 PV accumulators are not precise enough at long context.
//
// One CTA owns TokenRows consecutive query tokens for all Group query heads that share one KV
// head; each warp owns 8 (token, head) rows. The CTA walks the visible key range in 32-key tiles.
// A tile never crosses a 64-token page, so its K and V codes are each one contiguous 8 KiB block;
// the next tile's codes are prefetched into registers while the current one is consumed. Codes are
// dequantized once into row-major shared memory, then every warp runs QK^T (B = K, .col) and P.V
// (B = V, .row) on mma.m8n8k4 with an FP32 online softmax. Compared with the gather + llama.cpp
// route this reads INT8 instead of FP16 keys, needs no dense mask and no contiguous FP16 copy of
// the prefix, accumulates in FP32 throughout, and writes BF16 output directly.
//
// QK fragment column c (0..7) of quadpair qp maps to key qp*4 + (c&3) + (c>>2)*16; PV column c of
// output block cb maps to dim (cb>>1)*64 + qp*8 + (c>>2)*32 + (cb&1)*4 + (c&3), so one 16-byte
// load feeds two blocks. With the strides below every B-fragment shared load is conflict-free.

#include "core/device.h"
#include "core/tensor.h"
#include "ops/common/volta_mma.cuh"
#include "ops/kernel/paged_kv_address.cuh"
#include "ops/kv_cache/hadamard_d256.cuh"
#include "ops/kv_cache/int8_g64_codec.cuh"
#include "ops/softmax_attention/dense/causal_cache/launch.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr int kPfD       = 256;
constexpr int kPfKeys    = 32;
constexpr int kPfQStride = kPfD + 8;  // 132 words: rows step 4 banks
constexpr int kPfKStride = kPfD + 8;
constexpr int kPfVStride = kPfD + 16; // 136 words: rows step 8 banks
constexpr int kPfPStride = kPfKeys + 8;
constexpr int kPfChunks  = kPfKeys * kPfD / 16; // 16-code chunks per operand tile

static_assert(kPagedKVPageSize % kPfKeys == 0);

template <int Group>
struct PfShape {
    static constexpr int kTokens      = Group <= 6 ? 16 : 8;
    static constexpr int kTokenGroups = kTokens / 8;
    static constexpr int kWarps       = Group * kTokenGroups;
    static constexpr int kThreads     = kWarps * 32;
    static constexpr int kRows        = kWarps * 8;
    static constexpr int kStageIters  = (kPfChunks + kThreads - 1) / kThreads; // per operand
    static constexpr int kQBytes      = kRows * kPfQStride * 2;
    static constexpr int kKBytes      = kPfKeys * kPfKStride * 2;
    static constexpr int kVBytes      = kPfKeys * kPfVStride * 2;
    static constexpr int kPBytes      = kWarps * 8 * kPfPStride * 2;
    static constexpr int kSmem        = kQBytes + kKBytes + kVBytes + kPBytes;
    static_assert(kSmem <= 96 * 1024);
};

__device__ __forceinline__ void pf_mma_row_row(float (&d)[8], unsigned a0, unsigned a1,
                                               unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m8n8k4.row.row.f32.f16.f16.f32 "
                 "{%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9}, {%10, %11}, "
                 "{%0, %1, %2, %3, %4, %5, %6, %7};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]),
                   "+f"(d[6]), "+f"(d[7])
                 : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
}

// 16 signed codes -> 16 halves (code * scale), exact: 0x64XX is 1024 + XX, so subtracting
// 1152 recovers the signed code before the single rounding multiply.
// Qwen3.6-27B (G=6, 12 warps): split byte_perm from hsub/hmul so the ALU pipe can dual-issue
// more of the dequant against the other warps' HMMA (dequant is ~15% of the fused FA).
__device__ __forceinline__ void pf_dequant16(uint4 codes, half scale, uint4& lo, uint4& hi) {
    const half2 bias  = __half2half2(__ushort_as_half(0x6480));
    const half2 s2    = __half2half2(scale);
    const unsigned w[4] = {codes.x ^ 0x80808080u, codes.y ^ 0x80808080u, codes.z ^ 0x80808080u,
                           codes.w ^ 0x80808080u};
    unsigned p0[4], p1[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        p0[i] = __byte_perm(w[i], 0x64646464u, 0x4140);
        p1[i] = __byte_perm(w[i], 0x64646464u, 0x4342);
    }
    unsigned out[8];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        half2 h0 = __hmul2(__hsub2(*reinterpret_cast<const half2*>(&p0[i]), bias), s2);
        half2 h1 = __hmul2(__hsub2(*reinterpret_cast<const half2*>(&p1[i]), bias), s2);
        out[2 * i]     = *reinterpret_cast<const unsigned*>(&h0);
        out[2 * i + 1] = *reinterpret_cast<const unsigned*>(&h1);
    }
    lo = make_uint4(out[0], out[1], out[2], out[3]);
    hi = make_uint4(out[4], out[5], out[6], out[7]);
}

// NINFER_PAGED_FA_FAST_EXP=1 selects the bit-cast 2^x. It is piecewise linear in the fraction, so
// each attention weight carries up to +6.15% relative error that normalization does not cancel
// (it changes generated text at 90K context), and re-measured it is no faster than the exact
// instruction: 43.1 vs 42.2-43.6 TFLOPS at 2048x43K. Kept only for A/B.
#ifndef NINFER_PAGED_FA_FAST_EXP
#define NINFER_PAGED_FA_FAST_EXP 0
#endif

__device__ __forceinline__ float pf_ex2(float x) {
#if NINFER_PAGED_FA_FAST_EXP
    x = fminf(fmaxf(x, -126.0f), 126.0f);
    const int i = static_cast<int>(x * 8388608.0f) + (127 << 23);
    return __int_as_float(i);
#else
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
#endif
}

__device__ __forceinline__ float pf_max3(float v) {
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 2));
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 4));
    return fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 8));
}

__device__ __forceinline__ float pf_sum3(float v) {
    v += __shfl_xor_sync(0xffffffffu, v, 2);
    v += __shfl_xor_sync(0xffffffffu, v, 4);
    return v + __shfl_xor_sync(0xffffffffu, v, 8);
}

template <int KVHeads, int Group>
__global__ void __launch_bounds__(PfShape<Group>::kThreads, 1) volta_paged_i8_prefill_kernel(
        const __nv_bfloat16* __restrict__ q, const std::int32_t* __restrict__ positions, int width,
        const std::int8_t* __restrict__ k_pages, const std::int8_t* __restrict__ v_pages,
        const half* __restrict__ k_scales, const half* __restrict__ v_scales,
        const std::int32_t* __restrict__ block_tables, const std::int32_t* __restrict__ table_rows,
        int logical_pages, float scale_log2, __nv_bfloat16* __restrict__ out) {
    using S                 = PfShape<Group>;
    constexpr int kQHeads   = KVHeads * Group;
    constexpr float kNegInf = -INFINITY;

    extern __shared__ __align__(16) unsigned char pf_smem[];
    half* q_s   = reinterpret_cast<half*>(pf_smem);
    half* k_s   = reinterpret_cast<half*>(pf_smem + S::kQBytes);
    half* v_s   = reinterpret_cast<half*>(pf_smem + S::kQBytes + S::kKBytes);
    half* p_all = reinterpret_cast<half*>(pf_smem + S::kQBytes + S::kKBytes + S::kVBytes);

    // Later tiles see more keys; issue them first so the tail wave is short.
    const int tile     = gridDim.x - 1 - blockIdx.x;
    const int kvh      = blockIdx.y;
    const int tid      = threadIdx.x;
    const int lane     = tid & 31;
    const int warp     = tid >> 5;
    const int qh       = kvh * Group + warp % Group;
    const int tok_cta  = tile * S::kTokens;
    const int tok_base = tok_cta + (warp / Group) * 8;
    const std::int32_t* block_table =
            block_tables + static_cast<std::int64_t>(table_rows[0]) * logical_pages;

    for (int rr = 0; rr < 8; ++rr) {
        const int t = tok_base + rr;
        float vals[8];
        if (t < width) {
            const __nv_bfloat16* src = q + (static_cast<std::int64_t>(t) * kQHeads + qh) * kPfD;
#pragma unroll
            for (int p = 0; p < 8; ++p) { vals[p] = __bfloat162float(src[lane + 32 * p]); }
            normalized_hadamard_d256_inplace(vals, lane);
        } else {
#pragma unroll
            for (int p = 0; p < 8; ++p) { vals[p] = 0.0f; }
        }
        half* dst = q_s + (warp * 8 + rr) * kPfQStride;
#pragma unroll
        for (int p = 0; p < 8; ++p) { dst[lane + 32 * p] = __float2half_rn(vals[p] * scale_log2); }
    }

    const int c_row_lo = ((lane & 16) ? 4 : 0) | (lane & 1);
    const int pos_lo   = tok_base + c_row_lo < width ? positions[tok_base + c_row_lo] : -1;
    const int pos_hi   = tok_base + c_row_lo + 2 < width ? positions[tok_base + c_row_lo + 2] : -1;

    int warp_max = (lane < 8 && tok_base + lane < width) ? positions[tok_base + lane] : -1;
    int warp_min = (lane < 8 && tok_base + lane < width) ? positions[tok_base + lane] : INT32_MAX;
    int cta_max  = (lane < S::kTokens && tok_cta + lane < width) ? positions[tok_cta + lane] : -1;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        warp_max = max(warp_max, __shfl_xor_sync(0xffffffffu, warp_max, off));
        warp_min = min(warp_min, __shfl_xor_sync(0xffffffffu, warp_min, off));
        cta_max  = max(cta_max, __shfl_xor_sync(0xffffffffu, cta_max, off));
    }
    const int cta_end = cta_max + 1;
    const int n_tiles = (cta_end + kPfKeys - 1) / kPfKeys;

    const int r     = (lane & 3) + ((lane & 16) ? 4 : 0);
    const int qp    = (lane >> 2) & 3;
    const int b_sel = qp * 4 + (lane & 3) + ((lane & 16) ? 16 : 0); // B column -> key (QK)
    const int c_sel = qp * 4 + (((lane >> 1) & 1) << 1);             // C column base
    half* p_s       = p_all + warp * 8 * kPfPStride;
    const half* qrow = q_s + (warp * 8 + r) * kPfQStride;
    const half* krow = k_s + b_sel * kPfKStride;
    const half* prow = p_s + r * kPfPStride;
    const half* vcol = v_s + (lane & 3) * kPfVStride + qp * 8 + ((lane & 16) ? 32 : 0);

    uint4 codes[S::kStageIters];
    half scales[S::kStageIters];
    auto load_half = [&](int kt, bool is_v) {
        const int k0     = kt * kPfKeys;
        const int page   = block_table[k0 >> kPagedKVPageShift];
        const int off0   = k0 & kPagedKVPageMask;
        const auto* src  = reinterpret_cast<const uint4*>(
                (is_v ? v_pages : k_pages) + paged_kv_element_offset<kPfD, KVHeads>(page, kvh, off0, 0));
        const half* sc   = (is_v ? v_scales : k_scales) +
                         paged_kv_element_offset<kKVCacheInt8Groups, KVHeads>(page, kvh, off0, 0);
#pragma unroll
        for (int it = 0; it < S::kStageIters; ++it) {
            const int c = tid + it * S::kThreads;
            if (c < kPfChunks) {
                codes[it]  = __ldg(src + c);
                scales[it] = sc[(c >> 4) * 4 + ((c & 15) >> 2)];
            }
        }
    };
    auto store_half = [&](int kt, bool is_v) {
        const int k0 = kt * kPfKeys;
#pragma unroll
        for (int it = 0; it < S::kStageIters; ++it) {
            const int c = tid + it * S::kThreads;
            if (c < kPfChunks) {
                const int key = c >> 4;
                uint4 lo = make_uint4(0, 0, 0, 0), hi = lo;
                if (k0 + key < cta_end) { pf_dequant16(codes[it], scales[it], lo, hi); }
                half* dst = (is_v ? v_s + key * kPfVStride : k_s + key * kPfKStride) + (c & 15) * 16;
                reinterpret_cast<uint4*>(dst)[0] = lo;
                reinterpret_cast<uint4*>(dst)[1] = hi;
            }
        }
    };

    float o[8][8];
#pragma unroll
    for (int cb = 0; cb < 8; ++cb) {
#pragma unroll
        for (int i = 0; i < 8; ++i) { o[cb][i] = 0.0f; }
    }
    float m_lo = kNegInf, m_hi = kNegInf, l_lo = 0.0f, l_hi = 0.0f;

    // Two barrier intervals per tile; the staging of one operand overlaps the MMAs on the other:
    //   [store V(kt), prefetch K(kt+1), QK(kt) + softmax]  |  [store K(kt+1), prefetch V(kt+1), PV(kt)]
    load_half(0, false);
    store_half(0, false);
    load_half(0, true);
    __syncthreads();
    for (int kt = 0; kt < n_tiles; ++kt) {
        const int k0      = kt * kPfKeys;
        const bool active = k0 <= warp_max;
        const bool more   = kt + 1 < n_tiles;
        store_half(kt, true);
        if (more) { load_half(kt + 1, false); }

        uint4 pa[4];
        if (active) {
        float s[8]  = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        float s2[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll 8
        for (int kk = 0; kk < kPfD / 8; ++kk) {
            const uint4 a = *reinterpret_cast<const uint4*>(qrow + kk * 8);
            const uint4 b = *reinterpret_cast<const uint4*>(krow + kk * 8);
            volta_mma_qp_n(s, a.x, a.y, b.x, b.y);
            volta_mma_qp_n(s2, a.z, a.w, b.z, b.w);
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) { s[i] += s2[i]; }

        if (k0 + kPfKeys - 1 > warp_min) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int key     = k0 + c_sel + (i & 1) + ((i >> 2) << 4);
                const int row_pos = (i & 2) ? pos_hi : pos_lo;
                s[i]              = key <= row_pos ? s[i] : kNegInf;
            }
        }
        float mx_lo = fmaxf(fmaxf(s[0], s[1]), fmaxf(s[4], s[5]));
        float mx_hi = fmaxf(fmaxf(s[2], s[3]), fmaxf(s[6], s[7]));
        // Stale running max: P = 2^(s - m) stays <= 2^8 (exact in FP16, FP32 accumulation) unless
        // some score passes m + 8, so the row reduction and O rescale run only on those tiles.
        if (__any_sync(0xffffffffu, mx_lo > m_lo + 8.0f || mx_hi > m_hi + 8.0f)) {
            const float new_lo   = fmaxf(m_lo, pf_max3(mx_lo));
            const float new_hi   = fmaxf(m_hi, pf_max3(mx_hi));
            const float alpha_lo = new_lo == kNegInf ? 1.0f : pf_ex2(m_lo - new_lo);
            const float alpha_hi = new_hi == kNegInf ? 1.0f : pf_ex2(m_hi - new_hi);
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
            s[i] = pf_ex2(s[i] - ((i & 2) ? base_hi : base_lo));
            if (i & 2) {
                l_hi += s[i];
            } else {
                l_lo += s[i];
            }
        }

#pragma unroll
        for (int i = 0; i < 8; i += 2) {
            const int row = (i & 2) | c_row_lo;
            const int col = c_sel + ((i >> 2) << 4);
            *reinterpret_cast<half2*>(p_s + row * kPfPStride + col) = __floats2half2_rn(s[i], s[i + 1]);
        }
        __syncwarp();
#pragma unroll
        for (int j = 0; j < 4; ++j) { pa[j] = *reinterpret_cast<const uint4*>(prow + j * 8); }
        }
        __syncthreads();

        if (more) {
            store_half(kt + 1, false);
            load_half(kt + 1, true);
        }
        if (active) {
#pragma unroll
        for (int ks = 0; ks < kPfKeys / 4; ++ks) {
            const unsigned a0 = (ks & 1) ? pa[ks >> 1].z : pa[ks >> 1].x;
            const unsigned a1 = (ks & 1) ? pa[ks >> 1].w : pa[ks >> 1].y;
            const half* vrow  = vcol + ks * 4 * kPfVStride;
#pragma unroll
            for (int cp = 0; cp < 4; ++cp) {
                const uint4 b = *reinterpret_cast<const uint4*>(vrow + cp * 64);
                pf_mma_row_row(o[2 * cp], a0, a1, b.x, b.y);
                pf_mma_row_row(o[2 * cp + 1], a0, a1, b.z, b.w);
            }
        }
        }
        __syncthreads();
    }

    l_lo               = pf_sum3(l_lo);
    l_hi               = pf_sum3(l_hi);
    const float inv_lo = l_lo > 0.0f ? 1.0f / l_lo : 0.0f;
    const float inv_hi = l_hi > 0.0f ? 1.0f / l_hi : 0.0f;
#pragma unroll
    for (int i = 0; i < 8; i += 2) {
        const int t = tok_base + ((i & 2) | c_row_lo);
        if (t >= width) { continue; }
        const float inv        = (i & 2) ? inv_hi : inv_lo;
        __nv_bfloat16* dst_row = out + (static_cast<std::int64_t>(t) * kQHeads + qh) * kPfD;
#pragma unroll
        for (int cb = 0; cb < 8; ++cb) {
            const int col = (cb >> 1) * 64 + qp * 8 + ((i >> 2) << 5) + (cb & 1) * 4 +
                            (((lane >> 1) & 1) << 1);
            *reinterpret_cast<__nv_bfloat162*>(dst_row + col) =
                    __floats2bfloat162_rn(o[cb][i] * inv, o[cb][i + 1] * inv);
        }
    }
}

template <int KVHeads, int Group>
void launch_paged_prefill(const Tensor& q, const Tensor& positions, const Tensor& table_rows,
                          float scale, PagedKVBatchLayerView cache, Tensor& out,
                          cudaStream_t stream) {
    using S          = PfShape<Group>;
    const int width  = q.ne[2];
    auto kernel      = volta_paged_i8_prefill_kernel<KVHeads, Group>;
    static bool once = [&] {
        CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        S::kSmem));
        return true;
    }();
    (void) once;
    const dim3 grid((width + S::kTokens - 1) / S::kTokens, KVHeads);
    kernel<<<grid, S::kThreads, S::kSmem, stream>>>(
            static_cast<const __nv_bfloat16*>(q.data),
            static_cast<const std::int32_t*>(positions.data), width,
            static_cast<const std::int8_t*>(cache.k_pages.data),
            static_cast<const std::int8_t*>(cache.v_pages.data),
            static_cast<const half*>(cache.k_scale_pages.data),
            static_cast<const half*>(cache.v_scale_pages.data),
            static_cast<const std::int32_t*>(cache.block_tables.data),
            static_cast<const std::int32_t*>(table_rows.data), cache.block_tables.ne[0],
            scale * 1.4426950408889634f, static_cast<__nv_bfloat16*>(out.data));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void causal_attention_volta_paged_i8_prefill_launch(const Tensor& q, const Tensor& positions,
                                                    const Tensor& table_rows, float scale,
                                                    PagedKVBatchLayerView cache, Tensor& out,
                                                    cudaStream_t stream) {
    if (q.ne[1] == 24) {
        launch_paged_prefill<4, 6>(q, positions, table_rows, scale, cache, out, stream);
        return;
    }
    if (q.ne[1] == 16) {
        launch_paged_prefill<2, 8>(q, positions, table_rows, scale, cache, out, stream);
        return;
    }
    throw std::invalid_argument("gqa_attention volta paged prefill: unsupported Q head geometry");
}

} // namespace ninfer::ops::detail
