#pragma once

#include "ops/common/math.cuh"
#include "ops/common/volta_mma.cuh"
#include "ops/common/warp.cuh"
#include "ops/softmax_attention/common/context_query.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math_constants.h>

namespace ninfer::ops {

inline constexpr int kSlidingWindowVoltaThreads = kContextQueryHeadDim;

// DFlash2's production K=7 round is an eight-column masked block.  The original sm_70
// fallback launched one CTA per (query token, query head), so all eight CTAs streamed the
// same 2K K/V window independently.  This split-KV path makes the proposal width the unit
// of work: one warp owns one query token, while the eight warps in a CTA share each K/V
// vector loaded from the cyclic cache.  Splitting the window supplies enough CTAs for an
// 80-SM V100 and preserves the existing partial/reduce contract.
inline constexpr int kSlidingWindowVoltaOctetThreads = 8 * 32;

template <int Tokens>
__launch_bounds__(kSlidingWindowVoltaOctetThreads, 2) __global__ void
sliding_window_attention_volta_octet_partial_kernel(
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ query_k,
    const __nv_bfloat16* __restrict__ query_v, const std::int32_t* __restrict__ positions,
    const std::int32_t* __restrict__ valid_columns, const std::int32_t* __restrict__ lanes,
    const __nv_bfloat16* __restrict__ context_k, const __half* __restrict__ context_v,
    int padded_context, int max_context, int split_capacity, float scale,
    float* __restrict__ partial_acc, float* __restrict__ partial_m,
    float* __restrict__ partial_l) {
    // One warp a query token: eight for the full K=7 block, fewer for the narrowed blocks the
    // kept-draft choice drafts under load (a narrow block took the per-token fallback, which
    // streams the 2K window once per token: 146 us a call).
    static_assert(Tokens >= 1 && Tokens <= 8, "one warp per query token, at most eight");
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ == 700
    constexpr int D                 = kContextQueryHeadDim;
    constexpr int Window            = 2048;
    constexpr int KeyBlock          = 32;
    constexpr std::int64_t QElems   = static_cast<std::int64_t>(D) * kContextQueryQHeads * Tokens;
    constexpr std::int64_t KVElems  = static_cast<std::int64_t>(D) * kContextQueryKVHeads * Tokens;
    constexpr std::int64_t StatElems = static_cast<std::int64_t>(kContextQueryQHeads) * Tokens;
    constexpr float Log2E           = 1.4426950408889634074f;
    constexpr unsigned FullMask     = 0xffffffffu;

    // Context keys are consumed 32 at a time: the CTA stages one KeyBlock of K and V (widened to
    // FP32, rows padded to an odd stride so lane j reading row j is conflict-free) behind one
    // barrier, each warp scores its token against all 32 keys with one key per lane, updates its
    // running softmax once per block and accumulates P.V four dimensions per lane. The previous
    // form took one key per iteration behind two CTA barriers and a five-step shuffle reduction,
    // which held this ~40 MB per layer read near 35 GB/s.
    constexpr int RowStride = D + 1;
    __shared__ float key_s[KeyBlock * RowStride];
    __shared__ float value_s[KeyBlock * RowStride];
    __shared__ float query_s[Tokens * D];
    __shared__ float prob_s[Tokens * KeyBlock];

    const int q_head = static_cast<int>(blockIdx.x);
    const int split  = static_cast<int>(blockIdx.y);
    const int batch  = static_cast<int>(blockIdx.z);
    const int tid    = static_cast<int>(threadIdx.x);
    const int warp   = tid >> 5;
    const int lane   = tid & 31;
    const int token  = warp;
    const int valid  = valid_columns[batch];
    const int length = positions[static_cast<std::int64_t>(batch) * Tokens];

    if (q_head >= kContextQueryQHeads || split >= split_capacity || valid < 0 || valid > Tokens ||
        length < 0 || length > max_context) {
        return;
    }

    q += QElems * batch;
    query_k += KVElems * batch;
    query_v += KVElems * batch;
    positions += static_cast<std::int64_t>(Tokens) * batch;
    partial_acc += QElems * split_capacity * batch;
    partial_m += StatElems * split_capacity * batch;
    partial_l += StatElems * split_capacity * batch;

    const int context_count = min(length, Window);
    const int context_start = length - context_count;
    const int context_tiles = (context_count + KeyBlock - 1) / KeyBlock;
    const int active_splits = context_tiles > 0 ? min(context_tiles, split_capacity) : 1;
    if (split >= active_splits) { return; }

    const int tile_begin =
        static_cast<int>((static_cast<std::int64_t>(context_tiles) * split) / active_splits);
    const int tile_end = static_cast<int>(
        (static_cast<std::int64_t>(context_tiles) * (split + 1)) / active_splits);
    const int key_begin = context_start + tile_begin * KeyBlock;
    const int key_end   = min(length, context_start + tile_end * KeyBlock);
    const bool owns_query = split == active_splits - 1;
    const int kv_head     = q_head / kContextQueryGroup;
    const bool live       = token < valid;
    const int query_position = live ? positions[token] : 0;

    float q_frag[4];
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
    for (int item = 0; item < 4; ++item) {
        const int d = lane + item * 32;
        q_frag[item] = live ? __bfloat162float(q[context_query_q_index(q_head, d, token)]) : 0.0f;
    }
    float row_m = -CUDART_INF_F;
    float row_l = 0.0f;

    const std::int64_t lane_base =
        static_cast<std::int64_t>(lanes[batch]) * D * padded_context * kContextQueryKVHeads;

    // This warp's query row, shared by its 32 lanes during scoring.
    for (int d = lane; d < D; d += 32) {
        query_s[token * D + d] =
            live ? __bfloat162float(q[context_query_q_index(q_head, d, token)]) : 0.0f;
    }

    float* prob_w = prob_s + token * KeyBlock;
    for (int block_key = key_begin; block_key < key_end; block_key += KeyBlock) {
        const int count = min(KeyBlock, key_end - block_key);
        // 32 keys x 128 dims of K and V: one 16-byte load (8 elements) per thread and plane per
        // step, keys in cyclic-slot order so a block may wrap the 2048-slot ring.
        constexpr int ChunksPerKey = D / 8;
        for (int c = tid; c < KeyBlock * ChunksPerKey; c += Tokens * 32) {
            const int j     = c / ChunksPerKey;
            const int piece = c % ChunksPerKey;
            float* krow = key_s + j * RowStride + piece * 8;
            float* vrow = value_s + j * RowStride + piece * 8;
            if (j < count) {
                const int slot = (block_key + j) & (Window - 1);
                const std::int64_t index =
                    lane_base + static_cast<std::int64_t>(D) * (slot + padded_context * kv_head) +
                    piece * 8;
                const uint4 kraw = *reinterpret_cast<const uint4*>(context_k + index);
                const uint4 vraw = *reinterpret_cast<const uint4*>(context_v + index);
                const auto* kb   = reinterpret_cast<const __nv_bfloat16*>(&kraw);
                const auto* vh   = reinterpret_cast<const __half*>(&vraw);
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    krow[e] = __bfloat162float(kb[e]);
                    vrow[e] = __half2float(vh[e]);
                }
            } else {
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    krow[e] = 0.0f;
                    vrow[e] = 0.0f;
                }
            }
        }
        __syncthreads();

        // Lane j scores key block_key + j.
        const float* qrow = query_s + token * D;
        const float* krow = key_s + lane * RowStride;
        float dot = 0.0f;
#pragma unroll 16
        for (int d = 0; d < D; ++d) { dot += qrow[d] * krow[d]; }
        const int key       = block_key + lane;
        const bool allowed  = live && lane < count && abs(key - query_position) < Window;
        const float score   = allowed ? dot * scale : -CUDART_INF_F;
        float block_m = score;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            block_m = fmaxf(block_m, __shfl_xor_sync(FullMask, block_m, off));
        }
        const float next_m = fmaxf(row_m, block_m);
        const float alpha  = (row_m == -CUDART_INF_F || next_m == -CUDART_INF_F)
                                 ? (row_m == -CUDART_INF_F ? 0.0f : 1.0f)
                                 : exp2_approx((row_m - next_m) * Log2E);
        const float probability =
            score == -CUDART_INF_F ? 0.0f : exp2_approx((score - next_m) * Log2E);
        float block_l = probability;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            block_l += __shfl_xor_sync(FullMask, block_l, off);
        }
        prob_w[lane] = probability;
        __syncwarp();

#pragma unroll
        for (int item = 0; item < 4; ++item) { acc[item] *= alpha; }
#pragma unroll 8
        for (int j = 0; j < KeyBlock; ++j) {
            const float pj    = prob_w[j];
            const float* vrow = value_s + j * RowStride;
#pragma unroll
            for (int item = 0; item < 4; ++item) { acc[item] += pj * vrow[lane + item * 32]; }
        }
        row_l = row_l * alpha + block_l;
        row_m = next_m;
        __syncthreads(); // every warp is done with this block's K, V and P
    }

    if (owns_query) {
        for (int key_token = 0; key_token < valid; ++key_token) {
            // Fewer than D threads at one to three query tokens: stride over the row.
            for (int d = tid; d < D; d += Tokens * 32) {
                const auto index = context_query_query_kv_index(kv_head, d, key_token);
                key_s[d]         = __bfloat162float(query_k[index]);
                value_s[d]       = __bfloat162float(query_v[index]);
            }
            __syncthreads();

            float dot = 0.0f;
#pragma unroll
            for (int item = 0; item < 4; ++item) {
                const int d = lane + item * 32;
                dot += q_frag[item] * key_s[d];
            }
            dot = warp_sum<32>(dot, FullMask);
            dot = __shfl_sync(FullMask, dot, 0);
            const bool allowed =
                live && abs(positions[key_token] - query_position) < Window;
            const float score  = allowed ? dot * scale : -CUDART_INF_F;
            const float next_m = fmaxf(row_m, score);
            const float alpha  = row_m == -CUDART_INF_F ? 0.0f
                                                         : exp2_approx((row_m - next_m) * Log2E);
            const float probability = score == -CUDART_INF_F
                                          ? 0.0f
                                          : exp2_approx((score - next_m) * Log2E);
#pragma unroll
            for (int item = 0; item < 4; ++item) {
                const int d = lane + item * 32;
                acc[item] = acc[item] * alpha + probability * value_s[d];
            }
            row_l = row_l * alpha + probability;
            row_m = next_m;
            __syncthreads();
        }
    }

    if (live) {
        if (lane == 0) {
            const auto stat = context_query_stat_index<Tokens>(q_head, token, split);
            partial_m[stat] = row_m;
            partial_l[stat] = row_l;
        }
#pragma unroll
        for (int item = 0; item < 4; ++item) {
            const int d = lane + item * 32;
            partial_acc[context_query_partial_index<Tokens>(q_head, d, token, split)] = acc[item];
        }
    }
#endif
}

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ == 700
__device__ __forceinline__ void swa_mma_row_row(float (&d)[8], unsigned a0, unsigned a1,
                                                unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m8n8k4.row.row.f32.f16.f16.f32 "
                 "{%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9}, {%10, %11}, "
                 "{%0, %1, %2, %3, %4, %5, %6, %7};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]),
                   "+f"(d[6]), "+f"(d[7])
                 : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
}

__device__ __forceinline__ float swa_ex2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ float swa_max3(float v) {
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 2));
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 4));
    return fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 8));
}

__device__ __forceinline__ float swa_sum3(float v) {
    v += __shfl_xor_sync(0xffffffffu, v, 2);
    v += __shfl_xor_sync(0xffffffffu, v, 4);
    return v + __shfl_xor_sync(0xffffffffu, v, 8);
}
#endif

template <int Tokens>
constexpr int sliding_window_volta_gqa_warps() {
    return (kContextQueryGroup * Tokens + 7) / 8;
}

template <int Tokens>
constexpr int sliding_window_volta_gqa_smem_bytes() {
    constexpr int kWarps   = sliding_window_volta_gqa_warps<Tokens>();
    constexpr int kQStride = kContextQueryHeadDim + 8;  // 136 halves (4-bank step)
    constexpr int kKStride = kContextQueryHeadDim + 8;  // 136 halves
    constexpr int kVStride = kContextQueryHeadDim + 16; // 144 halves (8-bank step)
    constexpr int kPStride = 32 + 8;                    // 40 halves
    return (kWarps * 8 * kQStride + 32 * kKStride + 32 * kVStride + kWarps * 8 * kPStride) *
           static_cast<int>(sizeof(half));
}

// Tensor-core (mma.sync.m8n8k4) grouped-query draft sliding-window attention on Volta.
// One CTA owns one K/V head (serving its 4 query heads x Tokens = 4..32 query rows packed into
// 1..4 warps of 8 rows each). FP16 shared memory drops smem from 54 KiB (1 CTA/SM) to <=28.5 KiB
// (3 CTAs/SM) and runs both QK^T and P.V on Tensor Cores with FP32 online softmax.
template <int Tokens>
__launch_bounds__(sliding_window_volta_gqa_warps<Tokens>() * 32, 2) __global__ void
sliding_window_attention_volta_gqa_partial_kernel(
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ query_k,
    const __nv_bfloat16* __restrict__ query_v, const std::int32_t* __restrict__ positions,
    const std::int32_t* __restrict__ valid_columns, const std::int32_t* __restrict__ lanes,
    const __nv_bfloat16* __restrict__ context_k, const __half* __restrict__ context_v,
    int padded_context, int max_context, int split_capacity, float scale,
    float* __restrict__ partial_acc, float* __restrict__ partial_m,
    float* __restrict__ partial_l) {
    static_assert(Tokens >= 1 && Tokens <= 8, "at most eight query tokens");
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ == 700
    constexpr int D                  = kContextQueryHeadDim;
    constexpr int Window             = 2048;
    constexpr int KeyBlock           = 32;
    constexpr int kWarps             = sliding_window_volta_gqa_warps<Tokens>();
    constexpr int kThreads           = kWarps * 32;
    constexpr int kRows              = kContextQueryGroup * Tokens;
    constexpr int kQStride           = D + 8;
    constexpr int kKStride           = D + 8;
    constexpr int kVStride           = D + 16;
    constexpr int kPStride           = KeyBlock + 8;
    constexpr std::int64_t QElems    = static_cast<std::int64_t>(D) * kContextQueryQHeads * Tokens;
    constexpr std::int64_t KVElems   = static_cast<std::int64_t>(D) * kContextQueryKVHeads * Tokens;
    constexpr std::int64_t StatElems = static_cast<std::int64_t>(kContextQueryQHeads) * Tokens;
    constexpr float Log2E            = 1.4426950408889634074f;
    constexpr float kLn2             = 0.6931471805599453094f;
    constexpr float kNegInf          = -INFINITY;

    extern __shared__ __align__(16) unsigned char gqa_smem_raw[];
    half* q_s   = reinterpret_cast<half*>(gqa_smem_raw);
    half* k_s   = q_s + kWarps * 8 * kQStride;
    half* v_s   = k_s + KeyBlock * kKStride;
    half* p_all = v_s + KeyBlock * kVStride;

    const int kv_group = static_cast<int>(blockIdx.x);
    const int split    = static_cast<int>(blockIdx.y);
    const int batch    = static_cast<int>(blockIdx.z);
    const int tid      = static_cast<int>(threadIdx.x);
    const int warp     = tid >> 5;
    const int lane     = tid & 31;
    const int valid    = valid_columns[batch];
    const int length   = positions[static_cast<std::int64_t>(batch) * Tokens];

    if (kv_group >= kContextQueryKVHeads || split >= split_capacity || valid < 0 ||
        valid > Tokens || length < 0 || length > max_context) {
        return;
    }

    q += QElems * batch;
    query_k += KVElems * batch;
    query_v += KVElems * batch;
    positions += static_cast<std::int64_t>(Tokens) * batch;
    partial_acc += QElems * split_capacity * batch;
    partial_m += StatElems * split_capacity * batch;
    partial_l += StatElems * split_capacity * batch;

    const int context_count = min(length, Window);
    const int context_start = length - context_count;
    const int context_tiles = (context_count + KeyBlock - 1) / KeyBlock;
    const int active_splits = context_tiles > 0 ? min(context_tiles, split_capacity) : 1;
    if (split >= active_splits) { return; }

    const int tile_begin =
        static_cast<int>((static_cast<std::int64_t>(context_tiles) * split) / active_splits);
    const int tile_end = static_cast<int>(
        (static_cast<std::int64_t>(context_tiles) * (split + 1)) / active_splits);
    const int key_begin   = context_start + tile_begin * KeyBlock;
    const int key_end     = min(length, context_start + tile_end * KeyBlock);
    const bool owns_query = split == active_splits - 1;
    const int kv_head     = kv_group;
    const float scale_l2  = scale * Log2E;

    // Stage this warp's 8 (token, q_head) rows into FP16 shared memory, pre-scaled by scale*log2(e).
    // Row rr maps to token = rr >> 2, q_head = kv_group * 4 + (rr & 3).
    const int row0       = warp * 8;
    const int live_rows  = valid * kContextQueryGroup;
    for (int rr = 0; rr < 8; ++rr) {
        const int row   = row0 + rr;
        half* dst       = q_s + (warp * 8 + rr) * kQStride;
        const int tok   = row >> 2;
        const int qh    = kv_group * kContextQueryGroup + (row & 3);
        const bool live = row < live_rows;
#pragma unroll
        for (int item = 0; item < 4; ++item) {
            const int d = lane + item * 32;
            dst[d] = live ? __float2half_rn(
                                __bfloat162float(q[context_query_q_index(qh, d, tok)]) * scale_l2)
                          : __ushort_as_half(0);
        }
    }

    const int r        = (lane & 3) + ((lane & 16) ? 4 : 0);
    const int qp       = (lane >> 2) & 3;
    const int b_sel    = qp * 4 + (lane & 3) + ((lane & 16) ? 16 : 0);
    const int c_sel    = qp * 4 + (((lane >> 1) & 1) << 1);
    const int c_row_lo = ((lane & 16) ? 4 : 0) | (lane & 1);
    const int row_lo   = row0 + c_row_lo;
    const int row_hi   = row_lo + 2;
    const bool live_lo = row_lo < live_rows;
    const bool live_hi = row_hi < live_rows;
    const int pos_lo   = live_lo ? positions[row_lo >> 2] : 0;
    const int pos_hi   = live_hi ? positions[row_hi >> 2] : 0;

    half* p_s        = p_all + warp * 8 * kPStride;
    const half* qrow = q_s + (warp * 8 + r) * kQStride;
    const half* krow = k_s + b_sel * kKStride;
    const half* prow = p_s + r * kPStride;
    const half* vcol = v_s + (lane & 3) * kVStride + qp * 8 + ((lane & 16) ? 32 : 0);

    float o[4][8];
#pragma unroll
    for (int cb = 0; cb < 4; ++cb) {
#pragma unroll
        for (int i = 0; i < 8; ++i) { o[cb][i] = 0.0f; }
    }
    float m_lo = kNegInf, m_hi = kNegInf, l_lo = 0.0f, l_hi = 0.0f;

    auto attend_tile = [&](int key_count, int key_base_pos, bool use_query_positions) {
        float s[8]  = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        float s2[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
        for (int kk = 0; kk < D / 8; ++kk) {
            const uint4 a = *reinterpret_cast<const uint4*>(qrow + kk * 8);
            const uint4 b = *reinterpret_cast<const uint4*>(krow + kk * 8);
            volta_mma_qp_n(s, a.x, a.y, b.x, b.y);
            volta_mma_qp_n(s2, a.z, a.w, b.z, b.w);
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) { s[i] += s2[i]; }

#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int key_idx = c_sel + (i & 1) + ((i >> 2) << 4);
            const bool is_hi  = (i & 2) != 0;
            const bool live   = is_hi ? live_hi : live_lo;
            const int qpos    = is_hi ? pos_hi : pos_lo;
            bool allowed      = false;
            if (live && key_idx < key_count) {
                const int kpos = use_query_positions ? positions[key_idx] : (key_base_pos + key_idx);
                allowed        = abs(kpos - qpos) < Window;
            }
            s[i] = allowed ? s[i] : kNegInf;
        }

        const float mx_lo    = swa_max3(fmaxf(fmaxf(s[0], s[1]), fmaxf(s[4], s[5])));
        const float mx_hi    = swa_max3(fmaxf(fmaxf(s[2], s[3]), fmaxf(s[6], s[7])));
        const float new_lo   = fmaxf(m_lo, mx_lo);
        const float new_hi   = fmaxf(m_hi, mx_hi);
        const float alpha_lo = (m_lo == kNegInf || new_lo == kNegInf)
                                   ? (m_lo == kNegInf ? 0.0f : 1.0f)
                                   : swa_ex2(m_lo - new_lo);
        const float alpha_hi = (m_hi == kNegInf || new_hi == kNegInf)
                                   ? (m_hi == kNegInf ? 0.0f : 1.0f)
                                   : swa_ex2(m_hi - new_hi);
        l_lo *= alpha_lo;
        l_hi *= alpha_hi;
#pragma unroll
        for (int cb = 0; cb < 4; ++cb) {
#pragma unroll
            for (int i = 0; i < 8; ++i) { o[cb][i] *= (i & 2) ? alpha_hi : alpha_lo; }
        }
        m_lo = new_lo;
        m_hi = new_hi;

#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const bool is_hi = (i & 2) != 0;
            const float base = is_hi ? m_hi : m_lo;
            const float p    = (s[i] == kNegInf) ? 0.0f : swa_ex2(s[i] - base);
            s[i]             = p;
            if (is_hi) {
                l_hi += p;
            } else {
                l_lo += p;
            }
        }

#pragma unroll
        for (int i = 0; i < 8; i += 2) {
            const int row = (i & 2) | c_row_lo;
            const int col = c_sel + ((i >> 2) << 4);
            *reinterpret_cast<half2*>(p_s + row * kPStride + col) =
                __floats2half2_rn(s[i], s[i + 1]);
        }
        __syncwarp();

        uint4 pa[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) { pa[j] = *reinterpret_cast<const uint4*>(prow + j * 8); }
#pragma unroll
        for (int ks = 0; ks < KeyBlock / 4; ++ks) {
            const unsigned a0 = (ks & 1) ? pa[ks >> 1].z : pa[ks >> 1].x;
            const unsigned a1 = (ks & 1) ? pa[ks >> 1].w : pa[ks >> 1].y;
            const half* vrow  = vcol + ks * 4 * kVStride;
#pragma unroll
            for (int cp = 0; cp < 2; ++cp) {
                const uint4 b = *reinterpret_cast<const uint4*>(vrow + cp * 64);
                swa_mma_row_row(o[2 * cp], a0, a1, b.x, b.y);
                swa_mma_row_row(o[2 * cp + 1], a0, a1, b.z, b.w);
            }
        }
    };

    const std::int64_t lane_base =
        static_cast<std::int64_t>(lanes[batch]) * D * padded_context * kContextQueryKVHeads;
    constexpr int ChunksPerKey = D / 8;

    for (int block_key = key_begin; block_key < key_end; block_key += KeyBlock) {
        const int count = min(KeyBlock, key_end - block_key);
        for (int c = tid; c < KeyBlock * ChunksPerKey; c += kThreads) {
            const int j     = c / ChunksPerKey;
            const int piece = c % ChunksPerKey;
            uint4 kpacked   = make_uint4(0, 0, 0, 0);
            uint4 vpacked   = make_uint4(0, 0, 0, 0);
            if (j < count) {
                const int slot = (block_key + j) & (Window - 1);
                const std::int64_t index =
                    lane_base + static_cast<std::int64_t>(D) * (slot + padded_context * kv_head) +
                    piece * 8;
                const uint4 kraw = __ldg(reinterpret_cast<const uint4*>(context_k + index));
                vpacked          = __ldg(reinterpret_cast<const uint4*>(context_v + index));
                const auto* k2   = reinterpret_cast<const __nv_bfloat162*>(&kraw);
                auto* kdst       = reinterpret_cast<half2*>(&kpacked);
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const float2 f = __bfloat1622float2(k2[e]);
                    kdst[e]        = __floats2half2_rn(f.x, f.y);
                }
            }
            *reinterpret_cast<uint4*>(k_s + j * kKStride + piece * 8) = kpacked;
            *reinterpret_cast<uint4*>(v_s + j * kVStride + piece * 8) = vpacked;
        }
        __syncthreads();
        attend_tile(count, block_key, false);
        __syncthreads();
    }

    if (owns_query && valid > 0) {
        for (int c = tid; c < KeyBlock * ChunksPerKey; c += kThreads) {
            const int j     = c / ChunksPerKey;
            const int piece = c % ChunksPerKey;
            uint4 kpacked   = make_uint4(0, 0, 0, 0);
            uint4 vpacked   = make_uint4(0, 0, 0, 0);
            if (j < valid) {
                const auto index = context_query_query_kv_index(kv_head, piece * 8, j);
                const uint4 kraw = __ldg(reinterpret_cast<const uint4*>(query_k + index));
                const uint4 vraw = __ldg(reinterpret_cast<const uint4*>(query_v + index));
                const auto* k2   = reinterpret_cast<const __nv_bfloat162*>(&kraw);
                const auto* v2   = reinterpret_cast<const __nv_bfloat162*>(&vraw);
                auto* kdst       = reinterpret_cast<half2*>(&kpacked);
                auto* vdst       = reinterpret_cast<half2*>(&vpacked);
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const float2 kf = __bfloat1622float2(k2[e]);
                    const float2 vf = __bfloat1622float2(v2[e]);
                    kdst[e]         = __floats2half2_rn(kf.x, kf.y);
                    vdst[e]         = __floats2half2_rn(vf.x, vf.y);
                }
            }
            *reinterpret_cast<uint4*>(k_s + j * kKStride + piece * 8) = kpacked;
            *reinterpret_cast<uint4*>(v_s + j * kVStride + piece * 8) = vpacked;
        }
        __syncthreads();
        attend_tile(valid, 0, true);
        __syncthreads();
    }

    l_lo = swa_sum3(l_lo);
    l_hi = swa_sum3(l_hi);
    if ((lane & 14) == 0) {
        if (live_lo) {
            const int tok   = row_lo >> 2;
            const int qh    = kv_group * kContextQueryGroup + (row_lo & 3);
            const auto stat = context_query_stat_index<Tokens>(qh, tok, split);
            partial_m[stat] = l_lo > 0.0f ? m_lo * kLn2 : kNegInf;
            partial_l[stat] = l_lo;
        }
        if (live_hi) {
            const int tok   = row_hi >> 2;
            const int qh    = kv_group * kContextQueryGroup + (row_hi & 3);
            const auto stat = context_query_stat_index<Tokens>(qh, tok, split);
            partial_m[stat] = l_hi > 0.0f ? m_hi * kLn2 : kNegInf;
            partial_l[stat] = l_hi;
        }
    }
#pragma unroll
    for (int i = 0; i < 8; i += 2) {
        const int row = row0 + ((i & 2) | c_row_lo);
        if (row >= live_rows) { continue; }
        const int tok = row >> 2;
        const int qh  = kv_group * kContextQueryGroup + (row & 3);
#pragma unroll
        for (int cb = 0; cb < 4; ++cb) {
            const int col = (cb >> 1) * 64 + qp * 8 + ((i >> 2) << 5) + (cb & 1) * 4 +
                            (((lane >> 1) & 1) << 1);
            *reinterpret_cast<float2*>(
                partial_acc + context_query_partial_index<Tokens>(qh, col, tok, split)) =
                make_float2(o[cb][i], o[cb][i + 1]);
        }
    }
#endif
}

__launch_bounds__(kSlidingWindowVoltaThreads, 1) __global__ void
sliding_window_attention_volta_kernel(
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ query_k,
    const __nv_bfloat16* __restrict__ query_v, const std::int32_t* __restrict__ positions,
    const std::int32_t* __restrict__ valid_columns, const std::int32_t* __restrict__ lanes,
    const __nv_bfloat16* __restrict__ context_k,
    const __half* __restrict__ context_v, int padded_context, int tokens, int window,
    float scale, __nv_bfloat16* __restrict__ out) {
    constexpr int D       = kContextQueryHeadDim;
    constexpr int QHeads  = kContextQueryQHeads;
    constexpr int KVHeads = kContextQueryKVHeads;
    // Window is 2048 (DFlash2 local layers) or 4096 (DFlash1); both are powers of two so the
    // ring slot stays `key & (Window - 1)`.
    const int Window = window;

    __shared__ float warp_sums[kSlidingWindowVoltaThreads / kWarpSize];
    __shared__ float score_s;

    const int token  = static_cast<int>(blockIdx.x);
    const int q_head = static_cast<int>(blockIdx.y);
    const int batch  = static_cast<int>(blockIdx.z);
    const int d      = static_cast<int>(threadIdx.x);
    const int valid  = valid_columns[batch];
    const std::int64_t q_batch = static_cast<std::int64_t>(batch) * D * QHeads * tokens;
    const std::int64_t kv_batch = static_cast<std::int64_t>(batch) * D * KVHeads * tokens;
    const std::int64_t out_index =
        q_batch + d + static_cast<std::int64_t>(D) * (q_head + QHeads * token);
    if (token >= valid) {
        out[out_index] = __float2bfloat16(0.0f);
        return;
    }

    const auto* batch_positions = positions + static_cast<std::int64_t>(batch) * tokens;
    const int context_end        = batch_positions[0];
    const int query_position     = batch_positions[token];
    const int context_begin      = max(0, max(context_end - Window, query_position - (Window - 1)));
    const int kv_head            = q_head / (QHeads / KVHeads);
    const float q_value          = __bfloat162float(q[out_index]);

    const std::int64_t lane_base =
        static_cast<std::int64_t>(lanes[batch]) * D * padded_context * KVHeads;
    float row_max = -CUDART_INF_F;
    float row_sum = 0.0f;
    float acc     = 0.0f;

    for (int key = context_begin; key < context_end; ++key) {
        const int slot = key & (Window - 1);
        const std::int64_t index = lane_base + d + static_cast<std::int64_t>(D) *
                                                       (slot + padded_context * kv_head);
        const float key_value = __bfloat162float(context_k[index]);
        const float value     = __half2float(context_v[index]);
        const float dot = block_reduce_sum<kSlidingWindowVoltaThreads>(q_value * key_value,
                                                                       warp_sums);
        if (d == 0) { score_s = dot * scale; }
        __syncthreads();
        const float score      = score_s;
        const float next_max   = fmaxf(row_max, score);
        const float old_scale  = row_max == -CUDART_INF_F
                                     ? 0.0f
                                     : exp2_approx((row_max - next_max) * 1.4426950408889634f);
        const float probability = exp2_approx((score - next_max) * 1.4426950408889634f);
        acc     = acc * old_scale + probability * value;
        row_sum = row_sum * old_scale + probability;
        row_max = next_max;
        __syncthreads();
    }

    for (int key_token = 0; key_token < valid; ++key_token) {
        const int key_position = batch_positions[key_token];
        if (abs(key_position - query_position) >= Window) { continue; }
        const std::int64_t index =
            kv_batch + d + static_cast<std::int64_t>(D) * (kv_head + KVHeads * key_token);
        const float key_value = __bfloat162float(query_k[index]);
        const float value     = __bfloat162float(query_v[index]);
        const float dot = block_reduce_sum<kSlidingWindowVoltaThreads>(q_value * key_value,
                                                                       warp_sums);
        if (d == 0) { score_s = dot * scale; }
        __syncthreads();
        const float score       = score_s;
        const float next_max    = fmaxf(row_max, score);
        const float old_scale   = row_max == -CUDART_INF_F
                                      ? 0.0f
                                      : exp2_approx((row_max - next_max) * 1.4426950408889634f);
        const float probability = exp2_approx((score - next_max) * 1.4426950408889634f);
        acc     = acc * old_scale + probability * value;
        row_sum = row_sum * old_scale + probability;
        row_max = next_max;
        __syncthreads();
    }

    out[out_index] = __float2bfloat16_rn(row_sum > 0.0f ? acc / row_sum : 0.0f);
}

} // namespace ninfer::ops
