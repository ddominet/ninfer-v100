#pragma once

#include "ops/common/math.cuh"
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

template <int Tokens>
__launch_bounds__(Tokens * kContextQueryGroup * 32, 1) __global__ void
sliding_window_attention_volta_gqa_partial_kernel(
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ query_k,
    const __nv_bfloat16* __restrict__ query_v, const std::int32_t* __restrict__ positions,
    const std::int32_t* __restrict__ valid_columns, const std::int32_t* __restrict__ lanes,
    const __nv_bfloat16* __restrict__ context_k, const __half* __restrict__ context_v,
    int padded_context, int max_context, int split_capacity, float scale,
    float* __restrict__ partial_acc, float* __restrict__ partial_m,
    float* __restrict__ partial_l) {
    // Grouped-query form: one CTA per K/V head, one warp per (query head of the group, token).
    // Each staged K/V block serves the group's four query heads, so a request's window is read
    // once per K/V head instead of once per query head (the octet form read it four times).
    // Dynamic shared memory: key, value, query and probability tiles (see the launcher).
    static_assert(Tokens >= 1 && Tokens <= 8, "at most eight query tokens");
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
    constexpr int Rows     = kContextQueryGroup * Tokens;
    constexpr int kThreads = Rows * 32;
    extern __shared__ float gqa_smem[];
    float* key_s   = gqa_smem;
    float* value_s = key_s + KeyBlock * RowStride;
    float* query_s = value_s + KeyBlock * RowStride;
    float* prob_s  = query_s + Rows * D;

    const int kv_group = static_cast<int>(blockIdx.x);
    const int split  = static_cast<int>(blockIdx.y);
    const int batch  = static_cast<int>(blockIdx.z);
    const int tid    = static_cast<int>(threadIdx.x);
    const int warp   = tid >> 5;
    const int lane   = tid & 31;
    const int token  = warp % Tokens;
    const int q_head = kv_group * kContextQueryGroup + warp / Tokens;
    const int valid  = valid_columns[batch];
    const int length = positions[static_cast<std::int64_t>(batch) * Tokens];

    if (kv_group >= kContextQueryKVHeads || split >= split_capacity || valid < 0 || valid > Tokens ||
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
    const int kv_head     = kv_group;
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
        query_s[warp * D + d] =
            live ? __bfloat162float(q[context_query_q_index(q_head, d, token)]) : 0.0f;
    }

    float* prob_w = prob_s + warp * KeyBlock;
    for (int block_key = key_begin; block_key < key_end; block_key += KeyBlock) {
        const int count = min(KeyBlock, key_end - block_key);
        // 32 keys x 128 dims of K and V: one 16-byte load (8 elements) per thread and plane per
        // step, keys in cyclic-slot order so a block may wrap the 2048-slot ring.
        constexpr int ChunksPerKey = D / 8;
        for (int c = tid; c < KeyBlock * ChunksPerKey; c += kThreads) {
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
        const float* qrow = query_s + warp * D;
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
            for (int d = tid; d < D; d += kThreads) {
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
