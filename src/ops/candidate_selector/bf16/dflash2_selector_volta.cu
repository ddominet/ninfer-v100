// DFlash2 coherent-path candidate selector — sm_70.
//
// For each request, walk positions 0..K-1 left to right. At position i the predecessor token is
// the anchor (i=0) or the token chosen at i-1. The rank-256 edge score for candidate c is
//
//   edge[c] = unary[c,i,b] + sum_r pred_cb[r, pred] * g[r,i,b] * succ_cb[r, cand_ids[c,i,b]]
//
// Greedy (temperature <= 0): pick the lowest rank attaining max(edge); q is its one-hot.
// Stochastic (temperature > 0): q is softmax(edge / temperature) over the 16 candidates, and the
// draw is the inverse-CDF of q against sampling_uniform(seed, base_pos + i, DFlash2Proposal, 0)
// -- the same counter-based RNG contract as upstream, safe under CUDA-graph replay.

#include "ops/candidate_selector/bf16/dflash2_selector_volta.h"

#include "core/device.h"
#include "ops/common/warp.cuh"
#include "ops/kernel/sampling_device.cuh"
#include "ninfer/ops/sampling.h"

#include <cuda_bf16.h>
#include <cstdint>

namespace ninfer::ops::detail {
namespace {

constexpr int kRank       = 256;
constexpr int kCandidates = 16;

// Parallel-warp selector walk: 16 warps (512 threads) per request CTA — one warp per candidate
// c in 0..15. Every warp issues 16-byte coalesced loads of its successor codebook row in parallel
// (16x memory-level parallelism over host-mapped PCIe codebooks) and reduces with warp shuffles
// only, cutting CTA barriers from 224 down to 14 per request.
__global__ __launch_bounds__(512) void dflash2_selector_walk_kernel(
    const std::int32_t* __restrict__ cand_ids, const float* __restrict__ unary,
    const __nv_bfloat16* __restrict__ g, const std::int32_t* __restrict__ anchors,
    const std::int32_t* __restrict__ base_positions, const SamplingConfig* __restrict__ configs,
    const __nv_bfloat16* __restrict__ pred_cb, const __nv_bfloat16* __restrict__ succ_cb, int steps,
    std::int32_t* __restrict__ drafts, float* __restrict__ proposal_q,
    float* __restrict__ draft_confidence) {
    const int b    = static_cast<int>(blockIdx.x);
    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5; // 0..15 == candidate index c
    const int lane = tid & 31; // 0..31 == 8 elements per lane -> 256 rank dims

    __shared__ float pred_g[kRank];
    __shared__ float edge[kCandidates];
    __shared__ int predecessor;
    __shared__ float temperature;
    __shared__ unsigned long long seed;
    __shared__ int base_pos;
    if (tid == 0) {
        predecessor = anchors[b];
        temperature = configs[b].temperature;
        seed        = configs[b].seed;
        base_pos    = base_positions[b];
    }
    __syncthreads();

    for (int i = 0; i < steps; ++i) {
        const int col  = i + steps * b;
        const int pred = predecessor;

        // Warp 0 loads pred_cb[0..255] and g[0..255] via 16-byte vectorized loads.
        if (warp == 0) {
            const uint4 p_raw = __ldg(reinterpret_cast<const uint4*>(
                pred_cb + static_cast<std::int64_t>(pred) * kRank + lane * 8));
            const uint4 g_raw = __ldg(reinterpret_cast<const uint4*>(
                g + static_cast<std::int64_t>(col) * kRank + lane * 8));
            const auto* p2 = reinterpret_cast<const __nv_bfloat162*>(&p_raw);
            const auto* g2 = reinterpret_cast<const __nv_bfloat162*>(&g_raw);
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const float2 pf            = __bfloat1622float2(p2[e]);
                const float2 gf            = __bfloat1622float2(g2[e]);
                pred_g[lane * 8 + 2 * e]     = pf.x * gf.x;
                pred_g[lane * 8 + 2 * e + 1] = pf.y * gf.y;
            }
        }
        __syncthreads();

        // All 16 warps evaluate all 16 candidates concurrently with 16-byte coalesced loads.
        const int cid   = cand_ids[warp + kCandidates * col];
        const uint4 s_raw = __ldg(reinterpret_cast<const uint4*>(
            succ_cb + static_cast<std::int64_t>(cid) * kRank + lane * 8));
        const auto* s2 = reinterpret_cast<const __nv_bfloat162*>(&s_raw);
        float acc      = 0.0f;
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const float2 sf = __bfloat1622float2(s2[e]);
            acc             = fmaf(pred_g[lane * 8 + 2 * e], sf.x, acc);
            acc             = fmaf(pred_g[lane * 8 + 2 * e + 1], sf.y, acc);
        }
        acc = warp_reduce_sum(acc);
        if (lane == 0) { edge[warp] = unary[warp + kCandidates * col] + acc; }
        __syncthreads();

        if (tid == 0) {
            float best = edge[0];
            for (int c = 1; c < kCandidates; ++c) { best = fmaxf(best, edge[c]); }

            int j;
            float conf = 1.0f;
            if (temperature <= 0.0f) {
                j         = 0;
                float top = edge[0];
                for (int c = 1; c < kCandidates; ++c) {
                    if (edge[c] > top) {
                        top = edge[c];
                        j   = c;
                    }
                }
                for (int c = 0; c < kCandidates; ++c) {
                    proposal_q[c + kCandidates * col] = (c == j) ? 1.0f : 0.0f;
                }
                if (draft_confidence != nullptr) {
                    float wsum = 0.0f;
                    for (int c = 0; c < kCandidates; ++c) { wsum += __expf(edge[c] - best); }
                    conf = 1.0f / wsum;
                }
            } else {
                float wsum = 0.0f;
                float w[kCandidates];
                for (int c = 0; c < kCandidates; ++c) {
                    w[c] = __expf((edge[c] - best) / temperature);
                    wsum += w[c];
                }
                for (int c = 0; c < kCandidates; ++c) {
                    proposal_q[c + kCandidates * col] = w[c] / wsum;
                }
                const float u = sampling_uniform(seed, base_pos + i,
                                                 kSamplePurposeDFlash2Proposal, 0U);
                j             = kCandidates - 1;
                float cdf     = 0.0f;
                for (int c = 0; c < kCandidates; ++c) {
                    cdf += w[c] / wsum;
                    if (u < cdf) {
                        j = c;
                        break;
                    }
                }
                conf = w[j] / wsum;
            }

            const int chosen = cand_ids[j + kCandidates * col];
            drafts[col]      = chosen;
            predecessor      = chosen;
            if (draft_confidence != nullptr) { draft_confidence[col] = conf; }
        }
        __syncthreads();
    }
}

} // namespace

void dflash2_candidate_selector_walk_launch(const Tensor& candidate_ids, const Tensor& unary_scores,
                                            const Tensor& projected_hidden, const Tensor& anchors,
                                            const Tensor& base_positions,
                                            const SamplingConfig* configs,
                                            const Tensor& predecessor_codebook,
                                            const Tensor& successor_codebook, Tensor& drafts,
                                            Tensor& proposal_q, cudaStream_t stream,
                                            float* draft_confidence) {
    const int steps = candidate_ids.ne[1];
    const int batch = candidate_ids.ne[2];
    dflash2_selector_walk_kernel<<<batch, kCandidates * 32, 0, stream>>>(
        static_cast<const std::int32_t*>(candidate_ids.data),
        static_cast<const float*>(unary_scores.data),
        static_cast<const __nv_bfloat16*>(projected_hidden.data),
        static_cast<const std::int32_t*>(anchors.data),
        static_cast<const std::int32_t*>(base_positions.data), configs,
        static_cast<const __nv_bfloat16*>(predecessor_codebook.data),
        static_cast<const __nv_bfloat16*>(successor_codebook.data), steps,
        static_cast<std::int32_t*>(drafts.data), static_cast<float*>(proposal_q.data),
        draft_confidence);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
