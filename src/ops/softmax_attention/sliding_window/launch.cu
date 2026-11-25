#include "ops/softmax_attention/sliding_window/launch.h"

#include "core/device.h"
#include "ops/softmax_attention/sliding_window/kernel.cuh"
#ifdef NINFER_VOLTA_BUILD
#include "ops/softmax_attention/sliding_window/volta.cuh"
#endif

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

template <int Tokens, class Launch>
void dispatch_token_case(Launch&& launch) {
    constexpr int Warps = (Tokens + 3) / 4;
    launch.template operator()<Tokens, Warps>();
}

template <class Launch>
void dispatch_tokens(std::int32_t tokens, Launch&& launch) {
    switch (tokens) {
#define NINFER_SLIDING_WINDOW_ATTENTION_TOKEN_CASE(TOKENS)                                         \
    case TOKENS:                                                                                   \
        dispatch_token_case<TOKENS>(launch);                                                       \
        return
        NINFER_SLIDING_WINDOW_ATTENTION_TOKEN_CASE(1);
        NINFER_SLIDING_WINDOW_ATTENTION_TOKEN_CASE(2);
        NINFER_SLIDING_WINDOW_ATTENTION_TOKEN_CASE(3);
        NINFER_SLIDING_WINDOW_ATTENTION_TOKEN_CASE(4);
        NINFER_SLIDING_WINDOW_ATTENTION_TOKEN_CASE(5);
        NINFER_SLIDING_WINDOW_ATTENTION_TOKEN_CASE(6);
        NINFER_SLIDING_WINDOW_ATTENTION_TOKEN_CASE(7);
        NINFER_SLIDING_WINDOW_ATTENTION_TOKEN_CASE(8);
        NINFER_SLIDING_WINDOW_ATTENTION_TOKEN_CASE(9);
        NINFER_SLIDING_WINDOW_ATTENTION_TOKEN_CASE(10);
        NINFER_SLIDING_WINDOW_ATTENTION_TOKEN_CASE(11);
        NINFER_SLIDING_WINDOW_ATTENTION_TOKEN_CASE(12);
        NINFER_SLIDING_WINDOW_ATTENTION_TOKEN_CASE(13);
        NINFER_SLIDING_WINDOW_ATTENTION_TOKEN_CASE(14);
        NINFER_SLIDING_WINDOW_ATTENTION_TOKEN_CASE(15);
        NINFER_SLIDING_WINDOW_ATTENTION_TOKEN_CASE(16);
#undef NINFER_SLIDING_WINDOW_ATTENTION_TOKEN_CASE
    default:
        throw std::invalid_argument("sliding_window_attention: unsupported T");
    }
}

} // namespace

SlidingWindowAttentionPlan
sliding_window_attention_resolve_plan(std::int32_t tokens,
                                      SlidingWindowAttentionExecutionEnvelope envelope,
                                      std::int32_t batch_size) {
    if (tokens < 1 || tokens > 16) {
        throw std::invalid_argument("sliding_window_attention plan: T must be 1..16");
    }
    if (envelope.min_context > envelope.max_context) {
        throw std::invalid_argument("sliding_window_attention plan: invalid envelope");
    }
    // Graph envelopes whose longest context fits three key tiles avoid the second kernel and
    // workspace round trip. At four tiles, split-KV is already faster for every qualified T.
    constexpr std::uint32_t direct_context_limit = 96;
    const bool direct                            = envelope.max_context <= direct_context_limit;
    constexpr std::int32_t key_block             = 32;
    const std::uint32_t context_rows             = std::min(envelope.max_context, 4095u);
    const std::int32_t context_tiles =
        static_cast<std::int32_t>((context_rows + key_block - 1u) / key_block);
#ifdef NINFER_VOLTA_BUILD
    // 80 SMs on V100: grid is (8 KV heads, splits, batch_size). Target ~160 CTAs (2 waves at
    // 1 CTA/SM, 1 wave at 2 CTAs/SM) so multi-stream batches do not oversplit and burn time in
    // the split reducer (was 2,048 CTAs at B=8 with hardcoded split_limit=32).
    const std::int32_t eff_batch   = std::max<std::int32_t>(1, batch_size);
    const std::int32_t target_ctas = 160;
    const std::int32_t split_limit =
        std::clamp((target_ctas + 8 * eff_batch - 1) / (8 * eff_batch), 2, 20);
#else
    (void)batch_size;
    constexpr std::int32_t split_limit = 32;
#endif
    return {
        .route =
            direct ? SlidingWindowAttentionRoute::Direct : SlidingWindowAttentionRoute::SplitKv,
        .tokens         = tokens,
        .warps          = (tokens + 3) / 4,
        .split_capacity = direct ? 1 : std::min(split_limit, std::max(1, context_tiles)),
        .max_context    = static_cast<std::int32_t>(envelope.max_context),
    };
}

const char* sliding_window_attention_route_name(SlidingWindowAttentionRoute route) {
    switch (route) {
    case SlidingWindowAttentionRoute::Direct:
        return "direct";
    case SlidingWindowAttentionRoute::SplitKv:
        return "split_kv";
    }
    return "unknown";
}

void sliding_window_attention_launch(const Tensor& q, const Tensor& query_k, const Tensor& query_v,
                                     const Tensor& positions, const Tensor& valid_columns,
                                     const Tensor& lanes, float scale,
                                     const CyclicKVCacheLayerView& context,
                                     const SlidingWindowAttentionPlan& plan, Tensor& partial_acc,
                                     Tensor& partial_m, Tensor& partial_l, Tensor& out,
                                     cudaStream_t stream) {
#ifdef NINFER_VOLTA_BUILD
    if (q.ne[2] >= 1 && q.ne[2] <= 8 && context.capacity == 2048 &&
        plan.route == SlidingWindowAttentionRoute::SplitKv) {
        auto run = [&]<int Tokens>() {
            // One CTA per K/V head serving its 4*Tokens query rows via Volta mma.m8n8k4.
            constexpr int kThreads = sliding_window_volta_gqa_warps<Tokens>() * 32;
            constexpr int kSmem    = sliding_window_volta_gqa_smem_bytes<Tokens>();
            static const bool configured = [] {
                CUDA_CHECK(cudaFuncSetAttribute(
                    sliding_window_attention_volta_gqa_partial_kernel<Tokens>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem));
                return true;
            }();
            (void)configured;
            const dim3 partial_grid(kContextQueryKVHeads, plan.split_capacity,
                                    static_cast<unsigned>(q.ne[3]));
            sliding_window_attention_volta_gqa_partial_kernel<Tokens>
                <<<partial_grid, kThreads, kSmem, stream>>>(
                    static_cast<const __nv_bfloat16*>(q.data),
                    static_cast<const __nv_bfloat16*>(query_k.data),
                    static_cast<const __nv_bfloat16*>(query_v.data),
                    static_cast<const std::int32_t*>(positions.data),
                    static_cast<const std::int32_t*>(valid_columns.data),
                    static_cast<const std::int32_t*>(lanes.data),
                    static_cast<const __nv_bfloat16*>(context.k.data),
                    static_cast<const __half*>(context.v.data),
                    static_cast<int>(context.padded_capacity), plan.max_context,
                    plan.split_capacity, scale, static_cast<float*>(partial_acc.data),
                    static_cast<float*>(partial_m.data), static_cast<float*>(partial_l.data));
            CUDA_CHECK(cudaGetLastError());

            constexpr int ReduceWarps = 1;
            constexpr int ReduceRows  = kContextQueryQHeads * Tokens;
            const dim3 reduce_grid((ReduceRows + ReduceWarps - 1) / ReduceWarps, 1, q.ne[3]);
            sliding_window_attention_reduce_kernel<Tokens, 32, ReduceWarps, float>
                <<<reduce_grid, ReduceWarps * 32, 0, stream>>>(
                    static_cast<const float*>(partial_acc.data),
                    static_cast<const float*>(partial_m.data),
                    static_cast<const float*>(partial_l.data),
                    static_cast<const std::int32_t*>(positions.data),
                    static_cast<const std::int32_t*>(valid_columns.data), plan.max_context,
                    plan.split_capacity, static_cast<__nv_bfloat16*>(out.data));
            CUDA_CHECK(cudaGetLastError());
        };
        switch (q.ne[2]) {
        case 1: run.template operator()<1>(); break;
        case 2: run.template operator()<2>(); break;
        case 3: run.template operator()<3>(); break;
        case 4: run.template operator()<4>(); break;
        case 5: run.template operator()<5>(); break;
        case 6: run.template operator()<6>(); break;
        case 7: run.template operator()<7>(); break;
        default: run.template operator()<8>(); break;
        }
        return;
    }
    const dim3 grid(static_cast<unsigned>(q.ne[2]), kContextQueryQHeads,
                    static_cast<unsigned>(q.ne[3]));
    sliding_window_attention_volta_kernel<<<grid, kSlidingWindowVoltaThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(q.data),
        static_cast<const __nv_bfloat16*>(query_k.data),
        static_cast<const __nv_bfloat16*>(query_v.data),
        static_cast<const std::int32_t*>(positions.data),
        static_cast<const std::int32_t*>(valid_columns.data),
        static_cast<const std::int32_t*>(lanes.data),
        static_cast<const __nv_bfloat16*>(context.k.data),
        static_cast<const __half*>(context.v.data),
        static_cast<int>(context.padded_capacity), q.ne[2],
        static_cast<int>(context.capacity), scale, static_cast<__nv_bfloat16*>(out.data));
    CUDA_CHECK(cudaGetLastError());
    return;
#else
    dispatch_tokens(q.ne[2], [&]<int Tokens, int Warps>() {
        const bool direct = plan.route == SlidingWindowAttentionRoute::Direct;
        if (plan.warps != Warps || plan.split_capacity < 1 ||
            plan.split_capacity > kSlidingWindowMaxCandidateSplit ||
            (direct && plan.split_capacity != 1)) {
            throw std::invalid_argument("sliding_window_attention: inconsistent plan");
        }
        constexpr int KeyBlock = 32;
        constexpr std::size_t SmemBytes =
            2u * KeyBlock * kContextQueryHeadDim * sizeof(__nv_bfloat16);
        if (direct) {
            const dim3 direct_grid(kContextQueryKVHeads, 1, q.ne[3]);
            sliding_window_attention_split_partial_kernel<Tokens, Warps, KeyBlock, true>
                <<<direct_grid, Warps * 32, SmemBytes, stream>>>(
                    static_cast<const __nv_bfloat16*>(q.data),
                    static_cast<const __nv_bfloat16*>(query_k.data),
                    static_cast<const __nv_bfloat16*>(query_v.data),
                    static_cast<const std::int32_t*>(positions.data),
                    static_cast<const std::int32_t*>(valid_columns.data),
                    static_cast<const std::int32_t*>(lanes.data),
                    static_cast<const __nv_bfloat16*>(context.k.data),
                    static_cast<const __half*>(context.v.data),
                    static_cast<int>(context.padded_capacity), plan.max_context, 1, scale,
                    static_cast<__nv_bfloat16*>(partial_acc.data),
                    static_cast<float*>(partial_m.data), static_cast<float*>(partial_l.data),
                    static_cast<__nv_bfloat16*>(out.data));
            CUDA_CHECK(cudaGetLastError());
            return;
        }

        const dim3 partial_grid(kContextQueryKVHeads, plan.split_capacity, q.ne[3]);
        sliding_window_attention_split_partial_kernel<Tokens, Warps, KeyBlock, false>
            <<<partial_grid, Warps * 32, SmemBytes, stream>>>(
                static_cast<const __nv_bfloat16*>(q.data),
                static_cast<const __nv_bfloat16*>(query_k.data),
                static_cast<const __nv_bfloat16*>(query_v.data),
                static_cast<const std::int32_t*>(positions.data),
                static_cast<const std::int32_t*>(valid_columns.data),
                static_cast<const std::int32_t*>(lanes.data),
                static_cast<const __nv_bfloat16*>(context.k.data),
                static_cast<const __half*>(context.v.data),
                static_cast<int>(context.padded_capacity), plan.max_context, plan.split_capacity,
                scale, static_cast<__nv_bfloat16*>(partial_acc.data),
                static_cast<float*>(partial_m.data), static_cast<float*>(partial_l.data),
                static_cast<__nv_bfloat16*>(out.data));
        CUDA_CHECK(cudaGetLastError());

        constexpr int ReduceWarps = 1;
        constexpr int ReduceRows  = kContextQueryQHeads * Tokens;
        const dim3 reduce_grid((ReduceRows + ReduceWarps - 1) / ReduceWarps, 1, q.ne[3]);
        sliding_window_attention_reduce_kernel<Tokens, KeyBlock, ReduceWarps>
            <<<reduce_grid, ReduceWarps * 32, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(partial_acc.data),
                static_cast<const float*>(partial_m.data),
                static_cast<const float*>(partial_l.data),
                static_cast<const std::int32_t*>(positions.data),
                static_cast<const std::int32_t*>(valid_columns.data), plan.max_context,
                plan.split_capacity, static_cast<__nv_bfloat16*>(out.data));
        CUDA_CHECK(cudaGetLastError());
    });
#endif
}

} // namespace ninfer::ops::detail
