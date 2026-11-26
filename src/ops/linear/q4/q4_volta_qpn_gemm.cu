#include "core/device.h"
#include "ops/linear/q4/q4_launch.h"
#include "ops/linear/q4/q4_volta_qpn_gemm.cuh"
#include "ops/linear/volta_fp16_stage.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <stdexcept>

namespace ninfer::ops::detail {

#ifdef NINFER_VOLTA_BUILD

namespace {
using S = Q4VoltaQpnSchedule;
} // namespace

bool q4_volta_qpn_supported(std::int32_t n, std::int32_t k, std::int32_t t) noexcept {
    if (n <= 0 || t <= 0) { return false; }
    // Whole Q4 groups per lane (the scale is per group and held across it), and one uint4 of
    // activations per mma pair.
    if (k % Q4RowSplitStorage::kGroupK != 0 || k % 8 != 0) { return false; }
    // The CTA's four warps split K by whole groups, so every warp needs at least one.
    if (k / Q4RowSplitStorage::kGroupK < S::kWarps) { return false; }
    return t <= 2 * S::kRowsPerTile;
}

void launch_q4_volta_qpn(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream,
                         std::int32_t weight_row_offset) {
    if (w.layout == QuantLayout::VoltaQpnPrepacked) {
        throw std::invalid_argument("Q4 QPN: a prepacked weight is readable only by fused top-K");
    }
    const std::int32_t n      = out.ne[0];
    const std::int32_t k      = x.ne[0];
    const std::int32_t t      = x.ne[1];
    const std::int32_t out_ld = static_cast<std::int32_t>(out.nb[1] / sizeof(__nv_bfloat16));
    const std::int32_t padded_groups = w.padded_shape[1] / Q4RowSplitStorage::kGroupK;

    const std::int64_t roff = static_cast<std::int64_t>(weight_row_offset) * padded_groups;
    const auto* codes =
        static_cast<const std::uint8_t*>(w.qdata) + roff * Q4RowSplitStorage::kCodeBytesPerGroup;
    const auto* scales =
        static_cast<const std::uint8_t*>(w.scales) + roff * Q4RowSplitStorage::kScaleBytesPerGroup;

    const dim3 grid(static_cast<unsigned>((n + S::kColsPerCta - 1) / S::kColsPerCta));
    // kBlk = 4 groups = 128 B, one cache line per lane per iteration; measured best at every
    // shape (see q4_volta_qpn_gemm.cuh and the V100 implementation).
#define NINFER_QPN_LAUNCH(TILES, BLK)                                                             \
    q4_volta_qpn_gemm_kernel<TILES, BLK><<<grid, S::kThreads, 0, stream>>>(                       \
        codes, scales, static_cast<const __nv_bfloat16*>(x.data),                                 \
        static_cast<__nv_bfloat16*>(out.data), n, k, t, padded_groups, out_ld)
    if (t <= S::kRowsPerTile) {
        NINFER_QPN_LAUNCH(1, 4);
    } else {
        NINFER_QPN_LAUNCH(2, 4);
    }
#undef NINFER_QPN_LAUNCH
    CUDA_CHECK(cudaGetLastError());
}

void launch_q4_volta_qpn_topk(const Tensor& x, const Weight& w,
                              const Tensor& row_to_global_ids, std::uint64_t* partial_keys,
                              std::int32_t producer_groups, cudaStream_t stream) {
    const std::int32_t n             = w.n;
    const std::int32_t k             = x.ne[0];
    const std::int32_t t             = x.ne[1];
    const std::int32_t padded_groups = w.padded_shape[1] / Q4RowSplitStorage::kGroupK;
    const auto* ids                  = static_cast<const std::int32_t*>(row_to_global_ids.data);
    const auto* codes                = static_cast<const std::uint8_t*>(w.qdata);
    const auto* scales               = static_cast<const std::uint8_t*>(w.scales);

    if (w.layout == QuantLayout::VoltaQpnPrepacked && t > 32 && (n % 64) == 0 && t <= 64) {
        const VoltaGroupMajorHalf* x_gm = volta_stage_fp16_group_major(x, stream);
        const half* x_fp16 = x_gm == nullptr ? volta_stage_fp16_activation(x, stream) : nullptr;
        const int x_tiles  = volta_group_major_tiles(t);
        const dim3 tc_grid(static_cast<unsigned>(n / 64));
        const auto dispatch_tc = [&]<int TT>() {
            constexpr std::size_t kSmem =
                (2 * 64 * 72 + 2 * 16 * TT * 72) * sizeof(half);
            if (x_gm != nullptr) {
                q4_volta_tc_topk_kernel<TT, VoltaGroupMajorHalf>
                    <<<tc_grid, 128, kSmem, stream>>>(codes, scales, x_gm, n, k, t, x_tiles,
                                                      padded_groups, ids, partial_keys,
                                                      producer_groups);
            } else if (x_fp16 != nullptr) {
                q4_volta_tc_topk_kernel<TT, half><<<tc_grid, 128, kSmem, stream>>>(
                    codes, scales, x_fp16, n, k, t, x_tiles, padded_groups, ids, partial_keys,
                    producer_groups);
            } else {
                q4_volta_tc_topk_kernel<TT, __nv_bfloat16><<<tc_grid, 128, kSmem, stream>>>(
                    codes, scales, static_cast<const __nv_bfloat16*>(x.data), n, k, t, x_tiles,
                    padded_groups, ids, partial_keys, producer_groups);
            }
        };
        if (t <= 16) {
            dispatch_tc.template operator()<1>();
        } else if (t <= 32) {
            dispatch_tc.template operator()<2>();
        } else if (t <= 48) {
            dispatch_tc.template operator()<3>();
        } else {
            dispatch_tc.template operator()<4>();
        }
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    const dim3 grid(static_cast<unsigned>((n + S::kColsPerCta - 1) / S::kColsPerCta));
    const VoltaGroupMajorHalf* x_gm =
        w.layout == QuantLayout::VoltaQpnPrepacked ? volta_stage_fp16_group_major(x, stream)
                                                   : nullptr;
    const half* x_fp16 = x_gm == nullptr ? volta_stage_fp16_activation(x, stream) : nullptr;

    const auto dispatch_tiles = [&]<int kTiles, int kBlk, bool kPrepacked>() {
        if (x_gm != nullptr) {
            q4_volta_qpn_gemm_kernel<kTiles, kBlk, true, kPrepacked, VoltaGroupMajorHalf>
                <<<grid, S::kThreads, 0, stream>>>(codes, scales, x_gm, nullptr, n, k, t,
                                                   padded_groups, 0, ids, partial_keys,
                                                   producer_groups);
        } else if (x_fp16 != nullptr) {
            q4_volta_qpn_gemm_kernel<kTiles, kBlk, true, kPrepacked, half>
                <<<grid, S::kThreads, 0, stream>>>(codes, scales, x_fp16, nullptr, n, k, t,
                                                   padded_groups, 0, ids, partial_keys,
                                                   producer_groups);
        } else {
            q4_volta_qpn_gemm_kernel<kTiles, kBlk, true, kPrepacked, __nv_bfloat16>
                <<<grid, S::kThreads, 0, stream>>>(codes, scales,
                                                   static_cast<const __nv_bfloat16*>(x.data),
                                                   nullptr, n, k, t, padded_groups, 0, ids,
                                                   partial_keys, producer_groups);
        }
    };

    if (w.layout == QuantLayout::VoltaQpnPrepacked) {
        if (t <= 8) {
            dispatch_tiles.template operator()<1, 2, true>();
        } else if (t <= 16) {
            dispatch_tiles.template operator()<2, 2, true>();
        } else if (t <= 24) {
            dispatch_tiles.template operator()<3, 2, true>();
        } else if (t <= 32) {
            dispatch_tiles.template operator()<4, 2, true>();
        } else {
            throw std::invalid_argument("Q4 QPN top-K: prepacked launch supports T <= 64");
        }
    } else {
        if (t <= 8) {
            dispatch_tiles.template operator()<1, 4, false>();
        } else if (t <= 16) {
            dispatch_tiles.template operator()<2, 4, false>();
        } else {
            throw std::invalid_argument("Q4 QPN top-K: row-split launch supports T <= 16");
        }
    }
    CUDA_CHECK(cudaGetLastError());
}

#endif // NINFER_VOLTA_BUILD

} // namespace ninfer::ops::detail
