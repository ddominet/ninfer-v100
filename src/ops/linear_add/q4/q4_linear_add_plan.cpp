#include "ops/linear_add/q4/q4_linear_add_plan.h"

#include "core/layout.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/residual_add.h"
#ifdef NINFER_VOLTA_BUILD
#include "ops/linear/common/volta_qpn_groupwise.h"
#include "ops/linear/common/volta_rowsplit_stream_hmma.h"
#include "ops/linear/q4/q4_launch.h"
#include "ops/linear_add/q4/q4_linear_add_cutlass_sm70.h"
#endif

#include <algorithm>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

bool supported_shape(std::int32_t rows, std::int32_t k, std::int32_t padded_k) noexcept {
    return rows == 5120 && padded_k == k && (k == 6144 || k == 17408);
}

#ifdef NINFER_VOLTA_BUILD
constexpr std::int32_t kQ4LinearAddStreamMaxCols   = 16;
constexpr std::int32_t kQ4LinearAddVoltaMmaMaxCols = 64;

std::size_t workspace_at(std::int32_t rows, std::int32_t k, std::int32_t cols) {
    if (cols <= kQ4LinearAddStreamMaxCols) { return 0; }
    if (cols <= kQ4LinearAddVoltaMmaMaxCols && q4_volta_mma_supported(rows, k, cols)) {
        return q4_volta_mma_workspace_bytes(rows, k, cols);
    }
    return q4_linear_add_cutlass_workspace_bytes(rows, k, cols);
}
#else
std::size_t workspace_at(std::int32_t rows, std::int32_t, std::int32_t cols) {
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::BF16, {rows, cols});
    return layout.peak_bytes(1);
}
#endif

} // namespace

std::size_t q4_linear_add_capacity_workspace_bytes(std::int32_t rows, std::int32_t k,
                                                   std::int32_t padded_k, std::int32_t min_cols,
                                                   std::int32_t max_cols) {
    if (!supported_shape(rows, k, padded_k) || min_cols <= 0 || max_cols < min_cols) {
        throw std::invalid_argument("q4 linear_add: unsupported shape or token interval");
    }
    std::size_t peak = std::max(workspace_at(rows, k, min_cols), workspace_at(rows, k, max_cols));
#ifdef NINFER_VOLTA_BUILD
    const std::int32_t mma_lo = std::max(min_cols, kQ4LinearAddStreamMaxCols + 1);
    const std::int32_t mma_hi = std::min(max_cols, kQ4LinearAddVoltaMmaMaxCols);
    for (std::int32_t c = mma_lo; c <= mma_hi; ++c) {
        peak = std::max(peak, workspace_at(rows, k, c));
    }
#endif
    return peak;
}

void q4_linear_add_dispatch(const Tensor& x, const Weight& w, Tensor& residual_out,
                            WorkspaceArena& ws, cudaStream_t stream) {
    const std::int32_t rows = residual_out.ne[0];
    const std::int32_t k    = x.ne[0];
    const std::int32_t cols = x.ne[1];
    if (!supported_shape(rows, k, w.padded_shape[1]) || cols <= 0) {
        throw std::invalid_argument("q4 linear_add: unsupported shape or T");
    }
#ifdef NINFER_VOLTA_BUILD
    if (w.layout == QuantLayout::VoltaQpnPrepacked && cols <= 64) {
        launch_q4_prepacked_volta_residual(x, w, residual_out, stream);
        return;
    }
    if (cols <= kQ4LinearAddStreamMaxCols) {
        launch_q4_volta_stream_hmma_residual(x, w, residual_out, stream);
        return;
    }
    if (cols <= kQ4LinearAddVoltaMmaMaxCols && q4_volta_mma_supported(rows, k, cols)) {
        launch_q4_volta_mma(x, w, residual_out, ws, stream, /*weight_row_offset=*/0,
                            /*splits_override=*/0, /*add_residual=*/true);
        return;
    }
    q4_linear_add_cutlass_sm70_launch(x, w, residual_out, ws, stream);
#else
    auto scope   = ws.scope();
    Tensor delta = ws.alloc(DType::BF16, {rows, cols});
    linear(x, w, delta, stream);
    residual_add(delta, residual_out, stream);
#endif
}

} // namespace ninfer::ops::detail
