#include "ops/linear/volta_qpn_width.h"
#include "ops/linear_add/nvfp4/nvfp4_linear_add_plan.h"

#include "core/layout.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/residual_add.h"
#include "ops/linear/nvfp4/nvfp4_config.h"
#ifdef NINFER_VOLTA_BUILD
#include "ops/linear/nvfp4/nvfp4_cutlass_sm70.h"
#include "ops/linear/nvfp4/nvfp4_launch.h"
#endif

#include <cstdlib>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

enum class Nvfp4LinearAddRoute : std::uint8_t {
    A16,
#ifdef NINFER_VOLTA_BUILD
    LinearThenAdd,
#endif
    W4A4,
};

Nvfp4LinearAddRoute resolve_route(std::int32_t output_rows, std::int32_t input_rows,
                                  LinearPolicy policy, std::int32_t tokens) {
    if (tokens <= 0 || output_rows != 5120 || (input_rows != 6144 && input_rows != 17408)) {
        throw std::invalid_argument("nvfp4 linear_add: unsupported shape");
    }
    if (policy == LinearPolicy::A16Only) {
#ifdef NINFER_VOLTA_BUILD
        // Prepacked down projections are consumed by QPN2 for every decode width. Materialize the
        // projection and add the residual separately because the row-major fused kernels cannot
        // read that load-time layout.
        return Nvfp4LinearAddRoute::LinearThenAdd;
#else
        return Nvfp4LinearAddRoute::A16;
#endif
    }
    if (policy != LinearPolicy::AllowA4) {
        throw std::invalid_argument("nvfp4 linear_add: unsupported policy");
    }
    const std::int32_t first_w4a4 = input_rows == 6144 ? 7 : 8;
    if (tokens >= first_w4a4) { return Nvfp4LinearAddRoute::W4A4; }
#ifdef NINFER_VOLTA_BUILD
    return Nvfp4LinearAddRoute::LinearThenAdd;
#else
    return Nvfp4LinearAddRoute::A16;
#endif
}

void launch_a16(const Tensor& x, const Weight& weight, Tensor& residual, cudaStream_t stream) {
    constexpr std::int32_t kChunk = kNvfp4LastSmallT;
    for (std::int32_t token_begin = 0; token_begin < x.ne[1]; token_begin += kChunk) {
        const std::int32_t active = std::min(kChunk, x.ne[1] - token_begin);
        auto* input               = static_cast<std::uint8_t*>(x.data) +
                      static_cast<std::int64_t>(token_begin) * weight.k * sizeof(std::uint16_t);
        auto* output = static_cast<std::uint8_t*>(residual.data) +
                       static_cast<std::int64_t>(token_begin) * weight.n * sizeof(std::uint16_t);
        Tensor input_chunk(input, DType::BF16, {weight.k, active});
        Tensor residual_chunk(output, DType::BF16, {weight.n, active});
        if (active == 1) {
            nvfp4_linear_add_decode_launch(input_chunk, weight, residual_chunk, stream);
        } else {
            nvfp4_linear_add_small_t_launch(input_chunk, weight, residual_chunk, stream);
        }
    }
}

#ifdef NINFER_VOLTA_BUILD
template <class Allocator>
Tensor allocate_projected(Allocator& allocator, std::int32_t output_rows, std::int32_t tokens) {
    return allocator.alloc(DType::BF16, {output_rows, tokens}, 256);
}

// Opt-in (NINFER_NVFP4_DOWN_T1_GEMV=1): the SIMT GEMV measures 590 GB/s in situ, the tensor-core
// QPN pass with a staged FP16 activation ~690, so QPN is the default at T=1 as well.
bool nvfp4_down_t1_gemv_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("NINFER_NVFP4_DOWN_T1_GEMV");
        return value != nullptr && value[0] == '1';
    }();
    return enabled;
}

void launch_linear_then_add(const Tensor& x, const Weight& weight, Tensor& residual,
                             WorkspaceArena& workspace, cudaStream_t stream) {
    if (nvfp4_down_t1_gemv_enabled() && x.ne[1] == 1 && weight.layout == QuantLayout::VoltaQpnPrepacked) {
        auto scope = workspace.scope();
        // T=1: the shared prepack GEMV (~590 GB/s in-situ) replaces the QPN2 pass (~419 GB/s,
        // 6.71 ms/token over 56 layers in the d8k nsys decode profile).
        nvfp4_linear_add_gemv_t1_sm70_launch(x, weight, residual, workspace, stream);
        return;
    }
    if (x.ne[1] < kVoltaCutlassMinT &&
        nvfp4_volta_qpn_supported(weight.n, weight.k, std::min(x.ne[1], kVoltaQpnMaxTokens))) {
        for (std::int32_t begin = 0; begin < x.ne[1]; begin += kVoltaQpnMaxTokens) {
            const std::int32_t active = std::min(kVoltaQpnMaxTokens, x.ne[1] - begin);
            Tensor input(static_cast<std::uint8_t*>(x.data) +
                             static_cast<std::int64_t>(begin) * weight.k * sizeof(std::uint16_t),
                         DType::BF16, {weight.k, active});
            Tensor output(static_cast<std::uint8_t*>(residual.data) +
                              static_cast<std::int64_t>(begin) * weight.n * sizeof(std::uint16_t),
                          DType::BF16, {weight.n, active});
            nvfp4_linear_add_qpn_launch(input, weight, output, stream);
        }
        return;
    }
    auto scope       = workspace.scope();
    Tensor projected = allocate_projected(workspace, weight.n, x.ne[1]);
    if (x.ne[1] >= kVoltaCutlassMinT) {
        nvfp4_cutlass_sm70_launch(x, weight, projected, workspace, stream);
    } else {
        const std::size_t linear_bytes = std::max<std::size_t>(
            linear_workspace_capacity_bytes(QType::NVFP4, weight.n, weight.k,
                                            kNvfp4InternalPolicy, x.ne[1], x.ne[1]),
            256);
        WorkspaceArena linear_workspace(workspace.alloc_bytes(linear_bytes, 256));
        linear(x, weight, projected, kNvfp4InternalPolicy, linear_workspace, stream);
    }
    residual_add(projected, residual, stream);
}

std::size_t linear_then_add_workspace_bytes(std::int32_t output_rows, std::int32_t input_rows,
                                            std::int32_t tokens) {
    if (tokens < kVoltaCutlassMinT &&
        nvfp4_volta_qpn_supported(output_rows, input_rows, std::min(tokens, kVoltaQpnMaxTokens))) {
        return 0;
    }
    WorkspaceLayoutBuilder layout;
    (void)allocate_projected(layout, output_rows, tokens);
    const std::size_t linear_bytes = tokens >= kVoltaCutlassMinT
        ? nvfp4_cutlass_sm70_workspace_bytes(output_rows, input_rows, tokens)
        : std::max<std::size_t>(
              linear_workspace_capacity_bytes(QType::NVFP4, output_rows, input_rows,
                                              kNvfp4InternalPolicy, tokens, tokens),
              256);
    (void)layout.alloc_bytes(linear_bytes, 256);
    return layout.peak_bytes(1);
}

std::size_t linear_then_add_gemv_bytes(std::int32_t output_rows, std::int32_t input_rows) {
    return nvfp4_linear_add_gemv_t1_sm70_workspace_bytes(output_rows, input_rows);
}
#endif // NINFER_VOLTA_BUILD

} // namespace

std::size_t nvfp4_linear_add_workspace_capacity_bytes(std::int32_t output_rows,
                                                      std::int32_t input_rows, LinearPolicy policy,
                                                      std::int32_t min_tokens,
                                                      std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("nvfp4 linear_add workspace: invalid token interval");
    }
    (void)resolve_route(output_rows, input_rows, policy, min_tokens);
    const Nvfp4LinearAddRoute route = resolve_route(output_rows, input_rows, policy, max_tokens);
    if (route == Nvfp4LinearAddRoute::W4A4) {
        return nvfp4_w4a4_workspace_capacity_bytes(max_tokens, input_rows);
    }
#ifdef NINFER_VOLTA_BUILD
    if (route == Nvfp4LinearAddRoute::LinearThenAdd) {
        std::size_t bytes = linear_then_add_workspace_bytes(output_rows, input_rows, max_tokens);
        if (min_tokens <= 1 && nvfp4_down_t1_gemv_enabled()) {
            // The T=1 route allocates fp32 scratch + fp16 activation instead of a projected
            // plane; any interval that can reach T=1 must cover it.
            bytes = std::max(bytes, linear_then_add_gemv_bytes(output_rows, input_rows));
        }
        return bytes;
    }
#endif
    return 0;
}

void nvfp4_linear_add_dispatch(const Tensor& x, const Weight& weight, Tensor& residual,
                               LinearPolicy policy, WorkspaceArena& workspace,
                               cudaStream_t stream) {
    const Nvfp4LinearAddRoute route = resolve_route(weight.n, weight.k, policy, x.ne[1]);
    if (route == Nvfp4LinearAddRoute::A16) {
        launch_a16(x, weight, residual, stream);
        return;
    }
#ifdef NINFER_VOLTA_BUILD
    if (route == Nvfp4LinearAddRoute::LinearThenAdd) {
        launch_linear_then_add(x, weight, residual, workspace, stream);
        return;
    }
#endif
    auto scope                       = workspace.scope();
    const Nvfp4W4a4Workspace scratch = allocate_nvfp4_w4a4_workspace(workspace, x.ne[1], weight.k);
    nvfp4_linear_add_w4a4_launch(x, weight, residual, scratch, stream);
}

} // namespace ninfer::ops::detail
