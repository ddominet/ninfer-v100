#include "ops/linear_add/nvfp4/nvfp4_linear_add_plan.h"

#include "core/device.h"
#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_gemv.cuh"
#include "ops/linear_add/nvfp4/nvfp4_linear_add_epilogue.cuh"
#ifdef NINFER_VOLTA_BUILD
#include "ops/linear/nvfp4/nvfp4_volta_qpn_gemm.cuh"
#endif

namespace ninfer::ops::detail {
namespace {

template <class Geometry>
void launch(const Tensor& x, const Weight& weight, Tensor& residual, cudaStream_t stream) {
    using Schedule        = typename Nvfp4LinearDecodeProductionSchedule<Geometry>::Type;
    constexpr int kBlocks = Geometry::kOutputRows / Schedule::kRowsPerCta;
    const float inverse   = 1.0F / weight.weight_scale_divisor;
    auto* output          = static_cast<__nv_bfloat16*>(residual.data);
    nvfp4_gemv_kernel<Geometry, Schedule><<<kBlocks, Schedule::kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.scales), inverse,
        Nvfp4AddResidualEpilogue{output, Geometry::kOutputRows},
        Nvfp4ContiguousOutput{output, Geometry::kOutputRows});
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void nvfp4_linear_add_decode_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                                    cudaStream_t stream) {
    switch (resolve_nvfp4_problem(weight.n, weight.k)) {
    case Nvfp4Problem::Residual6144:
        launch<Nvfp4Residual6144Geometry>(x, weight, residual, stream);
        return;
    case Nvfp4Problem::Residual17408:
        launch<Nvfp4Residual17408Geometry>(x, weight, residual, stream);
        return;
    case Nvfp4Problem::AttnInput:
    case Nvfp4Problem::GdnInput:
    case Nvfp4Problem::MlpGateUp:
    case Nvfp4Problem::OutputHead:
        break;
    }
    throw std::invalid_argument("nvfp4 linear_add: unsupported problem");
}

#ifdef NINFER_VOLTA_BUILD
void nvfp4_linear_add_qpn_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                                 cudaStream_t stream) {
    const float inverse = 1.0F / weight.weight_scale_divisor;
    launch_nvfp4_volta_qpn_with_output(
        x, weight, Nvfp4ResidualOutput{static_cast<__nv_bfloat16*>(residual.data), weight.n},
        weight.n, inverse, stream);
}

void nvfp4_linear_add_qpn_fp16_launch(const Tensor& x_shape, const half* x_fp16,
                                      const Weight& weight, Tensor& residual,
                                      cudaStream_t stream) {
    const float inverse = 1.0F / weight.weight_scale_divisor;
    launch_nvfp4_volta_qpn_with_fp16_activation(
        x_shape, weight, x_fp16,
        Nvfp4ResidualOutput{static_cast<__nv_bfloat16*>(residual.data), weight.n}, weight.n,
        inverse, stream);
}

void nvfp4_linear_add_qpn_group_major_launch(const Tensor& x_shape,
                                             const VoltaGroupMajorHalf* x_gm, const Weight& weight,
                                             Tensor& residual, cudaStream_t stream) {
    const float inverse = 1.0F / weight.weight_scale_divisor;
    launch_nvfp4_volta_qpn_with_group_major_activation(
        x_shape, weight, x_gm,
        Nvfp4ResidualOutput{static_cast<__nv_bfloat16*>(residual.data), weight.n}, weight.n,
        inverse, stream);
}
#endif

} // namespace ninfer::ops::detail
