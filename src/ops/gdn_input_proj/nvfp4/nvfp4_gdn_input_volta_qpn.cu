#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_output.cuh"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_plan.h"
#include "ops/linear/nvfp4/nvfp4_launch.h"
#include "ops/linear/nvfp4/nvfp4_volta_qpn_gemm.cuh"

#include <cuda_bf16.h>

namespace ninfer::ops::detail {

#ifdef NINFER_VOLTA_BUILD

void launch_nvfp4_gdn_input_volta_qpn(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                      const void* x_fp16, cudaStream_t stream) {
    const Nvfp4GdnInputOutput output{static_cast<__nv_bfloat16*>(qkv.data),
                                     static_cast<__nv_bfloat16*>(z.data)};
    const float inverse_weight_divisor = 1.0F / weight.weight_scale_divisor;
    if (x_fp16 != nullptr) {
        launch_nvfp4_volta_qpn_with_fp16_activation(x, weight, static_cast<const half*>(x_fp16),
                                                    output, weight.n, inverse_weight_divisor,
                                                    stream);
    } else {
        launch_nvfp4_volta_qpn_with_output(x, weight, output, weight.n, inverse_weight_divisor,
                                           stream);
    }
}

bool launch_nvfp4_gdn_input_volta_qpn_scatter(const Tensor& x, const Weight& weight,
                                              Tensor& qkv_padded, Tensor& z,
                                              const Tensor& qkv_columns, cudaStream_t stream,
                                              const void* x_fp16) {
    if (!nvfp4_volta_qpn_supported(weight.n, weight.k, x.ne[1])) { return false; }
    const Nvfp4GdnInputOutput output{static_cast<__nv_bfloat16*>(qkv_padded.data),
                                     static_cast<__nv_bfloat16*>(z.data),
                                     static_cast<const std::int32_t*>(qkv_columns.data)};
    const float inverse_weight_divisor = 1.0F / weight.weight_scale_divisor;
    if (x_fp16 != nullptr) {
        launch_nvfp4_volta_qpn_with_fp16_activation(x, weight, static_cast<const half*>(x_fp16),
                                                    output, weight.n, inverse_weight_divisor,
                                                    stream);
    } else {
        launch_nvfp4_volta_qpn_with_output(x, weight, output, weight.n, inverse_weight_divisor,
                                           stream);
    }
    return true;
}

#endif // NINFER_VOLTA_BUILD

} // namespace ninfer::ops::detail
