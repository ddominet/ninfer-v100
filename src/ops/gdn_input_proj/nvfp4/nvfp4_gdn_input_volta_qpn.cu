#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_output.cuh"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_plan.h"
#include "ops/linear/nvfp4/nvfp4_launch.h"
#include "ops/linear/nvfp4/nvfp4_volta_qpn_gemm.cuh"

#include "ops/linear/volta_fp16_stage.h"

#include <cuda_bf16.h>

#include <cstdlib>

namespace ninfer::ops::detail {

#ifdef NINFER_VOLTA_BUILD

namespace {

// The caller's FP16 hidden plane is row-major, where the prepacked kernel's two 16-byte activation
// loads per weight group touch eight cache lines each instead of one. Repacking it group-major
// first (a K*T-half copy, a few microseconds) measured 83 -> 74 us per launch at one 8-row tile
// and 181 -> 110 us at three on the 27B GDN input projection, so every width repacks.
// NINFER_NVFP4_GDN_GM_MIN_T overrides the first token count that repacks; 0 disables it.
int gdn_group_major_min_tokens() {
    static const int value = [] {
        const char* env = std::getenv("NINFER_NVFP4_GDN_GM_MIN_T");
        return env != nullptr ? std::atoi(env) : 1;
    }();
    return value;
}

template <class OutputPolicy>
void launch_gdn_input_fp16(const Tensor& x, const Weight& weight, const half* x_fp16,
                           OutputPolicy output, float inverse_weight_divisor,
                           cudaStream_t stream) {
    const int min_tokens = gdn_group_major_min_tokens();
    if (min_tokens > 0 && x.ne[1] >= min_tokens && weight.layout == QuantLayout::VoltaQpnPrepacked &&
        !volta_tc_applies(weight.n, x.ne[0], x.ne[1], kVoltaTcNvfp4MinTokens)) {
        if (const VoltaGroupMajorHalf* staged =
                volta_stage_fp16_to_group_major(x_fp16, x.ne[0], x.ne[1], stream)) {
            launch_nvfp4_volta_qpn_with_group_major_activation(x, weight, staged, output, weight.n,
                                                               inverse_weight_divisor, stream);
            return;
        }
    }
    launch_nvfp4_volta_qpn_with_fp16_activation(x, weight, x_fp16, output, weight.n,
                                                inverse_weight_divisor, stream);
}

} // namespace

void launch_nvfp4_gdn_input_volta_qpn(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                      const void* x_fp16, cudaStream_t stream) {
    const Nvfp4GdnInputOutput output{static_cast<__nv_bfloat16*>(qkv.data),
                                     static_cast<__nv_bfloat16*>(z.data)};
    const float inverse_weight_divisor = 1.0F / weight.weight_scale_divisor;
    if (x_fp16 != nullptr) {
        launch_gdn_input_fp16(x, weight, static_cast<const half*>(x_fp16), output,
                              inverse_weight_divisor, stream);
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
        launch_gdn_input_fp16(x, weight, static_cast<const half*>(x_fp16), output,
                              inverse_weight_divisor, stream);
    } else {
        launch_nvfp4_volta_qpn_with_output(x, weight, output, weight.n, inverse_weight_divisor,
                                           stream);
    }
    return true;
}

#endif // NINFER_VOLTA_BUILD

} // namespace ninfer::ops::detail
