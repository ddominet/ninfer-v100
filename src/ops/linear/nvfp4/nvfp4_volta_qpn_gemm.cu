#include "core/device.h"
#include "ops/linear/nvfp4/nvfp4_launch.h"
#include "ops/linear/nvfp4/nvfp4_volta_qpn_gemm.cuh"

#include <cuda_bf16.h>

#include <algorithm>

namespace ninfer::ops::detail {

#ifdef NINFER_VOLTA_BUILD

namespace {
using S = Nvfp4VoltaQpnSchedule;
static_assert(kNvfp4VoltaQpnRowsPerTile == S::kRowsPerTile,
              "dispatch mirrors the tile height for host code");

__global__ void nvfp4_argmax_unpack_kernel(const unsigned long long* __restrict__ packed_max,
                                           std::int32_t* __restrict__ out_tokens, int t) {
    const int i = static_cast<int>(threadIdx.x);
    if (i < t) {
        const unsigned long long packed = packed_max[i];
        out_tokens[i] =
            packed == 0ULL ? 0 : static_cast<std::int32_t>(~static_cast<std::uint32_t>(packed));
    }
}
} // namespace

bool nvfp4_volta_qpn_supported(std::int32_t n, std::int32_t k, std::int32_t t) noexcept {
    if (n <= 0 || t <= 0) { return false; }
    // The scale plane's BlockScaleK16M128x4 swizzle tiles on 128 output rows.
    if (n % 128 != 0) { return false; }
    // A lane consumes whole 64-k chunks (four NVFP4 groups, matching the swizzle's contiguous
    // scale-byte run), and the CTA's SPLITK warps split K by whole chunks. SPLITK=8 is the
    // production floor across every kTiles bucket (see launch_nvfp4_volta_qpn_with_output).
    if (k % S::kChunkK != 0) { return false; }
    if (k / S::kChunkK < 8) { return false; }
    // Prepacked runtime weights also use QPN2 at T=1; checkpoint-native weights keep the existing
    // SIMT decode route through the semantic-op plan.
    return t >= 1 && t <= kVoltaQpnMaxTokens;
}

void launch_nvfp4_volta_qpn(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    const std::int32_t n               = out.ne[0];
    const float inverse_weight_divisor = 1.0F / w.weight_scale_divisor;
    launch_nvfp4_volta_qpn_with_output(
        x, w, Nvfp4ContiguousOutput{static_cast<__nv_bfloat16*>(out.data), n}, n,
        inverse_weight_divisor, stream);
}

namespace {

// The output head's FP16 hidden plane arrives row-major in the caller's workspace (never in the
// per-stream staging block, so the repack cannot alias it). Past one 8-row tile the prepacked
// kernel reads a group-major plane in one cache line per load instead of eight.
const VoltaGroupMajorHalf* head_group_major(const Tensor& x, const Weight& w, const void* x_fp16,
                                            cudaStream_t stream) {
    if (x.ne[1] <= 8 || w.layout != QuantLayout::VoltaQpnPrepacked ||
        volta_tc_applies(w.n, x.ne[0], x.ne[1], kVoltaTcNvfp4MinTokens)) {
        return nullptr;
    }
    return volta_stage_fp16_to_group_major(static_cast<const half*>(x_fp16), x.ne[0], x.ne[1],
                                           stream);
}

} // namespace

void launch_nvfp4_volta_qpn_fp16(const Tensor& x, const Weight& w, const void* x_fp16, Tensor& out,
                                 cudaStream_t stream) {
    const std::int32_t n               = out.ne[0];
    const float inverse_weight_divisor = 1.0F / w.weight_scale_divisor;
    const Nvfp4ContiguousOutput output{static_cast<__nv_bfloat16*>(out.data), n};
    if (const VoltaGroupMajorHalf* staged = head_group_major(x, w, x_fp16, stream)) {
        launch_nvfp4_volta_qpn_with_group_major_activation(x, w, staged, output, n,
                                                           inverse_weight_divisor, stream);
        return;
    }
    launch_nvfp4_volta_qpn_with_fp16_activation(x, w, static_cast<const half*>(x_fp16), output, n,
                                                inverse_weight_divisor, stream);
}

void launch_nvfp4_volta_qpn_argmax_fp16(const Tensor& x, const Weight& w, const void* x_fp16,
                                        Tensor& out_tokens, std::int32_t valid_rows,
                                        void* scratch_u64, cudaStream_t stream) {
    const std::int32_t t               = x.ne[1];
    const std::int32_t n               = w.n;
    const float inverse_weight_divisor = 1.0F / w.weight_scale_divisor;
    auto* packed_max                   = static_cast<unsigned long long*>(scratch_u64);
    CUDA_CHECK(cudaMemsetAsync(packed_max, 0,
                               static_cast<std::size_t>(t) * sizeof(unsigned long long), stream));
    const Nvfp4ArgmaxOutput output{packed_max, std::min(valid_rows, n)};
    if (const VoltaGroupMajorHalf* staged = head_group_major(x, w, x_fp16, stream)) {
        launch_nvfp4_volta_qpn_with_group_major_activation(x, w, staged, output, n,
                                                           inverse_weight_divisor, stream);
    } else {
        launch_nvfp4_volta_qpn_with_fp16_activation(x, w, static_cast<const half*>(x_fp16), output,
                                                    n, inverse_weight_divisor, stream);
    }
    nvfp4_argmax_unpack_kernel<<<1, static_cast<unsigned>(t), 0, stream>>>(
        packed_max, static_cast<std::int32_t*>(out_tokens.data), t);
    CUDA_CHECK(cudaGetLastError());
}

#endif // NINFER_VOLTA_BUILD

} // namespace ninfer::ops::detail
