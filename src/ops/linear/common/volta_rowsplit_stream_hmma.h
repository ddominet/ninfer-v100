#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

#ifdef NINFER_VOLTA_BUILD

inline constexpr std::int32_t kVoltaStreamHmmaMaxTokens = 16;

[[nodiscard]] inline bool volta_rowsplit_stream_hmma_supported(std::int32_t n, std::int32_t k,
                                                               std::int32_t padded_k,
                                                               std::int32_t t) noexcept {
    return n > 0 && k > 0 && t > 0 && (k % 128 == 0) && padded_k == k;
}

// Q4 / Q5 / Q6 contiguous linear (BF16 x -> BF16 out).
void launch_q4_volta_stream_hmma(const Tensor& x, const Weight& w, Tensor& out,
                                 cudaStream_t stream);
void launch_q5_volta_stream_hmma(const Tensor& x, const Weight& w, Tensor& out,
                                 cudaStream_t stream);
void launch_q6_volta_stream_hmma(const Tensor& x, const Weight& w, Tensor& out,
                                 cudaStream_t stream);

// Q4 / Q5 fused residual add (BF16 x -> BF16 residual_out += W @ x).
void launch_q4_volta_stream_hmma_residual(const Tensor& x, const Weight& w, Tensor& residual_out,
                                          cudaStream_t stream);
void launch_q5_volta_stream_hmma_residual(const Tensor& x, const Weight& w, Tensor& residual_out,
                                          cudaStream_t stream);

// Q4 / Q5 split-row output (parent weight rows [0, split_row) -> out0, [split_row, n) -> out1).
void launch_q4_volta_stream_hmma_split(const Tensor& x, const Weight& w, Tensor& out0, Tensor& out1,
                                       std::int32_t split_row, cudaStream_t stream);
void launch_q5_volta_stream_hmma_split(const Tensor& x, const Weight& w, Tensor& out0, Tensor& out1,
                                       std::int32_t split_row, cudaStream_t stream);

// Q4 fused SwiGLU (gate_up [2*half_n, k] -> out [half_n, t], zero workspace, zero silu_mul).
void launch_q4_volta_stream_hmma_swiglu(const Tensor& x, const Weight& w, Tensor& out,
                                        cudaStream_t stream);

// Q6 fused greedy argmax (FP16 staged activations -> out_ids [t], zero logits HBM writes).
// `argmax_scratch` must hold at least `t * sizeof(unsigned long long)` bytes.
void launch_q6_volta_stream_hmma_argmax_fp16(const Tensor& x_shape, const Weight& w,
                                             const void* x_fp16, Tensor& out_ids,
                                             std::int32_t valid_rows, void* argmax_scratch,
                                             cudaStream_t stream);

// Q6 linear reading FP16 staged activations directly (zero BF16->FP16 conversion in output head).
void launch_q6_volta_stream_hmma_fp16(const Tensor& x_shape, const Weight& w, const void* x_fp16,
                                      Tensor& out, cudaStream_t stream);

#endif // NINFER_VOLTA_BUILD

} // namespace ninfer::ops::detail
