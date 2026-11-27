#pragma once

#include "core/tensor.h"
#include "ops/linear/volta_fp16_stage.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

#ifdef NINFER_VOLTA_BUILD

[[nodiscard]] inline bool volta_qpn_groupwise_supported(const Weight& w) noexcept {
    return w.layout == QuantLayout::VoltaQpnPrepacked && w.n > 0 && w.k > 0 && (w.n % 64 == 0) &&
           (w.k % 128 == 0) && w.padded_shape[1] == w.k;
}

// Contiguous / strided linear (BF16 x -> BF16 out), supports any T >= 1 (1..64 multi-tile QPN
// with group-major FP16 activation staging, >64 chunked by 64).
void launch_q4_prepacked_volta(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q5_prepacked_volta(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q6_prepacked_volta(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);

// Same reading pre-staged row-major FP16 activations directly (T = 1..64; auto-repacks to
// group-major for T > 8).
void launch_q4_prepacked_volta_fp16(const Tensor& x_shape, const half* x_fp16, const Weight& w,
                                    Tensor& out, cudaStream_t stream);
void launch_q6_prepacked_volta_fp16(const Tensor& x_shape, const half* x_fp16, const Weight& w,
                                    Tensor& out, cudaStream_t stream);

// Residual add (linear_add: residual_out += W @ x), T = 1..64.
void launch_q4_prepacked_volta_residual(const Tensor& x, const Weight& w, Tensor& residual_out,
                                        cudaStream_t stream);
void launch_q5_prepacked_volta_residual(const Tensor& x, const Weight& w, Tensor& residual_out,
                                        cudaStream_t stream);
void launch_q4_prepacked_volta_residual_fp16(const Tensor& x_shape, const half* x_fp16,
                                             const Weight& w, Tensor& residual_out,
                                             cudaStream_t stream);
void launch_q5_prepacked_volta_residual_fp16(const Tensor& x_shape, const half* x_fp16,
                                             const Weight& w, Tensor& residual_out,
                                             cudaStream_t stream);
void launch_q4_prepacked_volta_residual_gm(const Tensor& x_shape, const VoltaGroupMajorHalf* x_gm,
                                           const Weight& w, Tensor& residual_out,
                                           cudaStream_t stream);
void launch_q5_prepacked_volta_residual_gm(const Tensor& x_shape, const VoltaGroupMajorHalf* x_gm,
                                           const Weight& w, Tensor& residual_out,
                                           cudaStream_t stream);

// Split-row output (parent rows [0, split_row) -> out0, [split_row, n) -> out1), T = 1..64.
void launch_q4_prepacked_volta_split(const Tensor& x, const Weight& w, Tensor& out0, Tensor& out1,
                                     std::int32_t split_row, cudaStream_t stream);
void launch_q5_prepacked_volta_split(const Tensor& x, const Weight& w, Tensor& out0, Tensor& out1,
                                     std::int32_t split_row, cudaStream_t stream);
void launch_q4_prepacked_volta_split_fp16(const Tensor& x_shape, const half* x_fp16,
                                          const Weight& w, Tensor& out0, Tensor& out1,
                                          std::int32_t split_row, cudaStream_t stream);
void launch_q5_prepacked_volta_split_fp16(const Tensor& x_shape, const half* x_fp16,
                                          const Weight& w, Tensor& out0, Tensor& out1,
                                          std::int32_t split_row, cudaStream_t stream);
void launch_q4_prepacked_volta_split_gm(const Tensor& x_shape, const VoltaGroupMajorHalf* x_gm,
                                        const Weight& w, Tensor& out0, Tensor& out1,
                                        std::int32_t split_row, cudaStream_t stream);
void launch_q5_prepacked_volta_split_gm(const Tensor& x_shape, const VoltaGroupMajorHalf* x_gm,
                                        const Weight& w, Tensor& out0, Tensor& out1,
                                        std::int32_t split_row, cudaStream_t stream);

// GDN compacted-verify scatter projection (FP16 staged x -> qkv_padded[col_map[t]] + z[t]),
// T = 1..64.
void launch_q4_prepacked_volta_scatter_fp16(const Tensor& x_shape, const half* x_fp16,
                                            const Weight& w, Tensor& out_padded,
                                            const std::int32_t* col_map, cudaStream_t stream);
void launch_q5_prepacked_volta_split_scatter_fp16(const Tensor& x_shape, const half* x_fp16,
                                                  const Weight& w, Tensor& out0_padded,
                                                  Tensor& out1_dense, std::int32_t split_row,
                                                  const std::int32_t* out0_col_map,
                                                  cudaStream_t stream);

// Q4 split SwiGLU (T = 1..64):
// - zero-workspace BF16 -> BF16
void launch_q4_prepacked_volta_swiglu(const Tensor& x, const Weight& w, Tensor& out,
                                      cudaStream_t stream);
// - fused group-major FP16-in / group-major FP16-out using FP32 gate scratch (matches NVFP4 path)
void launch_q4_prepacked_volta_swiglu_gm_io(const Tensor& x_shape, const VoltaGroupMajorHalf* x_gm,
                                            const Weight& w, VoltaGroupMajorHalf* out_gm,
                                            float* gate_scratch, cudaStream_t stream);

// Q6 fused greedy argmax reading FP16 staged activations directly (T = 1..64, zero logits HBM).
void launch_q6_prepacked_volta_argmax_fp16(const Tensor& x_shape, const half* x_fp16,
                                           const Weight& w, Tensor& out_ids,
                                           std::int32_t valid_rows, void* argmax_scratch,
                                           cudaStream_t stream);

#endif // NINFER_VOLTA_BUILD

} // namespace ninfer::ops::detail
