#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

[[nodiscard]] std::size_t nvfp4_cutlass_sm70_workspace_bytes(std::int32_t n, std::int32_t k,
                                                              std::int32_t cols);
void nvfp4_cutlass_sm70_launch(const Tensor& x, const Weight& w, Tensor& out, WorkspaceArena& ws,
                               cudaStream_t stream);

// Prefill-width SwiGLU: x {k, cols} BF16, w the fused gate/up weight (n = 2 * out rows), out
// {n / 2, cols} BF16. Gate and up stay FP32 through silu(gate) * up, so the output is rounded
// once like the decode-width split route; rounding both to BF16 first compounds to more than one
// output ULP.
[[nodiscard]] std::size_t nvfp4_linear_swiglu_cutlass_sm70_workspace_bytes(std::int32_t n,
                                                                           std::int32_t k,
                                                                           std::int32_t cols);
void nvfp4_linear_swiglu_cutlass_sm70_launch(const Tensor& x, const Weight& w, Tensor& out,
                                             WorkspaceArena& ws, cudaStream_t stream);

// T=1 prepack-GEMV route for the down projection with in-place residual add.
void nvfp4_linear_add_gemv_t1_sm70_launch(const Tensor& x, const Weight& w, Tensor& residual,
                                          WorkspaceArena& ws, cudaStream_t stream);
[[nodiscard]] std::size_t nvfp4_linear_add_gemv_t1_sm70_workspace_bytes(std::int32_t output_rows,
                                                                        std::int32_t input_rows);

} // namespace ninfer::ops::detail
