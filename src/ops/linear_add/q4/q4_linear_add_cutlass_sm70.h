#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

std::size_t q4_linear_add_cutlass_workspace_bytes(std::int32_t rows, std::int32_t k,
                                                  std::int32_t cols);

void q4_linear_add_cutlass_sm70_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                       WorkspaceArena& ws, cudaStream_t stream);

} // namespace ninfer::ops::detail
