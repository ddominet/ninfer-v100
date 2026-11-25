#pragma once

// ninfer::ops::detail - private launch prototype for rmsnorm.

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void rmsnorm_launch(const Tensor& x, const Tensor& weight, float eps, bool unit_offset,
                    const Tensor* z, Tensor& out, cudaStream_t stream);
// Gated, x read through a column map (one warp a row): see gated_rmsnorm_columns.
void gated_rmsnorm_columns_launch(const Tensor& x, const Tensor& weight, const Tensor& z,
                                  float eps, const Tensor& x_columns, Tensor& out,
                                  cudaStream_t stream);

} // namespace ninfer::ops::detail
