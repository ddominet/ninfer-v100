#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void prepare_ragged_prefix_launch(const Tensor& source, const Tensor& lanes, const Tensor& starts,
                                  const Tensor& ends, Tensor& destination, Tensor& positions,
                                  Tensor& counts, cudaStream_t stream);

void prepare_compact_prefix_launch(const Tensor& source, const Tensor& src_lanes,
                                   const Tensor& src_cols, Tensor& destination,
                                   cudaStream_t stream);

} // namespace ninfer::ops::detail
