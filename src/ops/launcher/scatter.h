#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void scatter_launch(const Tensor& src, const Tensor& indices, Tensor& dst, cudaStream_t stream);
void scatter_bf16_batch_launch(const Tensor& source, const Tensor& lanes,
                               const Tensor& valid_columns, Tensor& destination,
                               cudaStream_t stream);
void gather_columns_launch(const Tensor& source, const Tensor& indices, Tensor& destination,
                           cudaStream_t stream);
struct GatherColumnsGroup {
    const void* source[4]{};
    void* destination[4]{};
    std::int64_t words[4]{}; // 4-byte words per column
    int pairs         = 0;
    bool zero_dropped = true; // write zeros to columns whose index is negative
};
void gather_columns_group_launch(const GatherColumnsGroup& group, const Tensor& indices,
                                 std::int32_t columns, cudaStream_t stream);
} // namespace ninfer::ops::detail
