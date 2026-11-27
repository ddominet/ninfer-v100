#pragma once

#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/linear.h"
#include "ops/linear/nvfp4/nvfp4_w4a4_plan.h"
#ifdef NINFER_VOLTA_BUILD
#include "ops/linear/volta_fp16_stage.h"
#endif

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

[[nodiscard]] std::size_t nvfp4_linear_add_workspace_capacity_bytes(std::int32_t output_rows,
                                                                    std::int32_t input_rows,
                                                                    LinearPolicy policy,
                                                                    std::int32_t min_tokens,
                                                                    std::int32_t max_tokens);

void nvfp4_linear_add_decode_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                                    cudaStream_t stream);

void nvfp4_linear_add_small_t_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                                     cudaStream_t stream);

void nvfp4_linear_add_w4a4_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                                  Nvfp4W4a4Workspace workspace, cudaStream_t stream);

#ifdef NINFER_VOLTA_BUILD
void nvfp4_linear_add_qpn_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                                 cudaStream_t stream);
void nvfp4_linear_add_qpn_fp16_launch(const Tensor& x_shape, const half* x_fp16,
                                      const Weight& weight, Tensor& residual, cudaStream_t stream);
void nvfp4_linear_add_qpn_group_major_launch(const Tensor& x_shape,
                                             const VoltaGroupMajorHalf* x_gm, const Weight& weight,
                                             Tensor& residual, cudaStream_t stream);
#endif

void nvfp4_linear_add_dispatch(const Tensor& x, const Weight& weight, Tensor& residual,
                               LinearPolicy policy, WorkspaceArena& workspace, cudaStream_t stream);

} // namespace ninfer::ops::detail
