#pragma once

#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/linear.h"

#include <cuda_runtime.h>
#ifdef NINFER_VOLTA_BUILD
#include <cuda_fp16.h>
#endif

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

[[nodiscard]] std::size_t fp8_linear_swiglu_workspace_capacity_bytes(LinearPolicy policy,
                                                                     std::int32_t min_tokens,
                                                                     std::int32_t max_tokens);

void fp8_linear_swiglu_decode_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                     cudaStream_t stream);
void fp8_linear_swiglu_small_t_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                      cudaStream_t stream);
void fp8_linear_swiglu_a8_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                 WorkspaceArena& workspace, cudaStream_t stream);

#ifdef NINFER_VOLTA_BUILD
void fp8_linear_swiglu_qpn_split_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                        float* gate_scratch, float* up_scratch,
                                        void* activation_scratch,
                                        cudaStream_t stream);
void fp8_linear_swiglu_qpn_split_fp16_io_launch(const Tensor& x_shape, const half* x_fp16,
                                                const Weight& weight, half* out_fp16,
                                                float* gate_scratch, cudaStream_t stream);
[[nodiscard]] bool fp8_linear_swiglu_qpn_split_supported(std::int32_t k,
                                                          std::int32_t t) noexcept;
#endif

void fp8_linear_swiglu_dispatch(const Tensor& x, const Weight& weight, Tensor& out,
                                LinearPolicy policy, WorkspaceArena& workspace,
                                cudaStream_t stream);

} // namespace ninfer::ops::detail
