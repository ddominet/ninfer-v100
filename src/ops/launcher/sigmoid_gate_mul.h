#pragma once

// ninfer::ops::detail - private launch prototype for sigmoid_mul. Included by the wrapper
// (host) and defined by the launcher (.cu). Not part of the public api.
// See docs/op-development.md §2.

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

// Host entry; assumes inputs already validated by the wrapper.
void sigmoid_gate_mul_launch(const Tensor& gate, Tensor& x, cudaStream_t stream);

// Fixed-route launch control used by production and qualification benchmarks.
void sigmoid_gate_mul_bf16x8_launch(const Tensor& gate, Tensor& x, int block, cudaStream_t stream);

#ifdef NINFER_VOLTA_BUILD
// In-place BF16 sigmoid gate multiply emitting FP16 into x's storage for FP8 QPN/TC.
[[nodiscard]] bool sigmoid_gate_mul_to_fp16_launch(const Tensor& gate, Tensor& x,
                                                   cudaStream_t stream);
#endif

} // namespace ninfer::ops::detail
