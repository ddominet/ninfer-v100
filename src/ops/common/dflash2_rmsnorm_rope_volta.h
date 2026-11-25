#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

// Fused plain RMSNorm + full-128 split-half 1-D RoPE, in place. x is BF16 [128, heads, T],
// weight BF16 [128], positions I32 [T]. n stays FP32 through the rotation.
void dflash2_rmsnorm_rope_launch(const Tensor& x, const Tensor& weight, const Tensor& positions,
                                 float eps, float theta, cudaStream_t stream);

// Fused zero-centered RMSNorm + partial-64 1-D Text RoPE (theta = 1e7) for Qwen3.6/3.8 target
// attention: q/qn [256, n_q, T], k/kn [256, n_kv, T], q_norm/k_norm [256], positions I32 [T].
// Rounds normalized values to BF16 before rotation to stay bit-identical to rmsnorm + rope.
void target_rmsnorm_qk_rope_launch(const Tensor& q, const Tensor& k, const Tensor& q_norm,
                                   const Tensor& k_norm, const Tensor& positions, float eps,
                                   Tensor& qn, Tensor& kn, cudaStream_t stream);

} // namespace ninfer::ops::detail
