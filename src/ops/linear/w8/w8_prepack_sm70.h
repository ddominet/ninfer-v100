#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

// In-place load-time permutation of a W8G32 weight from its row-major codes and group scales to
// the fragment order the W8 quadpair kernel reads with one contiguous load per warp
// (w8_volta_qpn_gemm.cuh, kPrepacked). The payload size is unchanged. Only weights consumed
// exclusively through the prepacked-aware routes (QPN, one launch per 64 tokens) may be permuted: every
// other W8 route reads row-major and refuses a prepacked weight.
void w8_prepack_qpn_sm70(Weight& weight, cudaStream_t stream = nullptr);

} // namespace ninfer::ops::detail
