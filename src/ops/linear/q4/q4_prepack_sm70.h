#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

// In-place load-time permutation of a Q4G64 weight from its row-major codes and group scales to
// the order the fused Q4 quadpair top-K kernel reads with one contiguous load per warp
// (q4_volta_qpn_gemm.cuh, kPrepacked): codes [N/32 tile][group][lane][32 bytes], scales
// [N/32 tile][group][lane]. Measured on the DFlash2 131,072 x 5120 proposal head: 770-820 us a
// read against 1,140 us row-major, with identical outputs. Only a head read exclusively by
// linear_topk's fused route may be permuted; every other Q4 route refuses a prepacked weight.
void q4_prepack_qpn_sm70(Weight& weight, cudaStream_t stream = nullptr);

} // namespace ninfer::ops::detail
