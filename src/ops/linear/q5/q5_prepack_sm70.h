#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

// In-place load-time permutation of a Q5G64 weight into the coalesced half-split Volta QPN /
// WMMA skinny layout (QuantLayout::VoltaQpnPrepacked):
//   codes:  [N/32 tile][K/64 group][half(0..1)][lane(0..31)][16 B]  (512 B/warp load)
//   high:   [N/32 tile][K/64 group][half(0..1)][lane(0..31)][4 B]   (128 B/warp load)
//   scales: [N/32 tile][K/64 group][lane(0..31)][2 B]              ( 64 B/warp load)
// Requires N % 32 == 0, K % 64 == 0, padded_shape[1] == K.
void q5_prepack_qpn_sm70(Weight& weight, cudaStream_t stream = nullptr);

} // namespace ninfer::ops::detail
