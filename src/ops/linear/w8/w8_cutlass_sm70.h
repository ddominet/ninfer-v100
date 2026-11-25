#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

#ifdef NINFER_VOLTA_BUILD

// Opportunistic CUTLASS SM70 Tensor-Core routes for wide-T W8 prefill projections (DFlash2
// feature_projection 5120x25600 and context_kv_materialize prepacked 1024+1024x5120).
// Both return false (launching nothing) when `workspace` does not have enough free capacity or
// the shape is outside the wide-T band, allowing callers to fall back without inflating peak
// workspace reservations.
[[nodiscard]] bool w8_cutlass_sm70_try_launch(const Tensor& x, const Weight& w, Tensor& out,
                                              WorkspaceArena& workspace, cudaStream_t stream);

[[nodiscard]] bool w8_cutlass_sm70_try_launch_pair(const Tensor& x, const Weight& first,
                                                   const Weight& second, Tensor& first_out,
                                                   Tensor& second_out, WorkspaceArena& workspace,
                                                   cudaStream_t stream);

#endif

} // namespace ninfer::ops::detail
