#pragma once

// Decode-width QPN kernels read the activation tile once per weight group in every lane. With a
// BF16 activation that is 16 bf16->fp16 conversions per lane per group inside the weight-streaming
// loop, which caps the down / residual-output projections near 390-520 GB/s on a V100; staging
// the K*T activation as FP16 once (a few microseconds) lifts the same kernels to ~690 GB/s.
#include "core/tensor.h"
#include "ops/linear/volta_qpn_width.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace ninfer::ops::detail {

// Converts the BF16 tensor x to FP16 in a persistent per-stream device scratch and returns it, or
// nullptr when staging is disabled (NINFER_VOLTA_STAGE_FP16=0). The scratch is valid until the
// next call on the same stream.
const half* volta_stage_fp16_activation(const Tensor& x, cudaStream_t stream);

// Group-major staging for the prepacked NVFP4 QPN kernels: [K/16 group][row tile][k half][8 rows]
// [8 k], with rows past T zero-filled. A warp's two 16-byte activation loads per weight group
// then cover exactly one 128-byte line each; the row-major form touches eight lines per load,
// which is what made the 9..32-row kernels 1.45x slower. The wrapper type keeps
// a row-major pointer from being passed to a kernel that indexes this layout.
struct VoltaGroupMajorHalf {
    half value;
};

inline constexpr int volta_group_major_tiles(int tokens) { return volta_qpn_tiles(tokens); }

// x is BF16 {K, T} with K % 16 == 0 and T <= kVoltaQpnMaxTokens. Returns nullptr when disabled
// (NINFER_VOLTA_STAGE_FP16=0 or NINFER_VOLTA_GROUP_MAJOR=0). Shares the per-stream scratch with
// volta_stage_fp16_activation: the result is valid until the next staging call on the stream.
const VoltaGroupMajorHalf* volta_stage_fp16_group_major(const Tensor& x, cudaStream_t stream);

// A second, 1 MiB per-stream block for the attention query plane, independent of the activation
// staging above. Returns nullptr when the request does not fit or the block is unavailable;
// callers keep their unstaged route.
void* volta_stream_scratch(cudaStream_t stream, std::size_t bytes);

} // namespace ninfer::ops::detail
