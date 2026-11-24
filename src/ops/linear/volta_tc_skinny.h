#pragma once

// Host-side controls for the sm_70 tensor-core skinny GEMM route (volta_tc_skinny.cuh).

#include <cuda_runtime.h>

#include <cstddef>

namespace ninfer::ops::detail {

struct VoltaTcControl {
    // From NINFER_VOLTA_TC (default off); 0: off; 1: on.
    int enabled = -1;
    // Smallest token count the route takes (NINFER_VOLTA_TC_MIN_T, default 9: the quadpair
    // kernels own one to eight tokens).
    int min_tokens = 0;
    // Schedule overrides for tuning; 0 keeps the measured choice.
    int rows   = 0;
    int splits = 0;
};

// Process-wide controls (benchmarks and tests may change them between launches).
VoltaTcControl& volta_tc_control();

// True when the route is enabled and takes this shape: 9..64 tokens by default, N a multiple of
// 64 and K a multiple of 128 (whole prepacked blocks for every format).
bool volta_tc_applies(int n, int k, int t);

struct VoltaTcSplitWorkspace {
    float* partial        = nullptr; // FP32 [t][n] partial sums of a split-K launch
    unsigned* semaphores  = nullptr; // one per row block, zero between launches
};

// Persistent per-stream split-K workspace, zeroed once at allocation; the kernel leaves its
// semaphores at zero. Returns nulls when the request does not fit, the stream is capturing before
// its block exists, or the device had no memory left for it (callers then launch unsplit).
VoltaTcSplitWorkspace volta_tc_split_workspace(cudaStream_t stream, std::size_t partial_floats,
                                               std::size_t semaphores);

} // namespace ninfer::ops::detail
