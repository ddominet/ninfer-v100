// DFlash2 dynamic grouped convolution — sm_70.
//
// A short (2-tap) depthwise filter with per-group dynamic gains, applied inside one request
// block. 320 groups of 16 channels; taps = {current position, previous position}; two sides.
//
//   Conv_s(X)[c,i,b] = (base[c,0,s] + d_s0) * X[c,i,b]
//                    + 1{i>=1} * (base[c,1,s] + d_s1) * X[c,i-1,b]        (g = c/16)
//
// prepare (side 0) consumes the full W_delta projection `delta` [side(2) x tap(2) x group(320),
// col] (col = i + W*b), writes the filtered input to `prepared`, and stashes the side-1 gains as
// `finish_delta` [group(320) x tap(2) x col]. add (side 1) reads that `finish_delta` back.

#include "core/device.h"
#include "ops/common/warp.cuh"
#include "ops/linear/volta_tc_skinny.cuh"

#include <cuda_bf16.h>
#include <algorithm>
#include <cstdint>

namespace ninfer::ops::detail {

namespace {

constexpr int kHidden = 5120;
constexpr int kGroup  = 16;
constexpr int kGroups = kHidden / kGroup; // 320

__device__ __forceinline__ float bf(const __nv_bfloat16* p, std::int64_t i) {
    return __bfloat162float(p[i]);
}

__device__ __forceinline__ int delta_row(int side, int tap, int g) {
    return (side * 2 + tap) * kGroups + g;
}

__global__ void dflash2_dynamic_conv_prepare_kernel(const __nv_bfloat16* __restrict__ n,
                                                    const __nv_bfloat16* __restrict__ delta,
                                                    const __nv_bfloat16* __restrict__ base,
                                                    int width, int batch,
                                                    __nv_bfloat16* __restrict__ prepared,
                                                    __nv_bfloat16* __restrict__ finish) {
    const std::int64_t idx   = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::int64_t total = static_cast<std::int64_t>(kHidden) * width * batch;
    if (idx >= total) { return; }

    const int c   = static_cast<int>(idx % kHidden);
    const int rem = static_cast<int>(idx / kHidden);
    const int i   = rem % width;
    const int b   = rem / width;
    const int g   = c / kGroup;
    const int col = i + width * b;

    const float base_t0 = bf(base, c + static_cast<std::int64_t>(kHidden) * 0);         // tap0 side0
    const float base_t1 = bf(base, c + static_cast<std::int64_t>(kHidden) * 1);         // tap1 side0
    const float d_t0    = bf(delta, delta_row(0, 0, g) + 1280LL * col);
    const float d_t1    = bf(delta, delta_row(0, 1, g) + 1280LL * col);

    const std::int64_t out_index = c + static_cast<std::int64_t>(kHidden) * col;
    float acc                    = (base_t0 + d_t0) * bf(n, out_index);
    if (i >= 1) {
        acc += (base_t1 + d_t1) * bf(n, out_index - kHidden);
    }
    prepared[out_index] = __float2bfloat16_rn(acc);

    if (finish != nullptr && (c % kGroup) == 0) {
        finish[g + 320LL * (0 + 2 * col)] = delta[delta_row(1, 0, g) + 1280LL * col];
        finish[g + 320LL * (1 + 2 * col)] = delta[delta_row(1, 1, g) + 1280LL * col];
    }
}

__global__ void dflash2_dynamic_conv_finish_add_kernel(
    const __nv_bfloat16* __restrict__ projected, const __nv_bfloat16* __restrict__ finish_delta,
    const __nv_bfloat16* __restrict__ base, int width, int batch,
    __nv_bfloat16* __restrict__ residual) {
    const std::int64_t idx   = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::int64_t total = static_cast<std::int64_t>(kHidden) * width * batch;
    if (idx >= total) { return; }

    const int c   = static_cast<int>(idx % kHidden);
    const int rem = static_cast<int>(idx / kHidden);
    const int i   = rem % width;
    const int b   = rem / width;
    const int g   = c / kGroup;
    const int col = i + width * b;

    const float base_t0 = bf(base, c + static_cast<std::int64_t>(kHidden) * (0 + 2)); // tap0 side1
    const float base_t1 = bf(base, c + static_cast<std::int64_t>(kHidden) * (1 + 2)); // tap1 side1
    const float d_t0    = bf(finish_delta, g + 320LL * (0 + 2 * col));
    const float d_t1    = bf(finish_delta, g + 320LL * (1 + 2 * col));

    const std::int64_t out_index = c + static_cast<std::int64_t>(kHidden) * col;
    float acc                    = (base_t0 + d_t0) * bf(projected, out_index);
    if (i >= 1) {
        acc += (base_t1 + d_t1) * bf(projected, out_index - kHidden);
    }
    residual[out_index] = __float2bfloat16_rn(__bfloat162float(residual[out_index]) + acc);
}


// The W_delta control projection is BF16_CTRL [1280, 5120]; the fork general BF16 linear only
// admits a few fixed shapes, so a small dedicated matvec covers it. One block per (row, col).
__global__ void dflash2_bf16_matvec_kernel(const __nv_bfloat16* __restrict__ w,
                                           const __nv_bfloat16* __restrict__ x, int n_rows, int k,
                                           int cols, __nv_bfloat16* __restrict__ out) {
    const int row = static_cast<int>(blockIdx.x);
    const int col = static_cast<int>(blockIdx.y);
    if (row >= n_rows || col >= cols) { return; }
    const __nv_bfloat16* wrow = w + static_cast<std::int64_t>(row) * k;
    const __nv_bfloat16* xcol = x + static_cast<std::int64_t>(col) * k;
    float acc = 0.0f;
    for (int i = static_cast<int>(threadIdx.x); i < k; i += static_cast<int>(blockDim.x)) {
        acc += __bfloat162float(wrow[i]) * __bfloat162float(xcol[i]);
    }
    __shared__ float sums[4];
    const float total = block_reduce_sum<128>(acc, sums);
    if (threadIdx.x == 0) {
        out[row + static_cast<std::int64_t>(col) * n_rows] = __float2bfloat16_rn(total);
    }
}

// Four weight rows a CTA: each activation value is loaded once and used by four rows, 8 K values
// a thread per step in 16-byte loads. One row a CTA reread every activation column per weight row
// (~200 MB of cache traffic for the 13 MB weight at 16 columns, 104 us); scalar 2-byte loads still
// held four rows at 64 us.
template <int kCols>
__global__ __launch_bounds__(128) void dflash2_bf16_control_rows4_kernel(
    const __nv_bfloat16* __restrict__ w, const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ out, int cols) {
    constexpr int kRows  = 1280;
    constexpr int k      = 5120;
    constexpr int kTile  = 4;
    constexpr int kWarps = 4;
    constexpr int kVec   = 8;
    const int row0       = static_cast<int>(blockIdx.x) * kTile;
    const int lane       = static_cast<int>(threadIdx.x) & 31;
    const int warp       = static_cast<int>(threadIdx.x) >> 5;
    float acc[kTile][kCols] = {};

    for (int i = static_cast<int>(threadIdx.x) * kVec; i < k; i += 128 * kVec) {
        float wv[kTile][kVec];
#pragma unroll
        for (int r = 0; r < kTile; ++r) {
            const uint4 raw = __ldg(reinterpret_cast<const uint4*>(
                w + static_cast<std::int64_t>(row0 + r) * k + i));
            const auto* pairs = reinterpret_cast<const __nv_bfloat162*>(&raw);
#pragma unroll
            for (int e = 0; e < kVec / 2; ++e) {
                const float2 f = __bfloat1622float2(pairs[e]);
                wv[r][2 * e]     = f.x;
                wv[r][2 * e + 1] = f.y;
            }
        }
#pragma unroll
        for (int col = 0; col < kCols; ++col) {
            if (col < cols) {
                const uint4 raw = __ldg(
                    reinterpret_cast<const uint4*>(x + static_cast<std::int64_t>(col) * k + i));
                const auto* pairs = reinterpret_cast<const __nv_bfloat162*>(&raw);
#pragma unroll
                for (int e = 0; e < kVec / 2; ++e) {
                    const float2 f = __bfloat1622float2(pairs[e]);
#pragma unroll
                    for (int r = 0; r < kTile; ++r) {
                        acc[r][col] = fmaf(wv[r][2 * e], f.x, acc[r][col]);
                        acc[r][col] = fmaf(wv[r][2 * e + 1], f.y, acc[r][col]);
                    }
                }
            }
        }
    }

    __shared__ float sums[kWarps][kTile][kCols];
#pragma unroll
    for (int r = 0; r < kTile; ++r) {
#pragma unroll
        for (int col = 0; col < kCols; ++col) {
            const float total = warp_reduce_sum(acc[r][col]);
            if (lane == 0) { sums[warp][r][col] = total; }
        }
    }
    __syncthreads();
    for (int e = static_cast<int>(threadIdx.x); e < kTile * kCols; e += 128) {
        const int r   = e / kCols;
        const int col = e % kCols;
        if (col < cols) {
            const float total =
                (sums[0][r][col] + sums[2][r][col]) + (sums[1][r][col] + sums[3][r][col]);
            out[row0 + r + static_cast<std::int64_t>(col) * kRows] = __float2bfloat16_rn(total);
        }
    }
}

// Row-major BF16 weights for the tensor-core skinny route (volta_tc_skinny.cuh): one item is 8
// adjacent K of one row of a 32-row tile in a 64-K chunk, converted to FP16 in shared memory.
// The control weights are small coefficients well inside FP16's range.
struct Bf16VoltaTcDecoder {
    const __nv_bfloat16* weight;
    int k;

    static constexpr int kItemsPerTileChunk = 256; // 32 rows x 8 vectors of 8
    static constexpr int kMinTokens         = 1;
    static constexpr bool kMandatory        = false;
    static constexpr int kTargetCtas        = 320; // 20 row blocks: split K 16 ways
    static constexpr int kMinChunks         = 2;
    struct Raw {
        uint4 q;
    };

    __device__ __forceinline__ Raw load(int tile, int chunk, int item) const {
        const std::int64_t row = static_cast<std::int64_t>(tile) * 32 + (item >> 3);
        return {__ldg(reinterpret_cast<const uint4*>(weight + row * k + chunk * 64 +
                                                     (item & 7) * 8))};
    }
    __device__ __forceinline__ void store(const Raw& raw, int item, half* tile, int lds) const {
        const auto* pairs = reinterpret_cast<const __nv_bfloat162*>(&raw.q);
        half2 values[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) { values[i] = __float22half2_rn(__bfloat1622float2(pairs[i])); }
        *reinterpret_cast<uint4*>(tile + (item >> 3) * lds + (item & 7) * 8) =
            *reinterpret_cast<const uint4*>(values);
    }
    __device__ __forceinline__ float finish(int, float value) const { return value; }
};

struct Bf16ControlOutput {
    __nv_bfloat16* out;
    int rows;
    __device__ __forceinline__ void store(int row, int token, float value) const {
        out[static_cast<std::int64_t>(token) * rows + row] = __float2bfloat16_rn(value);
    }
};

} // namespace

void dflash2_dynamic_conv_prepare_launch(const __nv_bfloat16* n, const __nv_bfloat16* delta,
                                         const __nv_bfloat16* base, int width, int batch,
                                         __nv_bfloat16* prepared, __nv_bfloat16* finish_delta,
                                         cudaStream_t stream) {
    const std::int64_t total = static_cast<std::int64_t>(kHidden) * width * batch;
    const int block          = 256;
    const int grid           = static_cast<int>((total + block - 1) / block);
    dflash2_dynamic_conv_prepare_kernel<<<grid, block, 0, stream>>>(n, delta, base, width, batch,
                                                                    prepared, finish_delta);
    CUDA_CHECK(cudaGetLastError());
}

void dflash2_bf16_control_projection_launch(const __nv_bfloat16* w, const __nv_bfloat16* x,
                                           int n_rows, int k, int cols, __nv_bfloat16* out,
                                           cudaStream_t stream) {
    // Up to 16 columns a launch through the four-row kernel: each weight row is read once for all
    // of them, where the scalar fallback rereads every weight row per column (0.9 ms a call at
    // eight requests). The vectorized reduction order differs from the scalar fallback's, so the
    // two agree to rounding, not bit for bit (the projection feeds the draft only).
    int first = 0;
    const bool aligned16 =
        ((reinterpret_cast<std::uintptr_t>(w) | reinterpret_cast<std::uintptr_t>(x)) & 15u) == 0;
    // The tensor-core skinny route reads the 13 MB weight once for up to 64 columns (one launch a
    // call instead of 16-column chunks at ~49 us each).
    if (n_rows % 64 == 0 && k % 128 == 0 && aligned16) {
        const Bf16VoltaTcDecoder decoder{w, k};
        for (; first < cols; first += 64) {
            const int count = std::min(64, cols - first);
            if (!launch_volta_tc_skinny(decoder, x + static_cast<std::int64_t>(first) * k, n_rows,
                                        k, count, 0,
                                        Bf16ControlOutput{out + static_cast<std::int64_t>(first) *
                                                                    n_rows,
                                                          n_rows},
                                        stream)) {
                break;
            }
        }
        if (first >= cols) { return; }
    }
    if (n_rows == 1280 && k == 5120 && aligned16) {
        // Any remainder takes the narrowest wide form that holds it: a 15-column block (three
        // drafts for five requests) was 8 + 7 scalar columns, the scalar part 124 us a call.
        // The 16-byte loads need 16-byte aligned rows (k = 5120 keeps every row aligned).
        while (first < cols) {
            const int block = std::min(16, cols - first);
            if (block > 8) {
                dflash2_bf16_control_rows4_kernel<16><<<n_rows / 4, 128, 0, stream>>>(
                    w, x + static_cast<std::int64_t>(first) * k,
                    out + static_cast<std::int64_t>(first) * n_rows, block);
            } else {
                dflash2_bf16_control_rows4_kernel<8><<<n_rows / 4, 128, 0, stream>>>(
                    w, x + static_cast<std::int64_t>(first) * k,
                    out + static_cast<std::int64_t>(first) * n_rows, block);
            }
            CUDA_CHECK(cudaGetLastError());
            first += block;
        }
    }
    if (first == cols) { return; }
    const dim3 grid(static_cast<unsigned>(n_rows), static_cast<unsigned>(cols - first));
    dflash2_bf16_matvec_kernel<<<grid, 128, 0, stream>>>(
        w, x + static_cast<std::int64_t>(first) * k, n_rows, k, cols - first,
        out + static_cast<std::int64_t>(first) * n_rows);
    CUDA_CHECK(cudaGetLastError());
}

void dflash2_dynamic_conv_finish_add_launch(const __nv_bfloat16* projected,
                                            const __nv_bfloat16* finish_delta,
                                            const __nv_bfloat16* base, int width, int batch,
                                            __nv_bfloat16* residual, cudaStream_t stream) {
    const std::int64_t total = static_cast<std::int64_t>(kHidden) * width * batch;
    const int block          = 256;
    const int grid           = static_cast<int>((total + block - 1) / block);
    dflash2_dynamic_conv_finish_add_kernel<<<grid, block, 0, stream>>>(projected, finish_delta,
                                                                       base, width, batch,
                                                                       residual);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
