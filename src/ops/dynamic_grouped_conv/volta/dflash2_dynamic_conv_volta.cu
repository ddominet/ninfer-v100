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
#include "ops/common/volta_mma.cuh"
#include "ops/common/warp.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
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

// Volta HMMA (mma.sync.m8n8k4) BF16 projection for DFlash2 control (N=1280, K=5120) and
// candidate-selector hidden projection (N=256, K=5120). Each CTA owns 8 output rows with 4
// K-split warps (160 CTAs = 2.0 exact waves on 80 SMs for N=1280; zero global semaphores or
// partials). One 16-byte weight load feeds all 32*kColTiles columns via Tensor Cores.
__device__ __forceinline__ void bf16x8_to_half2x4(uint4 raw, unsigned& h0, unsigned& h1,
                                                  unsigned& h2, unsigned& h3) {
    const auto* p   = reinterpret_cast<const __nv_bfloat162*>(&raw);
    const float2 f0 = __bfloat1622float2(p[0]);
    const float2 f1 = __bfloat1622float2(p[1]);
    const float2 f2 = __bfloat1622float2(p[2]);
    const float2 f3 = __bfloat1622float2(p[3]);
    const __half2 q0 = __floats2half2_rn(f0.x, f0.y);
    const __half2 q1 = __floats2half2_rn(f1.x, f1.y);
    const __half2 q2 = __floats2half2_rn(f2.x, f2.y);
    const __half2 q3 = __floats2half2_rn(f3.x, f3.y);
    h0 = *reinterpret_cast<const unsigned*>(&q0);
    h1 = *reinterpret_cast<const unsigned*>(&q1);
    h2 = *reinterpret_cast<const unsigned*>(&q2);
    h3 = *reinterpret_cast<const unsigned*>(&q3);
}

// Coalesced 8-warp Volta HMMA (mma.sync.m8n8k4) BF16 projection for DFlash2 control (N=1280,
// K=5120) and candidate-selector hidden projection (N=256, K=5120). Each CTA owns 8 output rows
// with 8 K-split warps (160 CTAs x 8 warps = 16 warps/SM on 80 SMs for N=1280). Every warp loads
// 8 rows x 64 BF16 via 2 coalesced 128-byte cache-line uint4 loads per lane (read once, converted
// BF16->FP16 once into per-warp smem, register-prefetched 1 tile ahead, zero CTA barriers in the
// K-loop).
template <int kColTiles>
__global__ __launch_bounds__(256, 2) void dflash2_bf16_hmma_kernel(
    const __nv_bfloat16* __restrict__ w, const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ out, int n_rows, int k, int cols) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ == 700
    constexpr int kWarps  = 8;
    constexpr int kTileK  = 64;
    constexpr int kLdsW   = kTileK + 8; // padded half stride to eliminate smem bank conflicts
    constexpr int kMaxCol = kColTiles * 32;
    const int row0        = static_cast<int>(blockIdx.x) * 8;
    const int tid         = static_cast<int>(threadIdx.x);
    const int lane        = tid & 31;
    const int warp        = tid >> 5;
    const int r           = (lane & 3) + ((lane & 16) ? 4 : 0);
    const int qp          = (lane >> 2) & 3;
    const int b_sel       = qp * 4 + (lane & 3) + ((lane & 16) ? 16 : 0);

    // Coalesced weight loader mapping inside each warp: 8 lanes cover 64 BF16 (128B cache line).
    const int w_r  = lane >> 3;      // 0..3
    const int w_c8 = (lane & 7) * 8; // 0, 8, ..., 56

    union Shared {
        half w_tile[kWarps][8][kLdsW];
        float c_red[kWarps][8][kMaxCol];
    };
    __shared__ Shared sh;

    float d[kColTiles][8];
#pragma unroll
    for (int ct = 0; ct < kColTiles; ++ct) {
#pragma unroll
        for (int i = 0; i < 8; ++i) { d[ct][i] = 0.0f; }
    }

    const int total_tiles = k / kTileK;
    const int tile_begin  = (total_tiles * warp) / kWarps;
    const int tile_end    = (total_tiles * (warp + 1)) / kWarps;

    const __nv_bfloat16* w_lo_row = w + static_cast<std::int64_t>(row0 + w_r) * k + w_c8;
    const __nv_bfloat16* w_hi_row = w + static_cast<std::int64_t>(row0 + w_r + 4) * k + w_c8;

    uint4 cur_lo{};
    uint4 cur_hi{};
    if (tile_begin < tile_end) {
        const std::int64_t off0 = static_cast<std::int64_t>(tile_begin) * kTileK;
        cur_lo = __ldg(reinterpret_cast<const uint4*>(w_lo_row + off0));
        cur_hi = __ldg(reinterpret_cast<const uint4*>(w_hi_row + off0));
    }

    for (int tile = tile_begin; tile < tile_end; ++tile) {
        // Convert current weight tile BF16->FP16 once per element into warp-private smem.
        uint4 h_lo, h_hi;
        bf16x8_to_half2x4(cur_lo, h_lo.x, h_lo.y, h_lo.z, h_lo.w);
        bf16x8_to_half2x4(cur_hi, h_hi.x, h_hi.y, h_hi.z, h_hi.w);
        *reinterpret_cast<uint4*>(&sh.w_tile[warp][w_r][w_c8])     = h_lo;
        *reinterpret_cast<uint4*>(&sh.w_tile[warp][w_r + 4][w_c8]) = h_hi;
        __syncwarp();

        // Prefetch next K-tile's coalesced weight lines before executing 8 HMMA slices.
        if (tile + 1 < tile_end) {
            const std::int64_t next_off = static_cast<std::int64_t>(tile + 1) * kTileK;
            cur_lo = __ldg(reinterpret_cast<const uint4*>(w_lo_row + next_off));
            cur_hi = __ldg(reinterpret_cast<const uint4*>(w_hi_row + next_off));
        }

        const int k_tile_base = tile * kTileK;
#pragma unroll
        for (int ks = 0; ks < 8; ++ks) {
            const uint4 a_h = *reinterpret_cast<const uint4*>(&sh.w_tile[warp][r][ks * 8]);
            const int kk    = k_tile_base + ks * 8;
#pragma unroll
            for (int ct = 0; ct < kColTiles; ++ct) {
                const int col = ct * 32 + b_sel;
                unsigned b0 = 0u, b1 = 0u, b2 = 0u, b3 = 0u;
                if (col < cols) {
                    const uint4 b_raw = __ldg(
                        reinterpret_cast<const uint4*>(x + static_cast<std::int64_t>(col) * k + kk));
                    bf16x8_to_half2x4(b_raw, b0, b1, b2, b3);
                }
                volta_mma_qp_n(d[ct], a_h.x, a_h.y, b0, b1);
                volta_mma_qp_n(d[ct], a_h.z, a_h.w, b2, b3);
            }
        }
        __syncwarp();
    }

    __syncthreads(); // ensure all warps finish reading sh.w_tile before reusing union as sh.c_red

    const int c_row_lo = ((lane & 16) ? 4 : 0) | (lane & 1);
    const int c_sel    = qp * 4 + (((lane >> 1) & 1) << 1);
#pragma unroll
    for (int ct = 0; ct < kColTiles; ++ct) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int rr = (i & 2) | c_row_lo;
            const int cc = ct * 32 + c_sel + (i & 1) + ((i >> 2) << 4);
            sh.c_red[warp][rr][cc] = d[ct][i];
        }
    }
    __syncthreads();

    for (int e = tid; e < 8 * kMaxCol; e += 256) {
        const int rr = e & 7;
        const int cc = e >> 3;
        if (cc < cols) {
            float total = 0.0f;
#pragma unroll
            for (int ww = 0; ww < kWarps; ++ww) { total += sh.c_red[ww][rr][cc]; }
            out[row0 + rr + static_cast<std::int64_t>(cc) * n_rows] = __float2bfloat16_rn(total);
        }
    }
#endif
}

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
    int first = 0;
    const bool aligned16 =
        ((reinterpret_cast<std::uintptr_t>(w) | reinterpret_cast<std::uintptr_t>(x)) & 15u) == 0;
    if ((n_rows % 8) == 0 && (k % 64) == 0 && aligned16) {
        while (first < cols) {
            const int block = std::min(64, cols - first);
            if (block > 32) {
                dflash2_bf16_hmma_kernel<2><<<n_rows / 8, 256, 0, stream>>>(
                    w, x + static_cast<std::int64_t>(first) * k,
                    out + static_cast<std::int64_t>(first) * n_rows, n_rows, k, block);
            } else {
                dflash2_bf16_hmma_kernel<1><<<n_rows / 8, 256, 0, stream>>>(
                    w, x + static_cast<std::int64_t>(first) * k,
                    out + static_cast<std::int64_t>(first) * n_rows, n_rows, k, block);
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
