#pragma once

// Coalesced VoltaQpnPrepacked -> row-major FP16 dequantization helpers for CUTLASS Sm70 prefill.
// Reads half-split prepacked Q4/Q5 weights with 512B/128B warp-coalesced loads, decodes via FP16
// magic-number arithmetic into warp-local shared memory, and writes 64-byte coalesced uint4 row
// segments (4 lanes/row x 16B) to CUTLASS ColumnMajor B.

#include "core/device.h"

#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops::detail {

__device__ __forceinline__ int prepacked_dequant_local_row(int lane) {
    return ((lane >> 2) & 3) * 8 + (lane & 3) + ((lane & 16) ? 4 : 0);
}

__global__ __launch_bounds__(256, 4) inline void dequant_q4_prepacked_to_fp16_kernel(
    const uint4* __restrict__ codes, const std::uint16_t* __restrict__ scales, int n, int k,
    half* __restrict__ out) {
    constexpr int kWarpsPerBlock = 8;
    __shared__ __align__(16) half smem[kWarpsPerBlock][32][40];

    const std::int64_t groups     = k / 64;
    const std::int64_t total_warps = static_cast<std::int64_t>(n / 32) * groups * 2;
    const int warp_id             = static_cast<int>(threadIdx.x) >> 5;
    const int lane                = static_cast<int>(threadIdx.x) & 31;
    const std::int64_t warp_idx =
        static_cast<std::int64_t>(blockIdx.x) * kWarpsPerBlock + warp_id;
    if (warp_idx >= total_warps) { return; }

    const std::int64_t idx   = warp_idx * 32 + lane;
    const int half_idx       = static_cast<int>(warp_idx & 1);
    const std::int64_t tg    = warp_idx >> 1;
    const std::int64_t group = tg % groups;
    const std::int64_t tile  = tg / groups;

    const uint4 q               = __ldg(codes + idx);
    const std::uint16_t sc_bits = __ldg(scales + tg * 32 + lane);
    const half2 sc2             = __half2half2(__ushort_as_half(sc_bits));
    const half2 bias            = __half2half2(__ushort_as_half(0x6408)); // 1032.0

    const int local_row          = prepacked_dequant_local_row(lane);
    const std::uint32_t words[4] = {q.x, q.y, q.z, q.w};
    auto* smem_row               = reinterpret_cast<uint4*>(&smem[warp_id][local_row][0]);
#pragma unroll
    for (int w = 0; w < 4; ++w) {
        const std::uint32_t p4 = words[w] ^ 0x88888888u;
        half2 b[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const std::uint32_t lb = (p4 >> (8 * j)) & 0xffu;
            std::uint32_t bits     = ((lb & 0x0fu) | ((lb & 0xf0u) << 12)) | 0x64006400u;
            b[j]                   = __hmul2(__hsub2(*reinterpret_cast<half2*>(&bits), bias), sc2);
        }
        smem_row[w] = *reinterpret_cast<const uint4*>(b);
    }
    __syncwarp();

    const std::int64_t row_base = tile * 32;
    const std::int64_t k_off    = group * 64 + half_idx * 32;
    const int sub_row           = lane >> 2;
    const int vec_col           = (lane & 3) * 8;
#pragma unroll
    for (int rb = 0; rb < 32; rb += 8) {
        const int lr  = rb + sub_row;
        const uint4 v = *reinterpret_cast<const uint4*>(&smem[warp_id][lr][vec_col]);
        *reinterpret_cast<uint4*>(out + (row_base + lr) * k + k_off + vec_col) = v;
    }
}

__global__ __launch_bounds__(256, 4) inline void dequant_q5_prepacked_to_fp16_kernel(
    const uint4* __restrict__ codes, const std::uint32_t* __restrict__ high,
    const std::uint16_t* __restrict__ scales, int n, int k, half* __restrict__ out) {
    constexpr int kWarpsPerBlock = 8;
    __shared__ __align__(16) half smem[kWarpsPerBlock][32][40];

    const std::int64_t groups      = k / 64;
    const std::int64_t total_warps = static_cast<std::int64_t>(n / 32) * groups * 2;
    const int warp_id              = static_cast<int>(threadIdx.x) >> 5;
    const int lane                 = static_cast<int>(threadIdx.x) & 31;
    const std::int64_t warp_idx =
        static_cast<std::int64_t>(blockIdx.x) * kWarpsPerBlock + warp_id;
    if (warp_idx >= total_warps) { return; }

    const std::int64_t idx   = warp_idx * 32 + lane;
    const int half_idx       = static_cast<int>(warp_idx & 1);
    const std::int64_t tg    = warp_idx >> 1;
    const std::int64_t group = tg % groups;
    const std::int64_t tile  = tg / groups;

    const uint4 q               = __ldg(codes + idx);
    const std::uint32_t h32     = __ldg(high + idx);
    const std::uint16_t sc_bits = __ldg(scales + tg * 32 + lane);
    const half2 sc2             = __half2half2(__ushort_as_half(sc_bits));
    const half2 bias            = __half2half2(__ushort_as_half(0x6410)); // 1040.0

    const int local_row          = prepacked_dequant_local_row(lane);
    const std::uint32_t words[4] = {q.x, q.y, q.z, q.w};
    auto* smem_row               = reinterpret_cast<uint4*>(&smem[warp_id][local_row][0]);
#pragma unroll
    for (int w = 0; w < 4; ++w) {
        const std::uint32_t hb = ((h32 >> (8 * w)) & 0xffu) ^ 0xffu;
        half2 b[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const std::uint32_t lb = (words[w] >> (8 * j)) & 0xffu;
            std::uint32_t bits     = ((lb & 0x0fu) | ((lb & 0xf0u) << 12)) | 0x64006400u;
            bits |= (((hb >> (2 * j)) & 1u) << 4) | (((hb >> (2 * j + 1)) & 1u) << 20);
            b[j] = __hmul2(__hsub2(*reinterpret_cast<half2*>(&bits), bias), sc2);
        }
        smem_row[w] = *reinterpret_cast<const uint4*>(b);
    }
    __syncwarp();

    const std::int64_t row_base = tile * 32;
    const std::int64_t k_off    = group * 64 + half_idx * 32;
    const int sub_row           = lane >> 2;
    const int vec_col           = (lane & 3) * 8;
#pragma unroll
    for (int rb = 0; rb < 32; rb += 8) {
        const int lr  = rb + sub_row;
        const uint4 v = *reinterpret_cast<const uint4*>(&smem[warp_id][lr][vec_col]);
        *reinterpret_cast<uint4*>(out + (row_base + lr) * k + k_off + vec_col) = v;
    }
}

inline void launch_dequant_q4_prepacked_to_fp16(const void* codes, const void* scales, int n, int k,
                                                void* out_fp16, cudaStream_t stream) {
    const std::int64_t total_warps = static_cast<std::int64_t>(n / 32) * (k / 64) * 2;
    const int blocks               = static_cast<int>((total_warps + 7) / 8);
    dequant_q4_prepacked_to_fp16_kernel<<<blocks, 256, 0, stream>>>(
        static_cast<const uint4*>(codes), static_cast<const std::uint16_t*>(scales), n, k,
        static_cast<half*>(out_fp16));
    CUDA_CHECK(cudaGetLastError());
}

inline void launch_dequant_q5_prepacked_to_fp16(const void* codes, const void* high,
                                                const void* scales, int n, int k, void* out_fp16,
                                                cudaStream_t stream) {
    const std::int64_t total_warps = static_cast<std::int64_t>(n / 32) * (k / 64) * 2;
    const int blocks               = static_cast<int>((total_warps + 7) / 8);
    dequant_q5_prepacked_to_fp16_kernel<<<blocks, 256, 0, stream>>>(
        static_cast<const uint4*>(codes), static_cast<const std::uint32_t*>(high),
        static_cast<const std::uint16_t*>(scales), n, k, static_cast<half*>(out_fp16));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
