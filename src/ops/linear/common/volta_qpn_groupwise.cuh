#pragma once

// Unified Volta (sm_70) Prepacked Q4 / Q5 / Q6 Multi-Tile Quadpair-split-N Tensor-Core GEMM:
//   - Covers T = 1..64 in a single weight pass via kTiles in {1, 2, 3, 4, 6, 8} (mma.sync.m8n8k4).
//   - Software-pipelined scalar Carry cur/nxt (0 stack frames, 0 local spills, 512B/128B/256B
//     warp-coalesced weight loads).
//   - Native VoltaGroupMajorHalf activation indexing (each warp's 16B activation loads hit a
//     single 128B L1 cache line across all 4 quadpairs -> 1.3-1.8x faster than WMMA skinny across
//     all shapes and T = 1..64, with zero split-K semaphores and zero global partial atomics).

#include "ops/common/volta_mma.cuh"
#include "ops/linear/fp8/fp8_output.cuh"
#include "ops/linear/nvfp4/nvfp4_output.cuh"
#include "ops/linear/volta_fp16_stage.h"
#include "ops/linear/volta_qpn_width.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <type_traits>

namespace ninfer::ops::detail {

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ == 700

enum class GroupwiseQpnFormat : int {
    Q4 = 4,
    Q5 = 5,
    Q6 = 6,
};

// -------------------------------------------------------------------------------------------------
// Output policies
// -------------------------------------------------------------------------------------------------

struct GroupwiseContiguousOutput {
    __nv_bfloat16* out;
    std::int32_t out_ld;

    __device__ __forceinline__ void store(int row, int tok, float value) const {
        out[static_cast<std::int64_t>(tok) * out_ld + row] = __float2bfloat16_rn(value);
    }
};

struct GroupwiseResidualOutput {
    __nv_bfloat16* out;
    std::int32_t out_ld;

    __device__ __forceinline__ void store(int row, int tok, float value) const {
        const std::int64_t idx = static_cast<std::int64_t>(tok) * out_ld + row;
        out[idx]               = __float2bfloat16_rn(value + __bfloat162float(out[idx]));
    }
};

struct GroupwiseSplitRowOutput {
    __nv_bfloat16* out0;
    __nv_bfloat16* out1;
    std::int32_t out0_ld;
    std::int32_t out1_ld;
    std::int32_t split_row;

    __device__ __forceinline__ void store(int row, int tok, float value) const {
        const __nv_bfloat16 bf = __float2bfloat16_rn(value);
        if (row < split_row) {
            out0[static_cast<std::int64_t>(tok) * out0_ld + row] = bf;
        } else {
            out1[static_cast<std::int64_t>(tok) * out1_ld + (row - split_row)] = bf;
        }
    }
};

struct GroupwiseScatterColOutput {
    __nv_bfloat16* out;
    const std::int32_t* col_map;
    std::int32_t out_ld;

    __device__ __forceinline__ void store(int row, int tok, float value) const {
        const std::int32_t dst_col = col_map[tok];
        out[static_cast<std::int64_t>(dst_col) * out_ld + row] = __float2bfloat16_rn(value);
    }
};

struct GroupwiseSplitScatterOutput {
    __nv_bfloat16* out0;
    __nv_bfloat16* out1;
    const std::int32_t* out0_col_map;
    std::int32_t out0_ld;
    std::int32_t out1_ld;
    std::int32_t split_row;

    __device__ __forceinline__ void store(int row, int tok, float value) const {
        const __nv_bfloat16 bf = __float2bfloat16_rn(value);
        if (row < split_row) {
            const std::int32_t dst_col = out0_col_map[tok];
            out0[static_cast<std::int64_t>(dst_col) * out0_ld + row] = bf;
        } else {
            out1[static_cast<std::int64_t>(tok) * out1_ld + (row - split_row)] = bf;
        }
    }
};

struct GroupwiseSwiGluInPlaceBf16Output {
    __nv_bfloat16* out;
    std::int32_t out_ld;

    __device__ __forceinline__ void store(int row, int tok, float up_value) const {
        const std::int64_t idx = static_cast<std::int64_t>(tok) * out_ld + row;
        const float g          = __bfloat162float(out[idx]);
        const float silu_g     = g / (1.0f + __expf(-g));
        const float up         = __bfloat162float(__float2bfloat16_rn(up_value));
        out[idx]               = __float2bfloat16_rn(silu_g * up);
    }
};

// -------------------------------------------------------------------------------------------------
// Fast activation loader for QPN (VoltaGroupMajorHalf, half, or BF16 -> half2)
// -------------------------------------------------------------------------------------------------

__device__ __forceinline__ half2 qpn_bf16_pair_to_half2(std::uint32_t bf16x2_bits) {
    const float f0 = __uint_as_float(bf16x2_bits << 16);
    const float f1 = __uint_as_float(bf16x2_bits & 0xffff0000u);
    return __floats2half2_rn(f0, f1);
}

template <typename Activation>
__device__ __forceinline__ uint4 qpn_load_act8(const Activation* __restrict__ src) {
    const uint4 raw = __ldg(reinterpret_cast<const uint4*>(src));
    if constexpr (std::is_same_v<Activation, half>) {
        return raw;
    } else {
        half2 h[4];
        h[0] = qpn_bf16_pair_to_half2(raw.x);
        h[1] = qpn_bf16_pair_to_half2(raw.y);
        h[2] = qpn_bf16_pair_to_half2(raw.z);
        h[3] = qpn_bf16_pair_to_half2(raw.w);
        return *reinterpret_cast<const uint4*>(h);
    }
}

// -------------------------------------------------------------------------------------------------
// Prepacked Q4 / Q5 / Q6 group decoders
// -------------------------------------------------------------------------------------------------

template <GroupwiseQpnFormat Fmt>
struct PrepackedQpnDecoder;

template <>
struct PrepackedQpnDecoder<GroupwiseQpnFormat::Q4> {
    struct Carry {
        uint4 c0;
        uint4 c1;
        std::uint16_t sc;
    };

    __device__ __forceinline__ static void load(const uint4* __restrict__ codes,
                                                const void* __restrict__,
                                                const std::uint16_t* __restrict__ scales,
                                                std::int64_t tg, int lane, Carry& out) {
        out.c0 = __ldg(codes + tg * 64 + lane);
        out.c1 = __ldg(codes + tg * 64 + 32 + lane);
        out.sc = __ldg(scales + tg * 32 + lane);
    }

    __device__ __forceinline__ static void decode_word(const Carry& w, int half_idx, int word_idx,
                                                       half2 sc2, half2 (&b)[4]) {
        const uint4& q          = (half_idx == 0) ? w.c0 : w.c1;
        const std::uint32_t raw = (word_idx == 0) ? q.x
                                  : ((word_idx == 1) ? q.y : ((word_idx == 2) ? q.z : q.w));
        const std::uint32_t p4  = raw ^ 0x88888888u;
        const half2 bias        = __half2half2(__ushort_as_half(0x6408)); // 1032.0
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const std::uint32_t lb = (p4 >> (8 * j)) & 0xffu;
            std::uint32_t bits     = ((lb & 0x0fu) | ((lb & 0xf0u) << 12)) | 0x64006400u;
            b[j]                   = __hmul2(__hsub2(*reinterpret_cast<half2*>(&bits), bias), sc2);
        }
    }
};

template <>
struct PrepackedQpnDecoder<GroupwiseQpnFormat::Q5> {
    struct Carry {
        uint4 c0;
        uint4 c1;
        std::uint32_t h0;
        std::uint32_t h1;
        std::uint16_t sc;
    };

    __device__ __forceinline__ static void load(const uint4* __restrict__ codes,
                                                const void* __restrict__ high_void,
                                                const std::uint16_t* __restrict__ scales,
                                                std::int64_t tg, int lane, Carry& out) {
        const auto* high = static_cast<const std::uint32_t*>(high_void);
        out.c0           = __ldg(codes + tg * 64 + lane);
        out.c1           = __ldg(codes + tg * 64 + 32 + lane);
        out.h0           = __ldg(high + tg * 64 + lane);
        out.h1           = __ldg(high + tg * 64 + 32 + lane);
        out.sc           = __ldg(scales + tg * 32 + lane);
    }

    __device__ __forceinline__ static void decode_word(const Carry& w, int half_idx, int word_idx,
                                                       half2 sc2, half2 (&b)[4]) {
        const uint4& q           = (half_idx == 0) ? w.c0 : w.c1;
        const std::uint32_t h32  = (half_idx == 0) ? w.h0 : w.h1;
        const std::uint32_t raw  = (word_idx == 0) ? q.x
                                   : ((word_idx == 1) ? q.y : ((word_idx == 2) ? q.z : q.w));
        const std::uint32_t hb   = ((h32 >> (8 * word_idx)) & 0xffu) ^ 0xffu;
        const half2 bias         = __half2half2(__ushort_as_half(0x6410)); // 1040.0
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const std::uint32_t lb = (raw >> (8 * j)) & 0xffu;
            std::uint32_t bits     = ((lb & 0x0fu) | ((lb & 0xf0u) << 12)) | 0x64006400u;
            bits |= (((hb >> (2 * j)) & 1u) << 4) | (((hb >> (2 * j + 1)) & 1u) << 20);
            b[j] = __hmul2(__hsub2(*reinterpret_cast<half2*>(&bits), bias), sc2);
        }
    }
};

template <>
struct PrepackedQpnDecoder<GroupwiseQpnFormat::Q6> {
    struct Carry {
        uint4 c0;
        uint4 c1;
        uint2 h0;
        uint2 h1;
        std::uint16_t sc;
    };

    __device__ __forceinline__ static void load(const uint4* __restrict__ codes,
                                                const void* __restrict__ high_void,
                                                const std::uint16_t* __restrict__ scales,
                                                std::int64_t tg, int lane, Carry& out) {
        const auto* high = static_cast<const uint2*>(high_void);
        out.c0           = __ldg(codes + tg * 64 + lane);
        out.c1           = __ldg(codes + tg * 64 + 32 + lane);
        out.h0           = __ldg(high + tg * 64 + lane);
        out.h1           = __ldg(high + tg * 64 + 32 + lane);
        out.sc           = __ldg(scales + tg * 32 + lane);
    }

    __device__ __forceinline__ static void decode_word(const Carry& w, int half_idx, int word_idx,
                                                       half2 sc2, half2 (&b)[4]) {
        const uint4& q           = (half_idx == 0) ? w.c0 : w.c1;
        const uint2& h64         = (half_idx == 0) ? w.h0 : w.h1;
        const std::uint32_t raw  = (word_idx == 0) ? q.x
                                   : ((word_idx == 1) ? q.y : ((word_idx == 2) ? q.z : q.w));
        const std::uint32_t h32  = (word_idx < 2) ? h64.x : h64.y;
        const std::uint32_t h16  = ((h32 >> (16 * (word_idx & 1))) & 0xffffu) ^ 0xaaaau;
        const half2 bias         = __half2half2(__ushort_as_half(0x6420)); // 1056.0
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const std::uint32_t lb = (raw >> (8 * j)) & 0xffu;
            std::uint32_t bits     = ((lb & 0x0fu) | ((lb & 0xf0u) << 12)) | 0x64006400u;
            bits |= (((h16 >> (4 * j)) & 3u) << 4) | (((h16 >> (4 * j + 2)) & 3u) << 20);
            b[j] = __hmul2(__hsub2(*reinterpret_cast<half2*>(&bits), bias), sc2);
        }
    }
};

// -------------------------------------------------------------------------------------------------
// Software-pipelined Multi-Tile QPN kernel (T = 1..64, kTiles in {1,2,3,4,6,8}, SPLITK = 4/8/16)
// -------------------------------------------------------------------------------------------------

template <GroupwiseQpnFormat Fmt, int kTiles, int SPLITK, typename Activation,
          typename OutputPolicy>
__global__ __launch_bounds__(SPLITK * 32, volta_qpn_min_blocks(kTiles, SPLITK)) void
volta_qpn_groupwise_kernel(const uint4* __restrict__ codes, const void* __restrict__ high,
                           const std::uint16_t* __restrict__ scales,
                           const Activation* __restrict__ x, int n, int k, int t,
                           OutputPolicy output) {
    using Decoder = PrepackedQpnDecoder<Fmt>;

    __shared__ float cs[SPLITK][kTiles * 8 * 32];

    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int qp   = (lane >> 2) & 3;
    const int r    = (lane & 3) + ((lane & 16) != 0 ? 4 : 0);

    const std::int64_t tile_n = static_cast<std::int64_t>(blockIdx.x);
    const int groups          = k / 64;
    const int gq              = groups / SPLITK;
    const int g0              = warp * gq;
    const int gend            = (warp == SPLITK - 1) ? groups : (g0 + gq);
    const std::int64_t tg0    = tile_n * groups;

    float c[kTiles][8];
#pragma unroll
    for (int ti = 0; ti < kTiles; ++ti) {
#pragma unroll
        for (int i = 0; i < 8; ++i) { c[ti][i] = 0.0f; }
    }

    typename Decoder::Carry cur, nxt;
    if (g0 < gend) { Decoder::load(codes, high, scales, tg0 + g0, lane, cur); }

    for (int g = g0; g < gend; ++g) {
        if (g + 1 < gend) { Decoder::load(codes, high, scales, tg0 + g + 1, lane, nxt); }
        const half2 sc2 = __half2half2(__ushort_as_half(cur.sc));
        const int g_k   = g * 64;

#pragma unroll
        for (int half_idx = 0; half_idx < 2; ++half_idx) {
#pragma unroll
            for (int w = 0; w < 4; ++w) {
                half2 b[4];
                Decoder::decode_word(cur, half_idx, w, sc2, b);
                const unsigned* B = reinterpret_cast<const unsigned*>(b);
                const int k_off   = g_k + half_idx * 32 + w * 8;

#pragma unroll
                for (int ti = 0; ti < kTiles; ++ti) {
                    uint4 araw;
                    if constexpr (std::is_same_v<Activation, VoltaGroupMajorHalf>) {
                        const int group16 = g * 4 + half_idx * 2 + (w >> 1);
                        const int khalf   = w & 1;
                        const half* xp =
                            reinterpret_cast<const half*>(x) +
                            (((static_cast<std::int64_t>(group16) * kTiles + ti) * 16 +
                              khalf * 8 + r) *
                             8);
                        araw = __ldg(reinterpret_cast<const uint4*>(xp));
                    } else {
                        const int row = ti * 8 + r;
                        if (row < t) {
                            araw = qpn_load_act8(x + static_cast<std::int64_t>(row) * k + k_off);
                        } else {
                            araw = make_uint4(0u, 0u, 0u, 0u);
                        }
                    }
                    const unsigned* A = reinterpret_cast<const unsigned*>(&araw);
                    volta_mma_qp_n(c[ti], A[0], A[1], B[0], B[1]);
                    volta_mma_qp_n(c[ti], A[2], A[3], B[2], B[3]);
                }
            }
        }
        cur = nxt;
    }

#pragma unroll
    for (int ti = 0; ti < kTiles; ++ti) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int row = (i & 2) | ((lane & 16) != 0 ? 4 : 0) | (lane & 1);
            const int cl  = (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
            cs[warp][(ti * 8 + row) * 32 + qp * 8 + cl] = c[ti][i];
        }
    }
    __syncthreads();

    if constexpr (requires { output.make_argmax_key(0, 0.0f); }) {
        const int ocol = static_cast<int>(blockIdx.x) * 32 + lane;
        for (int row = warp; row < t; row += SPLITK) {
            float v = 0.0f;
#pragma unroll
            for (int w = 0; w < SPLITK; ++w) { v += cs[w][row * 32 + lane]; }
            const unsigned long long key = (ocol < n) ? output.make_argmax_key(ocol, v) : 0ULL;
            output.commit_warp(row, key, lane, true);
        }
    } else {
        constexpr int kOut = kTiles * 8 * 32;
        for (int e = static_cast<int>(threadIdx.x); e < kOut; e += SPLITK * 32) {
            const int row  = e >> 5;
            const int cl   = e & 31;
            const int ocol = static_cast<int>(blockIdx.x) * 32 + cl;
            if (ocol < n) {
                if (row < t) {
                    float v = 0.0f;
#pragma unroll
                    for (int w = 0; w < SPLITK; ++w) { v += cs[w][e]; }
                    output.store(ocol, row, v);
                } else if constexpr (requires { output.store_zero(0, 0); }) {
                    output.store_zero(ocol, row);
                }
            }
        }
    }
}

#endif // !defined(__CUDA_ARCH__) || __CUDA_ARCH__ == 700

} // namespace ninfer::ops::detail
