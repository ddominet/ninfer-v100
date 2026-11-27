#pragma once

#include "ops/common/memory.cuh"
#ifdef NINFER_VOLTA_BUILD
#include "ops/linear/volta_fp16_stage.h"
#endif

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops::detail {

struct Nvfp4IdentityEpilogue {
    __device__ __forceinline__ float apply(std::int32_t, std::int32_t, float value) const {
        return value;
    }
};

struct Nvfp4ContiguousOutput {
    __nv_bfloat16* data;
    std::int32_t rows;

    __device__ __forceinline__ void store(std::int32_t parent_row, std::int32_t token,
                                          float value) const {
        data[static_cast<std::int64_t>(token) * rows + parent_row] = __float2bfloat16_rn(value);
    }

    __device__ __forceinline__ void store_vector(std::int32_t parent_row, std::int32_t token,
                                                 uint4 values) const {
        auto* destination = data + static_cast<std::int64_t>(token) * rows + parent_row;
        store_vec(destination, values);
    }
};

// Writes the mma accumulator straight through, no BF16 round. Exists for split-projection SwiGLU
// (see nvfp4_linear_swiglu_qpn_split.cuh): two independent QPN2 launches -- one per weight half,
// unmodified, at whatever schedule QPN2 already measured fastest for this shape -- write gate and
// up into two fp32 scratch planes, and a small combine kernel applies silu(gate)*up in fp32 before
// the single BF16 round. Splitting keeps each launch at QPN2's own tuned register/occupancy
// profile instead of the fused kernel's doubled one; see the plan.cpp route for the measured
// comparison.
struct Nvfp4Fp32ContiguousOutput {
    float* data;
    std::int32_t rows;

    __device__ __forceinline__ void store(std::int32_t parent_row, std::int32_t token,
                                          float value) const {
        data[static_cast<std::int64_t>(token) * rows + parent_row] = value;
    }
};

// Fused SwiGLU epilogue for the up projection half: reads the FP32 gate plane written by the first
// half, computes silu(gate) * up in FP32, and performs the single observable BF16 round straight
// into the output tensor (eliminating up's FP32 DRAM traffic and the combine kernel launch).
struct Nvfp4QpnSwiGluOutput {
    const float* gate;
    __nv_bfloat16* out;
    std::int32_t rows;

    __device__ __forceinline__ void store(std::int32_t parent_row, std::int32_t token,
                                          float up_value) const {
        const std::int64_t idx = static_cast<std::int64_t>(token) * rows + parent_row;
        const float g          = gate[idx];
        const float silu_g     = g / (1.0f + __expf(-g));
        out[idx]               = __float2bfloat16_rn(silu_g * up_value);
    }
};

// Fused residual-add epilogue for NVFP4 projections (mirrors Fp8ResidualOutput): adds the FP32
// projection result into the BF16 residual tensor with a single observable BF16 round.
struct Nvfp4ResidualOutput {
    __nv_bfloat16* data;
    std::int32_t rows;

    __device__ __forceinline__ void store(std::int32_t parent_row, std::int32_t token,
                                          float value) const {
        const std::int64_t index = static_cast<std::int64_t>(token) * rows + parent_row;
        data[index] = __float2bfloat16_rn(value + __bfloat162float(data[index]));
    }
};

#ifdef NINFER_VOLTA_BUILD
// Fused SwiGLU epilogue that emits the BF16-rounded intermediate directly as FP16 in Volta
// group-major staging layout ([K/16 group][row tile][k half][8 rows][8 k]) and zero-fills padded
// rows past T via store_zero, eliminating the BF16 intermediate write/read and staging kernel.
struct Nvfp4QpnSwiGluGroupMajorOutput {
    const float* gate;
    VoltaGroupMajorHalf* out;
    std::int32_t rows;
    std::int32_t tiles;

    __device__ __forceinline__ std::int64_t group_major_index(std::int32_t parent_row,
                                                              std::int32_t token) const {
        const std::int64_t group = static_cast<std::int64_t>(parent_row >> 4);
        const std::int64_t tile  = static_cast<std::int64_t>(token >> 3);
        const std::int64_t khalf = static_cast<std::int64_t>((parent_row >> 3) & 1);
        const std::int64_t r     = static_cast<std::int64_t>(token & 7);
        const std::int64_t k8    = static_cast<std::int64_t>(parent_row & 7);
        return (((group * tiles + tile) * 2 + khalf) * 8 + r) * 8 + k8;
    }

    __device__ __forceinline__ void store(std::int32_t parent_row, std::int32_t token,
                                          float up_value) const {
        const std::int64_t idx = static_cast<std::int64_t>(token) * rows + parent_row;
        const float g          = gate[idx];
        const float silu_g     = g / (1.0f + __expf(-g));
        const float rounded    = __bfloat162float(__float2bfloat16_rn(silu_g * up_value));
        out[group_major_index(parent_row, token)].value = __float2half_rn(rounded);
    }

    __device__ __forceinline__ void store_zero(std::int32_t parent_row, std::int32_t token) const {
        out[group_major_index(parent_row, token)].value = __ushort_as_half(0);
    }
};

// Order-preserving (BF16 value, ~token_idx) key for bit-exact greedy argmax matching
// ops::linear(BF16) + ops::argmax(valid_rows, lower-id tie-break) with zero logits HBM traffic.
__device__ __forceinline__ unsigned long long nvfp4_bf16_argmax_key(float value, std::int32_t row) {
    const __nv_bfloat16 bf = __float2bfloat16_rn(value);
    std::uint16_t bits     = __bfloat16_as_ushort(bf);
    if ((bits & 0x7FFFu) > 0x7F80u) { return 0ULL; }
    if (bits == 0x8000u) { bits = 0x0000u; }
    const std::uint32_t ordered = (bits & 0x8000u) == 0
                                      ? (static_cast<std::uint32_t>(bits) ^ 0x8000u)
                                      : (static_cast<std::uint32_t>(~bits) & 0xFFFFu);
    return (static_cast<unsigned long long>(ordered) << 32) |
           static_cast<unsigned long long>(~static_cast<std::uint32_t>(row));
}

struct Nvfp4ArgmaxOutput {
    unsigned long long* packed_max;
    std::int32_t valid_rows;

    __device__ __forceinline__ unsigned long long make_argmax_key(std::int32_t parent_row,
                                                                  float value) const {
        if (parent_row >= valid_rows) { return 0ULL; }
        return nvfp4_bf16_argmax_key(value, parent_row);
    }

    __device__ __forceinline__ void store(std::int32_t parent_row, std::int32_t token,
                                          float value) const {
        const unsigned long long key = make_argmax_key(parent_row, value);
        if (key > *reinterpret_cast<const volatile unsigned long long*>(packed_max + token)) {
            atomicMax(packed_max + token, key);
        }
    }

    __device__ __forceinline__ void commit_warp(std::int32_t token, unsigned long long key,
                                                int lane, bool active) const {
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            const unsigned long long other = __shfl_xor_sync(0xffffffffu, key, offset);
            if (other > key) { key = other; }
        }
        if (lane == 0 && active &&
            key > *reinterpret_cast<const volatile unsigned long long*>(packed_max + token)) {
            atomicMax(packed_max + token, key);
        }
    }

    __device__ __forceinline__ void commit_halfwarp(std::int32_t token, unsigned long long key,
                                                    int lane, bool active) const {
#pragma unroll
        for (int offset = 8; offset > 0; offset >>= 1) {
            const unsigned long long other = __shfl_xor_sync(0xffffffffu, key, offset);
            if (other > key) { key = other; }
        }
        if ((lane & 15) == 0 && active &&
            key > *reinterpret_cast<const volatile unsigned long long*>(packed_max + token)) {
            atomicMax(packed_max + token, key);
        }
    }
};
#endif

} // namespace ninfer::ops::detail
