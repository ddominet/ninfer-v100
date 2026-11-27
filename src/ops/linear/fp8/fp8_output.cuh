#pragma once

#include "ops/common/memory.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops::detail {

struct Fp8IdentityEpilogue {
    __device__ __forceinline__ float apply(std::int32_t, std::int32_t, float value) const {
        return value;
    }
};

struct Fp8ContiguousOutput {
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

struct Fp8ResidualOutput {
    __nv_bfloat16* data;
    std::int32_t rows;

    __device__ __forceinline__ void store(std::int32_t parent_row, std::int32_t token,
                                          float value) const {
        const std::int64_t index = static_cast<std::int64_t>(token) * rows + parent_row;
        data[index] = __float2bfloat16_rn(value + __bfloat162float(data[index]));
    }
};

// Split-projection SwiGLU keeps both projection results in FP32 until the fused
// SiLU/multiply epilogue performs the single observable BF16 rounding.
struct Fp8Fp32ContiguousOutput {
    float* data;
    std::int32_t rows;

    __device__ __forceinline__ void store(std::int32_t parent_row, std::int32_t token,
                                          float value) const {
        data[static_cast<std::int64_t>(token) * rows + parent_row] = value;
    }
};

struct Fp8QpnSwiGluOutput {
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

// Same BF16-rounded SwiGLU epilogue, emitted directly as row-major FP16 for consumption by an
// immediately following FP8 QPN/TC projection without an extra BF16->FP16 staging launch.
struct Fp8QpnSwiGluFp16Output {
    const float* gate;
    half* out;
    std::int32_t rows;

    __device__ __forceinline__ void store(std::int32_t parent_row, std::int32_t token,
                                          float up_value) const {
        const std::int64_t idx = static_cast<std::int64_t>(token) * rows + parent_row;
        const float g          = gate[idx];
        const float silu_g     = g / (1.0f + __expf(-g));
        const float rounded    = __bfloat162float(__float2bfloat16_rn(silu_g * up_value));
        out[idx]               = __float2half_rn(rounded);
    }
};

// Order-preserving (BF16 value, ~token_idx) key for bit-exact greedy argmax matching
// ops::linear(BF16) + ops::argmax(valid_rows, lower-id tie-break) with zero logits HBM traffic.
__device__ __forceinline__ unsigned long long fp8_bf16_argmax_key(float value, std::int32_t row) {
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

struct Fp8ArgmaxOutput {
    unsigned long long* packed_max;
    std::int32_t valid_rows;

    __device__ __forceinline__ unsigned long long make_argmax_key(std::int32_t parent_row,
                                                                  float value) const {
        if (parent_row >= valid_rows) { return 0ULL; }
        return fp8_bf16_argmax_key(value, parent_row);
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

} // namespace ninfer::ops::detail
