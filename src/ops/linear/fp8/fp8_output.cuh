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

} // namespace ninfer::ops::detail
