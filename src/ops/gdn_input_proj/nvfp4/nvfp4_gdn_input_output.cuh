#pragma once

#include "ops/common/memory.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {

struct Nvfp4GdnInputOutput {
    static constexpr std::int32_t kQkvRows = 10240;
    static constexpr std::int32_t kZRows   = 6144;

    __nv_bfloat16* qkv;
    __nv_bfloat16* z;
    // Optional: token t's q/k/v column is qkv_columns[t] (a wider, padded qkv plane).
    const std::int32_t* qkv_columns = nullptr;

    __device__ __forceinline__ __nv_bfloat16* destination(std::int32_t parent_row,
                                                          std::int32_t token) const {
        if (parent_row < kQkvRows) {
            const std::int32_t column = qkv_columns != nullptr ? qkv_columns[token] : token;
            return qkv + static_cast<std::int64_t>(column) * kQkvRows + parent_row;
        }
        return z + static_cast<std::int64_t>(token) * kZRows + parent_row - kQkvRows;
    }

    __device__ __forceinline__ void store(std::int32_t parent_row, std::int32_t token,
                                          float value) const {
        *destination(parent_row, token) = __float2bfloat16_rn(value);
    }

    __device__ __forceinline__ void store_vector(std::int32_t parent_row, std::int32_t token,
                                                 uint4 values) const {
        store_vec(destination(parent_row, token), values);
    }
};

static_assert((Nvfp4GdnInputOutput::kQkvRows % 128) == 0);
static_assert((Nvfp4GdnInputOutput::kZRows % 128) == 0);

} // namespace ninfer::ops::detail
