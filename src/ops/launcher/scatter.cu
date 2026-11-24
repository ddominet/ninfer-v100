#include "ops/launcher/scatter.h"

#include "ops/kernel/scatter.cuh"
#include "core/device.h"
#include "core/dtype.h"

#include <cstdint>
#include <algorithm>

namespace ninfer::ops::detail {

void scatter_launch(const Tensor& src, const Tensor& indices, Tensor& dst, cudaStream_t stream) {
    constexpr int block        = 256;
    constexpr int vector_block = 128;
    const int d                = src.ne[0];
    const int vision           = src.ne[1];
    const auto src_addr        = reinterpret_cast<std::uintptr_t>(src.data);
    const auto dst_addr        = reinterpret_cast<std::uintptr_t>(dst.data);
    if ((d % 8) == 0 && ((src_addr | dst_addr) & 0xfu) == 0) {
        scatter_bf16x8_kernel<<<vision, vector_block, 0, stream>>>(
            static_cast<const uint4*>(src.data), static_cast<const std::int32_t*>(indices.data),
            static_cast<uint4*>(dst.data), d / 8);
    } else if ((d & 1) == 0 && ((src_addr | dst_addr) & 0x3u) == 0) {
        scatter_bf16x2_kernel<<<vision, block, 0, stream>>>(
            static_cast<const __nv_bfloat162*>(src.data),
            static_cast<const std::int32_t*>(indices.data), static_cast<__nv_bfloat162*>(dst.data),
            d / 2);
    } else {
        scatter_scalar_kernel<<<vision, block, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(src.data),
            static_cast<const std::int32_t*>(indices.data), static_cast<__nv_bfloat16*>(dst.data),
            d);
    }
    CUDA_CHECK(cudaGetLastError());
}

void scatter_bf16_batch_launch(const Tensor& source, const Tensor& lanes,
                               const Tensor& valid_columns, Tensor& destination,
                               cudaStream_t stream) {
    constexpr int block = kScatterBatchThreads;
    // Keep a grid-stride feature loop beyond CUDA's z-dimension limit.
    const int feature_tiles = std::min((source.ne[0] / 8 + block - 1) / block, 65535);
    const dim3 grid(source.ne[1], source.ne[2], feature_tiles);
    scatter_bf16_batch_kernel<<<grid, block, 0, stream>>>(
        static_cast<const uint4*>(source.data), static_cast<const std::int32_t*>(lanes.data),
        static_cast<const std::int32_t*>(valid_columns.data), static_cast<uint4*>(destination.data),
        source.ne[0] / 8, source.ne[1],
        destination.nb[1] / static_cast<std::int64_t>(sizeof(uint4)),
        destination.nb[2] / static_cast<std::int64_t>(sizeof(uint4)));
    CUDA_CHECK(cudaGetLastError());
}

namespace {

// One CTA per destination column, 16-byte copies when the column bytes allow it.
template <class Word>
__global__ void gather_columns_kernel(const Word* __restrict__ source,
                                      const std::int32_t* __restrict__ indices,
                                      Word* __restrict__ destination, std::int64_t words) {
    const std::int32_t column = static_cast<std::int32_t>(blockIdx.x);
    const std::int32_t from   = indices[column];
    Word* out                 = destination + static_cast<std::int64_t>(column) * words;
    if (from < 0) {
        for (std::int64_t i = threadIdx.x; i < words; i += blockDim.x) { out[i] = Word{}; }
        return;
    }
    const Word* in = source + static_cast<std::int64_t>(from) * words;
    for (std::int64_t i = threadIdx.x; i < words; i += blockDim.x) { out[i] = in[i]; }
}

// blockIdx.y selects the pair; 4-byte words (every BF16 column here has an even row count).
__global__ void gather_columns_group_kernel(GatherColumnsGroup group,
                                            const std::int32_t* __restrict__ indices) {
    const int pair            = static_cast<int>(blockIdx.y);
    const std::int32_t column = static_cast<std::int32_t>(blockIdx.x);
    const std::int64_t words  = group.words[pair];
    const std::int32_t from   = indices[column];
    auto* out = static_cast<std::uint32_t*>(group.destination[pair]) + column * words;
    if (from < 0) {
        for (std::int64_t i = threadIdx.x; i < words; i += blockDim.x) { out[i] = 0; }
        return;
    }
    const auto* in = static_cast<const std::uint32_t*>(group.source[pair]) + from * words;
    for (std::int64_t i = threadIdx.x; i < words; i += blockDim.x) { out[i] = in[i]; }
}

} // namespace

void gather_columns_group_launch(const GatherColumnsGroup& group, const Tensor& indices,
                                 std::int32_t columns, cudaStream_t stream) {
    if (group.pairs <= 0 || columns <= 0) { return; }
    const dim3 grid(static_cast<unsigned>(columns), static_cast<unsigned>(group.pairs));
    gather_columns_group_kernel<<<grid, 256, 0, stream>>>(
        group, static_cast<const std::int32_t*>(indices.data));
    CUDA_CHECK(cudaGetLastError());
}

void gather_columns_launch(const Tensor& source, const Tensor& indices, Tensor& destination,
                           cudaStream_t stream) {
    const std::int64_t column_bytes = static_cast<std::int64_t>(source.ne[0]) * dtype_size(source.dtype);
    const std::int32_t columns      = destination.ne[1];
    const auto aligned = ((reinterpret_cast<std::uintptr_t>(source.data) |
                           reinterpret_cast<std::uintptr_t>(destination.data)) & 0xfu) == 0;
    const auto* ids = static_cast<const std::int32_t*>(indices.data);
    if (aligned && column_bytes % 16 == 0) {
        gather_columns_kernel<uint4><<<columns, 128, 0, stream>>>(
            static_cast<const uint4*>(source.data), ids, static_cast<uint4*>(destination.data),
            column_bytes / 16);
    } else if (column_bytes % 4 == 0) {
        gather_columns_kernel<std::uint32_t><<<columns, 128, 0, stream>>>(
            static_cast<const std::uint32_t*>(source.data), ids,
            static_cast<std::uint32_t*>(destination.data), column_bytes / 4);
    } else {
        gather_columns_kernel<std::uint16_t><<<columns, 128, 0, stream>>>(
            static_cast<const std::uint16_t*>(source.data), ids,
            static_cast<std::uint16_t*>(destination.data), column_bytes / 2);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
