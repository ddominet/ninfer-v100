#include "ops/linear/q6/q6_prepack_sm70.h"

#include "core/arena.h"
#include "core/device.h"
#include "ops/linear/q6/q6_rowsplit_storage.cuh"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

__device__ __forceinline__ int q6_prepack_row(std::int64_t tile, int lane) {
    return static_cast<int>(tile * 32) + ((lane >> 2) & 3) * 8 + (lane & 3) +
           ((lane & 16) ? 4 : 0);
}

__global__ void prepack_q6_codes_kernel(const std::uint8_t* __restrict__ input,
                                        std::uint8_t* __restrict__ output, int n,
                                        std::int64_t groups) {
    const std::int64_t idx   = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::int64_t count = static_cast<std::int64_t>(n / 32) * groups * 64;
    if (idx >= count) { return; }
    const int lane           = static_cast<int>(idx & 31);
    const int half_idx       = static_cast<int>((idx >> 5) & 1);
    const std::int64_t tg    = idx >> 6;
    const std::int64_t group = tg % groups;
    const std::int64_t tile  = tg / groups;
    const int row            = q6_prepack_row(tile, lane);
    const auto* source       = reinterpret_cast<const uint4*>(
        input + (static_cast<std::int64_t>(row) * groups + group) *
                    Q6RowSplitStorage::kCodeBytesPerGroup) +
        half_idx;
    reinterpret_cast<uint4*>(output)[idx] = *source;
}

__global__ void prepack_q6_high_kernel(const std::uint8_t* __restrict__ input,
                                       std::uint8_t* __restrict__ output, int n,
                                       std::int64_t groups) {
    const std::int64_t idx   = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::int64_t count = static_cast<std::int64_t>(n / 32) * groups * 64;
    if (idx >= count) { return; }
    const int lane           = static_cast<int>(idx & 31);
    const int half_idx       = static_cast<int>((idx >> 5) & 1);
    const std::int64_t tg    = idx >> 6;
    const std::int64_t group = tg % groups;
    const std::int64_t tile  = tg / groups;
    const int row            = q6_prepack_row(tile, lane);
    const auto* source       = reinterpret_cast<const uint2*>(
        input + (static_cast<std::int64_t>(row) * groups + group) *
                    Q6RowSplitStorage::kHighBytesPerGroup) +
        half_idx;
    reinterpret_cast<uint2*>(output)[idx] = *source;
}

__global__ void prepack_q6_scales_kernel(const std::uint16_t* __restrict__ input,
                                         std::uint16_t* __restrict__ output, int n,
                                         std::int64_t groups) {
    const std::int64_t index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::int64_t count = static_cast<std::int64_t>(n / 32) * groups * 32;
    if (index >= count) { return; }
    const int lane           = static_cast<int>(index & 31);
    const std::int64_t tg    = index >> 5;
    const std::int64_t group = tg % groups;
    const std::int64_t tile  = tg / groups;
    output[index] = input[static_cast<std::int64_t>(q6_prepack_row(tile, lane)) * groups + group];
}

} // namespace

void q6_prepack_qpn_sm70(Weight& weight, cudaStream_t stream) {
    if (weight.qtype != QType::Q6G64_F16S) {
        throw std::invalid_argument("Q6 QPN prepack: weight is not Q6G64_F16S");
    }
    if (weight.layout == QuantLayout::VoltaQpnPrepacked) { return; }
    const int n = weight.n;
    const int k = weight.k;
    if (n % 32 != 0 || k % Q6RowSplitStorage::kGroupK != 0 || weight.padded_shape[1] != k ||
        weight.qhigh == nullptr) {
        throw std::invalid_argument("Q6 QPN prepack: shape must tile N32 without K padding");
    }
    const std::int64_t groups      = k / Q6RowSplitStorage::kGroupK;
    const std::int64_t items       = static_cast<std::int64_t>(n / 32) * groups * 32;
    const std::int64_t code_bytes  = items * Q6RowSplitStorage::kCodeBytesPerGroup;
    const std::int64_t high_bytes  = items * Q6RowSplitStorage::kHighBytesPerGroup;
    const std::int64_t scale_bytes = items * Q6RowSplitStorage::kScaleBytesPerGroup;
    DeviceBuffer scratch(static_cast<std::size_t>(code_bytes));
    const int half_blocks  = static_cast<int>((items * 2 + 255) / 256);
    const int scale_blocks = static_cast<int>((items + 255) / 256);

    prepack_q6_codes_kernel<<<half_blocks, 256, 0, stream>>>(
        static_cast<const std::uint8_t*>(weight.qdata), static_cast<std::uint8_t*>(scratch.p), n,
        groups);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpyAsync(const_cast<void*>(weight.qdata), scratch.p,
                               static_cast<std::size_t>(code_bytes), cudaMemcpyDeviceToDevice,
                               stream));

    prepack_q6_high_kernel<<<half_blocks, 256, 0, stream>>>(
        static_cast<const std::uint8_t*>(weight.qhigh), static_cast<std::uint8_t*>(scratch.p), n,
        groups);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpyAsync(const_cast<void*>(weight.qhigh), scratch.p,
                               static_cast<std::size_t>(high_bytes), cudaMemcpyDeviceToDevice,
                               stream));

    prepack_q6_scales_kernel<<<scale_blocks, 256, 0, stream>>>(
        static_cast<const std::uint16_t*>(weight.scales), static_cast<std::uint16_t*>(scratch.p),
        n, groups);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpyAsync(const_cast<void*>(weight.scales), scratch.p,
                               static_cast<std::size_t>(scale_bytes), cudaMemcpyDeviceToDevice,
                               stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));
    weight.layout = QuantLayout::VoltaQpnPrepacked;
}

} // namespace ninfer::ops::detail
