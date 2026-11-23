#include "ops/linear/w8/w8_prepack_sm70.h"

#include "core/arena.h"
#include "core/device.h"
#include "ops/linear/w8/w8_rowsplit_storage.cuh"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

// The quadpair kernel's lane -> row map within a 32-row tile.
__device__ __forceinline__ int w8_prepack_row(int tile, int lane) {
    return tile * 32 + ((lane >> 2) & 3) * 8 + (lane & 3) + ((lane & 16) ? 4 : 0);
}

// Codes: [N/32 tile][K/128 block][8 sixteen-byte slices][lane]. Slice s of block b holds the
// lane's row bytes b*128 + s*16 .. +16, so a lane still receives its row's 128 adjacent K.
__global__ void prepack_w8_codes_kernel(const std::uint8_t* __restrict__ input,
                                        std::uint8_t* __restrict__ output, int n, int k,
                                        std::int64_t row_bytes) {
    const std::int64_t index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int blocks         = k / 128;
    const std::int64_t count = static_cast<std::int64_t>(n / 32) * blocks * 8 * 32;
    if (index >= count) { return; }
    const int lane  = static_cast<int>(index % 32);
    const int slice = static_cast<int>((index / 32) % 8);
    const int block = static_cast<int>((index / (32 * 8)) % blocks);
    const int tile  = static_cast<int>(index / (static_cast<std::int64_t>(blocks) * 8 * 32));
    const int row   = w8_prepack_row(tile, lane);
    reinterpret_cast<uint4*>(output)[index] = *reinterpret_cast<const uint4*>(
        input + static_cast<std::int64_t>(row) * row_bytes + block * 128 + slice * 16);
}

// Scales: [N/32 tile][K/128 block][lane] x the block's four FP16 group scales (8 bytes).
__global__ void prepack_w8_scales_kernel(const std::uint8_t* __restrict__ input,
                                         std::uint8_t* __restrict__ output, int n, int k,
                                         std::int64_t row_scale_bytes) {
    const std::int64_t index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int blocks         = k / 128;
    const std::int64_t count = static_cast<std::int64_t>(n / 32) * blocks * 32;
    if (index >= count) { return; }
    const int lane  = static_cast<int>(index % 32);
    const int block = static_cast<int>((index / 32) % blocks);
    const int tile  = static_cast<int>(index / (static_cast<std::int64_t>(blocks) * 32));
    const int row   = w8_prepack_row(tile, lane);
    reinterpret_cast<uint2*>(output)[index] = *reinterpret_cast<const uint2*>(
        input + static_cast<std::int64_t>(row) * row_scale_bytes + block * 8);
}

} // namespace

void w8_prepack_qpn_sm70(Weight& weight, cudaStream_t stream) {
    if (weight.qtype != QType::W8G32_F16S) {
        throw std::invalid_argument("W8 QPN prepack: weight is not W8G32_F16S");
    }
    if (weight.layout == QuantLayout::VoltaQpnPrepacked) { return; }
    const int n = weight.n;
    const int k = weight.k;
    if (n % 32 != 0 || k % 128 != 0 || weight.padded_shape[1] != k) {
        throw std::invalid_argument("W8 QPN prepack: shape must tile N32 K128 without padding");
    }
    const std::int64_t groups      = k / W8RowSplitStorage::kGroupK;
    const std::int64_t code_bytes  = static_cast<std::int64_t>(n) * groups *
                                    W8RowSplitStorage::kCodeBytesPerGroup;
    const std::int64_t scale_bytes = static_cast<std::int64_t>(n) * groups *
                                     W8RowSplitStorage::kScaleBytesPerGroup;
    DeviceBuffer scratch(static_cast<std::size_t>(code_bytes));

    const std::int64_t code_items = static_cast<std::int64_t>(n / 32) * (k / 128) * 8 * 32;
    prepack_w8_codes_kernel<<<static_cast<int>((code_items + 255) / 256), 256, 0, stream>>>(
        static_cast<const std::uint8_t*>(weight.qdata), static_cast<std::uint8_t*>(scratch.p), n,
        k, groups * W8RowSplitStorage::kCodeBytesPerGroup);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpyAsync(const_cast<void*>(weight.qdata), scratch.p,
                               static_cast<std::size_t>(code_bytes), cudaMemcpyDeviceToDevice,
                               stream));

    const std::int64_t scale_items = static_cast<std::int64_t>(n / 32) * (k / 128) * 32;
    prepack_w8_scales_kernel<<<static_cast<int>((scale_items + 255) / 256), 256, 0, stream>>>(
        static_cast<const std::uint8_t*>(weight.scales), static_cast<std::uint8_t*>(scratch.p), n,
        k, groups * W8RowSplitStorage::kScaleBytesPerGroup);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpyAsync(const_cast<void*>(weight.scales), scratch.p,
                               static_cast<std::size_t>(scale_bytes), cudaMemcpyDeviceToDevice,
                               stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    weight.layout = QuantLayout::VoltaQpnPrepacked;
}

} // namespace ninfer::ops::detail
