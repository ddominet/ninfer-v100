#include "ops/linear/volta_fp16_stage.h"
#include "ops/linear/volta_tc_skinny.h"
#include <algorithm>
#include <cstdlib>
#include <mutex>
#include <unordered_map>
#include "core/device.h"
#include "ops/linear/fp8/fp8_launch.h"
#include "ops/linear/fp8/fp8_volta_qpn_gemm.cuh"

#include <cuda_bf16.h>

namespace ninfer::ops::detail {

#ifdef NINFER_VOLTA_BUILD

namespace {
using S = Fp8VoltaQpnSchedule;
static_assert(kFp8VoltaQpnRowsPerTile == S::kRowsPerTile,
              "dispatch mirrors the tile height for host code");

__global__ void fp8_stage_bf16_activation_kernel(const __nv_bfloat16* __restrict__ input,
                                                 half* __restrict__ output,
                                                 std::int64_t count) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) { output[i] = __float2half(__bfloat162float(input[i])); }
}
} // namespace

void fp8_stage_bf16_activation_sm70(const Tensor& x, void* fp16_out, cudaStream_t stream) {
    const std::int64_t count = x.numel();
    fp8_stage_bf16_activation_kernel<<<static_cast<int>((count + 255) / 256), 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<half*>(fp16_out), count);
    CUDA_CHECK(cudaGetLastError());
}

namespace {
// Persistent per-stream device scratch, one fixed block per slot: 4 MiB for activation staging
// (K=17408 at kVoltaQpnMaxTokens rows is 2.2 MiB in either layout) and 1 MiB for the attention
// query plane (eight streams of eight tokens are 0.75 MiB). A block is allocated once and never
// grown or freed, so pointers baked into captured graphs stay valid. Callers keep their unstaged
// route when this returns nullptr: the request is larger than the block, the stream is capturing
// a graph before its block exists, or the device had no memory left for it.
void* volta_stage_scratch(cudaStream_t stream, std::size_t bytes, int slot = 0) {
    constexpr std::size_t kCapacity[2] = {std::size_t{4} << 20, std::size_t{1} << 20};
    struct Scratch {
        void* data = nullptr;
        bool tried = false;
    };
    static std::mutex mutex;
    static std::unordered_map<cudaStream_t, Scratch> scratch_by_stream[2];
    if (bytes > kCapacity[slot]) { return nullptr; }
    const std::lock_guard<std::mutex> lock(mutex);
    Scratch& scratch = scratch_by_stream[slot][stream];
    if (!scratch.tried) {
        // Allocation is illegal while a stream is capturing, and even a refused cudaMalloc
        // invalidates the capture, so ask first and try again on a later launch.
        cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
        CUDA_CHECK(cudaStreamIsCapturing(stream, &capture));
        if (capture != cudaStreamCaptureStatusNone) { return nullptr; }
        scratch.tried = true;
        if (cudaMalloc(&scratch.data, kCapacity[slot]) != cudaSuccess) {
            (void)cudaGetLastError();
            scratch.data = nullptr;
        }
    }
    return scratch.data;
}

bool volta_stage_enabled(const char* name) {
    const char* value = std::getenv(name);
    return value == nullptr || value[0] != '0';
}

__global__ void volta_stage_group_major_kernel(const __nv_bfloat16* __restrict__ input,
                                               half* __restrict__ output, int k, int t, int tiles,
                                               std::int64_t count) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) { return; }
    const int j     = static_cast<int>(i & 7);
    const int r     = static_cast<int>((i >> 3) & 7);
    const int h     = static_cast<int>((i >> 6) & 1);
    const int tile  = static_cast<int>((i >> 7) % tiles);
    const int group = static_cast<int>((i >> 7) / tiles);
    const int row   = tile * 8 + r;
    output[i] = row < t ? __float2half(__bfloat162float(
                              input[static_cast<std::int64_t>(row) * k + group * 16 + h * 8 + j]))
                        : __ushort_as_half(0);
}
} // namespace

void* volta_stream_scratch(cudaStream_t stream, std::size_t bytes) {
    return volta_stage_scratch(stream, bytes, 1);
}

VoltaTcControl& volta_tc_control() {
    static VoltaTcControl control = [] {
        VoltaTcControl initial;
        const char* enabled = std::getenv("NINFER_VOLTA_TC");
        // Off by default: on the 27B shapes the quadpair kernels still win at 9..32 tokens.
        initial.enabled     = enabled != nullptr && enabled[0] == '1' ? 1 : 0;
        const char* minimum = std::getenv("NINFER_VOLTA_TC_MIN_T");
        initial.min_tokens  = minimum != nullptr && std::atoi(minimum) > 0 ? std::atoi(minimum) : 9;
        return initial;
    }();
    return control;
}

bool volta_tc_applies(int n, int k, int t) {
    const VoltaTcControl& control = volta_tc_control();
    return control.enabled == 1 && t >= control.min_tokens && t <= 64 && n > 0 && n % 64 == 0 &&
           k > 0 && k % 128 == 0;
}

VoltaTcSplitWorkspace volta_tc_split_workspace(cudaStream_t stream, std::size_t partial_floats,
                                               std::size_t semaphores) {
    // FP32 partials of the widest split shape (17408 rows x 64 tokens is 4.25 MiB) and one
    // semaphore per row block, allocated and zeroed once per stream and never freed (pointers
    // baked into captured graphs stay valid).
    constexpr std::size_t kPartialBytes   = std::size_t{17408} * 64 * sizeof(float);
    constexpr std::size_t kSemaphoreCount = 1024;
    struct Workspace {
        void* data = nullptr;
        bool tried = false;
    };
    static std::mutex mutex;
    static std::unordered_map<cudaStream_t, Workspace> workspace_by_stream;
    if (partial_floats * sizeof(float) > kPartialBytes || semaphores > kSemaphoreCount) {
        return {};
    }
    const std::lock_guard<std::mutex> lock(mutex);
    Workspace& workspace = workspace_by_stream[stream];
    if (!workspace.tried) {
        cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
        CUDA_CHECK(cudaStreamIsCapturing(stream, &capture));
        if (capture != cudaStreamCaptureStatusNone) { return {}; }
        workspace.tried         = true;
        const std::size_t bytes = kPartialBytes + kSemaphoreCount * sizeof(unsigned);
        if (cudaMalloc(&workspace.data, bytes) != cudaSuccess) {
            (void)cudaGetLastError();
            workspace.data = nullptr;
        } else {
            CUDA_CHECK(cudaMemsetAsync(workspace.data, 0, bytes, stream));
        }
    }
    if (workspace.data == nullptr) { return {}; }
    return {static_cast<float*>(workspace.data),
            reinterpret_cast<unsigned*>(static_cast<char*>(workspace.data) + kPartialBytes)};
}

const half* volta_stage_fp16_activation(const Tensor& x, cudaStream_t stream) {
    static const bool enabled = volta_stage_enabled("NINFER_VOLTA_STAGE_FP16");
    if (!enabled) { return nullptr; }
    void* data = volta_stage_scratch(stream, static_cast<std::size_t>(x.numel()) * sizeof(half));
    if (data == nullptr) { return nullptr; }
    fp8_stage_bf16_activation_sm70(x, data, stream);
    return static_cast<const half*>(data);
}

const VoltaGroupMajorHalf* volta_stage_fp16_group_major(const Tensor& x, cudaStream_t stream) {
    static const bool enabled = volta_stage_enabled("NINFER_VOLTA_STAGE_FP16") &&
                                volta_stage_enabled("NINFER_VOLTA_GROUP_MAJOR");
    const int k = x.ne[0];
    const int t = x.ne[1];
    if (!enabled || k % 16 != 0 || t > kVoltaQpnMaxTokens) { return nullptr; }
    const int tiles          = volta_group_major_tiles(t);
    const std::int64_t count = static_cast<std::int64_t>(k / 16) * tiles * 128;
    void* data = volta_stage_scratch(stream, static_cast<std::size_t>(count) * sizeof(half));
    if (data == nullptr) { return nullptr; }
    volta_stage_group_major_kernel<<<static_cast<int>((count + 255) / 256), 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<half*>(data), k, t, tiles, count);
    CUDA_CHECK(cudaGetLastError());
    return static_cast<const VoltaGroupMajorHalf*>(data);
}

bool fp8_volta_qpn_supported(std::int32_t n, std::int32_t k, std::int32_t t) noexcept {
    if (n <= 0 || t <= 0) { return false; }
    // A lane consumes whole 128-byte lines, and the CTA's SPLITK warps split K by whole lines.
    // 8 is the floor every shipped SPLITK choice (8 or 16) divides down to (see the launcher).
    if (k % S::kKPerBlock != 0) { return false; }
    if (k / S::kKPerBlock < 8) { return false; }
    // Two 8-row A tiles cover T=9..16 in one pass over the weights. The W8 sibling rejected its
    // two-tile form because the 32x8 fused route already absorbed those rows for free; FP8 has no
    // 32x8 route, so here the alternative is chunking -- reading the whole weight twice -- and the
    // second tile wins easily.
    //
    // The portable layout historically kept T=1 on the slightly faster SIMT decoder. Volta
    // artifacts are now prepacked for QPN, so the row-major decoder is no longer a legal consumer;
    // QPN owns that width as well as verification.
    return t <= kVoltaQpnMaxTokens;
}

void launch_fp8_volta_qpn(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    const std::int32_t n = out.ne[0];
    launch_fp8_volta_qpn_with_output(x, w, Fp8ContiguousOutput{static_cast<__nv_bfloat16*>(out.data),
                                                               n},
                                     n, stream);
}

void launch_fp8_volta_qpn_fp16(const Tensor& x, const Weight& w, const void* x_fp16, Tensor& out,
                               cudaStream_t stream) {
    const std::int32_t n = out.ne[0];
    launch_fp8_volta_qpn_with_fp16_activation(
        x, w, static_cast<const half*>(x_fp16),
        Fp8ContiguousOutput{static_cast<__nv_bfloat16*>(out.data), n}, n, stream);
}

#endif // NINFER_VOLTA_BUILD

} // namespace ninfer::ops::detail
