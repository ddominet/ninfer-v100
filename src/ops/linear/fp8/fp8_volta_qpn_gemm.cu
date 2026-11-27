#include "ops/linear/volta_fp16_stage.h"
#include "ops/linear/volta_tc_skinny.h"
#include <algorithm>
#include <cstdlib>
#include <mutex>
#include <unordered_map>
#include "core/device.h"
#include "ops/linear/fp8/fp8_launch.h"
#include "ops/linear/fp8/fp8_volta_qpn_gemm.cuh"
#include "ops/common/warp.cuh"

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
    const std::int64_t i =
        (static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * 8;
    if (i + 8 <= count) {
        const uint4 raw = __ldg(reinterpret_cast<const uint4*>(input + i));
        const auto* src = reinterpret_cast<const __nv_bfloat162*>(&raw);
        uint4 packed;
        auto* dst = reinterpret_cast<half2*>(&packed);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            dst[j] = __float22half2_rn(__bfloat1622float2(src[j]));
        }
        *reinterpret_cast<uint4*>(output + i) = packed;
    } else {
        for (std::int64_t j = i; j < count; ++j) {
            output[j] = __float2half(__bfloat162float(input[j]));
        }
    }
}
} // namespace

void fp8_stage_bf16_activation_sm70(const Tensor& x, void* fp16_out, cudaStream_t stream) {
    const std::int64_t count = x.numel();
    const std::int64_t vecs  = (count + 7) / 8;
    fp8_stage_bf16_activation_kernel<<<static_cast<int>((vecs + 255) / 256), 256, 0, stream>>>(
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
                                               std::int64_t vec_count) {
    // Each vector is 8 contiguous elements (j = 0..7) at [group][tile][h][r][0..7].
    const std::int64_t vec = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (vec >= vec_count) { return; }
    const int r     = static_cast<int>(vec & 7);
    const int h     = static_cast<int>((vec >> 3) & 1);
    const std::int64_t gt = vec >> 4;
    const int tile  = static_cast<int>(gt % tiles);
    const int group = static_cast<int>(gt / tiles);
    const int row   = tile * 8 + r;
    uint4 packed;
    if (row < t) {
        const uint4 raw = __ldg(reinterpret_cast<const uint4*>(
            input + static_cast<std::int64_t>(row) * k + group * 16 + h * 8));
        const auto* src = reinterpret_cast<const __nv_bfloat162*>(&raw);
        auto* dst       = reinterpret_cast<half2*>(&packed);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            dst[j] = __float22half2_rn(__bfloat1622float2(src[j]));
        }
    } else {
        packed = make_uint4(0u, 0u, 0u, 0u);
    }
    reinterpret_cast<uint4*>(output)[vec] = packed;
}

__global__ void volta_repack_fp16_group_major_kernel(const half* __restrict__ input,
                                                     half* __restrict__ output, int k, int t,
                                                     int tiles, std::int64_t vec_count) {
    const std::int64_t vec = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (vec >= vec_count) { return; }
    const int r           = static_cast<int>(vec & 7);
    const int h           = static_cast<int>((vec >> 3) & 1);
    const std::int64_t gt = vec >> 4;
    const int tile        = static_cast<int>(gt % tiles);
    const int group       = static_cast<int>(gt / tiles);
    const int row         = tile * 8 + r;
    const uint4 packed    = (row < t)
                                ? __ldg(reinterpret_cast<const uint4*>(
                                      input + static_cast<std::int64_t>(row) * k + group * 16 +
                                      h * 8))
                                : make_uint4(0u, 0u, 0u, 0u);
    reinterpret_cast<uint4*>(output)[vec] = packed;
}

template <bool kGroupMajor, bool kWriteBf16 = false>
__launch_bounds__(640, 2) __global__
    void volta_rmsnorm_offset_d5120_staged_kernel(const __nv_bfloat16* __restrict__ x,
                                                  const __nv_bfloat16* __restrict__ weight,
                                                  half* __restrict__ out,
                                                  __nv_bfloat16* __restrict__ out_bf16, int t,
                                                  int tiles, float eps) {
    constexpr int kBlock = 640; // 20 warps, 1 uint4 (8 elements) per thread -> 5120 exact
    constexpr int kWarps = kBlock / 32;
    const int row        = static_cast<int>(blockIdx.x);
    const int tid        = static_cast<int>(threadIdx.x);

    if constexpr (kGroupMajor) {
        const int tile  = row >> 3;
        const int r     = row & 7;
        const int group = tid >> 1;
        const int khalf = tid & 1;
        if (row >= t) {
            const std::int64_t vec =
                ((static_cast<std::int64_t>(group) * tiles + tile) * 2 + khalf) * 8 + r;
            reinterpret_cast<uint4*>(out)[vec] = make_uint4(0u, 0u, 0u, 0u);
            return;
        }
    } else {
        if (row >= t) { return; }
    }

    const uint4 xv =
        __ldg(reinterpret_cast<const uint4*>(x + static_cast<std::int64_t>(row) * 5120) + tid);
    const uint4 wv = __ldg(reinterpret_cast<const uint4*>(weight) + tid);
    const auto* xp = reinterpret_cast<const __nv_bfloat162*>(&xv);
    float sum      = 0.0f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float2 xf = __bfloat1622float2(xp[j]);
        sum += xf.x * xf.x + xf.y * xf.y;
    }

    const int lane = tid & 31;
    const int warp = tid >> 5;
    sum            = ninfer::ops::warp_reduce_sum(sum);
    __shared__ float warp_sums[kWarps];
    __shared__ float inv_shared;
    if (lane == 0) { warp_sums[warp] = sum; }
    __syncthreads();
    if (warp == 0) {
        float wsum = lane < kWarps ? warp_sums[lane] : 0.0f;
        wsum       = ninfer::ops::warp_reduce_sum(wsum);
        if (lane == 0) { inv_shared = rsqrtf(wsum * (1.0f / 5120.0f) + eps); }
    }
    __syncthreads();
    const float inv = inv_shared;

    const auto* wp = reinterpret_cast<const __nv_bfloat162*>(&wv);
    uint4 packed;
    uint4 packed_bf16;
    auto* dst    = reinterpret_cast<half2*>(&packed);
    auto* dst_bf = reinterpret_cast<__nv_bfloat162*>(&packed_bf16);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float2 xf         = __bfloat1622float2(xp[j]);
        const float2 wf         = __bfloat1622float2(wp[j]);
        const __nv_bfloat162 bf = __floats2bfloat162_rn(xf.x * inv * (wf.x + 1.0f),
                                                        xf.y * inv * (wf.y + 1.0f));
        if constexpr (kWriteBf16) { dst_bf[j] = bf; }
        dst[j] = __float22half2_rn(__bfloat1622float2(bf));
    }
    if constexpr (kWriteBf16) {
        reinterpret_cast<uint4*>(out_bf16)[static_cast<std::int64_t>(row) * 640 + tid] =
            packed_bf16;
    }
    if constexpr (kGroupMajor) {
        const int tile         = row >> 3;
        const int r            = row & 7;
        const int group        = tid >> 1;
        const int khalf        = tid & 1;
        const std::int64_t vec =
            ((static_cast<std::int64_t>(group) * tiles + tile) * 2 + khalf) * 8 + r;
        reinterpret_cast<uint4*>(out)[vec] = packed;
    } else {
        reinterpret_cast<uint4*>(out)[static_cast<std::int64_t>(row) * 640 + tid] = packed;
    }
}

__global__ void fp8_argmax_unpack_kernel(const unsigned long long* __restrict__ packed_max,
                                         std::int32_t* __restrict__ out_tokens, int t) {
    const int i = static_cast<int>(threadIdx.x);
    if (i < t) {
        const unsigned long long packed = packed_max[i];
        out_tokens[i] =
            packed == 0ULL ? 0 : static_cast<std::int32_t>(~static_cast<std::uint32_t>(packed));
    }
}
} // namespace

bool volta_stage_fp16_enabled() noexcept {
    static const bool enabled = volta_stage_enabled("NINFER_VOLTA_STAGE_FP16");
    return enabled;
}

bool volta_group_major_enabled() noexcept {
    static const bool enabled =
        volta_stage_enabled("NINFER_VOLTA_STAGE_FP16") &&
        volta_stage_enabled("NINFER_VOLTA_GROUP_MAJOR");
    return enabled;
}

void volta_rmsnorm_offset_d5120_to_fp16(const Tensor& x, const Tensor& weight, float eps,
                                        half* out_fp16, cudaStream_t stream) {
    const int t = x.ne[1];
    volta_rmsnorm_offset_d5120_staged_kernel<false, false><<<static_cast<unsigned>(t), 640, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const __nv_bfloat16*>(weight.data),
        out_fp16, nullptr, t, 0, eps);
    CUDA_CHECK(cudaGetLastError());
}

void volta_rmsnorm_offset_d5120_to_bf16_and_fp16(const Tensor& x, const Tensor& weight, float eps,
                                                 Tensor& out_bf16, half* out_fp16,
                                                 cudaStream_t stream) {
    const int t = x.ne[1];
    volta_rmsnorm_offset_d5120_staged_kernel<false, true><<<static_cast<unsigned>(t), 640, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const __nv_bfloat16*>(weight.data),
        out_fp16, static_cast<__nv_bfloat16*>(out_bf16.data), t, 0, eps);
    CUDA_CHECK(cudaGetLastError());
}

void volta_rmsnorm_offset_d5120_to_group_major(const Tensor& x, const Tensor& weight, float eps,
                                               VoltaGroupMajorHalf* out_gm, cudaStream_t stream) {
    const int t     = x.ne[1];
    const int tiles = volta_group_major_tiles(t);
    volta_rmsnorm_offset_d5120_staged_kernel<true, false>
        <<<static_cast<unsigned>(tiles * 8), 640, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const __nv_bfloat16*>(weight.data), reinterpret_cast<half*>(out_gm),
            nullptr, t, tiles, eps);
    CUDA_CHECK(cudaGetLastError());
}

void* volta_stream_scratch(cudaStream_t stream, std::size_t bytes) {
    return volta_stage_scratch(stream, bytes, 1);
}

VoltaTcControl& volta_tc_control() {
    static VoltaTcControl control = [] {
        VoltaTcControl initial;
        const char* enabled = std::getenv("NINFER_VOLTA_TC");
        initial.enabled     = enabled != nullptr && enabled[0] == '0' ? 0 : 1;
        const char* minimum = std::getenv("NINFER_VOLTA_TC_MIN_T");
        initial.min_tokens  = minimum != nullptr && std::atoi(minimum) > 0 ? std::atoi(minimum) : 0;
        return initial;
    }();
    return control;
}

bool volta_tc_applies(int n, int k, int t, int format_min_tokens) {
    const VoltaTcControl& control = volta_tc_control();
    const int floor = control.min_tokens > 0 ? control.min_tokens : format_min_tokens;
    return control.enabled == 1 && t >= floor && t <= 64 && n > 0 && n % 64 == 0 && k > 0 &&
           k % 128 == 0;
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
    const std::int64_t vecs  = count / 8;
    void* data = volta_stage_scratch(stream, static_cast<std::size_t>(count) * sizeof(half));
    if (data == nullptr) { return nullptr; }
    volta_stage_group_major_kernel<<<static_cast<int>((vecs + 255) / 256), 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<half*>(data), k, t, tiles, vecs);
    CUDA_CHECK(cudaGetLastError());
    return static_cast<const VoltaGroupMajorHalf*>(data);
}

const VoltaGroupMajorHalf* volta_stage_fp16_to_group_major(const half* x_fp16, std::int32_t k,
                                                           std::int32_t t, cudaStream_t stream) {
    static const bool enabled = volta_stage_enabled("NINFER_VOLTA_STAGE_FP16") &&
                                volta_stage_enabled("NINFER_VOLTA_GROUP_MAJOR");
    if (!enabled || x_fp16 == nullptr || k % 16 != 0 || t <= 0 || t > kVoltaQpnMaxTokens) {
        return nullptr;
    }
    const int tiles          = volta_group_major_tiles(t);
    const std::int64_t count = static_cast<std::int64_t>(k / 16) * tiles * 128;
    const std::int64_t vecs  = count / 8;
    void* data = volta_stage_scratch(stream, static_cast<std::size_t>(count) * sizeof(half));
    if (data == nullptr) { return nullptr; }
    volta_repack_fp16_group_major_kernel<<<static_cast<int>((vecs + 255) / 256), 256, 0, stream>>>(
        x_fp16, static_cast<half*>(data), k, t, tiles, vecs);
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

void launch_fp8_volta_qpn_argmax_fp16(const Tensor& x, const Weight& w, const void* x_fp16,
                                      Tensor& out_tokens, std::int32_t valid_rows,
                                      void* scratch_u64, cudaStream_t stream) {
    const std::int32_t t = x.ne[1];
    const std::int32_t n = w.n;
    auto* packed_max     = static_cast<unsigned long long*>(scratch_u64);
    CUDA_CHECK(cudaMemsetAsync(packed_max, 0, static_cast<std::size_t>(t) * sizeof(unsigned long long),
                               stream));
    launch_fp8_volta_qpn_with_fp16_activation(
        x, w, static_cast<const half*>(x_fp16),
        Fp8ArgmaxOutput{packed_max, std::min(valid_rows, n)}, n, stream);
    fp8_argmax_unpack_kernel<<<1, static_cast<unsigned>(t), 0, stream>>>(
        packed_max, static_cast<std::int32_t*>(out_tokens.data), t);
    CUDA_CHECK(cudaGetLastError());
}

#endif // NINFER_VOLTA_BUILD

} // namespace ninfer::ops::detail
