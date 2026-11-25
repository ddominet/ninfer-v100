#include "ops/linear/w8/w8_cutlass_sm70.h"

#include "core/device.h"
#include "core/layout.h"
#include "ops/linear/w8/w8_rowsplit_storage.cuh"

#include "cutlass/bfloat16.h"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/half.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {

#ifdef NINFER_VOLTA_BUILD

namespace {

constexpr std::int32_t kW8CutlassMinTokens = 256;

__device__ __forceinline__ void decode_w8_u4_to_fp16(const uint4& q, __half scale,
                                                     uint4 (&dst)[2]) {
    const half2 bias  = __half2half2(__ushort_as_half(0x6480)); // 1152.0
    const half2 sc2   = __half2half2(scale);
    const unsigned w0 = q.x ^ 0x80808080u;
    const unsigned w1 = q.y ^ 0x80808080u;
    const unsigned w2 = q.z ^ 0x80808080u;
    const unsigned w3 = q.w ^ 0x80808080u;
    const unsigned ws[4] = {w0, w1, w2, w3};
    half2 vals[8];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const unsigned lo = __byte_perm(ws[i], 0x64646464u, 0x4140);
        const unsigned hi = __byte_perm(ws[i], 0x64646464u, 0x4342);
        vals[2 * i]       = __hmul2(__hsub2(*reinterpret_cast<const half2*>(&lo), bias), sc2);
        vals[2 * i + 1]   = __hmul2(__hsub2(*reinterpret_cast<const half2*>(&hi), bias), sc2);
    }
    dst[0] = reinterpret_cast<const uint4*>(vals)[0];
    dst[1] = reinterpret_cast<const uint4*>(vals)[1];
}

// Coalesced RowSplit W8 slice dequant: 16 codes (half a K=32 group) per thread -> 1x uint4 read,
// 2x uint4 FP16 stores. `row_begin` slices N so large weights like 5120x25600 fit in 50 MiB.
__global__ void dequant_w8_rowsplit_slice_to_fp16(const std::uint8_t* __restrict__ codes,
                                                  const std::uint8_t* __restrict__ scales,
                                                  int row_begin, int slice_rows, int k,
                                                  int padded_groups,
                                                  cutlass::half_t* __restrict__ out) {
    const std::int64_t chunk =
        static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int chunks_per_row = k / 16;
    const std::int64_t total = static_cast<std::int64_t>(slice_rows) * chunks_per_row;
    if (chunk >= total) { return; }

    const int local_row = static_cast<int>(chunk / chunks_per_row);
    const int col_chunk = static_cast<int>(chunk - static_cast<std::int64_t>(local_row) *
                                                       chunks_per_row);
    const int src_row   = row_begin + local_row;
    const int group     = col_chunk >> 1;

    const std::int64_t row_code_stride =
        static_cast<std::int64_t>(padded_groups) * W8RowSplitStorage::kCodeBytesPerGroup;
    const uint4 q = __ldg(reinterpret_cast<const uint4*>(
        codes + static_cast<std::int64_t>(src_row) * row_code_stride + col_chunk * 16));
    const __half sc = __ldg(reinterpret_cast<const __half*>(
        scales + (static_cast<std::int64_t>(src_row) * padded_groups + group) * 2));

    uint4 packed[2];
    decode_w8_u4_to_fp16(q, sc, packed);
    auto* dst = reinterpret_cast<uint4*>(out + static_cast<std::int64_t>(local_row) * k +
                                         col_chunk * 16);
    dst[0] = packed[0];
    dst[1] = packed[1];
}

// Coalesced Prepacked W8 dequant: 4-warp CTA owning 1 tile (32 rows) x 4 slices (64 K = 128B FP16
// per row), staging through a padded 4.5 KiB shared-memory tile so BOTH warp reads (512B) and
// warp stores (4x 128B cache lines) are 100% coalesced.
__global__ void dequant_w8_prepacked_to_fp16(const uint4* __restrict__ codes,
                                             const __half* __restrict__ scales, int n, int k,
                                             cutlass::half_t* __restrict__ out) {
    __shared__ uint4 smem[32][9];

    const int warp       = static_cast<int>(threadIdx.x) >> 5; // 0..3
    const int lane       = static_cast<int>(threadIdx.x) & 31;
    const int blocks     = k / 128;
    const int half_block = static_cast<int>(blockIdx.x) & 1;
    const int block      = static_cast<int>(blockIdx.x) >> 1;
    const int tile       = static_cast<int>(blockIdx.y);
    const int slice      = half_block * 4 + warp;
    const int local_row  = ((lane >> 2) & 3) * 8 + (lane & 3) + ((lane & 16) ? 4 : 0);
    const int row        = tile * 32 + local_row;

    if (row < n) {
        const std::int64_t blk = static_cast<std::int64_t>(tile) * blocks + block;
        const uint4 q          = __ldg(codes + (blk * 8 + slice) * 32 + lane);
        const __half sc        = __ldg(scales + (blk * 32 + lane) * 4 + (slice >> 1));
        uint4 packed[2];
        decode_w8_u4_to_fp16(q, sc, packed);
        smem[local_row][warp * 2 + 0] = packed[0];
        smem[local_row][warp * 2 + 1] = packed[1];
    }
    __syncthreads();

    const int vec   = lane & 7;
    const int k_off = block * 128 + half_block * 64 + vec * 8;
#pragma unroll
    for (int p = 0; p < 2; ++p) {
        const int lr      = warp * 8 + p * 4 + (lane >> 3);
        const int out_row = tile * 32 + lr;
        if (out_row < n) {
            *reinterpret_cast<uint4*>(out + static_cast<std::int64_t>(out_row) * k + k_off) =
                smem[lr][vec];
        }
    }
}

__global__ void w8_bf16_to_fp16_vec8_kernel(const __nv_bfloat16* __restrict__ in,
                                            cutlass::half_t* __restrict__ out,
                                            std::int64_t count) {
    const std::int64_t i =
        (static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * 8;
    if (i + 8 <= count) {
        const uint4 raw = __ldg(reinterpret_cast<const uint4*>(in + i));
        const auto* src = reinterpret_cast<const __nv_bfloat162*>(&raw);
        uint4 packed;
        auto* dst = reinterpret_cast<__half2*>(&packed);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float2 v = __bfloat1622float2(src[j]);
            dst[j]         = __floats2half2_rn(v.x, v.y);
        }
        *reinterpret_cast<uint4*>(out + i) = packed;
    } else {
        for (std::int64_t j = i; j < count; ++j) {
            out[j] = cutlass::half_t(__bfloat162float(in[j]));
        }
    }
}

using ElementAccumulator     = float;
using ElementComputeEpilogue = float;
using ElementInput           = cutlass::half_t;
using ElementOutput          = cutlass::bfloat16_t;
using Gemm = cutlass::gemm::device::Gemm<
    ElementInput, cutlass::layout::RowMajor, ElementInput, cutlass::layout::ColumnMajor,
    ElementOutput, cutlass::layout::RowMajor, ElementAccumulator, cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm70, cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>, cutlass::gemm::GemmShape<8, 8, 4>,
    cutlass::epilogue::thread::LinearCombination<
        ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator,
        ElementComputeEpilogue>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

std::size_t gemm_ws_bytes(std::int32_t n, std::int32_t k, std::int32_t cols) {
    const cutlass::gemm::GemmCoord shape(cols, n, k);
    typename Gemm::Arguments args{shape, {nullptr, k}, {nullptr, k}, {nullptr, n}, {nullptr, n},
                                  {1.0F, 0.0F}, 1};
    return Gemm::get_workspace_size(args);
}

void run_w8_cutlass_gemm(const cutlass::half_t* x_fp16, const cutlass::half_t* w_fp16,
                         ElementOutput* out, std::int32_t out_ld, std::int32_t n, std::int32_t k,
                         std::int32_t cols, void* gemm_ws, cudaStream_t stream) {
    const cutlass::gemm::GemmCoord shape(cols, n, k);
    typename Gemm::Arguments args{shape,          {x_fp16, k},    {w_fp16, k},
                                  {out, out_ld},  {out, out_ld},  {1.0F, 0.0F}, 1};
    Gemm op;
    cutlass::Status status = op.can_implement(args);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("w8_cutlass_sm70: CUTLASS can_implement failed");
    }
    status = op.initialize(args, gemm_ws, stream);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("w8_cutlass_sm70: CUTLASS initialize failed");
    }
    status = op(stream);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("w8_cutlass_sm70: CUTLASS gemm failed");
    }
    CUDA_CHECK(cudaGetLastError());
}

void launch_bf16_to_fp16(const Tensor& x, cutlass::half_t* x_fp16, cudaStream_t stream) {
    const std::int64_t count = static_cast<std::int64_t>(x.ne[0]) * x.ne[1];
    w8_bf16_to_fp16_vec8_kernel<<<static_cast<unsigned>((count + 2047) / 2048), 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), x_fp16, count);
    CUDA_CHECK(cudaGetLastError());
}

std::size_t round_up_256(std::size_t bytes) { return (bytes + 255U) & ~std::size_t(255U); }

} // namespace

bool w8_cutlass_sm70_try_launch(const Tensor& x, const Weight& w, Tensor& out,
                                WorkspaceArena& workspace, cudaStream_t stream) {
    const std::int32_t k    = x.ne[0];
    const std::int32_t cols = x.ne[1];
    const std::int32_t n    = w.n;
    if (cols < kW8CutlassMinTokens || (n % 8) != 0 || (k % 32) != 0) { return false; }

    const bool prepacked = w.layout == QuantLayout::VoltaQpnPrepacked;
    if (prepacked && ((n % 32) != 0 || (k % 128) != 0 || w.padded_shape[1] != k)) { return false; }
    if (!prepacked && w.layout != QuantLayout::RowSplit) { return false; }

    const std::size_t x_bytes = round_up_256(static_cast<std::size_t>(k) * cols * sizeof(half));
    const std::size_t free_bytes =
        workspace.capacity() > workspace.used() ? workspace.capacity() - workspace.used() : 0;

    // Choose the largest N slice (128-row multiple) that fits inside available workspace (up to
    // 64 MiB for the dequantized FP16 weight tile). Prepacked dequants full N in one shot.
    std::int32_t slice_rows = n;
    if (!prepacked) {
        constexpr std::size_t kMaxWeightBytes = 64ULL * 1024ULL * 1024ULL;
        const std::size_t full_bytes          = static_cast<std::size_t>(n) * k * sizeof(half);
        if (full_bytes > kMaxWeightBytes) {
            const std::int32_t max_rows =
                static_cast<std::int32_t>(kMaxWeightBytes / (static_cast<std::size_t>(k) * 2U));
            slice_rows = std::max<std::int32_t>(128, (max_rows / 128) * 128);
            slice_rows = std::min(slice_rows, n);
        }
    }
    const std::size_t w_bytes =
        round_up_256(static_cast<std::size_t>(slice_rows) * k * sizeof(half));
    const std::size_t g_bytes = round_up_256(gemm_ws_bytes(slice_rows, k, cols));
    // Include generous alignment margin (1 KiB) so workspace.alloc cannot throw.
    if (x_bytes + w_bytes + g_bytes + 1024U > free_bytes) { return false; }

    auto scope         = workspace.scope();
    Tensor x_fp16_t    = workspace.alloc(DType::FP16, {k, cols});
    Tensor w_fp16_t    = workspace.alloc(DType::FP16, {k, slice_rows});
    DeviceSpan gemm_ws = g_bytes > 0 ? workspace.alloc_bytes(g_bytes) : DeviceSpan{};
    auto* x_fp16       = static_cast<cutlass::half_t*>(x_fp16_t.data);
    auto* w_fp16       = static_cast<cutlass::half_t*>(w_fp16_t.data);
    launch_bf16_to_fp16(x, x_fp16, stream);

    const std::int32_t out_ld = static_cast<std::int32_t>(out.nb[1] / sizeof(__nv_bfloat16));
    auto* out_ptr             = static_cast<ElementOutput*>(out.data);

    if (prepacked) {
        const dim3 grid(static_cast<unsigned>((k / 128) * 2), static_cast<unsigned>(n / 32));
        dequant_w8_prepacked_to_fp16<<<grid, 128, 0, stream>>>(
            static_cast<const uint4*>(w.qdata), static_cast<const __half*>(w.scales), n, k, w_fp16);
        CUDA_CHECK(cudaGetLastError());
        run_w8_cutlass_gemm(x_fp16, w_fp16, out_ptr, out_ld, n, k, cols, gemm_ws.data, stream);
        return true;
    }

    const int padded_groups = w.padded_shape[1] / W8RowSplitStorage::kGroupK;
    for (std::int32_t r0 = 0; r0 < n; r0 += slice_rows) {
        const std::int32_t rlen   = std::min(slice_rows, n - r0);
        const std::int64_t chunks = static_cast<std::int64_t>(rlen) * (k / 16);
        dequant_w8_rowsplit_slice_to_fp16<<<static_cast<unsigned>((chunks + 255) / 256), 256, 0,
                                            stream>>>(
            static_cast<const std::uint8_t*>(w.qdata), static_cast<const std::uint8_t*>(w.scales),
            r0, rlen, k, padded_groups, w_fp16);
        CUDA_CHECK(cudaGetLastError());
        run_w8_cutlass_gemm(x_fp16, w_fp16, out_ptr + r0, out_ld, rlen, k, cols, gemm_ws.data,
                            stream);
    }
    return true;
}

bool w8_cutlass_sm70_try_launch_pair(const Tensor& x, const Weight& first, const Weight& second,
                                     Tensor& first_out, Tensor& second_out,
                                     WorkspaceArena& workspace, cudaStream_t stream) {
    const std::int32_t k    = x.ne[0];
    const std::int32_t cols = x.ne[1];
    if (cols < kW8CutlassMinTokens || (k % 128) != 0 || first.n != second.n ||
        (first.n % 32) != 0 || first.layout != QuantLayout::VoltaQpnPrepacked ||
        second.layout != QuantLayout::VoltaQpnPrepacked || first.padded_shape[1] != k ||
        second.padded_shape[1] != k) {
        return false;
    }

    const std::int32_t n = first.n;
    const std::size_t x_bytes = round_up_256(static_cast<std::size_t>(k) * cols * sizeof(half));
    const std::size_t w_bytes = round_up_256(static_cast<std::size_t>(n) * k * sizeof(half));
    const std::size_t g_bytes = round_up_256(gemm_ws_bytes(n, k, cols));
    const std::size_t free_bytes =
        workspace.capacity() > workspace.used() ? workspace.capacity() - workspace.used() : 0;
    if (x_bytes + w_bytes + g_bytes + 1024U > free_bytes) { return false; }

    auto scope         = workspace.scope();
    Tensor x_fp16_t    = workspace.alloc(DType::FP16, {k, cols});
    Tensor w_fp16_t    = workspace.alloc(DType::FP16, {k, n});
    DeviceSpan gemm_ws = g_bytes > 0 ? workspace.alloc_bytes(g_bytes) : DeviceSpan{};
    auto* x_fp16       = static_cast<cutlass::half_t*>(x_fp16_t.data);
    auto* w_fp16       = static_cast<cutlass::half_t*>(w_fp16_t.data);
    launch_bf16_to_fp16(x, x_fp16, stream);

    const dim3 grid(static_cast<unsigned>((k / 128) * 2), static_cast<unsigned>(n / 32));
    const std::int32_t ld0 = static_cast<std::int32_t>(first_out.nb[1] / sizeof(__nv_bfloat16));
    const std::int32_t ld1 = static_cast<std::int32_t>(second_out.nb[1] / sizeof(__nv_bfloat16));

    dequant_w8_prepacked_to_fp16<<<grid, 128, 0, stream>>>(
        static_cast<const uint4*>(first.qdata), static_cast<const __half*>(first.scales), n, k,
        w_fp16);
    CUDA_CHECK(cudaGetLastError());
    run_w8_cutlass_gemm(x_fp16, w_fp16, static_cast<ElementOutput*>(first_out.data), ld0, n, k,
                        cols, gemm_ws.data, stream);

    dequant_w8_prepacked_to_fp16<<<grid, 128, 0, stream>>>(
        static_cast<const uint4*>(second.qdata), static_cast<const __half*>(second.scales), n, k,
        w_fp16);
    CUDA_CHECK(cudaGetLastError());
    run_w8_cutlass_gemm(x_fp16, w_fp16, static_cast<ElementOutput*>(second_out.data), ld1, n, k,
                        cols, gemm_ws.data, stream);
    return true;
}

#endif // NINFER_VOLTA_BUILD

} // namespace ninfer::ops::detail
