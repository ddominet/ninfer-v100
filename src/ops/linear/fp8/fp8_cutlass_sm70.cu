#include "ops/linear/fp8/fp8_cutlass_sm70.h"

#include "core/device.h"
#include "core/layout.h"
#include "ops/linear/fp8/fp8_gemv.cuh"
#include "ops/linear/fp8/fp8_prepack_sm70.cuh"

#include "cutlass/bfloat16.h"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/half.h"

#include <cuda_bf16.h>

#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

__global__ void dequant_fp8_row_to_fp16(const std::uint8_t* __restrict__ codes, int n, int k,
                                        bool prepacked,
                                        cutlass::half_t* __restrict__ out) {
    const int row      = static_cast<int>(blockIdx.y);
    const int pair_idx = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= n || pair_idx >= k / 2) { return; }
    const int k0 = pair_idx * 2;
    const std::uint8_t* source = nullptr;
    if (prepacked) {
        source = codes + fp8_qpn_prepacked_offset(row, k0, k);
    } else {
        source = codes + static_cast<std::int64_t>(row) * k + k0;
    }
    const std::uint16_t packed = *reinterpret_cast<const std::uint16_t*>(source);
    const float2 weight = decode_fp8_e4m3x2(packed);
    cutlass::half_t* out_row = out + static_cast<std::int64_t>(row) * k;
    out_row[pair_idx * 2]     = cutlass::half_t(weight.x);
    out_row[pair_idx * 2 + 1] = cutlass::half_t(weight.y);
}

// Prepacked code plane in storage order: [N/32 tile][K/128 block][8 slices][32 lanes][16 codes],
// where a lane's 16 codes are 16 consecutive columns of one row (fp8_qpn_prepacked_offset). One
// thread moves one 16-code chunk: a warp reads 512 contiguous bytes and each thread writes 32
// contiguous bytes of its row. Same per-value decode as the row-order kernel, which gathers two
// bytes per thread (~505 GB/s on the p45k prefill trace).
__global__ void dequant_fp8_prepacked_to_fp16(const std::uint8_t* __restrict__ codes, int n, int k,
                                              cutlass::half_t* __restrict__ out) {
    const std::int64_t chunk =
        static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int blocks = k / 128;
    const int lane   = static_cast<int>(chunk & 31);
    const int slice  = static_cast<int>((chunk >> 5) & 7);
    const std::int64_t tile_block = chunk >> 8;
    const int block  = static_cast<int>(tile_block % blocks);
    const int tile   = static_cast<int>(tile_block / blocks);
    const int row    = tile * 32 + ((lane >> 2) & 3) * 8 + ((lane & 3) | ((lane & 16) ? 4 : 0));
    if (row >= n) { return; }
    const uint4 raw      = *reinterpret_cast<const uint4*>(codes + chunk * 16);
    const auto* pairs    = reinterpret_cast<const std::uint16_t*>(&raw);
    uint4 packed[2];
    auto* dst = reinterpret_cast<__half2*>(packed);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const float2 weight = decode_fp8_e4m3x2(pairs[j]);
        dst[j]              = __floats2half2_rn(weight.x, weight.y);
    }
    auto* destination = reinterpret_cast<uint4*>(out + static_cast<std::int64_t>(row) * k +
                                                 block * 128 + slice * 16);
    destination[0] = packed[0];
    destination[1] = packed[1];
}

__global__ void bf16_to_fp16_kernel(const __nv_bfloat16* __restrict__ in,
                                    cutlass::half_t* __restrict__ out, std::int64_t count) {
    // Eight elements (one 16-byte transaction each way) per thread; the scalar form ran at
    // ~370 GB/s on the p45k prefill trace.
    const std::int64_t i =
        (static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * 8;
    if (i + 8 <= count) {
        const uint4 raw = *reinterpret_cast<const uint4*>(in + i);
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

// data is row-major {rows=n, cols}; element (r, c) at c*n + r is scaled by scales[r].
// The old flat form paid an int64 modulo per element (237 GB/s on the p45k trace); the
// 2-D grid drops the division entirely and vectorizes rows in bf16 pairs.
__global__ void scale_rows_kernel(__nv_bfloat16* __restrict__ data,
                                  const __nv_bfloat16* __restrict__ scales, std::int64_t cols,
                                  int rows) {
    const int pair = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::int64_t col = static_cast<std::int64_t>(blockIdx.y);
    if (pair * 2 + 1 >= rows) {
        if (pair * 2 < rows) {
            __nv_bfloat16* d = data + col * rows + pair * 2;
            *d = __float2bfloat16(__bfloat162float(*d) * __bfloat162float(scales[pair * 2]));
        }
        return;
    }
    const __nv_bfloat162 s2 = reinterpret_cast<const __nv_bfloat162*>(scales)[pair];
    __nv_bfloat162* d2      = reinterpret_cast<__nv_bfloat162*>(data + col * rows) + pair;
    const float2 d0         = __bfloat1622float2(*d2);
    const float2 sf         = __bfloat1622float2(s2);
    *d2                     = __floats2bfloat162_rn(d0.x * sf.x, d0.y * sf.y);
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

template <class Allocator>
struct Scratch {
    Tensor weight;
    Tensor input;
    DeviceSpan gemm;
};

template <class Allocator>
Scratch<Allocator> allocate_scratch(Allocator& allocator, int n, int k, int cols,
                                    std::size_t gemm_bytes) {
    Scratch<Allocator> out;
    out.weight = allocator.alloc(DType::FP16, {k, n});
    out.input  = allocator.alloc(DType::FP16, {k, cols});
    if (gemm_bytes != 0) { out.gemm = allocator.alloc_bytes(gemm_bytes); }
    return out;
}

std::size_t gemm_workspace_bytes(int n, int k, int cols) {
    const cutlass::gemm::GemmCoord shape(cols, n, k);
    typename Gemm::Arguments args{shape, {nullptr, k}, {nullptr, k}, {nullptr, n}, {nullptr, n},
                                  {1.0F, 0.0F}, 1};
    return Gemm::get_workspace_size(args);
}

} // namespace

std::size_t fp8_cutlass_sm70_workspace_bytes(std::int32_t n, std::int32_t k, std::int32_t cols) {
    WorkspaceLayoutBuilder layout;
    (void)allocate_scratch(layout, n, k, cols, gemm_workspace_bytes(n, k, cols));
    return layout.peak_bytes(1);
}

void fp8_cutlass_sm70_launch(const Tensor& x, const Weight& w, Tensor& out, WorkspaceArena& ws,
                             cudaStream_t stream) {
    const int n = w.n;
    const int k = w.k;
    const int t = x.ne[1];
    const std::size_t gemm_bytes = gemm_workspace_bytes(n, k, t);
    auto scope = ws.scope();
    Scratch<WorkspaceArena> scratch = allocate_scratch(ws, n, k, t, gemm_bytes);
    auto* weight = static_cast<cutlass::half_t*>(scratch.weight.data);
    auto* input  = static_cast<cutlass::half_t*>(scratch.input.data);

    if (w.layout == QuantLayout::VoltaQpnPrepacked && n % 32 == 0 && k % 128 == 0) {
        const std::int64_t chunks = static_cast<std::int64_t>(n) * k / 16;
        dequant_fp8_prepacked_to_fp16<<<static_cast<unsigned>((chunks + 255) / 256), 256, 0,
                                        stream>>>(static_cast<const std::uint8_t*>(w.qdata), n, k,
                                                  weight);
    } else {
        const dim3 block(256);
        const dim3 grid(static_cast<unsigned>((k / 2 + 255) / 256), static_cast<unsigned>(n), 1u);
        dequant_fp8_row_to_fp16<<<grid, block, 0, stream>>>(
            static_cast<const std::uint8_t*>(w.qdata), n, k,
            w.layout == QuantLayout::VoltaQpnPrepacked, weight);
    }
    CUDA_CHECK(cudaGetLastError());
    const std::int64_t input_count = static_cast<std::int64_t>(t) * k;
    bf16_to_fp16_kernel<<<static_cast<int>((input_count + 2047) / 2048), 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), input, input_count);
    CUDA_CHECK(cudaGetLastError());

    const cutlass::gemm::GemmCoord shape(t, n, k);
    typename Gemm::Arguments args{shape, {input, k}, {weight, k},
                                  {static_cast<ElementOutput*>(out.data), n},
                                  {static_cast<ElementOutput*>(out.data), n}, {1.0F, 0.0F}, 1};
    Gemm op;
    cutlass::Status status = op.can_implement(args);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("fp8_cutlass_sm70: CUTLASS can_implement failed");
    }
    status = op.initialize(args, scratch.gemm.data, stream);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("fp8_cutlass_sm70: CUTLASS initialize failed");
    }
    status = op(stream);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("fp8_cutlass_sm70: CUTLASS gemm failed");
    }
    CUDA_CHECK(cudaGetLastError());

    const dim3 scale_grid(static_cast<unsigned>(((n + 1) / 2 + 255) / 256),
                          static_cast<unsigned>(t), 1u);
    scale_rows_kernel<<<scale_grid, 256, 0, stream>>>(
        static_cast<__nv_bfloat16*>(out.data), static_cast<const __nv_bfloat16*>(w.scales),
        static_cast<std::int64_t>(t), n);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
