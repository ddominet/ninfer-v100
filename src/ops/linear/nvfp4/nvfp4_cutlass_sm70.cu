#include "ops/linear/nvfp4/nvfp4_cutlass_sm70.h"

#include "ops/linear/nvfp4/nvfp4_prepack_gemv_t1.cuh"

#include "core/device.h"
#include "core/layout.h"
#include "ops/common/math.cuh"
#include "ops/linear/nvfp4/nvfp4_codec.cuh"

#include "cutlass/bfloat16.h"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/half.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

__device__ __forceinline__ half2 decode_e4m3_scale_shift(std::uint8_t value) {
    const std::uint16_t bits = static_cast<std::uint16_t>((value & 0x80u) << 8) |
                               static_cast<std::uint16_t>((value & 0x7fu) << 7);
    return __half2half2(__ushort_as_half(bits));
}

__device__ __forceinline__ void decode_e2m1_word_shift(std::uint32_t packed, half2 rebias,
                                                       half2 (&out)[4]) {
    constexpr std::uint32_t sign = 0x80008000u;
    constexpr std::uint32_t expm = 0x0e000e00u;
    std::uint32_t v0 = ((packed << 12) & sign) | ((packed << 9) & expm);
    std::uint32_t v1 = ((packed << 8) & sign) | ((packed << 5) & expm);
    std::uint32_t v2 = ((packed << 4) & sign) | ((packed << 1) & expm);
    std::uint32_t v3 = (packed & sign) | ((packed >> 3) & expm);
    out[0] = __hmul2(*reinterpret_cast<half2*>(&v0), rebias);
    out[1] = __hmul2(*reinterpret_cast<half2*>(&v1), rebias);
    out[2] = __hmul2(*reinterpret_cast<half2*>(&v2), rebias);
    out[3] = __hmul2(*reinterpret_cast<half2*>(&v3), rebias);
}

__device__ __forceinline__ void unshuffle_e2m1_word(const half2 (&in)[4], half2 (&out)[4]) {
    const auto* source = reinterpret_cast<const std::uint32_t*>(in);
    auto* destination  = reinterpret_cast<std::uint32_t*>(out);
    destination[0] = (source[0] & 0x0000ffffu) | (source[1] << 16);
    destination[1] = (source[2] & 0x0000ffffu) | (source[3] << 16);
    destination[2] = (source[0] >> 16) | (source[1] & 0xffff0000u);
    destination[3] = (source[2] >> 16) | (source[3] & 0xffff0000u);
}

// The artifact stores adjacent E2M1 values in each code byte and one E4M3 scale per K16 group.
// Scales use the BlockScaleK16M128x4 swizzle. Materializing once is intentionally a wide-T
// strategy: CUTLASS can reuse the resulting FP16 matrix across every token tile instead of
// decoding the packed weights again for each tile.
__global__ void dequant_nvfp4_row_to_fp16(const std::uint8_t* __restrict__ codes,
                                           const std::uint8_t* __restrict__ scales, int n, int k,
                                           float inverse_weight_divisor,
                                           cutlass::half_t* __restrict__ out) {
    // A thread owns half of one K16 scale group. The former byte-per-thread,
    // two-dimensional launch created hundreds of thousands of tiny blocks for
    // gate_up; this preserves the exact native artifact decode while matching
    // the prepacked decoder's eight-output work granularity.
    const std::int64_t segment = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int segments_per_row = k / 8;
    const std::int64_t count   = static_cast<std::int64_t>(n) * segments_per_row;
    if (segment >= count) { return; }
    const int row        = static_cast<int>(segment / segments_per_row);
    const int row_segment = static_cast<int>(segment -
        static_cast<std::int64_t>(row) * segments_per_row);
    const int group      = row_segment / 2;
    const int group_half = row_segment & 1;
    const int scale_tile    = group / 4;
    const int scale_lane    = group & 3;
    const int row_inner     = row & 127;
    const int scales_per_m128 = k / 64;
    const std::int64_t scale_offset =
        static_cast<std::int64_t>((row / 128) * scales_per_m128 + scale_tile) * 512 +
        (row_inner & 31) * 16 + (row_inner >> 5) * 4 + scale_lane;
    const int byte_begin = group * 8 + group_half * 4;
    const std::uint8_t* source = codes + static_cast<std::int64_t>(row) * (k / 2) + byte_begin;
    const std::uint32_t packed = *reinterpret_cast<const std::uint32_t*>(source);
    const half2 rebias = __float2half2_rn(16384.0f);
    const half2 divisor = __float2half2_rn(inverse_weight_divisor * 256.0f);
    const half2 coefficient = __hmul2(decode_e4m3_scale_shift(scales[scale_offset]), divisor);
    half2 interleaved[4];
    decode_e2m1_word_shift(packed, rebias, interleaved);
    half2 adjacent[4];
    unshuffle_e2m1_word(interleaved, adjacent);
#pragma unroll
    for (half2& value : adjacent) { value = __hmul2(value, coefficient); }
    auto* destination = reinterpret_cast<uint4*>(
        out + static_cast<std::int64_t>(row) * k + byte_begin * 2);
    *destination = *reinterpret_cast<const uint4*>(adjacent);
}

__global__ void dequant_nvfp4_qpn_to_fp16(const std::uint8_t* __restrict__ codes,
                                          const std::uint8_t* __restrict__ scales, int n, int k,
                                          float inverse_weight_divisor,
                                          cutlass::half_t* __restrict__ out) {
    const int row      = static_cast<int>(blockIdx.y);
    const int segment  = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int segments_per_row = k / 8;
    if (row >= n || segment >= segments_per_row) { return; }

    const int group      = segment / 2;
    const int group_half = segment & 1;
    const int local_row  = row & 31;
    const int qp         = local_row / 8;
    const int r          = local_row & 7;
    const int lane       = (qp << 2) | (r & 3) | ((r & 4) << 2);
    const int groups     = k / 16;
    const std::int64_t tuple =
        (static_cast<std::int64_t>(row / 32) * groups + group) * 32 + lane;
    const std::uint8_t* packed = codes + tuple * 8 + group_half * 4;
    const std::uint32_t packed_word = *reinterpret_cast<const std::uint32_t*>(packed);
    const half2 rebias = __float2half2_rn(16384.0f);
    const half2 divisor = __float2half2_rn(inverse_weight_divisor * 256.0f);
    const half2 coefficient = __hmul2(decode_e4m3_scale_shift(scales[tuple]), divisor);
    half2 values[4];
    decode_e2m1_word_shift(packed_word, rebias, values);
#pragma unroll
    for (int pair = 0; pair < 4; ++pair) { values[pair] = __hmul2(values[pair], coefficient); }
    auto* destination = reinterpret_cast<uint4*>(
        out + static_cast<std::int64_t>(row) * k + segment * 8);
    *destination = *reinterpret_cast<const uint4*>(values);
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

using ElementAccumulator     = float;
using ElementComputeEpilogue = ElementAccumulator;
using ElementInputA          = cutlass::half_t;
using ElementInputB          = cutlass::half_t;
using ElementOutput          = cutlass::bfloat16_t;
using LayoutInputA           = cutlass::layout::RowMajor;
using LayoutInputB           = cutlass::layout::ColumnMajor;
using LayoutOutput           = cutlass::layout::RowMajor;
using MMAOp                  = cutlass::arch::OpClassTensorOp;
using SmArch                 = cutlass::arch::Sm70;
using ShapeMMAThreadBlock    = cutlass::gemm::GemmShape<128, 128, 32>;
using ShapeMMAWarp           = cutlass::gemm::GemmShape<64, 64, 32>;
using ShapeMMAOp             = cutlass::gemm::GemmShape<8, 8, 4>;
using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator,
    ElementComputeEpilogue>;
constexpr int kNumStages = 2;
using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB,
                                         ElementOutput, LayoutOutput, ElementAccumulator, MMAOp,
                                         SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp,
                                         EpilogueOp, SwizzleThreadBlock, kNumStages>;

// The SwiGLU route keeps gate and up in FP32 until after silu(gate) * up.
using EpilogueOpF32 = cutlass::epilogue::thread::LinearCombination<
    float, 128 / cutlass::sizeof_bits<float>::value, ElementAccumulator, ElementComputeEpilogue>;
using GemmF32 = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB,
                                            LayoutInputB, float, LayoutOutput, ElementAccumulator,
                                            MMAOp, SmArch, ShapeMMAThreadBlock, ShapeMMAWarp,
                                            ShapeMMAOp, EpilogueOpF32, SwizzleThreadBlock,
                                            kNumStages>;

template <class GemmT>
std::size_t gemm_workspace_bytes(std::int32_t n, std::int32_t k, std::int32_t cols) {
    cutlass::gemm::GemmCoord problem_size(cols, n, k);
    typename GemmT::Arguments arguments{
        problem_size, {nullptr, k}, {nullptr, k}, {nullptr, n}, {nullptr, n},
        {ElementComputeEpilogue(1), ElementComputeEpilogue(0)}, 1};
    return GemmT::get_workspace_size(arguments);
}

// Dequantizes the weight to FP16 {k, n} and converts the BF16 activation {k, cols} to FP16.
void stage_operands(const Tensor& x, const Weight& w, cutlass::half_t* w_fp16,
                    cutlass::half_t* x_fp16, cudaStream_t stream) {
    const std::int32_t k    = x.ne[0];
    const std::int32_t cols = x.ne[1];
    const std::int32_t n    = w.n;
    constexpr int kThreads  = 256;
    if (w.layout == QuantLayout::VoltaQpnPrepacked) {
        const dim3 grid(static_cast<unsigned>((k / 8 + kThreads - 1) / kThreads),
                        static_cast<unsigned>(n), 1u);
        dequant_nvfp4_qpn_to_fp16<<<grid, kThreads, 0, stream>>>(
            static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const std::uint8_t*>(w.scales), n, k,
            1.0F / w.weight_scale_divisor, w_fp16);
    } else {
        const std::int64_t segments = static_cast<std::int64_t>(n) * (k / 8);
        const int blocks = static_cast<int>((segments + kThreads - 1) / kThreads);
        dequant_nvfp4_row_to_fp16<<<blocks, kThreads, 0, stream>>>(
            static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const std::uint8_t*>(w.scales), n, k,
            1.0F / w.weight_scale_divisor, w_fp16);
    }
    CUDA_CHECK(cudaGetLastError());

    const std::int64_t x_count = static_cast<std::int64_t>(cols) * k;
    const int x_blocks         = static_cast<int>((x_count + 2047) / 2048);
    bf16_to_fp16_kernel<<<x_blocks, 256, 0, stream>>>(static_cast<const __nv_bfloat16*>(x.data),
                                                      x_fp16, x_count);
    CUDA_CHECK(cudaGetLastError());
}

// out is RowMajor [cols, n]: element (token, row) at token * n + row.
template <class GemmT, class Output>
void run_gemm(const cutlass::half_t* x_fp16, const cutlass::half_t* w_fp16, Output* out,
              std::int32_t n, std::int32_t k, std::int32_t cols, void* gemm_workspace,
              cudaStream_t stream) {
    cutlass::gemm::GemmCoord problem_size(cols, n, k);
    GemmT gemm_op;
    typename GemmT::Arguments arguments{
        problem_size, {x_fp16, k}, {w_fp16, k}, {out, n}, {out, n},
        {ElementComputeEpilogue(1), ElementComputeEpilogue(0)}, 1};
    cutlass::Status status = gemm_op.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("nvfp4_cutlass_sm70: CUTLASS can_implement failed");
    }
    status = gemm_op.initialize(arguments, gemm_workspace, stream);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("nvfp4_cutlass_sm70: CUTLASS initialize failed");
    }
    status = gemm_op(stream);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("nvfp4_cutlass_sm70: CUTLASS gemm() failed");
    }
    CUDA_CHECK(cudaGetLastError());
}

template <class Allocator>
struct CutlassWorkspace {
    Tensor w_fp16;
    Tensor x_fp16;
    DeviceSpan gemm_workspace;
};

template <class Allocator>
CutlassWorkspace<Allocator> allocate_cutlass_workspace(Allocator& allocator, std::int32_t n,
                                                       std::int32_t k, std::int32_t cols,
                                                       std::size_t gemm_workspace_bytes) {
    CutlassWorkspace<Allocator> out;
    out.w_fp16 = allocator.alloc(DType::FP16, {k, n});
    out.x_fp16 = allocator.alloc(DType::FP16, {k, cols});
    if (gemm_workspace_bytes > 0) { out.gemm_workspace = allocator.alloc_bytes(gemm_workspace_bytes); }
    return out;
}

// The FP32 gate/up plane of the SwiGLU route covers at most 1024 columns at a time, so at the
// 2048-token prefill chunk it is no larger than the BF16 {n, cols} projection it replaces. Each
// pass streams the whole FP16 weight, so narrower inputs stay a single pass.
constexpr std::int32_t kSwigluPassColumns = 1024;
constexpr std::int32_t swiglu_pass_columns(std::int32_t cols) {
    return cols < kSwigluPassColumns ? cols : kSwigluPassColumns;
}

template <class Allocator>
struct SwigluWorkspace {
    CutlassWorkspace<Allocator> operands;
    DeviceSpan gate_up;
};

template <class Allocator>
SwigluWorkspace<Allocator> allocate_swiglu_workspace(Allocator& allocator, std::int32_t n,
                                                     std::int32_t k, std::int32_t cols) {
    const std::int32_t pass = swiglu_pass_columns(cols);
    SwigluWorkspace<Allocator> out;
    out.operands =
        allocate_cutlass_workspace(allocator, n, k, cols, gemm_workspace_bytes<GemmF32>(n, k, pass));
    out.gate_up =
        allocator.alloc_bytes(static_cast<std::size_t>(n) * pass * sizeof(float), 256);
    return out;
}

// gate_up is [tokens][2 * half] FP32 (gate rows, then up rows); out is BF16 {half, tokens}.
__global__ void swiglu_fp32_rows_kernel(const float* __restrict__ gate_up,
                                        __nv_bfloat16* __restrict__ out, int half,
                                        std::int64_t quads) {
    const std::int64_t quad = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (quad >= quads) { return; }
    const int quads_per_token = half / 4;
    const std::int64_t token  = quad / quads_per_token;
    const int row             = static_cast<int>(quad - token * quads_per_token) * 4;
    const float* source       = gate_up + token * 2 * half + row;
    const float4 gate         = *reinterpret_cast<const float4*>(source);
    const float4 up           = *reinterpret_cast<const float4*>(source + half);
    __nv_bfloat162 packed[2];
    packed[0] = __floats2bfloat162_rn(silu(gate.x) * up.x, silu(gate.y) * up.y);
    packed[1] = __floats2bfloat162_rn(silu(gate.z) * up.z, silu(gate.w) * up.w);
    *reinterpret_cast<uint2*>(out + token * half + row) = *reinterpret_cast<const uint2*>(packed);
}

} // namespace

std::size_t nvfp4_cutlass_sm70_workspace_bytes(std::int32_t n, std::int32_t k,
                                               std::int32_t cols) {
    WorkspaceLayoutBuilder layout;
    (void)allocate_cutlass_workspace(layout, n, k, cols, gemm_workspace_bytes<Gemm>(n, k, cols));
    return layout.peak_bytes(1);
}

void nvfp4_cutlass_sm70_launch(const Tensor& x, const Weight& w, Tensor& out, WorkspaceArena& ws,
                               cudaStream_t stream) {
    const std::int32_t k    = x.ne[0];
    const std::int32_t cols = x.ne[1];
    const std::int32_t n    = w.n;
    auto scratch_scope      = ws.scope();
    CutlassWorkspace<WorkspaceArena> scratch =
        allocate_cutlass_workspace(ws, n, k, cols, gemm_workspace_bytes<Gemm>(n, k, cols));
    auto* w_fp16 = static_cast<cutlass::half_t*>(scratch.w_fp16.data);
    auto* x_fp16 = static_cast<cutlass::half_t*>(scratch.x_fp16.data);
    stage_operands(x, w, w_fp16, x_fp16, stream);
    run_gemm<Gemm>(x_fp16, w_fp16, static_cast<ElementOutput*>(out.data), n, k, cols,
                   scratch.gemm_workspace.data, stream);
}

std::size_t nvfp4_linear_swiglu_cutlass_sm70_workspace_bytes(std::int32_t n, std::int32_t k,
                                                             std::int32_t cols) {
    WorkspaceLayoutBuilder layout;
    (void)allocate_swiglu_workspace(layout, n, k, cols);
    return layout.peak_bytes(1);
}

void nvfp4_linear_swiglu_cutlass_sm70_launch(const Tensor& x, const Weight& w, Tensor& out,
                                             WorkspaceArena& ws, cudaStream_t stream) {
    const std::int32_t k    = x.ne[0];
    const std::int32_t cols = x.ne[1];
    const std::int32_t n    = w.n;
    const std::int32_t half = n / 2;
    if (n % 8 != 0 || out.ne[0] != half || out.ne[1] != cols) {
        throw std::invalid_argument("nvfp4 linear_swiglu cutlass: invalid shape");
    }
    auto scratch_scope = ws.scope();
    SwigluWorkspace<WorkspaceArena> scratch = allocate_swiglu_workspace(ws, n, k, cols);
    auto* w_fp16  = static_cast<cutlass::half_t*>(scratch.operands.w_fp16.data);
    auto* x_fp16  = static_cast<cutlass::half_t*>(scratch.operands.x_fp16.data);
    auto* gate_up = static_cast<float*>(scratch.gate_up.data);
    stage_operands(x, w, w_fp16, x_fp16, stream);

    const std::int32_t pass = swiglu_pass_columns(cols);
    for (std::int32_t begin = 0; begin < cols; begin += pass) {
        const std::int32_t active = std::min(pass, cols - begin);
        run_gemm<GemmF32>(x_fp16 + static_cast<std::int64_t>(begin) * k, w_fp16, gate_up, n, k,
                          active, scratch.operands.gemm_workspace.data, stream);
        const std::int64_t quads = static_cast<std::int64_t>(active) * (half / 4);
        swiglu_fp32_rows_kernel<<<static_cast<int>((quads + 255) / 256), 256, 0, stream>>>(
            gate_up, static_cast<__nv_bfloat16*>(out.data) + static_cast<std::int64_t>(begin) * half,
            half, quads);
        CUDA_CHECK(cudaGetLastError());
    }
}

// T=1 down-projection route: prepack GEMV + fp32-to-residual combine. Replaces the QPN2
// single-token pass (~419 GB/s measured in the d8k nsys decode profile: 6.71 ms/token over
// 56 layers) with the shared nvfp4_prepack_gemv_t1 (~590 GB/s in-situ).
__global__ void residual_add_fp32_kernel(const float* __restrict__ projected,
                                          __nv_bfloat16* __restrict__ residual, std::int64_t count) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) { return; }
    residual[i] = __float2bfloat16(__bfloat162float(residual[i]) + projected[i]);
}

void nvfp4_linear_add_gemv_t1_sm70_launch(const Tensor& x, const Weight& w, Tensor& residual,
                                          WorkspaceArena& workspace, cudaStream_t stream) {
    const std::int32_t n = w.n;
    const std::int32_t k = w.k;
    if (w.layout != QuantLayout::VoltaQpnPrepacked) {
        // Fall back to the caller's general route; only the prepack layout is wired here.
        throw std::invalid_argument("nvfp4 linear_add gemv t1 requires prepacked weights");
    }
    auto scope   = workspace.scope();
    float* scratch = static_cast<float*>(
        workspace.alloc_bytes(static_cast<std::size_t>(n) * sizeof(float), 256).data);
    half* x16  = static_cast<half*>(
        workspace.alloc_bytes(static_cast<std::size_t>(k) * sizeof(half), 256).data);
    CUDA_CHECK(cudaMemsetAsync(scratch, 0, static_cast<std::size_t>(n) * sizeof(float), stream));
    bf16_to_fp16_kernel<<<(static_cast<int>(k) + 2047) / 2048, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), reinterpret_cast<cutlass::half_t*>(x16), k);
    nvfp4_prepack_gemv_t1<32><<<dim3(static_cast<unsigned>(n / 32), 32), 32, 0, stream>>>(
        static_cast<const std::uint8_t*>(w.qdata), static_cast<const std::uint8_t*>(w.scales), x16,
        scratch, n, k, 1.0F / w.weight_scale_divisor);
    residual_add_fp32_kernel<<<(static_cast<int>(n) + 255) / 256, 256, 0, stream>>>(
        scratch, static_cast<__nv_bfloat16*>(residual.data), n);
    CUDA_CHECK(cudaGetLastError());
}

std::size_t nvfp4_linear_add_gemv_t1_sm70_workspace_bytes(std::int32_t output_rows,
                                                          std::int32_t input_rows) {
    return static_cast<std::size_t>(output_rows) * sizeof(float) +
           static_cast<std::size_t>(input_rows) * sizeof(half);
}

} // namespace ninfer::ops::detail
