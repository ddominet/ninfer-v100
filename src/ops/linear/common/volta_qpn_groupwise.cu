#include "ops/linear/common/volta_qpn_groupwise.h"

#ifdef NINFER_VOLTA_BUILD

#include "core/device.h"
#include "ops/linear/common/volta_qpn_groupwise.cuh"
#include "ops/linear/volta_fp16_stage.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

__global__ void groupwise_argmax_unpack_kernel(const unsigned long long* __restrict__ packed,
                                               std::int32_t* __restrict__ out_ids, int t) {
    const int i = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) +
                  static_cast<int>(threadIdx.x);
    if (i < t) {
        const unsigned long long v = packed[i];
        out_ids[i] = v == 0ULL ? 0 : static_cast<std::int32_t>(~static_cast<std::uint32_t>(v));
    }
}

inline std::int32_t bf16_ld(const Tensor& t) {
    return static_cast<std::int32_t>(t.nb[1] / sizeof(__nv_bfloat16));
}

template <GroupwiseQpnFormat Fmt, int kTiles, typename Activation, typename OutputPolicy>
void dispatch_qpn_tiles(const Weight& w, const Activation* x, int t, OutputPolicy output,
                        cudaStream_t stream) {
    const int n        = w.n;
    const int k        = w.k;
    const int groups   = k / 64;
    const dim3 grid(static_cast<unsigned>(n / 32));
    const auto* codes  = static_cast<const uint4*>(w.qdata);
    const auto* scales = static_cast<const std::uint16_t*>(w.scales);

    if constexpr (kTiles == 1) {
        if (n <= 5120 && groups >= 128) {
            volta_qpn_groupwise_kernel<Fmt, 1, 16, Activation, OutputPolicy>
                <<<grid, 16 * 32, 0, stream>>>(codes, w.qhigh, scales, x, n, k, t, output);
        } else if (n <= 8192 && groups >= 16) {
            volta_qpn_groupwise_kernel<Fmt, 1, 8, Activation, OutputPolicy>
                <<<grid, 8 * 32, 0, stream>>>(codes, w.qhigh, scales, x, n, k, t, output);
        } else {
            volta_qpn_groupwise_kernel<Fmt, 1, 4, Activation, OutputPolicy>
                <<<grid, 4 * 32, 0, stream>>>(codes, w.qhigh, scales, x, n, k, t, output);
        }
    } else if constexpr (kTiles <= 4) {
        if (n <= 8192 && groups >= 16) {
            volta_qpn_groupwise_kernel<Fmt, kTiles, 8, Activation, OutputPolicy>
                <<<grid, 8 * 32, 0, stream>>>(codes, w.qhigh, scales, x, n, k, t, output);
        } else {
            volta_qpn_groupwise_kernel<Fmt, kTiles, 4, Activation, OutputPolicy>
                <<<grid, 4 * 32, 0, stream>>>(codes, w.qhigh, scales, x, n, k, t, output);
        }
    } else {
        volta_qpn_groupwise_kernel<Fmt, kTiles, 4, Activation, OutputPolicy>
            <<<grid, 4 * 32, 0, stream>>>(codes, w.qhigh, scales, x, n, k, t, output);
    }
    CUDA_CHECK(cudaGetLastError());
}

template <GroupwiseQpnFormat Fmt, typename Activation, typename OutputPolicy>
void dispatch_qpn_typed(const Weight& w, const Activation* x, int t, OutputPolicy output,
                        cudaStream_t stream) {
    if (t <= 8) {
        dispatch_qpn_tiles<Fmt, 1>(w, x, t, output, stream);
    } else if (t <= 16) {
        dispatch_qpn_tiles<Fmt, 2>(w, x, t, output, stream);
    } else if (t <= 24) {
        dispatch_qpn_tiles<Fmt, 3>(w, x, t, output, stream);
    } else if (t <= 32) {
        dispatch_qpn_tiles<Fmt, 4>(w, x, t, output, stream);
    } else if (t <= 48) {
        dispatch_qpn_tiles<Fmt, 6>(w, x, t, output, stream);
    } else {
        dispatch_qpn_tiles<Fmt, 8>(w, x, t, output, stream);
    }
}

template <GroupwiseQpnFormat Fmt, typename OutputPolicy>
void dispatch_from_fp16(const Weight& w, const half* x_fp16, int t, OutputPolicy output,
                        cudaStream_t stream) {
    if (t > 8) {
        if (const VoltaGroupMajorHalf* x_gm =
                volta_stage_fp16_to_group_major(x_fp16, w.k, t, stream)) {
            dispatch_qpn_typed<Fmt>(w, x_gm, t, output, stream);
            return;
        }
    }
    dispatch_qpn_typed<Fmt>(w, x_fp16, t, output, stream);
}

template <GroupwiseQpnFormat Fmt, typename OutputPolicy>
void dispatch_from_bf16(const Tensor& x, const Weight& w, OutputPolicy output,
                        cudaStream_t stream) {
    const int t = x.ne[1];
    if (const VoltaGroupMajorHalf* x_gm = volta_stage_fp16_group_major(x, stream)) {
        dispatch_qpn_typed<Fmt>(w, x_gm, t, output, stream);
        return;
    }
    if (const half* x_fp16 = volta_stage_fp16_activation(x, stream)) {
        dispatch_qpn_typed<Fmt>(w, x_fp16, t, output, stream);
        return;
    }
    dispatch_qpn_typed<Fmt>(w, static_cast<const __nv_bfloat16*>(x.data), t, output, stream);
}

template <GroupwiseQpnFormat Fmt>
void launch_prepacked_contiguous_any_t(const Tensor& x, const Weight& w, Tensor& out,
                                       cudaStream_t stream) {
    const int k       = x.ne[0];
    const int total   = x.ne[1];
    const int out_ld  = bf16_ld(out);
    const auto* x_all = static_cast<const __nv_bfloat16*>(x.data);
    auto* out_all     = static_cast<__nv_bfloat16*>(out.data);
    for (int first = 0; first < total; first += 64) {
        const int count = std::min(64, total - first);
        Tensor x_chunk(const_cast<__nv_bfloat16*>(x_all + static_cast<std::int64_t>(first) * k),
                       DType::BF16, {k, count});
        dispatch_from_bf16<Fmt>(
            x_chunk, w,
            GroupwiseContiguousOutput{out_all + static_cast<std::int64_t>(first) * out_ld, out_ld},
            stream);
    }
}

Weight make_q4_half_view(const Weight& w, std::int32_t row_begin, std::int32_t half_n) {
    const std::int64_t groups = w.k / 64;
    Weight out                = w;
    out.n                     = half_n;
    out.shape[0]              = half_n;
    out.padded_shape[0]       = half_n;
    out.qdata                 = static_cast<const std::uint8_t*>(w.qdata) +
                static_cast<std::int64_t>(row_begin) * groups * 32;
    out.scales = static_cast<const std::uint8_t*>(w.scales) +
                 static_cast<std::int64_t>(row_begin) * groups * 2;
    return out;
}

} // namespace

void launch_q4_prepacked_volta(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_prepacked_contiguous_any_t<GroupwiseQpnFormat::Q4>(x, w, out, stream);
}

void launch_q5_prepacked_volta(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_prepacked_contiguous_any_t<GroupwiseQpnFormat::Q5>(x, w, out, stream);
}

void launch_q6_prepacked_volta(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_prepacked_contiguous_any_t<GroupwiseQpnFormat::Q6>(x, w, out, stream);
}

void launch_q4_prepacked_volta_fp16(const Tensor& x_shape, const half* x_fp16, const Weight& w,
                                    Tensor& out, cudaStream_t stream) {
    dispatch_from_fp16<GroupwiseQpnFormat::Q4>(
        w, x_fp16, x_shape.ne[1],
        GroupwiseContiguousOutput{static_cast<__nv_bfloat16*>(out.data), bf16_ld(out)}, stream);
}

void launch_q6_prepacked_volta_fp16(const Tensor& x_shape, const half* x_fp16, const Weight& w,
                                    Tensor& out, cudaStream_t stream) {
    dispatch_from_fp16<GroupwiseQpnFormat::Q6>(
        w, x_fp16, x_shape.ne[1],
        GroupwiseContiguousOutput{static_cast<__nv_bfloat16*>(out.data), bf16_ld(out)}, stream);
}

void launch_q4_prepacked_volta_residual(const Tensor& x, const Weight& w, Tensor& residual_out,
                                        cudaStream_t stream) {
    dispatch_from_bf16<GroupwiseQpnFormat::Q4>(
        x, w,
        GroupwiseResidualOutput{static_cast<__nv_bfloat16*>(residual_out.data),
                                bf16_ld(residual_out)},
        stream);
}

void launch_q5_prepacked_volta_residual(const Tensor& x, const Weight& w, Tensor& residual_out,
                                        cudaStream_t stream) {
    dispatch_from_bf16<GroupwiseQpnFormat::Q5>(
        x, w,
        GroupwiseResidualOutput{static_cast<__nv_bfloat16*>(residual_out.data),
                                bf16_ld(residual_out)},
        stream);
}

void launch_q4_prepacked_volta_residual_fp16(const Tensor& x_shape, const half* x_fp16,
                                             const Weight& w, Tensor& residual_out,
                                             cudaStream_t stream) {
    dispatch_from_fp16<GroupwiseQpnFormat::Q4>(
        w, x_fp16, x_shape.ne[1],
        GroupwiseResidualOutput{static_cast<__nv_bfloat16*>(residual_out.data),
                                bf16_ld(residual_out)},
        stream);
}

void launch_q5_prepacked_volta_residual_fp16(const Tensor& x_shape, const half* x_fp16,
                                             const Weight& w, Tensor& residual_out,
                                             cudaStream_t stream) {
    dispatch_from_fp16<GroupwiseQpnFormat::Q5>(
        w, x_fp16, x_shape.ne[1],
        GroupwiseResidualOutput{static_cast<__nv_bfloat16*>(residual_out.data),
                                bf16_ld(residual_out)},
        stream);
}

void launch_q4_prepacked_volta_residual_gm(const Tensor& x_shape, const VoltaGroupMajorHalf* x_gm,
                                           const Weight& w, Tensor& residual_out,
                                           cudaStream_t stream) {
    dispatch_qpn_typed<GroupwiseQpnFormat::Q4>(
        w, x_gm, x_shape.ne[1],
        GroupwiseResidualOutput{static_cast<__nv_bfloat16*>(residual_out.data),
                                bf16_ld(residual_out)},
        stream);
}

void launch_q5_prepacked_volta_residual_gm(const Tensor& x_shape, const VoltaGroupMajorHalf* x_gm,
                                           const Weight& w, Tensor& residual_out,
                                           cudaStream_t stream) {
    dispatch_qpn_typed<GroupwiseQpnFormat::Q5>(
        w, x_gm, x_shape.ne[1],
        GroupwiseResidualOutput{static_cast<__nv_bfloat16*>(residual_out.data),
                                bf16_ld(residual_out)},
        stream);
}

void launch_q4_prepacked_volta_split(const Tensor& x, const Weight& w, Tensor& out0, Tensor& out1,
                                     std::int32_t split_row, cudaStream_t stream) {
    dispatch_from_bf16<GroupwiseQpnFormat::Q4>(
        x, w,
        GroupwiseSplitRowOutput{static_cast<__nv_bfloat16*>(out0.data),
                                static_cast<__nv_bfloat16*>(out1.data), bf16_ld(out0),
                                bf16_ld(out1), split_row},
        stream);
}

void launch_q5_prepacked_volta_split(const Tensor& x, const Weight& w, Tensor& out0, Tensor& out1,
                                     std::int32_t split_row, cudaStream_t stream) {
    dispatch_from_bf16<GroupwiseQpnFormat::Q5>(
        x, w,
        GroupwiseSplitRowOutput{static_cast<__nv_bfloat16*>(out0.data),
                                static_cast<__nv_bfloat16*>(out1.data), bf16_ld(out0),
                                bf16_ld(out1), split_row},
        stream);
}

void launch_q4_prepacked_volta_split_fp16(const Tensor& x_shape, const half* x_fp16,
                                          const Weight& w, Tensor& out0, Tensor& out1,
                                          std::int32_t split_row, cudaStream_t stream) {
    dispatch_from_fp16<GroupwiseQpnFormat::Q4>(
        w, x_fp16, x_shape.ne[1],
        GroupwiseSplitRowOutput{static_cast<__nv_bfloat16*>(out0.data),
                                static_cast<__nv_bfloat16*>(out1.data), bf16_ld(out0),
                                bf16_ld(out1), split_row},
        stream);
}

void launch_q5_prepacked_volta_split_fp16(const Tensor& x_shape, const half* x_fp16,
                                          const Weight& w, Tensor& out0, Tensor& out1,
                                          std::int32_t split_row, cudaStream_t stream) {
    dispatch_from_fp16<GroupwiseQpnFormat::Q5>(
        w, x_fp16, x_shape.ne[1],
        GroupwiseSplitRowOutput{static_cast<__nv_bfloat16*>(out0.data),
                                static_cast<__nv_bfloat16*>(out1.data), bf16_ld(out0),
                                bf16_ld(out1), split_row},
        stream);
}

void launch_q4_prepacked_volta_split_gm(const Tensor& x_shape, const VoltaGroupMajorHalf* x_gm,
                                        const Weight& w, Tensor& out0, Tensor& out1,
                                        std::int32_t split_row, cudaStream_t stream) {
    dispatch_qpn_typed<GroupwiseQpnFormat::Q4>(
        w, x_gm, x_shape.ne[1],
        GroupwiseSplitRowOutput{static_cast<__nv_bfloat16*>(out0.data),
                                static_cast<__nv_bfloat16*>(out1.data), bf16_ld(out0),
                                bf16_ld(out1), split_row},
        stream);
}

void launch_q5_prepacked_volta_split_gm(const Tensor& x_shape, const VoltaGroupMajorHalf* x_gm,
                                        const Weight& w, Tensor& out0, Tensor& out1,
                                        std::int32_t split_row, cudaStream_t stream) {
    dispatch_qpn_typed<GroupwiseQpnFormat::Q5>(
        w, x_gm, x_shape.ne[1],
        GroupwiseSplitRowOutput{static_cast<__nv_bfloat16*>(out0.data),
                                static_cast<__nv_bfloat16*>(out1.data), bf16_ld(out0),
                                bf16_ld(out1), split_row},
        stream);
}

void launch_q4_prepacked_volta_scatter_fp16(const Tensor& x_shape, const half* x_fp16,
                                            const Weight& w, Tensor& out_padded,
                                            const std::int32_t* col_map, cudaStream_t stream) {
    dispatch_from_fp16<GroupwiseQpnFormat::Q4>(
        w, x_fp16, x_shape.ne[1],
        GroupwiseScatterColOutput{static_cast<__nv_bfloat16*>(out_padded.data), col_map,
                                  bf16_ld(out_padded)},
        stream);
}

void launch_q5_prepacked_volta_split_scatter_fp16(const Tensor& x_shape, const half* x_fp16,
                                                  const Weight& w, Tensor& out0_padded,
                                                  Tensor& out1_dense, std::int32_t split_row,
                                                  const std::int32_t* out0_col_map,
                                                  cudaStream_t stream) {
    dispatch_from_fp16<GroupwiseQpnFormat::Q5>(
        w, x_fp16, x_shape.ne[1],
        GroupwiseSplitScatterOutput{static_cast<__nv_bfloat16*>(out0_padded.data),
                                    static_cast<__nv_bfloat16*>(out1_dense.data), out0_col_map,
                                    bf16_ld(out0_padded), bf16_ld(out1_dense), split_row},
        stream);
}

void launch_q4_prepacked_volta_swiglu(const Tensor& x, const Weight& w, Tensor& out,
                                      cudaStream_t stream) {
    const std::int32_t half_n = out.ne[0];
    const std::int32_t out_ld = bf16_ld(out);
    const std::int32_t t      = x.ne[1];
    const Weight gate_w       = make_q4_half_view(w, 0, half_n);
    const Weight up_w         = make_q4_half_view(w, half_n, half_n);
    auto* out_ptr             = static_cast<__nv_bfloat16*>(out.data);
    if (const VoltaGroupMajorHalf* x_gm = volta_stage_fp16_group_major(x, stream)) {
        dispatch_qpn_typed<GroupwiseQpnFormat::Q4>(
            gate_w, x_gm, t, GroupwiseContiguousOutput{out_ptr, out_ld}, stream);
        dispatch_qpn_typed<GroupwiseQpnFormat::Q4>(
            up_w, x_gm, t, GroupwiseSwiGluInPlaceBf16Output{out_ptr, out_ld}, stream);
    } else if (const half* x_fp16 = volta_stage_fp16_activation(x, stream)) {
        dispatch_qpn_typed<GroupwiseQpnFormat::Q4>(
            gate_w, x_fp16, t, GroupwiseContiguousOutput{out_ptr, out_ld}, stream);
        dispatch_qpn_typed<GroupwiseQpnFormat::Q4>(
            up_w, x_fp16, t, GroupwiseSwiGluInPlaceBf16Output{out_ptr, out_ld}, stream);
    } else {
        const auto* x_bf16 = static_cast<const __nv_bfloat16*>(x.data);
        dispatch_qpn_typed<GroupwiseQpnFormat::Q4>(
            gate_w, x_bf16, t, GroupwiseContiguousOutput{out_ptr, out_ld}, stream);
        dispatch_qpn_typed<GroupwiseQpnFormat::Q4>(
            up_w, x_bf16, t, GroupwiseSwiGluInPlaceBf16Output{out_ptr, out_ld}, stream);
    }
}

void launch_q4_prepacked_volta_swiglu_gm_io(const Tensor& x_shape, const VoltaGroupMajorHalf* x_gm,
                                            const Weight& w, VoltaGroupMajorHalf* out_gm,
                                            float* gate_scratch, cudaStream_t stream) {
    const std::int32_t half_n = w.n / 2;
    const std::int32_t t      = x_shape.ne[1];
    const int tiles           = volta_group_major_tiles(t);
    const Weight gate_w       = make_q4_half_view(w, 0, half_n);
    const Weight up_w         = make_q4_half_view(w, half_n, half_n);
    dispatch_qpn_typed<GroupwiseQpnFormat::Q4>(
        gate_w, x_gm, t, Fp8Fp32ContiguousOutput{gate_scratch, half_n}, stream);
    dispatch_qpn_typed<GroupwiseQpnFormat::Q4>(
        up_w, x_gm, t, Nvfp4QpnSwiGluGroupMajorOutput{gate_scratch, out_gm, half_n, tiles}, stream);
}

void launch_q6_prepacked_volta_argmax_fp16(const Tensor& x_shape, const half* x_fp16,
                                           const Weight& w, Tensor& out_ids,
                                           std::int32_t valid_rows, void* argmax_scratch,
                                           cudaStream_t stream) {
    const std::int32_t t = x_shape.ne[1];
    auto* packed_max     = static_cast<unsigned long long*>(argmax_scratch);
    CUDA_CHECK(cudaMemsetAsync(packed_max, 0, static_cast<std::size_t>(t) * sizeof(unsigned long long),
                               stream));
    dispatch_from_fp16<GroupwiseQpnFormat::Q6>(
        w, x_fp16, t, Fp8ArgmaxOutput{packed_max, std::min(valid_rows, w.n)}, stream);
    groupwise_argmax_unpack_kernel<<<1, static_cast<unsigned>((t + 31) & ~31), 0, stream>>>(
        packed_max, static_cast<std::int32_t*>(out_ids.data), t);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail

#endif // NINFER_VOLTA_BUILD
