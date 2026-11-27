#include "ops/linear/common/volta_rowsplit_stream_hmma.h"

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/linear/common/volta_rowsplit_stream_hmma.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {

#ifdef NINFER_VOLTA_BUILD

namespace {

template <RowSplitStreamFormat Fmt, typename Activation, typename Epilogue>
void dispatch_splitk_slices(const Weight& w, const Activation* x_ptr, std::int32_t n,
                            std::int32_t k, std::int32_t t, Epilogue epilogue,
                            cudaStream_t stream) {
    const auto* codes  = static_cast<const std::uint8_t*>(w.qdata);
    const auto* high   = static_cast<const std::uint8_t*>(w.qhigh);
    const auto* scales = static_cast<const std::uint16_t*>(w.scales);
    const dim3 grid(static_cast<unsigned>(div_up(n, 8)), 1u, 1u);
    constexpr dim3 block(128u, 1u, 1u);

    for (std::int32_t col = 0; col < t; col += 16) {
        const std::int32_t rem = t - col;
        if (rem <= 8) {
            volta_rowsplit_stream_hmma_splitk_kernel<Fmt, 1, Activation, Epilogue>
                <<<grid, block, 0, stream>>>(codes, high, scales, x_ptr, n, k, t, col, epilogue);
        } else {
            volta_rowsplit_stream_hmma_splitk_kernel<Fmt, 2, Activation, Epilogue>
                <<<grid, block, 0, stream>>>(codes, high, scales, x_ptr, n, k, t, col, epilogue);
        }
    }
    CUDA_CHECK(cudaGetLastError());
}

template <RowSplitStreamFormat Fmt, bool kArgmax, typename Activation>
void dispatch_splitn_slices(const Weight& w, const Activation* x_ptr, __nv_bfloat16* out_ptr,
                            unsigned long long* packed_max, std::int32_t n, std::int32_t valid_rows,
                            std::int32_t k, std::int32_t t, std::int32_t out_ld,
                            cudaStream_t stream) {
    const auto* codes  = static_cast<const std::uint8_t*>(w.qdata);
    const auto* high   = static_cast<const std::uint8_t*>(w.qhigh);
    const auto* scales = static_cast<const std::uint16_t*>(w.scales);
    const dim3 grid(static_cast<unsigned>(div_up(n, 32)), 1u, 1u);
    constexpr dim3 block(128u, 1u, 1u);

    for (std::int32_t col = 0; col < t; col += 16) {
        const std::int32_t rem = t - col;
        if (rem <= 8) {
            volta_rowsplit_stream_hmma_splitn_kernel<Fmt, 1, kArgmax, Activation>
                <<<grid, block, 0, stream>>>(codes, high, scales, x_ptr, out_ptr, packed_max, n,
                                             valid_rows, k, t, col, out_ld);
        } else {
            volta_rowsplit_stream_hmma_splitn_kernel<Fmt, 2, kArgmax, Activation>
                <<<grid, block, 0, stream>>>(codes, high, scales, x_ptr, out_ptr, packed_max, n,
                                             valid_rows, k, t, col, out_ld);
        }
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void launch_q4_volta_stream_hmma(const Tensor& x, const Weight& w, Tensor& out,
                                 cudaStream_t stream) {
    const std::int32_t n      = out.ne[0];
    const std::int32_t k      = x.ne[0];
    const std::int32_t t      = x.ne[1];
    const std::int32_t out_ld = static_cast<std::int32_t>(out.nb[1] / sizeof(__nv_bfloat16));
    if (n >= 65536) {
        dispatch_splitn_slices<RowSplitStreamFormat::Q4, false>(
            w, static_cast<const __nv_bfloat16*>(x.data), static_cast<__nv_bfloat16*>(out.data),
            nullptr, n, n, k, t, out_ld, stream);
        return;
    }
    dispatch_splitk_slices<RowSplitStreamFormat::Q4>(
        w, static_cast<const __nv_bfloat16*>(x.data), n, k, t,
        StreamContiguousEpilogue{static_cast<__nv_bfloat16*>(out.data), out_ld}, stream);
}

void launch_q5_volta_stream_hmma(const Tensor& x, const Weight& w, Tensor& out,
                                 cudaStream_t stream) {
    const std::int32_t n      = out.ne[0];
    const std::int32_t k      = x.ne[0];
    const std::int32_t t      = x.ne[1];
    const std::int32_t out_ld = static_cast<std::int32_t>(out.nb[1] / sizeof(__nv_bfloat16));
    dispatch_splitk_slices<RowSplitStreamFormat::Q5>(
        w, static_cast<const __nv_bfloat16*>(x.data), n, k, t,
        StreamContiguousEpilogue{static_cast<__nv_bfloat16*>(out.data), out_ld}, stream);
}

void launch_q6_volta_stream_hmma(const Tensor& x, const Weight& w, Tensor& out,
                                 cudaStream_t stream) {
    const std::int32_t n      = out.ne[0];
    const std::int32_t k      = x.ne[0];
    const std::int32_t t      = x.ne[1];
    const std::int32_t out_ld = static_cast<std::int32_t>(out.nb[1] / sizeof(__nv_bfloat16));
    if (n >= 32768) {
        dispatch_splitn_slices<RowSplitStreamFormat::Q6, false>(
            w, static_cast<const __nv_bfloat16*>(x.data), static_cast<__nv_bfloat16*>(out.data),
            nullptr, n, n, k, t, out_ld, stream);
        return;
    }
    dispatch_splitk_slices<RowSplitStreamFormat::Q6>(
        w, static_cast<const __nv_bfloat16*>(x.data), n, k, t,
        StreamContiguousEpilogue{static_cast<__nv_bfloat16*>(out.data), out_ld}, stream);
}

void launch_q6_volta_stream_hmma_fp16(const Tensor& x_shape, const Weight& w, const void* x_fp16,
                                      Tensor& out, cudaStream_t stream) {
    const std::int32_t n      = out.ne[0];
    const std::int32_t k      = x_shape.ne[0];
    const std::int32_t t      = x_shape.ne[1];
    const std::int32_t out_ld = static_cast<std::int32_t>(out.nb[1] / sizeof(__nv_bfloat16));
    dispatch_splitn_slices<RowSplitStreamFormat::Q6, false>(
        w, static_cast<const half*>(x_fp16), static_cast<__nv_bfloat16*>(out.data), nullptr, n, n,
        k, t, out_ld, stream);
}

void launch_q6_volta_stream_hmma_argmax_fp16(const Tensor& x_shape, const Weight& w,
                                             const void* x_fp16, Tensor& out_ids,
                                             std::int32_t valid_rows, void* argmax_scratch,
                                             cudaStream_t stream) {
    const std::int32_t n = w.n;
    const std::int32_t k = x_shape.ne[0];
    const std::int32_t t = x_shape.ne[1];
    auto* packed_max     = static_cast<unsigned long long*>(argmax_scratch);
    CUDA_CHECK(cudaMemsetAsync(packed_max, 0,
                               static_cast<std::size_t>(t) * sizeof(unsigned long long), stream));
    dispatch_splitn_slices<RowSplitStreamFormat::Q6, true>(
        w, static_cast<const half*>(x_fp16), nullptr, packed_max, n, std::min(n, valid_rows), k, t,
        0, stream);
    const int blocks = std::max(1, div_up(t, 64));
    stream_argmax_extract_kernel<<<blocks, 64, 0, stream>>>(
        packed_max, static_cast<std::int32_t*>(out_ids.data), t);
    CUDA_CHECK(cudaGetLastError());
}

void launch_q4_volta_stream_hmma_residual(const Tensor& x, const Weight& w, Tensor& residual_out,
                                          cudaStream_t stream) {
    const std::int32_t n      = residual_out.ne[0];
    const std::int32_t k      = x.ne[0];
    const std::int32_t t      = x.ne[1];
    const std::int32_t out_ld = static_cast<std::int32_t>(residual_out.nb[1] / sizeof(__nv_bfloat16));
    dispatch_splitk_slices<RowSplitStreamFormat::Q4>(
        w, static_cast<const __nv_bfloat16*>(x.data), n, k, t,
        StreamResidualAddEpilogue{static_cast<__nv_bfloat16*>(residual_out.data), out_ld}, stream);
}

void launch_q5_volta_stream_hmma_residual(const Tensor& x, const Weight& w, Tensor& residual_out,
                                          cudaStream_t stream) {
    const std::int32_t n      = residual_out.ne[0];
    const std::int32_t k      = x.ne[0];
    const std::int32_t t      = x.ne[1];
    const std::int32_t out_ld = static_cast<std::int32_t>(residual_out.nb[1] / sizeof(__nv_bfloat16));
    dispatch_splitk_slices<RowSplitStreamFormat::Q5>(
        w, static_cast<const __nv_bfloat16*>(x.data), n, k, t,
        StreamResidualAddEpilogue{static_cast<__nv_bfloat16*>(residual_out.data), out_ld}, stream);
}

void launch_q4_volta_stream_hmma_split(const Tensor& x, const Weight& w, Tensor& out0, Tensor& out1,
                                       std::int32_t split_row, cudaStream_t stream) {
    const std::int32_t n       = w.n;
    const std::int32_t k       = x.ne[0];
    const std::int32_t t       = x.ne[1];
    const std::int32_t out0_ld = static_cast<std::int32_t>(out0.nb[1] / sizeof(__nv_bfloat16));
    const std::int32_t out1_ld = static_cast<std::int32_t>(out1.nb[1] / sizeof(__nv_bfloat16));
    dispatch_splitk_slices<RowSplitStreamFormat::Q4>(
        w, static_cast<const __nv_bfloat16*>(x.data), n, k, t,
        StreamSplitRowEpilogue{static_cast<__nv_bfloat16*>(out0.data),
                               static_cast<__nv_bfloat16*>(out1.data), out0_ld, out1_ld,
                               split_row},
        stream);
}

void launch_q5_volta_stream_hmma_split(const Tensor& x, const Weight& w, Tensor& out0, Tensor& out1,
                                       std::int32_t split_row, cudaStream_t stream) {
    const std::int32_t n       = w.n;
    const std::int32_t k       = x.ne[0];
    const std::int32_t t       = x.ne[1];
    const std::int32_t out0_ld = static_cast<std::int32_t>(out0.nb[1] / sizeof(__nv_bfloat16));
    const std::int32_t out1_ld = static_cast<std::int32_t>(out1.nb[1] / sizeof(__nv_bfloat16));
    dispatch_splitk_slices<RowSplitStreamFormat::Q5>(
        w, static_cast<const __nv_bfloat16*>(x.data), n, k, t,
        StreamSplitRowEpilogue{static_cast<__nv_bfloat16*>(out0.data),
                               static_cast<__nv_bfloat16*>(out1.data), out0_ld, out1_ld,
                               split_row},
        stream);
}

void launch_q4_volta_stream_hmma_swiglu(const Tensor& x, const Weight& w, Tensor& out,
                                        cudaStream_t stream) {
    const std::int32_t half_n = out.ne[0];
    const std::int32_t k      = x.ne[0];
    const std::int32_t t      = x.ne[1];
    const std::int32_t out_ld = static_cast<std::int32_t>(out.nb[1] / sizeof(__nv_bfloat16));
    if (t <= 0 || t > kVoltaStreamHmmaMaxTokens) {
        throw std::invalid_argument("launch_q4_volta_stream_hmma_swiglu: T must be in [1,16]");
    }
    const auto* codes  = static_cast<const std::uint8_t*>(w.qdata);
    const auto* scales = static_cast<const std::uint16_t*>(w.scales);
    const dim3 grid(static_cast<unsigned>(div_up(half_n, 32)), 1u, 1u);
    constexpr dim3 block(128u, 1u, 1u);
    if (t <= 8) {
        volta_q4_stream_hmma_swiglu_kernel<1, __nv_bfloat16><<<grid, block, 0, stream>>>(
            codes, scales, static_cast<const __nv_bfloat16*>(x.data),
            static_cast<__nv_bfloat16*>(out.data), half_n, k, t, out_ld);
    } else {
        volta_q4_stream_hmma_swiglu_kernel<2, __nv_bfloat16><<<grid, block, 0, stream>>>(
            codes, scales, static_cast<const __nv_bfloat16*>(x.data),
            static_cast<__nv_bfloat16*>(out.data), half_n, k, t, out_ld);
    }
    CUDA_CHECK(cudaGetLastError());
}

#endif // NINFER_VOLTA_BUILD

} // namespace ninfer::ops::detail
