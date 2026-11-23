#include "core/device.h"
#include "ops/linear/w8/w8_launch.h"
#include "ops/linear/w8/w8_volta_qpn_gemm.cuh"
#include "ops/linear/volta_fp16_stage.h"

#include <cuda_bf16.h>

namespace ninfer::ops::detail {

#ifdef NINFER_VOLTA_BUILD

namespace {
using S = W8VoltaQpnSchedule;

// Per-shape launch settings: warps per CTA splitting K, and whether the activation is staged as
// FP16 first. Measured at eight tokens on a V100 (us per call, best of split 4/8/16 x BF16/FP16):
//
//   17408 x 5120  (draft MLP gate, up)       split 8,  BF16   221  (split 4 BF16: 234)
//    5120 x 17408 (draft MLP down)           split 8,  BF16   279  (split 4 BF16: 320)
//    6144 x 5120  (draft q|k|v)              split 8,  BF16    89  (warp-per-row SIMT: 194)
//    5120 x 4096  (draft attention output)   split 8,  FP16    42  (split 4 BF16: 69)
//    5120 x 25600 (draft feature projection) split 16, FP16   399  (split 8 FP16: 551)
//
// Staging only wins where many warps share an output row; on the wide-output shapes the staged
// FP16 read measured 1.1-1.8x slower. Shapes not listed keep the shipped split 4 on BF16.
struct W8QpnLaunch {
    int splitk;
    bool stage_fp16;
};

W8QpnLaunch w8_qpn_launch_settings(std::int32_t n, std::int32_t k) {
    if (n == 17408 && k == 5120) { return {8, false}; }
    if (n == 5120 && k == 17408) { return {8, false}; }
    if (n == 6144 && k == 5120) { return {8, false}; }
    if (n == 5120 && k == 4096) { return {8, true}; }
    if (n == 5120 && k == 25600) { return {16, true}; }
    return {4, false};
}

// The activation for every launch of one product: the FP16 copy when the shape stages and
// staging is enabled, the caller's BF16 rows otherwise.
struct W8QpnActivation {
    const half* fp16;
    const __nv_bfloat16* bf16;
};

W8QpnActivation w8_qpn_activation(const Tensor& x, const W8QpnLaunch& launch,
                                  cudaStream_t stream) {
    const half* staged = launch.stage_fp16 ? volta_stage_fp16_activation(x, stream) : nullptr;
    return {staged, static_cast<const __nv_bfloat16*>(x.data)};
}

template <W8QpnEpilogue E, int SPLITK, class Activation>
void w8_qpn_launch_typed(const dim3& grid, const Activation* x, const std::uint8_t* codes,
                         const std::uint8_t* scales, __nv_bfloat16* out, int n, int k, int t,
                         int padded_groups, int out_ld, const __nv_bfloat16* finish_delta,
                         const __nv_bfloat16* base, __nv_bfloat16* residual, int width,
                         __nv_bfloat16* out1, __nv_bfloat16* out2, cudaStream_t stream) {
    w8_volta_qpn_gemm_kernel<1, 4, SPLITK, Activation, E><<<grid, SPLITK * 32, 0, stream>>>(
        codes, scales, x, out, n, k, t, padded_groups, out_ld, finish_delta, base, residual, width,
        out1, out2);
}

template <W8QpnEpilogue E>
void w8_qpn_launch(const W8QpnActivation& a, int splitk, const std::uint8_t* codes,
                   const std::uint8_t* scales, __nv_bfloat16* out, int n, int k, int t,
                   int padded_groups, int out_ld, cudaStream_t stream,
                   const __nv_bfloat16* finish_delta = nullptr,
                   const __nv_bfloat16* base = nullptr, __nv_bfloat16* residual = nullptr,
                   int width = 0, __nv_bfloat16* out1 = nullptr, __nv_bfloat16* out2 = nullptr) {
    const dim3 grid(static_cast<unsigned>((n + S::kColsPerCta - 1) / S::kColsPerCta));
#define NINFER_W8_QPN_LAUNCH(SPLITK)                                                             \
    do {                                                                                         \
        if (a.fp16 != nullptr) {                                                                 \
            w8_qpn_launch_typed<E, SPLITK>(grid, a.fp16, codes, scales, out, n, k, t,            \
                                           padded_groups, out_ld, finish_delta, base, residual,  \
                                           width, out1, out2, stream);                           \
        } else {                                                                                 \
            w8_qpn_launch_typed<E, SPLITK>(grid, a.bf16, codes, scales, out, n, k, t,            \
                                           padded_groups, out_ld, finish_delta, base, residual,  \
                                           width, out1, out2, stream);                           \
        }                                                                                        \
    } while (false)
    if (splitk == 16) {
        NINFER_W8_QPN_LAUNCH(16);
    } else if (splitk == 8) {
        NINFER_W8_QPN_LAUNCH(8);
    } else {
        NINFER_W8_QPN_LAUNCH(4);
    }
#undef NINFER_W8_QPN_LAUNCH
    CUDA_CHECK(cudaGetLastError());
}

std::int32_t w8_qpn_padded_groups(const Weight& w) {
    return w.padded_shape[1] / W8RowSplitStorage::kGroupK;
}

} // namespace

bool w8_volta_qpn_supported(std::int32_t n, std::int32_t k, std::int32_t t) noexcept {
    if (n <= 0 || t <= 0) { return false; }
    // Whole W8 groups per lane (the scale is per group and held across it), and one uint4 of
    // activations per mma pair.
    if (k % W8RowSplitStorage::kGroupK != 0 || k % 8 != 0) { return false; }
    // At least one 128-byte block of four groups; a split larger than the block count leaves the
    // leading warps idle and the last warp with every block, which stays correct.
    if (k / W8RowSplitStorage::kGroupK < 4) { return false; }
    // One 8-row A tile only. A two-tile form covering T=9..16 was measured and loses: it doubles
    // the MMA work for a second tile the shape does not need, reading 5115-5658us against the
    // 32x8 fused route's 5390-5502 over that span. The 32x8 geometry absorbs those rows for free,
    // which is exactly the crossover the Q4 sibling shows.
    return t <= S::kRowsPerTile;
}

void launch_w8_volta_qpn(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    const std::int32_t n      = out.ne[0];
    const std::int32_t k      = x.ne[0];
    const std::int32_t t      = x.ne[1];
    const std::int32_t out_ld = static_cast<std::int32_t>(out.nb[1] / sizeof(__nv_bfloat16));
    const W8QpnLaunch launch  = w8_qpn_launch_settings(n, k);
    w8_qpn_launch<W8QpnEpilogue::Store>(
        w8_qpn_activation(x, launch, stream), launch.splitk,
        static_cast<const std::uint8_t*>(w.qdata),
        static_cast<const std::uint8_t*>(w.scales), static_cast<__nv_bfloat16*>(out.data), n, k,
        t, w8_qpn_padded_groups(w), out_ld, stream);
}

void launch_w8_volta_qpn_split3(const Tensor& x, const Weight& w, Tensor& q, Tensor& k,
                                Tensor& v, cudaStream_t stream) {
    constexpr std::int32_t kParentRows = 6144;
    const W8QpnLaunch launch = w8_qpn_launch_settings(kParentRows, x.ne[0]);
    w8_qpn_launch<W8QpnEpilogue::Split3>(
        w8_qpn_activation(x, launch, stream), launch.splitk,
        static_cast<const std::uint8_t*>(w.qdata),
        static_cast<const std::uint8_t*>(w.scales), static_cast<__nv_bfloat16*>(q.data),
        kParentRows, x.ne[0], x.ne[1], w8_qpn_padded_groups(w), 0, stream, nullptr, nullptr,
        nullptr, 0, static_cast<__nv_bfloat16*>(k.data), static_cast<__nv_bfloat16*>(v.data));
}

void launch_w8_volta_qpn_dynamic_conv_add(const Tensor& x, const Weight& w, const Tensor& base,
                                          const Tensor& finish_delta, Tensor& residual,
                                          std::int32_t width, cudaStream_t stream) {
    const W8QpnLaunch launch = w8_qpn_launch_settings(residual.ne[0], x.ne[0]);
    w8_qpn_launch<W8QpnEpilogue::DynamicConvAdd>(
        w8_qpn_activation(x, launch, stream), launch.splitk,
        static_cast<const std::uint8_t*>(w.qdata),
        static_cast<const std::uint8_t*>(w.scales), nullptr, residual.ne[0], x.ne[0], x.ne[1],
        w8_qpn_padded_groups(w), 0, stream,
        static_cast<const __nv_bfloat16*>(finish_delta.data),
        static_cast<const __nv_bfloat16*>(base.data),
        static_cast<__nv_bfloat16*>(residual.data), width);
}

void w8_linear_swiglu_volta_qpn_split_launch(const Tensor& x, const Weight& w, Tensor& out,
                                              cudaStream_t stream) {
    constexpr std::int32_t kIntermediate = 17408;
    const std::int32_t k = x.ne[0];
    const std::int32_t t = x.ne[1];
    const std::int32_t padded_groups = w8_qpn_padded_groups(w);
    const std::int64_t row_bytes = static_cast<std::int64_t>(padded_groups) *
                                   W8RowSplitStorage::kCodeBytesPerGroup;
    const std::int64_t row_scale_bytes = static_cast<std::int64_t>(padded_groups) *
                                         W8RowSplitStorage::kScaleBytesPerGroup;
    const auto* gate_codes = static_cast<const std::uint8_t*>(w.qdata);
    const auto* gate_scales = static_cast<const std::uint8_t*>(w.scales);
    const auto* up_codes = gate_codes + static_cast<std::int64_t>(kIntermediate) * row_bytes;
    const auto* up_scales = gate_scales + static_cast<std::int64_t>(kIntermediate) * row_scale_bytes;
    auto* output = static_cast<__nv_bfloat16*>(out.data);

    // One activation for both halves: the gate pass writes its projection, the up pass reads it
    // back in its epilogue and writes silu(gate) * up in place.
    const W8QpnLaunch launch = w8_qpn_launch_settings(kIntermediate, k);
    const W8QpnActivation a  = w8_qpn_activation(x, launch, stream);
    w8_qpn_launch<W8QpnEpilogue::Store>(a, launch.splitk, gate_codes, gate_scales, output,
                                        kIntermediate, k, t, padded_groups, kIntermediate, stream);
    w8_qpn_launch<W8QpnEpilogue::SwiGluUp>(a, launch.splitk, up_codes, up_scales, output,
                                           kIntermediate, k, t, padded_groups, kIntermediate,
                                           stream);
}

#endif // NINFER_VOLTA_BUILD

} // namespace ninfer::ops::detail
