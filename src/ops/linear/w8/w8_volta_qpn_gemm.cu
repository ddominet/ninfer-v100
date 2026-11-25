#include "core/device.h"
#include "ops/dynamic_grouped_conv/volta/dflash2_dynamic_conv_volta.h"
#include "ops/linear/w8/w8_launch.h"
#include "ops/linear/w8/w8_volta_qpn_gemm.cuh"
#include "ops/linear/volta_fp16_stage.h"

#include <algorithm>
#include <stdexcept>

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

// Row-major weights at 9-64 tokens: one launch reads the weight once for up to four 8-row tiles
// (32 tokens), so a batch of draft requests no longer falls back to the SIMT or general
// tensor-core kernels. Measured on the DFlash2 draft shapes against 8-token chunks of the one-tile
// kernel (identical outputs at the same split): q|k|v 6144 x 5120 at 64 tokens 285 us (chunks:
// 870 us; the SIMT route it replaces: 932 us), feature projection 5120 x 25600 at 24 tokens
// 491-532 us (the general tensor-core route: 845 us).
template <W8QpnEpilogue E, int TILES, int SPLITK, class Activation>
void w8_qpn_launch_rm(const Activation* x, const std::uint8_t* codes, const std::uint8_t* scales,
                      __nv_bfloat16* out, int n, int k, int t, int padded_groups, int out_ld,
                      __nv_bfloat16* out1, __nv_bfloat16* out2, cudaStream_t stream) {
    const dim3 grid(static_cast<unsigned>((n + S::kColsPerCta - 1) / S::kColsPerCta));
    w8_volta_qpn_gemm_kernel<TILES, 4, SPLITK, Activation, E><<<grid, SPLITK * 32, 0, stream>>>(
        codes, scales, x, out, n, k, t, padded_groups, out_ld, nullptr, nullptr, nullptr, 0, out1,
        out2);
}

// One chunk of at most 32 tokens at a fixed split; the tile count follows the chunk.
template <W8QpnEpilogue E, int SPLITK, class Activation>
void w8_qpn_launch_rm_chunk(const Activation* x, const std::uint8_t* codes,
                            const std::uint8_t* scales, __nv_bfloat16* out, int n, int k, int t,
                            int padded_groups, int out_ld, __nv_bfloat16* out1,
                            __nv_bfloat16* out2, cudaStream_t stream) {
    if (t <= 8) {
        w8_qpn_launch_rm<E, 1, SPLITK>(x, codes, scales, out, n, k, t, padded_groups, out_ld,
                                       out1, out2, stream);
    } else if (t <= 16) {
        w8_qpn_launch_rm<E, 2, SPLITK>(x, codes, scales, out, n, k, t, padded_groups, out_ld,
                                       out1, out2, stream);
    } else {
        w8_qpn_launch_rm<E, 4, SPLITK>(x, codes, scales, out, n, k, t, padded_groups, out_ld,
                                       out1, out2, stream);
    }
}

std::int32_t w8_qpn_padded_groups(const Weight& w) {
    return w.padded_shape[1] / W8RowSplitStorage::kGroupK;
}

// Prepacked weights (w8_prepack_sm70.cu): one kernel per 64 tokens, with 1, 2, 4 or 8 row tiles,
// so a batch of requests reads the weight once. The activation is staged as FP16 whenever staging
// is available; on the prepacked layout that measured 1.4-2.3x faster at eight tokens (gate/up
// 17408 x 5120: 175 -> 121 us, down 5120 x 17408: 177 -> 123 us), close to the NVFP4 kernels'
// 750+ GB/s. Eight K-split warps for one or two tiles, four beyond, which keeps the reduce buffer
// inside 32 KB.
template <W8QpnEpilogue E, int TILES, int SPLITK, class Activation>
void w8_qpn_launch_packed_typed(const dim3& grid, const Activation* x, const std::uint8_t* codes,
                                const std::uint8_t* scales, __nv_bfloat16* out, int n, int k,
                                int t, int out_ld, const __nv_bfloat16* finish_delta,
                                const __nv_bfloat16* base, __nv_bfloat16* residual, int width,
                                cudaStream_t stream) {
    w8_volta_qpn_gemm_kernel<TILES, 4, SPLITK, Activation, E, true>
        <<<grid, SPLITK * 32, 0, stream>>>(codes, scales, x, out, n, k, t, k / 32, out_ld,
                                           finish_delta, base, residual, width);
}

// One launch reads at most 64 token rows (eight row tiles). Larger batches, such as eight
// DFlash2 requests of nine rows each, run as consecutive launches over row chunks. Each chunk
// re-reads the weight, so it costs bandwidth but stays exact. The DynamicConvAdd epilogue reads
// a row's predecessor from the same CTA, so its chunks hold whole requests: a multiple of width.
constexpr int kPackedMaxRows = 64;

// Store and SwiGLU-up epilogues of the quadpair kernel, for the tensor-core route.
template <W8QpnEpilogue E>
struct W8TcEpilogueOutput {
    __nv_bfloat16* out;
    std::int64_t ld;

    __device__ __forceinline__ void store(int row, int token, float value) const {
        const std::int64_t offset = token * ld + row;
        if constexpr (E == W8QpnEpilogue::SwiGluUp) {
            const float gate = __bfloat162float(out[offset]);
            const float up   = __bfloat162float(__float2bfloat16(value));
            out[offset]      = __float2bfloat16_rn(silu(gate) * up);
        } else {
            out[offset] = __float2bfloat16(value);
        }
    }
};

template <W8QpnEpilogue E>
void w8_qpn_launch_packed(const Tensor& x, const half* x_fp16, const std::uint8_t* codes,
                          const std::uint8_t* scales, __nv_bfloat16* out, int n, int out_ld,
                          cudaStream_t stream, const __nv_bfloat16* finish_delta = nullptr,
                          const __nv_bfloat16* base = nullptr, __nv_bfloat16* residual = nullptr,
                          int width = 0) {
    const int k     = x.ne[0];
    const int total = x.ne[1];
    const int chunk = E == W8QpnEpilogue::DynamicConvAdd
                          ? kPackedMaxRows / width * width
                          : kPackedMaxRows;
    const auto* x_bf16_all = static_cast<const __nv_bfloat16*>(x.data);
    const dim3 grid(static_cast<unsigned>((n + S::kColsPerCta - 1) / S::kColsPerCta));
    for (int start = 0; start < total; start += chunk) {
        const int t = total - start < chunk ? total - start : chunk;
        const std::int64_t first = start;
        const auto* x_bf16 = x_bf16_all + first * k;
        const half* x_half = x_fp16 != nullptr ? x_fp16 + first * k : nullptr;
        __nv_bfloat16* out_chunk = out != nullptr ? out + first * out_ld : nullptr;
        // Conv-add rows: two finish-delta taps of 320 groups each, one 5120-wide residual row.
        const __nv_bfloat16* delta_chunk =
            finish_delta != nullptr ? finish_delta + first * 640 : nullptr;
        __nv_bfloat16* residual_chunk = residual != nullptr ? residual + first * 5120 : nullptr;
        if constexpr (E == W8QpnEpilogue::Store || E == W8QpnEpilogue::SwiGluUp) {
            // Past two row tiles the tensor-core route reads the weight once for all rows.
            const W8VoltaTcOptionalDecoder decoder{
                {reinterpret_cast<const uint4*>(codes), reinterpret_cast<const __half*>(scales),
                 k / 128}};
            const W8TcEpilogueOutput<E> policy{out_chunk, out_ld};
            if (x_half != nullptr
                    ? launch_volta_tc_skinny(decoder, x_half, n, k, t, 0, policy, stream)
                    : launch_volta_tc_skinny(decoder, x_bf16, n, k, t, 0, policy, stream)) {
                continue;
            }
        } else if constexpr (E == W8QpnEpilogue::DynamicConvAdd) {
            // Draft down projections (5120 x 17408) past 16 tokens: project once through
            // volta_tc_skinny into the L2-resident 1 MiB stream scratch (<=640 KiB) and finish
            // with the 2-tap dynamic conv + residual add (~85 us vs 144 us on quadpair kTiles=4).
            if (t > 16 && n == 5120 && width > 0 && (t % width) == 0) {
                auto* scratch = static_cast<__nv_bfloat16*>(volta_stream_scratch(
                    stream, static_cast<std::size_t>(n) * t * sizeof(__nv_bfloat16)));
                if (scratch != nullptr) {
                    const W8VoltaTcOptionalDecoder decoder{
                        {reinterpret_cast<const uint4*>(codes),
                         reinterpret_cast<const __half*>(scales), k / 128}};
                    const W8TcEpilogueOutput<W8QpnEpilogue::Store> policy{scratch, n};
                    const bool launched =
                        x_half != nullptr
                            ? launch_volta_tc_skinny(decoder, x_half, n, k, t, 0, policy, stream)
                            : launch_volta_tc_skinny(decoder, x_bf16, n, k, t, 0, policy, stream);
                    if (launched) {
                        dflash2_dynamic_conv_finish_add_launch(scratch, delta_chunk, base, width,
                                                               t / width, residual_chunk, stream);
                        continue;
                    }
                }
            }
        }
#define NINFER_W8_PACKED(TILES, SPLITK)                                                          \
    do {                                                                                         \
        if (x_half != nullptr) {                                                                 \
            w8_qpn_launch_packed_typed<E, TILES, SPLITK>(grid, x_half, codes, scales, out_chunk, \
                                                         n, k, t, out_ld, delta_chunk, base,     \
                                                         residual_chunk, width, stream);         \
        } else {                                                                                 \
            w8_qpn_launch_packed_typed<E, TILES, SPLITK>(grid, x_bf16, codes, scales, out_chunk, \
                                                         n, k, t, out_ld, delta_chunk, base,     \
                                                         residual_chunk, width, stream);         \
        }                                                                                        \
    } while (false)
        if (t <= 8) {
            NINFER_W8_PACKED(1, 8);
        } else if (t <= 16) {
            NINFER_W8_PACKED(2, 8);
        } else if (t <= 32) {
            NINFER_W8_PACKED(4, 4);
        } else {
            NINFER_W8_PACKED(8, 4);
        }
#undef NINFER_W8_PACKED
        CUDA_CHECK(cudaGetLastError());
    }
}

void require_w8_packed(const Weight& w, std::int32_t n, std::int32_t k, std::int32_t t,
                       std::int32_t width = 1) {
    if (!w8_volta_qpn_prepacked_supported(n, k, t)) {
        throw std::invalid_argument(
            "W8 QPN: a prepacked weight is readable only with N % 32 == 0, K % 128 == 0 and at "
            "least one token");
    }
    if (width < 1 || width > kPackedMaxRows || t % width != 0) {
        throw std::invalid_argument(
            "W8 QPN: a prepacked conv-add needs whole requests of 1-64 rows each");
    }
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

bool w8_volta_qpn_tiled_supported(std::int32_t n, std::int32_t k, std::int32_t t) noexcept {
    // The single-tile domain's weight checks, at up to 64 tokens (the FP16 staging limit).
    return w8_volta_qpn_supported(n, k, 1) && t >= 1 && t <= 64;
}

bool w8_volta_qpn_prepacked_supported(std::int32_t n, std::int32_t k, std::int32_t t) noexcept {
    // Any token count: w8_qpn_launch_packed splits it into launches of at most 64 rows.
    return n > 0 && n % 32 == 0 && k > 0 && k % 128 == 0 && t >= 1;
}

void launch_w8_volta_qpn(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    const std::int32_t n      = out.ne[0];
    const std::int32_t k      = x.ne[0];
    const std::int32_t t      = x.ne[1];
    const std::int32_t out_ld = static_cast<std::int32_t>(out.nb[1] / sizeof(__nv_bfloat16));
    if (w.layout == QuantLayout::VoltaQpnPrepacked) {
        require_w8_packed(w, n, k, t);
        w8_qpn_launch_packed<W8QpnEpilogue::Store>(
            x, volta_stage_fp16_activation(x, stream), static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const std::uint8_t*>(w.scales), static_cast<__nv_bfloat16*>(out.data), n,
            out_ld, stream);
        return;
    }
    if (t > S::kRowsPerTile) {
        const half* staged = volta_stage_fp16_activation(x, stream);
        if (staged == nullptr) {
            throw std::logic_error("W8 QPN: tiled launch needs the FP16-staged activation");
        }
        auto* o = static_cast<__nv_bfloat16*>(out.data);
        for (std::int32_t first = 0; first < t; first += 32) {
            const std::int32_t chunk = t - first < 32 ? t - first : 32;
            w8_qpn_launch_rm_chunk<W8QpnEpilogue::Store, 4>(
                staged + static_cast<std::int64_t>(first) * k,
                static_cast<const std::uint8_t*>(w.qdata),
                static_cast<const std::uint8_t*>(w.scales), o + static_cast<std::int64_t>(first) * out_ld,
                n, k, chunk, w8_qpn_padded_groups(w), out_ld, nullptr, nullptr, stream);
        }
        CUDA_CHECK(cudaGetLastError());
        return;
    }
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
    if (w.layout == QuantLayout::VoltaQpnPrepacked) {
        // The q|k|v parent is shared with the row-major context K/V views; it is never packed.
        throw std::invalid_argument("W8 QPN split3: the q|k|v weight must stay row-major");
    }
    const W8QpnLaunch launch = w8_qpn_launch_settings(kParentRows, x.ne[0]);
    if (x.ne[1] > S::kRowsPerTile) {
        // The draft's measured setting (split 8, BF16 activations) at every chunk, so a request's
        // q|k|v rows are bit-identical whether it decodes alone or in a batch.
        if (launch.splitk != 8 || launch.stage_fp16) {
            throw std::logic_error("W8 QPN split3: tiled launch expects split 8 on BF16");
        }
        const std::int32_t kk = x.ne[0];
        const auto* xb = static_cast<const __nv_bfloat16*>(x.data);
        auto* qp = static_cast<__nv_bfloat16*>(q.data);
        auto* kp = static_cast<__nv_bfloat16*>(k.data);
        auto* vp = static_cast<__nv_bfloat16*>(v.data);
        for (std::int32_t first = 0; first < x.ne[1]; first += 32) {
            const std::int32_t t = x.ne[1] - first < 32 ? x.ne[1] - first : 32;
            w8_qpn_launch_rm_chunk<W8QpnEpilogue::Split3, 8>(
                xb + static_cast<std::int64_t>(first) * kk,
                static_cast<const std::uint8_t*>(w.qdata),
                static_cast<const std::uint8_t*>(w.scales), qp + first * 4096LL, kParentRows, kk,
                t, w8_qpn_padded_groups(w), 0, kp + first * 1024LL, vp + first * 1024LL, stream);
        }
        CUDA_CHECK(cudaGetLastError());
        return;
    }
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
    if (w.layout == QuantLayout::VoltaQpnPrepacked) {
        require_w8_packed(w, residual.ne[0], x.ne[0], x.ne[1], width);
        w8_qpn_launch_packed<W8QpnEpilogue::DynamicConvAdd>(
            x, volta_stage_fp16_activation(x, stream), static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const std::uint8_t*>(w.scales), nullptr, residual.ne[0], 0, stream,
            static_cast<const __nv_bfloat16*>(finish_delta.data),
            static_cast<const __nv_bfloat16*>(base.data),
            static_cast<__nv_bfloat16*>(residual.data), width);
        return;
    }
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

    if (w.layout == QuantLayout::VoltaQpnPrepacked) {
        // Packed tiles are 32 rows of whole K, so the up half starts kIntermediate rows in, at
        // the same byte offsets as the row-major planes.
        require_w8_packed(w, 2 * kIntermediate, k, t);
        const half* x_fp16 = volta_stage_fp16_activation(x, stream);
        w8_qpn_launch_packed<W8QpnEpilogue::Store>(x, x_fp16, gate_codes, gate_scales, output,
                                                   kIntermediate, kIntermediate, stream);
        w8_qpn_launch_packed<W8QpnEpilogue::SwiGluUp>(x, x_fp16, up_codes, up_scales, output,
                                                      kIntermediate, kIntermediate, stream);
        return;
    }
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

namespace {

// Row ranges of one prepacked parent weight scattered into separate [rows, T] outputs.
struct W8TcRowSplitOutput {
    __nv_bfloat16* first;
    __nv_bfloat16* second;
    __nv_bfloat16* third;
    int first_rows;
    int second_rows;
    std::int64_t first_ld;
    std::int64_t second_ld;
    std::int64_t third_ld;

    __device__ __forceinline__ void store(int row, int token, float value) const {
        if (row < first_rows) {
            first[token * first_ld + row] = __float2bfloat16(value);
        } else if (row < first_rows + second_rows) {
            second[token * second_ld + row - first_rows] = __float2bfloat16(value);
        } else {
            third[token * third_ld + row - first_rows - second_rows] = __float2bfloat16(value);
        }
    }
};

bool w8_tc_prepacked(const Weight& w, std::int32_t k) {
    return w.qtype == QType::W8G32_F16S && w.layout == QuantLayout::VoltaQpnPrepacked &&
           w.k == k && w.padded_shape[1] == k && k % 128 == 0 && w.n % 64 == 0;
}

// Every token count, in launches of at most 64 tokens, BF16 activations staged by the kernel.
void w8_tc_launch_chunks(const Tensor& x, const Weight& w, W8TcRowSplitOutput output,
                         cudaStream_t stream) {
    const std::int32_t k = x.ne[0];
    const std::int32_t t = x.ne[1];
    const W8VoltaTcDecoder decoder{static_cast<const uint4*>(w.qdata),
                                   static_cast<const __half*>(w.scales), k / 128};
    for (std::int32_t first = 0; first < t; first += 64) {
        const std::int32_t count = std::min<std::int32_t>(64, t - first);
        W8TcRowSplitOutput chunk = output;
        chunk.first += first * output.first_ld;
        if (chunk.second != nullptr) { chunk.second += first * output.second_ld; }
        if (chunk.third != nullptr) { chunk.third += first * output.third_ld; }
        const auto* activation =
            static_cast<const __nv_bfloat16*>(x.data) + static_cast<std::int64_t>(first) * k;
        if (!launch_volta_tc_skinny(decoder, activation, w.n, k, count, 0, chunk, stream)) {
            throw std::invalid_argument("W8 tensor-core route: unsupported prepacked shape");
        }
    }
}

std::int64_t leading(const Tensor& t) {
    return static_cast<std::int64_t>(t.nb[1] / sizeof(__nv_bfloat16));
}

} // namespace

bool launch_w8_volta_tc_split3(const Tensor& x, const Weight& w, Tensor& q, Tensor& k, Tensor& v,
                               cudaStream_t stream) {
    if (!w8_tc_prepacked(w, x.ne[0]) || w.n != q.ne[0] + k.ne[0] + v.ne[0]) { return false; }
    w8_tc_launch_chunks(x, w,
                        {static_cast<__nv_bfloat16*>(q.data), static_cast<__nv_bfloat16*>(k.data),
                         static_cast<__nv_bfloat16*>(v.data), q.ne[0], k.ne[0], leading(q),
                         leading(k), leading(v)},
                        stream);
    return true;
}

bool launch_w8_volta_tc_pair(const Tensor& x, const Weight& first, const Weight& second,
                             Tensor& first_out, Tensor& second_out, cudaStream_t stream) {
    const std::int32_t k = x.ne[0];
    if (!w8_tc_prepacked(first, k) || !w8_tc_prepacked(second, k)) { return false; }
    const auto* first_codes  = static_cast<const std::uint8_t*>(first.qdata);
    const auto* first_scales = static_cast<const std::uint8_t*>(first.scales);
    const bool adjacent =
        static_cast<const std::uint8_t*>(second.qdata) ==
            first_codes + static_cast<std::int64_t>(first.n) * k &&
        static_cast<const std::uint8_t*>(second.scales) ==
            first_scales + static_cast<std::int64_t>(first.n) * (k / 32) * 2;
    if (adjacent) {
        // Row views of one parent (the DFlash2 context key and value): one launch over both.
        Weight both          = first;
        both.n               = first.n + second.n;
        both.shape[0]        = both.n;
        both.padded_shape[0] = both.n;
        w8_tc_launch_chunks(x, both,
                            {static_cast<__nv_bfloat16*>(first_out.data),
                             static_cast<__nv_bfloat16*>(second_out.data), nullptr, first.n,
                             second.n, leading(first_out), leading(second_out), 0},
                            stream);
        return true;
    }
    w8_tc_launch_chunks(x, first,
                        {static_cast<__nv_bfloat16*>(first_out.data), nullptr, nullptr, first.n,
                         0, leading(first_out), 0, 0},
                        stream);
    w8_tc_launch_chunks(x, second,
                        {static_cast<__nv_bfloat16*>(second_out.data), nullptr, nullptr,
                         second.n, 0, leading(second_out), 0, 0},
                        stream);
    return true;
}

#endif // NINFER_VOLTA_BUILD

} // namespace ninfer::ops::detail
