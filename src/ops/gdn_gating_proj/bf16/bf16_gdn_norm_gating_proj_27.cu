#include "ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_kernels.h"
#include "core/device.h"
#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/warp.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace ninfer::ops::detail {
namespace {
// High-occupancy fused RMSNorm + GDN gating projection for 27B (D=5120, H=48, T<=128):
// - Threads=640 (Passes=1, exact 8 elems/thread, zero K-loop) for T<=16 (Tile=1/2 -> 192 CTAs at
//   T=8) saturates all 80 Volta SMs with zero wasted lanes.
// - Threads=320 (Passes=2, exact 16 elems/thread) for T>=17 (Tile=4 -> 288 CTAs at T=24, 384 CTAs
//   at T=32) processes the two K-halves sequentially so only 16 weight floats stay live in
//   registers while unpacking (a, b, 1+norm) ONCE per Tile tokens instead of Tile times.
// - Optional WriteFp16 emits __half2 directly into h, eliminating 48 fp8_stage_bf16 launches/round.
template <int Tile, int Threads, bool WriteFp16 = false>
__global__ __launch_bounds__(Threads) void gdn_norm_gating_27_simt(
    const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ nw,
    const __nv_bfloat16* __restrict__ aw, const __nv_bfloat16* __restrict__ bw,
    const float* __restrict__ alog, const float* __restrict__ bias, void* __restrict__ h,
    float* __restrict__ g, float* __restrict__ beta, const std::int32_t* __restrict__ scatter_cols,
    int tokens, float eps) {
    constexpr int D           = 5120;
    constexpr int H           = 48;
    constexpr int kWarps      = Threads / 32;
    constexpr int kPasses     = D / (Threads * 8);
    constexpr int PairsPerCta = (D / 2 + H - 1) / H; // 54
    static_assert(kPasses * Threads * 8 == D, "Threads * 8 must divide D=5120");

    const int tid        = static_cast<int>(threadIdx.x);
    const int lane       = tid & 31;
    const int warp       = tid >> 5;
    const int head       = static_cast<int>(blockIdx.x);
    const int first      = static_cast<int>(blockIdx.y) * Tile;
    const int tile_count = min(Tile, tokens - first);
    if (tile_count <= 0) { return; }

    float aa[Tile];
    float bb[Tile];
    float ss[Tile];
#pragma unroll
    for (int t = 0; t < Tile; ++t) {
        aa[t] = 0.0f;
        bb[t] = 0.0f;
        ss[t] = 0.0f;
    }

    __shared__ float partial[3][Tile][kWarps];
    __shared__ float inverse[Tile];
    __shared__ float2 s_np1[PairsPerCta];

    const int pair_base = head * PairsPerCta;
    const int my_pairs  = max(0, min(PairsPerCta, D / 2 - pair_base));
    if (tid < my_pairs) {
        const float2 nf =
            __bfloat1622float2(reinterpret_cast<const __nv_bfloat162*>(nw)[pair_base + tid]);
        s_np1[tid] = make_float2(1.0f + nf.x, 1.0f + nf.y);
    }

    const __nv_bfloat16* aw_head = aw + static_cast<std::int64_t>(head) * D;
    const __nv_bfloat16* bw_head = bw + static_cast<std::int64_t>(head) * D;

#pragma unroll
    for (int pass = 0; pass < kPasses; ++pass) {
        const int base = (tid + pass * Threads) * 8;
        const int4 av  = load_vec<int4>(aw_head + base);
        const int4 bv  = load_vec<int4>(bw_head + base);
        const int4 nv  = load_vec<int4>(nw + base);

        float an[8];
        float bn[8];
        const auto* ap = reinterpret_cast<const unsigned*>(&av);
        const auto* bp = reinterpret_cast<const unsigned*>(&bv);
        const auto* np = reinterpret_cast<const unsigned*>(&nv);
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            const float2 af = bf16x2_bits_to_float2(ap[p]);
            const float2 bf = bf16x2_bits_to_float2(bp[p]);
            const float2 nf = bf16x2_bits_to_float2(np[p]);
            const float nx  = 1.0f + nf.x;
            const float ny  = 1.0f + nf.y;
            an[2 * p]       = af.x * nx;
            an[2 * p + 1]   = af.y * ny;
            bn[2 * p]       = bf.x * nx;
            bn[2 * p + 1]   = bf.y * ny;
        }

#pragma unroll
        for (int t = 0; t < Tile; ++t) {
            if (t < tile_count) {
                const int4 xv  = load_vec<int4>(x + static_cast<std::int64_t>(first + t) * D + base);
                const auto* xp = reinterpret_cast<const unsigned*>(&xv);
#pragma unroll
                for (int p = 0; p < 4; ++p) {
                    const float2 v = bf16x2_bits_to_float2(xp[p]);
                    aa[t]          = fmaf(an[2 * p], v.x, aa[t]);
                    aa[t]          = fmaf(an[2 * p + 1], v.y, aa[t]);
                    bb[t]          = fmaf(bn[2 * p], v.x, bb[t]);
                    bb[t]          = fmaf(bn[2 * p + 1], v.y, bb[t]);
                    ss[t]          = fmaf(v.x, v.x, ss[t]);
                    ss[t]          = fmaf(v.y, v.y, ss[t]);
                }
            }
        }
    }

#pragma unroll
    for (int t = 0; t < Tile; ++t) {
        if (t < tile_count) {
            aa[t] = warp_reduce_sum(aa[t]);
            bb[t] = warp_reduce_sum(bb[t]);
            ss[t] = warp_reduce_sum(ss[t]);
            if (lane == 0) {
                partial[0][t][warp] = aa[t];
                partial[1][t][warp] = bb[t];
                partial[2][t][warp] = ss[t];
            }
        }
    }
    __syncthreads();

    if (tid < tile_count) {
        float a_sum = 0.0f;
        float b_sum = 0.0f;
        float s_sum = 0.0f;
#pragma unroll
        for (int w = 0; w < kWarps; ++w) {
            a_sum += partial[0][tid][w];
            b_sum += partial[1][tid][w];
            s_sum += partial[2][tid][w];
        }
        const float inv = rsqrtf(s_sum * (1.0f / static_cast<float>(D)) + eps);
        inverse[tid]    = inv;
        const int dst_col =
            scatter_cols != nullptr ? scatter_cols[first + tid] : (first + tid);
        const auto i = static_cast<std::int64_t>(dst_col) * H + head;
        g[i]         = -expf(alog[head]) * softplus(a_sum * inv + bias[head]);
        beta[i]      = sigmoid(b_sum * inv);
    }
    __syncthreads();

    if (my_pairs > 0) {
        const int total_writes = tile_count * my_pairs;
        for (int idx = tid; idx < total_writes; idx += Threads) {
            const int t    = idx / my_pairs;
            const int p    = idx - t * my_pairs;
            const int pair = pair_base + p;
            const auto i   = static_cast<std::int64_t>(first + t) * (D / 2) + pair;
            const float2 v = __bfloat1622float2(reinterpret_cast<const __nv_bfloat162*>(x)[i]);
            const float2 n = s_np1[p];
            const float iv = inverse[t];
            if constexpr (WriteFp16) {
                reinterpret_cast<__half2*>(h)[i] =
                    __floats2half2_rn(v.x * iv * n.x, v.y * iv * n.y);
            } else {
                reinterpret_cast<__nv_bfloat162*>(h)[i] =
                    __floats2bfloat162_rn(v.x * iv * n.x, v.y * iv * n.y);
            }
        }
    }
}

template <int Tile, int Threads, bool WriteFp16>
void launch_gdn_norm_gating_27(const Tensor& x, const Tensor& norm_weight, float eps, Tensor& h,
                               const Weight& a_weight, const Weight& b_weight, const Tensor& alog,
                               const Tensor& bias, Tensor& g, Tensor& beta,
                               const std::int32_t* scatter_cols, cudaStream_t stream) {
    gdn_norm_gating_27_simt<Tile, Threads, WriteFp16>
        <<<dim3(48, (x.ne[1] + Tile - 1) / Tile), Threads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const __nv_bfloat16*>(norm_weight.data),
            static_cast<const __nv_bfloat16*>(a_weight.qdata),
            static_cast<const __nv_bfloat16*>(b_weight.qdata),
            static_cast<const float*>(alog.data), static_cast<const float*>(bias.data), h.data,
            static_cast<float*>(g.data), static_cast<float*>(beta.data), scatter_cols, x.ne[1],
            eps);
}

template <bool WriteFp16>
void dispatch_gdn_norm_gating_27(const Tensor& x, const Tensor& norm_weight, float eps, Tensor& h,
                                 const Weight& a_weight, const Weight& b_weight, const Tensor& alog,
                                 const Tensor& bias, Tensor& g, Tensor& beta,
                                 const std::int32_t* scatter_cols, cudaStream_t stream) {
    const int tokens = x.ne[1];
    if (tokens <= 2) {
        launch_gdn_norm_gating_27<1, 640, WriteFp16>(x, norm_weight, eps, h, a_weight, b_weight,
                                                     alog, bias, g, beta, scatter_cols, stream);
    } else if (tokens <= 16) {
        launch_gdn_norm_gating_27<2, 640, WriteFp16>(x, norm_weight, eps, h, a_weight, b_weight,
                                                     alog, bias, g, beta, scatter_cols, stream);
    } else if (tokens <= 64) {
        launch_gdn_norm_gating_27<4, 320, WriteFp16>(x, norm_weight, eps, h, a_weight, b_weight,
                                                     alog, bias, g, beta, scatter_cols, stream);
    } else {
        launch_gdn_norm_gating_27<8, 320, WriteFp16>(x, norm_weight, eps, h, a_weight, b_weight,
                                                     alog, bias, g, beta, scatter_cols, stream);
    }
}
} // namespace

void bf16_gdn_norm_gating_proj_27_launch(const Tensor& x, const Tensor& norm_weight, float eps,
                                         Tensor& h, const Weight& a_weight, const Weight& b_weight,
                                         const Tensor& alog, const Tensor& bias, Tensor& g,
                                         Tensor& beta, cudaStream_t stream,
                                         const std::int32_t* scatter_cols, bool write_fp16) {
    if (write_fp16) {
        dispatch_gdn_norm_gating_27<true>(x, norm_weight, eps, h, a_weight, b_weight, alog, bias, g,
                                          beta, scatter_cols, stream);
    } else {
        dispatch_gdn_norm_gating_27<false>(x, norm_weight, eps, h, a_weight, b_weight, alog, bias,
                                           g, beta, scatter_cols, stream);
    }
    CUDA_CHECK(cudaGetLastError());
}
} // namespace ninfer::ops::detail
