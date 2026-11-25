// DFlash2 fused plain-RMSNorm + full-128 split-half 1-D RoPE (theta 1e7) — sm_70.
//
// n is kept in FP32 through the rotation; only the final rotated result is rounded to BF16 once.
// The angle uses the fork's precise double-precision inv-frequency table + range reduction.

#include "ops/common/dflash2_rmsnorm_rope_volta.h"

#include "core/device.h"
#include "ops/common/dflash_rope.cuh"
#include "ops/common/warp.cuh"

#include <cuda_bf16.h>
#include <cstdint>

namespace ninfer::ops::detail {
namespace {

constexpr int kHeadDim = 128;
constexpr int kHalf    = 64;

__global__ void dflash2_rmsnorm_rope_kernel(__nv_bfloat16* __restrict__ x,
                                            const __nv_bfloat16* __restrict__ weight,
                                            const std::int32_t* __restrict__ positions, int heads,
                                            int tokens, float eps) {
    // x is [128, heads, T] contiguous, so column col = head + heads * token.
    const int col = static_cast<int>(blockIdx.x);
    const int tok = col / heads;
    if (tok >= tokens) { return; }
    const int d = static_cast<int>(threadIdx.x); // 0..127

    __nv_bfloat16* row = x + static_cast<std::int64_t>(col) * kHeadDim;
    const float xd     = __bfloat162float(row[d]);

    __shared__ float sums[4];
    const float ss = block_reduce_sum<128>(xd * xd, sums);
    __shared__ float inv_norm;
    if (d == 0) { inv_norm = rsqrtf(ss / static_cast<float>(kHeadDim) + eps); }
    __syncthreads();

    __shared__ float n[kHeadDim];
    n[d] = xd * inv_norm * __bfloat162float(weight[d]);
    __syncthreads();

    if (d < kHalf) {
        const float lo = n[d];
        const float hi = n[d + kHalf];
        float s;
        float c;
        ninfer::ops::dflash_rope_sincos(positions, tok, d, &s, &c);
        row[d]         = __float2bfloat16_rn(lo * c - hi * s);
        row[d + kHalf] = __float2bfloat16_rn(hi * c + lo * s);
    }
}

} // namespace

void dflash2_rmsnorm_rope_launch(const Tensor& x, const Tensor& weight, const Tensor& positions,
                                 float eps, float theta, cudaStream_t stream) {
    (void)theta; // the inv-frequency table is fixed for theta = 1e7
    const int heads  = x.ne[1];
    const int tokens = x.ne[2];
    const int blocks = heads * tokens;
    dflash2_rmsnorm_rope_kernel<<<blocks, kHeadDim, 0, stream>>>(
        static_cast<__nv_bfloat16*>(x.data), static_cast<const __nv_bfloat16*>(weight.data),
        static_cast<const std::int32_t*>(positions.data), heads, tokens, eps);
    CUDA_CHECK(cudaGetLastError());
}

namespace {

static __device__ __constant__ float kTargetTextRopeInvFreq[32] = {
    1.000000000e+00F, 6.042963902e-01F, 3.651741273e-01F, 2.206734069e-01F, 1.333521432e-01F,
    8.058421878e-02F, 4.869675252e-02F, 2.942727176e-02F, 1.778279410e-02F, 1.074607828e-02F,
    6.493816316e-03F, 3.924189758e-03F, 2.371373706e-03F, 1.433012570e-03F, 8.659643234e-04F,
    5.232991147e-04F, 3.162277660e-04F, 1.910952975e-04F, 1.154781985e-04F, 6.978305849e-05F,
    4.216965034e-05F, 2.548296748e-05F, 1.539926526e-05F, 9.305720409e-06F, 5.623413252e-06F,
    3.398208329e-06F, 2.053525026e-06F, 1.240937761e-06F, 7.498942093e-07F, 4.531583638e-07F,
    2.738419634e-07F, 1.654817100e-07F,
};

__global__ __launch_bounds__(256) void target_rmsnorm_qk_rope_kernel(
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ q_norm, const __nv_bfloat16* __restrict__ k_norm,
    const std::int32_t* __restrict__ positions, __nv_bfloat16* __restrict__ qn,
    __nv_bfloat16* __restrict__ kn, int n_q, int n_kv, int tokens, float eps) {
    constexpr int kTargetHeadDim = 256;
    constexpr int kRotaryDim     = 64;
    constexpr int kRotaryHalf    = 32;

    const int head_idx = static_cast<int>(blockIdx.x);
    const int tok      = static_cast<int>(blockIdx.y);
    if (tok >= tokens) { return; }
    const int d = static_cast<int>(threadIdx.x); // 0..255

    const bool is_k = head_idx >= n_q;
    const int h     = is_k ? (head_idx - n_q) : head_idx;
    const std::int64_t off =
        (static_cast<std::int64_t>(tok) * (is_k ? n_kv : n_q) + h) * kTargetHeadDim;
    const __nv_bfloat16* src = (is_k ? k : q) + off;
    const __nv_bfloat16* nw  = is_k ? k_norm : q_norm;
    __nv_bfloat16* dst       = (is_k ? kn : qn) + off;

    const float xd = __bfloat162float(src[d]);
    __shared__ float sums[8];
    const float ss = block_reduce_sum<kTargetHeadDim>(xd * xd, sums);
    __shared__ float inv_norm;
    if (d == 0) { inv_norm = rsqrtf(ss / static_cast<float>(kTargetHeadDim) + eps); }
    __syncthreads();

    const __nv_bfloat16 nb =
        __float2bfloat16_rn(xd * inv_norm * (1.0f + __bfloat162float(nw[d])));

    __shared__ float rot[kRotaryDim];
    if (d >= kRotaryDim) {
        dst[d] = nb;
    } else {
        rot[d] = __bfloat162float(nb);
    }
    __syncthreads();

    if (d < kRotaryHalf) {
        const float angle = static_cast<float>(positions[tok]) * kTargetTextRopeInvFreq[d];
        float sn;
        float cs;
        sincosf(angle, &sn, &cs);
        const float lo   = rot[d];
        const float hi   = rot[d + kRotaryHalf];
        dst[d]           = __float2bfloat16_rn(lo * cs - hi * sn);
        dst[d + kRotaryHalf] = __float2bfloat16_rn(hi * cs + lo * sn);
    }
}

} // namespace

void target_rmsnorm_qk_rope_launch(const Tensor& q, const Tensor& k, const Tensor& q_norm,
                                   const Tensor& k_norm, const Tensor& positions, float eps,
                                   Tensor& qn, Tensor& kn, cudaStream_t stream) {
    const int n_q    = q.ne[1];
    const int n_kv   = k.ne[1];
    const int tokens = q.ne[2];
    const dim3 grid(static_cast<unsigned>(n_q + n_kv), static_cast<unsigned>(tokens));
    target_rmsnorm_qk_rope_kernel<<<grid, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(q.data), static_cast<const __nv_bfloat16*>(k.data),
        static_cast<const __nv_bfloat16*>(q_norm.data),
        static_cast<const __nv_bfloat16*>(k_norm.data),
        static_cast<const std::int32_t*>(positions.data),
        static_cast<__nv_bfloat16*>(qn.data), static_cast<__nv_bfloat16*>(kn.data), n_q, n_kv,
        tokens, eps);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
