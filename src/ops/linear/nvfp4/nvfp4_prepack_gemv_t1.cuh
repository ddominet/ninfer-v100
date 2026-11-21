#pragma once

// T=1 prepacked NVFP4 GEMV shared by the linear_swiglu split route and the linear_add
// (down projection + residual) route. One warp owns 32 rows; K is split across grid.y and
// atomic-added into fp32 scratch, which the caller zeroes first and combines after.
// Prepack index contract (nvfp4_prepack_sm70.cu): ((tile * groups + group) * 32 + lane);
// row = tile * 32 + ((lane >> 2) & 3) * 8 + (lane & 3) + ((lane & 16) ? 4 : 0);
// decoder pairs (nibble j, j+4) are adjacent K: (k0, k1) ... (k14, k15).
#include "ops/linear/nvfp4/nvfp4_volta_qpn_gemm.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace ninfer::ops::detail {

template <int kSplit>
__global__ void __launch_bounds__(32, 8)
nvfp4_prepack_gemv_t1(const std::uint8_t* __restrict__ codes,
                      const std::uint8_t* __restrict__ scales, const half* __restrict__ x,
                      float* __restrict__ out, int n, int k, float divisor) {
    const int lane   = threadIdx.x;
    const int tile   = static_cast<int>(blockIdx.x);
    const int sk     = static_cast<int>(blockIdx.y);
    const int groups = k / 16;
    const int chunk  = groups / kSplit;
    const int g0     = sk * chunk;
    const int gend   = sk == kSplit - 1 ? groups : g0 + chunk;
    const half2 rebias   = __float2half2_rn(16384.f);
    const half2 divisor2 = __float2half2_rn(256.f);
    float a0             = 0.f;
    for (int group = g0; group < gend; ++group) {
        const long long idx = (static_cast<long long>(tile) * groups + group) * 32 + lane;
        uint2 q2;
        unsigned sb;
        asm volatile("ld.global.cs.v2.u32 {%0, %1}, [%2];"
                     : "=r"(q2.x), "=r"(q2.y)
                     : "l"(codes + idx * 8));
        asm volatile("ld.global.cs.u8 %0, [%1];" : "=r"(sb) : "l"(scales + idx));
        half2 b[8];
        nvfp4_decode_e2m1_quad(q2.x, rebias, *reinterpret_cast<half2(*)[4]>(&b[0]));
        nvfp4_decode_e2m1_quad(q2.y, rebias, *reinterpret_cast<half2(*)[4]>(&b[4]));
        const half2 sc = __hmul2(nvfp4_decode_e4m3_scale(static_cast<unsigned char>(sb)), divisor2);
#pragma unroll
        for (int j = 0; j < 8; ++j) b[j] = __hmul2(b[j], sc);
        const int kbase = group * 16;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const float x0  = __half2float(x[kbase + 2 * j]);
            const float x1  = __half2float(x[kbase + 2 * j + 1]);
            const float2 bv = __half22float2(b[j]);
            a0              = fmaf(bv.x, x0, a0);
            a0              = fmaf(bv.y, x1, a0);
        }
    }
    const int row = tile * 32 + ((lane >> 2) & 3) * 8 + (lane & 3) + ((lane & 16) ? 4 : 0);
    if (row < n) atomicAdd(out + row, a0 * divisor);
}

} // namespace ninfer::ops::detail
