#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

using W8Launch = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);

#ifdef NINFER_VOLTA_BUILD
// Fused-dequant tensor-core route (w8_volta_mma_gemm.cuh). Unlike its Q4/Q5 siblings this one
// fits the W8Launch signature, because the shapes it is selected for supply far more CTAs than
// the machine holds resident and so need no split-K, and therefore no workspace.
void launch_w8_volta_mma(const Tensor&, const Weight&, Tensor&, cudaStream_t);
[[nodiscard]] bool w8_volta_mma_supported(std::int32_t n, std::int32_t k,
                                          std::int32_t t) noexcept;

// Quadpair-split-N form (w8_volta_qpn_gemm.cuh), for the narrow verify widths the 32x8 route
// pads its A rows away on. Also workspace-free: the CTA's warps split K and reduce in shared.
void launch_w8_volta_qpn(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_w8_volta_qpn_dynamic_conv_add(const Tensor& x, const Weight& weight,
                                          const Tensor& base, const Tensor& finish_delta,
                                          Tensor& residual, std::int32_t width,
                                          cudaStream_t stream);
[[nodiscard]] bool w8_volta_qpn_supported(std::int32_t n, std::int32_t k,
                                          std::int32_t t) noexcept;
// A weight permuted by w8_prepack_qpn_sm70 is read by the three QPN launchers above at 1-64
// tokens, and by nothing else.
[[nodiscard]] bool w8_volta_qpn_prepacked_supported(std::int32_t n, std::int32_t k,
                                                    std::int32_t t) noexcept;
// The DFlash2 draft's [6144, K] q|k|v projection, scattered into q [4096, T], k [1024, T] and
// v [1024, T]; T <= 8.
void launch_w8_volta_qpn_split3(const Tensor& x, const Weight& weight, Tensor& q, Tensor& k,
                                Tensor& v, cudaStream_t stream);
#endif

void launch_w8_decode_r4(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_w8_small_t(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_w8_exact_t_splitk(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_w8_exact_t_composite(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_w8_dflash_medium(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_w8_medium_splitk_c144(const Tensor&, const Weight&, Tensor&, cudaStream_t);

void launch_w8_simt_r8_c4(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_w8_simt_r8_c8(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_w8_simt_r4_c16(const Tensor&, const Weight&, Tensor&, cudaStream_t);

void launch_w8_mma_r32_c64(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_w8_mma_r32_c96(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_w8_mma_r32_c128(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_w8_mma_r48_c64(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_w8_mma_r48_c96(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_w8_mma_r48_c112(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_w8_mma_r48_c128(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_w8_mma_r64_c96(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_w8_mma_r64_c112(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_w8_mma_r64_c128(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_w8_mma_r96_c96(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_w8_mma_r128_c64(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_w8_mma_r128_c80(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_w8_mma_r64x16_c48_k128_a1(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_w8_mma_r64x32_c64_k128_a1(const Tensor&, const Weight&, Tensor&, cudaStream_t);

void launch_w8_exact_mma_r32_c96(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_w8_exact_mma_r32_c128(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_w8_exact_mma_r48_c96(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_w8_exact_mma_r48_c128(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_w8_exact_mma_r64_c96(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_w8_exact_mma_r64_c128(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_w8_exact_mma_r96_c96(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_w8_exact_mma_r128_c80(const Tensor&, const Weight&, Tensor&, cudaStream_t);

} // namespace ninfer::ops::detail
