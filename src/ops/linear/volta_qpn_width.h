#pragma once

// Row widths of the Volta quadpair-split (QPN) decode GEMMs, shared by the FP8 and NVFP4 kernels,
// their activation staging, and every plan that chooses between QPN and the CUTLASS prefill GEMM.

#include <cstdint>

namespace ninfer::ops::detail {

inline constexpr std::int32_t kVoltaQpnRowsPerTile = 8;

// One pass over the weights serves up to eight 8-row A tiles. Each tile is eight accumulator
// registers per lane and one mma per weight fragment, so cost grows with the tile count while the
// weight is still read once. The CUTLASS route dequantizes the whole weight to FP16 first, which
// at 40 rows cost ~1.1 ms per linear against ~0.2 ms here, so rounds of 33..64 rows (eight
// streams at four or more draft tokens, short prompt tails) stay on QPN.
inline constexpr std::int32_t kVoltaQpnMaxTokens = 64;

// Wider inputs up to three full passes are chunked through QPN. CUTLASS pays the weight
// dequantization whatever the width, so a prompt tail of 76..125 tokens cost ~375 ms of prefill
// against 155 ms for 55 tokens; chunked, 125 tokens take 259 ms. CUTLASS needs ~465 ms for
// 129..256 tokens, which three passes still beat and four do not.
inline constexpr std::int32_t kVoltaCutlassMinT = 3 * kVoltaQpnMaxTokens + 1;

// Resident CTAs per SM the quadpair kernels ask the compiler to fit (their __launch_bounds__
// minimum), from a target of resident warps per SM. Three- and four-tile kernels target sixteen
// warps, two eight-warp CTAs: at one CTA they took up to 206 registers a thread, eight warps an SM
// could not hide their weight and activation load latency, and two CTAs measured 21% faster on the
// 27B gate/up projection at 24 and 32 tokens (252 -> 198 and 295 -> 230 us). Six and eight tiles
// measured no gain from a second CTA.
constexpr int volta_qpn_min_blocks(int tiles, int splitk) {
    const int warps = tiles == 1 ? 32 : tiles <= 4 ? 16 : 4;
    return warps / splitk < 1 ? 1 : warps / splitk;
}

// Instantiated tile counts: 1, 2, 3, 4, 6, 8.
constexpr int volta_qpn_tiles(int tokens) {
    return tokens <= 8    ? 1
           : tokens <= 16 ? 2
           : tokens <= 24 ? 3
           : tokens <= 32 ? 4
           : tokens <= 48 ? 6
                          : 8;
}

} // namespace ninfer::ops::detail
