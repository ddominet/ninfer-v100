#pragma once

// Fused-dequant W8 x BF16 GEMM on Volta tensor cores, quadpair-split-N form (sm_70 only).
//
// The W8 sibling of q4_volta_qpn_gemm.cuh, for the same reason and on the shape where it matters
// most. The 27B output head is W8 [248320,5120] and the narrow verify widths are where it is worst:
//
//     T:            1       2       3       4       5       6       7       8
//     us:        1797    2122    3837    4469    5137    5794    5280    5302
//     GB/s:       752     637     352     303     263     234     257     256
//
// T=1 is already *at* the machine -- 752 GB/s against this card's ~794 GB/s measured ceiling, so
// the GEMV route leaves nothing on the table. Everything above it falls off a cliff, to a third of
// that by T=6, against a physical floor of ~1.8ms for reading the 1.31 GB weight once. Neither
// incumbent addresses it: sliced r8_c8 re-reads the weight per 8 output columns, and the 32x8
// fused route maps T to the 32-row A axis, so at T=4 it pads 28 of 32 rows with zeros and measures
// *worse* than SIMT (5702us).
//
// Quadpair-split-N puts T on the 8-row axis instead, which is the right shape for exactly this
// band. See the Q4 sibling for the mapping, the fragment maps (v100-skinny's, byte-verified) and
// why the four quadpairs share one activation tile.
//
// Two W8 specifics:
//
//   - kGroupK is 32 and kCodeBytesPerGroup is 32, so one group is 32 code bytes covering 32 k --
//     half the k per byte of the Q4 sibling, but the same 128B blocked read covers 4 groups, so
//     the loop structure is identical.
//   - The decode is the 8-bit magic-number identity already used by w8_volta_mma_gemm.cuh:
//     0x6400|u is exactly 1024+u as fp16, so biasing a code byte by XOR 0x80 into u = b+128 makes
//     it 1152+b, and one __hsub2 against 1152.0 recovers the signed code. One uint2 of codes is 8
//     adjacent k, decoding straight into the four half2 that two mma slices consume -- no
//     repacking, exactly as in Q4.

#include "ops/common/volta_mma.cuh"
#include "ops/linear/volta_tc_skinny.cuh"
#include "ops/common/math.cuh"
#include "ops/linear/w8/w8_rowsplit_storage.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <type_traits>

namespace ninfer::ops::detail {

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ == 700

struct W8VoltaQpnSchedule {
    static constexpr int kColsPerCta    = 32; // output rows per CTA (mma's N axis)
    static constexpr int kRowsPerTile   = 8;  // tokens per A tile (mma's M axis)
    static constexpr int kResidentWarps = 32; // per SM, held fixed across SPLITK
};

// Where a CTA's reduced outputs go. Store writes out[token * out_ld + row]; DynamicConvAdd and
// SwiGluUp are the DFlash2 draft's fused MLP epilogues; Split3 scatters a [6144] q|k|v parent into
// q [4096], k [1024] and v [1024] planes (out, out1, out2), each token-major. 4096 and 5120 are
// multiples of the 32-row CTA tile, so a CTA never straddles two planes.
// Prepacked W8 (w8_prepack_sm70.cu: codes [N/32 tile][K/128 block][8 slices][lane] x 16 bytes,
// scales [tile][block][lane] x 4 FP16) for the tensor-core route (volta_tc_skinny.cuh): one item
// is one lane's 16-byte slice of a 64-K chunk, decoded as code * group scale like the quadpair
// kernel. Mandatory: the DFlash2 draft's q|k|v weight is prepacked and read by nothing else.
struct W8VoltaTcDecoder {
    const uint4* codes;
    const __half* scales;
    int blocks; // K / 128

    static constexpr int kItemsPerTileChunk = 128;
    static constexpr int kMinTokens         = 1;
    static constexpr bool kMandatory        = true;
    struct Raw {
        uint4 q;
        __half s;
    };

    __device__ __forceinline__ Raw load(int tile, int chunk, int item) const {
        const std::int64_t block = static_cast<std::int64_t>(tile) * blocks + (chunk >> 1);
        const int slice          = item >> 5;
        const int lane           = item & 31;
        return {__ldg(codes + (block * 8 + (chunk & 1) * 4 + slice) * 32 + lane),
                scales[(block * 32 + lane) * 4 + (chunk & 1) * 2 + (slice >> 1)]};
    }
    __device__ __forceinline__ void store(const Raw& raw, int item, half* tile, int lds) const {
        // code + 1152 is exact in FP16 (1024 + 128 bias from the byte permute), so the subtract
        // is exact and the only rounding is the scale multiply.
        const half2 bias  = __half2half2(__ushort_as_half(0x6480));
        const half2 scale = __half2half2(raw.s);
        const unsigned words[4] = {raw.q.x ^ 0x80808080u, raw.q.y ^ 0x80808080u,
                                   raw.q.z ^ 0x80808080u, raw.q.w ^ 0x80808080u};
        half2 values[8];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const unsigned lo = __byte_perm(words[i], 0x64646464u, 0x4140);
            const unsigned hi = __byte_perm(words[i], 0x64646464u, 0x4342);
            values[2 * i]     = __hmul2(__hsub2(*reinterpret_cast<const half2*>(&lo), bias), scale);
            values[2 * i + 1] = __hmul2(__hsub2(*reinterpret_cast<const half2*>(&hi), bias), scale);
        }
        auto* destination =
            reinterpret_cast<uint4*>(tile + volta_tc_lane_row(item & 31) * lds + (item >> 5) * 16);
        destination[0] = reinterpret_cast<const uint4*>(values)[0];
        destination[1] = reinterpret_cast<const uint4*>(values)[1];
    }
    __device__ __forceinline__ float finish(int, float value) const { return value; }
};

// The same decoder where the quadpair kernel also reads the weight: optional, from its floor.
struct W8VoltaTcOptionalDecoder : W8VoltaTcDecoder {
    static constexpr int kMinTokens  = kVoltaTcW8MinTokens;
    static constexpr bool kMandatory = false;
};

enum class W8QpnEpilogue { Store, DynamicConvAdd, SwiGluUp, Split3 };

// `kTiles` is the number of 8-row A tiles (T <= 8 * kTiles); `kBlk` is how many W8 groups a lane
// reads before consuming any, in units of kCodeBytesPerGroup. See the Q4 sibling: a lane streams
// its own weight row, but a warp's 32 lanes stream 32 different rows at once, so consuming a whole
// 128B line per lane per iteration is what keeps that off DRAM.
//
// `SPLITK` is the number of warps per CTA splitting K (they split K, not N), and `Activation` is
// either the caller's BF16 rows or the FP16 copy staged by volta_stage_fp16_activation: the same
// two knobs as the FP8 sibling, for the same reason. Four warps a CTA leave too few lines in
// flight on the narrow (5120-row) shapes, and converting BF16 activations in the weight loop costs
// every lane 8 conversions per row tile per group once T > 1.
//
// `kPrepacked` reads codes and scales in the load-time fragment order of w8_prepack_sm70.cu:
// codes [N/32 tile][K/128 block][8 sixteen-byte slices][lane], scales [N/32 tile][K/128 block]
// [lane] x 4 FP16. Each lane still receives its own row's 128 adjacent K, but a warp's load is one
// contiguous 512-byte (codes) or 256-byte (scales) run instead of 32 rows a row-stride apart.
template <int kTiles, int kBlk, int SPLITK, class Activation,
          W8QpnEpilogue kEpilogue = W8QpnEpilogue::Store, bool kPrepacked = false>
__global__ __launch_bounds__(SPLITK * 32,
                             (W8VoltaQpnSchedule::kResidentWarps / SPLITK / kTiles) > 0
                                 ? W8VoltaQpnSchedule::kResidentWarps / SPLITK / kTiles
                                 : 1)
void w8_volta_qpn_gemm_kernel(
    const std::uint8_t* __restrict__ codes, const std::uint8_t* __restrict__ scales,
    const Activation* __restrict__ x, __nv_bfloat16* __restrict__ out, int n, int k, int t,
    int padded_groups, int out_ld, const __nv_bfloat16* __restrict__ finish_delta = nullptr,
    const __nv_bfloat16* __restrict__ base = nullptr,
    __nv_bfloat16* __restrict__ residual = nullptr, int width = 0,
    __nv_bfloat16* __restrict__ out1 = nullptr, __nv_bfloat16* __restrict__ out2 = nullptr) {
    using S = W8VoltaQpnSchedule;
    constexpr int kGroupK = W8RowSplitStorage::kGroupK;
    constexpr int kCodeB  = W8RowSplitStorage::kCodeBytesPerGroup;
    static_assert(kGroupK == kCodeB, "W8 stores one code byte per k");
    constexpr bool kDynamicConvAdd = kEpilogue == W8QpnEpilogue::DynamicConvAdd;
    constexpr bool kSwiGluUp       = kEpilogue == W8QpnEpilogue::SwiGluUp;

    __shared__ float cs[SPLITK][kTiles * S::kRowsPerTile * S::kColsPerCta];

    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    // Quadpair index, and this lane's position inside it. `r` is both the A row (token) and the
    // B column local to the quadpair.
    const int qp = (lane >> 2) & 3;
    const int r  = (lane & 3) + ((lane & 16) != 0 ? 4 : 0);

    const int col  = static_cast<int>(blockIdx.x) * S::kColsPerCta + qp * 8 + r;
    const int good = col < n;

    const int groups = k / kGroupK;
    // Whole kBlk-group blocks per warp, so every warp keeps the vector scale load below; the last
    // warp takes the remainder.
    const int gq     = (groups / kBlk / SPLITK) * kBlk;
    const int g0     = warp * gq;
    const int gend   = (warp == SPLITK - 1) ? groups : g0 + gq;

    static_assert(!kPrepacked || kBlk == 4, "prepacked blocks are four groups");
    const std::int64_t packed_tile =
        static_cast<std::int64_t>(blockIdx.x) * (k / (kBlk * kGroupK)) * 8 * 32;
    const std::uint8_t* crow =
        codes + static_cast<std::int64_t>(good ? col : 0) * padded_groups * kCodeB;
    const std::uint16_t* srow = reinterpret_cast<const std::uint16_t*>(
        scales + static_cast<std::int64_t>(good ? col : 0) * padded_groups *
                     W8RowSplitStorage::kScaleBytesPerGroup);

    float c[kTiles][8];
#pragma unroll
    for (int tile = 0; tile < kTiles; ++tile) {
#pragma unroll
        for (int i = 0; i < 8; ++i) { c[tile][i] = 0.0f; }
    }

    const half2 bias = __half2half2(__ushort_as_half(0x6480)); // 1152.0

    static_assert(kBlk == 4, "the vector scale load below assumes kBlk fp16 scales are 8 bytes");
    // 32 code bytes per group, so kBlk groups is kBlk * 2 uint4 and kBlk * 32 k-elements.
    uint4 cw[2 * kBlk];
    std::uint16_t scv[kBlk];

    for (int gb = g0; gb < gend; gb += kBlk) {
        const int blk  = min(kBlk, gend - gb);
        const int last = gend - 1;
        if constexpr (kPrepacked) {
            // k % 128 == 0 for this layout, so every block is whole.
            const std::int64_t base = packed_tile + static_cast<std::int64_t>(gb / kBlk) * 8 * 32 + lane;
#pragma unroll
            for (int e = 0; e < 2 * kBlk; ++e) {
                cw[e] = __ldg(reinterpret_cast<const uint4*>(codes) + base + e * 32);
            }
        } else {
#pragma unroll
            for (int e = 0; e < kBlk; ++e) {
                const int g           = min(gb + e, last);
                const std::uint8_t* p = crow + static_cast<std::int64_t>(g) * kCodeB;
                cw[2 * e]             = __ldg(reinterpret_cast<const uint4*>(p));
                cw[2 * e + 1]         = __ldg(reinterpret_cast<const uint4*>(p + 16));
            }
        }
        // Scales blocked as well, and this matters more than it looks. One fp16 scale per 32 k
        // means the plane is 79 MB on this shape, and a 2-byte load per lane fetches a whole 32B
        // sector for it -- a 16x amplification that measured as 1.89 GB of DRAM read against a
        // 1.31 GB weight. Reading the block's kBlk scales as one vector cuts that to 4x.
        if constexpr (kPrepacked) {
            const uint2 packed = __ldg(reinterpret_cast<const uint2*>(scales) +
                                       packed_tile / 8 + (gb / kBlk) * 32 + lane);
            const auto* sc = reinterpret_cast<const std::uint16_t*>(&packed);
#pragma unroll
            for (int e = 0; e < kBlk; ++e) { scv[e] = good ? sc[e] : static_cast<std::uint16_t>(0); }
        } else if (blk == kBlk && (gb % kBlk) == 0) {
            const uint2 packed =
                __ldg(reinterpret_cast<const uint2*>(srow + gb)); // kBlk == 4 fp16 scales
            const auto* sc = reinterpret_cast<const std::uint16_t*>(&packed);
#pragma unroll
            for (int e = 0; e < kBlk; ++e) { scv[e] = good ? sc[e] : static_cast<std::uint16_t>(0); }
        } else {
#pragma unroll
            for (int e = 0; e < kBlk; ++e) {
                const int g = min(gb + e, last);
                scv[e]      = good ? srow[g] : static_cast<std::uint16_t>(0);
            }
        }

        for (int e = 0; e < blk; ++e) {
            const int g     = gb + e;
            const half2 sc2 = __half2half2(__ushort_as_half(scv[e]));
            const std::uint32_t words[8] = {cw[2 * e].x,     cw[2 * e].y,     cw[2 * e].z,
                                            cw[2 * e].w,     cw[2 * e + 1].x, cw[2 * e + 1].y,
                                            cw[2 * e + 1].z, cw[2 * e + 1].w};

#pragma unroll
            for (int u = 0; u < 4; ++u) {
                // One uint2 = 8 code bytes = 8 adjacent k, decoding into the four half2 that two
                // mma slices consume.
                const std::uint32_t w0 = words[2 * u] ^ 0x80808080u;
                const std::uint32_t w1 = words[2 * u + 1] ^ 0x80808080u;
                half2 b[4];
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const std::uint32_t src = (j < 2) ? w0 : w1;
                    const int shift         = (j & 1) * 16;
                    std::uint32_t bits =
                        (((src >> shift) & 0xffu) | (((src >> shift) & 0xff00u) << 8)) |
                        0x64006400u;
                    b[j] = __hmul2(__hsub2(*reinterpret_cast<half2*>(&bits), bias), sc2);
                }
                const unsigned* B = reinterpret_cast<const unsigned*>(b);
                const int kbase   = g * kGroupK + u * 8;

#pragma unroll
                for (int tile = 0; tile < kTiles; ++tile) {
                    const int row = tile * S::kRowsPerTile + r;
                    half2 a[4];
                    if (row < t) {
                        const Activation* xrow = x + static_cast<std::int64_t>(row) * k + kbase;
                        const uint4 raw = *reinterpret_cast<const uint4*>(xrow);
                        const auto* src = reinterpret_cast<const Activation*>(&raw);
                        __half tmp[8];
                        if constexpr (std::is_same_v<Activation, half>) {
#pragma unroll
                            for (int j = 0; j < 8; ++j) { tmp[j] = src[j]; }
                        } else {
#pragma unroll
                            for (int j = 0; j < 8; ++j) {
                                tmp[j] = __float2half(__bfloat162float(src[j]));
                            }
                        }
#pragma unroll
                        for (int j = 0; j < 4; ++j) {
                            a[j] = *reinterpret_cast<const half2*>(tmp + 2 * j);
                        }
                    } else {
#pragma unroll
                        for (int j = 0; j < 4; ++j) { a[j] = __half2half2(__ushort_as_half(0)); }
                    }
                    const unsigned* A = reinterpret_cast<const unsigned*>(a);
                    volta_mma_qp_n(c[tile], A[0], A[1], B[0], B[1]); // k slice 0
                    volta_mma_qp_n(c[tile], A[2], A[3], B[2], B[3]); // k slice 1
                }
            }
        }
    }

    // C map (v100-skinny mma8_probe.cu, roles swapped); see the Q4 sibling.
#pragma unroll
    for (int tile = 0; tile < kTiles; ++tile) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int row = (i & 2) | ((lane & 16) != 0 ? 4 : 0) | (lane & 1);
            const int cl  = (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
            cs[warp][(tile * S::kRowsPerTile + row) * S::kColsPerCta + qp * 8 + cl] = c[tile][i];
        }
    }
    __syncthreads(); // the only barrier: cross-warp K reduce

    constexpr int kOut = kTiles * S::kRowsPerTile * S::kColsPerCta;
    for (int e = static_cast<int>(threadIdx.x); e < kOut; e += SPLITK * 32) {
        const int row  = e / S::kColsPerCta;
        const int cl   = e % S::kColsPerCta;
        const int ocol = static_cast<int>(blockIdx.x) * S::kColsPerCta + cl;
        if (row < t && ocol < n) {
            float v = 0.0f;
#pragma unroll
            for (int w = 0; w < SPLITK; ++w) { v += cs[w][e]; }
            if constexpr (kDynamicConvAdd) {
                // Preserve the old materialized projection boundary exactly: convolution sees
                // BF16-rounded current and predecessor projections, even though both live in
                // this CTA's reduction tile. Width is request-local, so row%width guards against
                // carrying the second tap across batched request boundaries.
                constexpr int kHidden = 5120;
                constexpr int kGroups = 320;
                const int group = ocol / 16;
                const float projected0 = __bfloat162float(__float2bfloat16(v));
                const float base0 = __bfloat162float(base[ocol + 2LL * kHidden]);
                const float delta0 = __bfloat162float(
                    finish_delta[group + static_cast<std::int64_t>(kGroups) * (2 * row)]);
                float acc = (base0 + delta0) * projected0;
                if ((row % width) != 0) {
                    float previous = 0.0f;
#pragma unroll
                    for (int w = 0; w < SPLITK; ++w) {
                        previous += cs[w][(row - 1) * S::kColsPerCta + cl];
                    }
                    const float projected1 = __bfloat162float(__float2bfloat16(previous));
                    const float base1 = __bfloat162float(base[ocol + 3LL * kHidden]);
                    const float delta1 = __bfloat162float(finish_delta[
                        group + static_cast<std::int64_t>(kGroups) * (1 + 2 * row)]);
                    acc += (base1 + delta1) * projected1;
                }
                const std::int64_t offset = ocol + static_cast<std::int64_t>(row) * kHidden;
                residual[offset] = __float2bfloat16_rn(__bfloat162float(residual[offset]) + acc);
            } else if constexpr (kSwiGluUp) {
                const std::int64_t offset = static_cast<std::int64_t>(row) * out_ld + ocol;
                const float gate = __bfloat162float(out[offset]);
                const float up = __bfloat162float(__float2bfloat16(v));
                out[offset] = __float2bfloat16_rn(silu(gate) * up);
            } else if constexpr (kEpilogue == W8QpnEpilogue::Split3) {
                constexpr int kQ = 4096;
                constexpr int kKv = 1024;
                if (ocol < kQ) {
                    out[static_cast<std::int64_t>(row) * kQ + ocol] = __float2bfloat16(v);
                } else if (ocol < kQ + kKv) {
                    out1[static_cast<std::int64_t>(row) * kKv + ocol - kQ] = __float2bfloat16(v);
                } else {
                    out2[static_cast<std::int64_t>(row) * kKv + ocol - kQ - kKv] =
                        __float2bfloat16(v);
                }
            } else {
                out[static_cast<std::int64_t>(row) * out_ld + ocol] = __float2bfloat16(v);
            }
        }
    }
}

#endif // sm_70

} // namespace ninfer::ops::detail
