#pragma once

// Fused-dequant Q4 x BF16 GEMM on Volta tensor cores, quadpair-split-N form (sm_70 only).
//
// The companion kernel (q4_volta_mma_gemm.cuh) maps T to mma.sync.m8n8k4's 32-row A axis and N to
// its 8-row B axis, giving a 32x8 warp tile. That is right at T=24-32 and badly wrong below: at
// T=4 and T=8 it pads 28 and 24 of its 32 A rows with zeros, so 87% / 75% of every MMA is wasted
// work. Those are exactly the widths MTP3 produces at C1 and C2, which is why the fused route is
// not selected there at all and those points have never moved.
//
// This kernel is the other assignment of the same instruction. m8n8k4 is issued per *quadpair*:
// one warp instruction is four independent 8x8x4 MMAs. The companion kernel gives all four
// quadpairs the same B (the I-major-mirrored replication volta_load_k implements) and splits the
// A rows between them. Here the four quadpairs split N instead -- quadpair q owns output rows
// q*8..q*8+7 -- and all four share the same 8x4 activation tile, because the A fragment map
// depends only on lane position *inside* the quadpair. So the warp tile is 8(T) x 32(N):
//
//     T:                4      8     16
//     32x8 A-row use:  12%    25%    50%
//     8x32  A-row use: 50%   100%    100% (two 8-row tiles)
//
// and activation traffic per weight byte drops 4x, because one A fragment feeds four MMAs
// instead of one.
//
// The design follows the production QPN mapping: fragment positions determine the operand layout.
// The maps below are byte-verified against an independent host oracle; do not re-derive them from
// an assumed row-major interpretation.
//
// Three things follow from the mapping and are load-bearing:
//
//   - No shared memory and no barrier in the main loop. Activations are read straight from global
//     (x is KB-scale and L1/L2-resident, and quadpair-sibling lanes hit the same line), weights
//     stream global -> register. The single __syncthreads() is the cross-warp K reduction at the
//     output, which also replaces the split-K workspace, memset, atomics and narrowing pass the
//     companion kernel needs: the CTA's four warps split K so the grid can stay at N/32.
//   - Q4's stored nibble order already IS the B-fragment order. Decoding one code byte as
//     ((b & 0x0f) | ((b & 0xf0) << 12)) puts weights k and k+1 in one half2, and a
//     mma.sync.m8n8k4 slice consumes exactly two such half2 as its four k. v100-skinny pays 8
//     pack instructions per group to rebuild adjacent-k pairs from its NVFP4 nibble order; this
//     costs nothing here.
//   - One fp16 group scale is held in a register across exactly its group's 16 MMA slices
//     (Q4's kGroupK = 64 = 16 slices x 4 k).
//
// The one cost v100-skinny does not pay: its activations are already FP16, while ninfer's are
// BF16, so every lane converts the 64 activations of its row per group. Quadpair siblings convert
// the same values, so that work is 4x redundant -- the price of keeping activations out of shared
// memory.

#include "ops/common/volta_mma.cuh"
#include "ops/common/score_id_order.cuh"
#include "ops/linear/q4/q4_rowsplit_storage.cuh"
#include "ops/linear/volta_fp16_stage.h"
#include "ops/linear/volta_tc_skinny.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <mma.h>

#include <cstdint>
#include <type_traits>

namespace ninfer::ops::detail {

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ == 700

// Register-only 32-element descending warp bitonic sort (lane 0 = largest key).
__device__ __forceinline__ std::uint64_t warp_bitonic_sort32_desc(std::uint64_t v, int lane) {
#pragma unroll
    for (int k = 2; k <= 32; k <<= 1) {
#pragma unroll
        for (int j = k >> 1; j > 0; j >>= 1) {
            const std::uint64_t other = __shfl_xor_sync(0xffffffffu, v, j);
            const bool want_max       = ((lane & k) == 0) == ((lane & j) == 0);
            v                         = want_max ? (v > other ? v : other) : (v < other ? v : other);
        }
    }
    return v;
}

// Register-only exact top-16 descending of 64 keys (2 per lane); lanes 0..15 hold the sorted top-16.
__device__ __forceinline__ std::uint64_t
warp_bitonic_top16_of_64(std::uint64_t v0, std::uint64_t v1, int lane) {
    const std::uint64_t a     = warp_bitonic_sort32_desc(v0, lane);
    const std::uint64_t b     = warp_bitonic_sort32_desc(v1, lane);
    const std::uint64_t b_rev = __shfl_sync(0xffffffffu, b, 31 - lane);
    // a[0..15] descending concatenated with b[15..0] ascending is a 32-element bitonic sequence.
    std::uint64_t m = (lane < 16) ? a : b_rev;
#pragma unroll
    for (int j = 16; j > 0; j >>= 1) {
        const std::uint64_t other = __shfl_xor_sync(0xffffffffu, m, j);
        const bool want_max       = ((lane & j) == 0);
        m                         = want_max ? (m > other ? m : other) : (m < other ? m : other);
    }
    return m;
}

struct Q4VoltaQpnSchedule {
    static constexpr int kWarps    = 4;  // warps per CTA; they split K, not N
    static constexpr int kColsPerCta = 32; // output rows per CTA (mma's N axis)
    static constexpr int kRowsPerTile = 8; // tokens per A tile (mma's M axis)
    static constexpr int kThreads  = kWarps * 32;
};

// `kTiles` is the number of 8-row A tiles, so T <= 8 * kTiles. The two-tile form decodes each
// weight group once and feeds both A tiles from it, so T=9..16 costs close to T=8 rather than
// double.
//
// `kBlk` is how many Q4 groups a lane reads before consuming any of them; kBlk * kCodeB bytes.
// It trades registers for read efficiency and the right value depends on n -- see the launcher.
//
// `kPrepacked` reads the load-time order of q4_prepack_sm70.cu: codes [tile][group][lane][32 B]
// and scales [tile][group][lane], so a warp's 32 lanes read one contiguous 1 KB line per group
// instead of 32 rows a row pitch apart (1.4x on the 131,072-row DFlash2 proposal head).
template <int kTiles, int kBlk, bool kProduceTopK = false, bool kPrepacked = false,
          typename Activation = __nv_bfloat16>
__global__ __launch_bounds__(Q4VoltaQpnSchedule::kThreads,
                             kTiles == 1 ? 8 : (kTiles == 2 ? 6 : 4)) void
q4_volta_qpn_gemm_kernel(
    const std::uint8_t* __restrict__ codes, const std::uint8_t* __restrict__ scales,
    const Activation* __restrict__ x, __nv_bfloat16* __restrict__ out, int n, int k, int t,
    int padded_groups, int out_ld, const std::int32_t* __restrict__ row_to_global_ids = nullptr,
    std::uint64_t* __restrict__ partial_keys = nullptr, int producer_groups = 0) {
    using S = Q4VoltaQpnSchedule;
    constexpr int kGroupK = Q4RowSplitStorage::kGroupK;
    constexpr int kCodeB  = Q4RowSplitStorage::kCodeBytesPerGroup;

    __shared__ float cs[S::kWarps][kTiles * S::kRowsPerTile * S::kColsPerCta];

    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    // Quadpair index, and this lane's position inside it. `r` is both the A row (token) and the
    // B column local to the quadpair -- that identity is what makes quadpair siblings share A.
    const int qp = (lane >> 2) & 3;
    const int r  = (lane & 3) + ((lane & 16) != 0 ? 4 : 0);

    const int col  = static_cast<int>(blockIdx.x) * S::kColsPerCta + qp * 8 + r;
    const int good = col < n;

    const int groups = k / kGroupK;
    const int gq     = groups / S::kWarps;
    const int g0     = warp * gq;
    const int gend   = (warp == S::kWarps - 1) ? groups : g0 + gq;

    // Per-group strides: one row's adjacent groups (row-major) or coalesced half-split prepacked
    // [tile][group][half(0..1)][lane(0..31)][16B] (1024B/group, 512B between halves).
    constexpr std::int64_t kScaleGroupStride = kPrepacked ? 32 : 1;
    const std::uint8_t* crow =
        kPrepacked ? codes + (static_cast<std::int64_t>(blockIdx.x) * padded_groups * 64 + lane) *
                                 16
                   : codes + static_cast<std::int64_t>(good ? col : 0) * padded_groups * kCodeB;
    const std::uint16_t* srow =
        kPrepacked ? reinterpret_cast<const std::uint16_t*>(scales) +
                         static_cast<std::int64_t>(blockIdx.x) * padded_groups * 32 + lane
                   : reinterpret_cast<const std::uint16_t*>(
                         scales + static_cast<std::int64_t>(good ? col : 0) * padded_groups *
                                      Q4RowSplitStorage::kScaleBytesPerGroup);

    float c[kTiles][8];
#pragma unroll
    for (int tile = 0; tile < kTiles; ++tile) {
#pragma unroll
        for (int i = 0; i < 8; ++i) { c[tile][i] = 0.0f; }
    }

    const half2 bias = __half2half2(__ushort_as_half(0x6408)); // 1032.0

    // Weight loads are the DRAM traffic and they are scattered: lane L owns output row
    // qp*8 + r, so a warp's 32 lanes read 32 *different* rows at the same group offset, strided
    // by the row pitch. Nothing coalesces, so each is a separate long-latency request -- and with
    // the loads issued immediately before the MMAs that consume them, ncu measured 25.53 of the
    // warp-cycles-per-issued-instruction in long_scoreboard, with the SM retiring 0.25
    // instructions per cycle. So the group loop issues a block of weight reads up front and then
    // runs their MMA slices while the rest are still in flight. It is blocked by
    // kBlk groups for a second reason: a lane streams its own weight row sequentially, but a
    // warp's 32 lanes stream 32 *different* rows at once, and at 640 resident CTAs that is ~82k
    // concurrent streams whose 128B lines would all have to stay resident in a 6 MB L2 to be
    // reused. They cannot -- at n=34816 DRAM read measured 2.44x the weight with L2 at 38%,
    // against 1.05x and 64% at n=6144, which is where 128B x streams stops fitting. Consuming a
    // whole line per lane per iteration means each line is touched once and fully used, so
    // residency stops mattering, and it puts 2*kBlk loads in flight to cover their latency.
    // Double-buffering the blocks on top of this does not work: at kBlk=4 the two live blocks are
    // 16 uint4 = 64 registers of weight data alone, against the 64-register budget
    // __launch_bounds__(128, 8) allows, so it spills and costs 1.6x (n=34816 T=5: 327.7us ->
    // 531.4). The block itself is the latency hiding.
    uint4 cw[2 * kBlk];
    std::uint16_t scv[kBlk];

    for (int gb = g0; gb < gend; gb += kBlk) {
        const int blk  = min(kBlk, gend - gb);
        const int last = gend - 1;
#pragma unroll
        for (int e = 0; e < kBlk; ++e) {
            const int g = min(gb + e, last);
            if constexpr (kPrepacked) {
                const std::uint8_t* p = crow + static_cast<std::int64_t>(g) * 1024;
                cw[2 * e]             = __ldg(reinterpret_cast<const uint4*>(p));
                cw[2 * e + 1]         = __ldg(reinterpret_cast<const uint4*>(p + 512));
            } else {
                const std::uint8_t* p = crow + static_cast<std::int64_t>(g) * kCodeB;
                cw[2 * e]             = __ldg(reinterpret_cast<const uint4*>(p));
                cw[2 * e + 1]         = __ldg(reinterpret_cast<const uint4*>(p + 16));
            }
            scv[e] = good ? srow[static_cast<std::int64_t>(g) * kScaleGroupStride]
                          : static_cast<std::uint16_t>(0);
        }

    for (int e = 0; e < blk; ++e) {
        const int g     = gb + e;
        const half2 sc2 = __half2half2(__ushort_as_half(scv[e]));
        // 32 code bytes = one Q4 group = 64 k-elements.
        const std::uint32_t words[8] = {cw[2 * e].x,     cw[2 * e].y,     cw[2 * e].z,
                                        cw[2 * e].w,     cw[2 * e + 1].x, cw[2 * e + 1].y,
                                        cw[2 * e + 1].z, cw[2 * e + 1].w};

#pragma unroll
        for (int w = 0; w < 8; ++w) {
            // One uint32 = 4 code bytes = 8 adjacent k. Byte j decodes straight into
            // half2(k=2j, k=2j+1), which is already the B-fragment register pair.
            const std::uint32_t p4 = words[w] ^ 0x88888888u;
            half2 b[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const std::uint32_t byte = (p4 >> (8 * j)) & 0xffu;
                std::uint32_t bits = ((byte & 0x0fu) | ((byte & 0xf0u) << 12)) | 0x64006400u;
                b[j] = __hmul2(__hsub2(*reinterpret_cast<half2*>(&bits), bias), sc2);
            }
            const unsigned* B = reinterpret_cast<const unsigned*>(b);
            const int kbase   = g * kGroupK + w * 8;

#pragma unroll
            for (int tile = 0; tile < kTiles; ++tile) {
                const int row = tile * S::kRowsPerTile + r;
                half2 a[4];
                if constexpr (std::is_same_v<Activation, VoltaGroupMajorHalf>) {
                    const int group16 = g * 4 + (w >> 1);
                    const int k_half  = w & 1;
                    const half* xrow =
                        reinterpret_cast<const half*>(x) +
                        (((static_cast<std::int64_t>(group16) * kTiles + tile) * 2 + k_half) * 8 +
                         r) *
                            8;
                    const uint4 raw = __ldg(reinterpret_cast<const uint4*>(xrow));
                    a[0]            = *reinterpret_cast<const half2*>(&raw.x);
                    a[1]            = *reinterpret_cast<const half2*>(&raw.y);
                    a[2]            = *reinterpret_cast<const half2*>(&raw.z);
                    a[3]            = *reinterpret_cast<const half2*>(&raw.w);
                } else if (row < t) {
                    const Activation* xrow = x + static_cast<std::int64_t>(row) * k + kbase;
                    if constexpr (std::is_same_v<Activation, half>) {
                        const uint4 raw = __ldg(reinterpret_cast<const uint4*>(xrow));
                        a[0]            = *reinterpret_cast<const half2*>(&raw.x);
                        a[1]            = *reinterpret_cast<const half2*>(&raw.y);
                        a[2]            = *reinterpret_cast<const half2*>(&raw.z);
                        a[3]            = *reinterpret_cast<const half2*>(&raw.w);
                    } else {
                        const uint4 raw = *reinterpret_cast<const uint4*>(xrow);
                        const auto* src = reinterpret_cast<const __nv_bfloat16*>(&raw);
                        __half tmp[8];
#pragma unroll
                        for (int j = 0; j < 8; ++j) {
                            tmp[j] = __float2half(__bfloat162float(src[j]));
                        }
#pragma unroll
                        for (int j = 0; j < 4; ++j) {
                            a[j] = *reinterpret_cast<const half2*>(tmp + 2 * j);
                        }
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

    // C map (v100-skinny mma8_probe.cu, roles swapped): register i of lane L holds
    //   A row  (i & 2) | ((L & 16) ? 4 : 0) | (L & 1)
    //   B col  (i & 1) | (((L >> 1) & 1) << 1) | ((i >> 2) << 2)   -- quadpair-local, + qp * 8
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

    if constexpr (!kProduceTopK) {
        constexpr int kOut = kTiles * S::kRowsPerTile * S::kColsPerCta;
        for (int e = static_cast<int>(threadIdx.x); e < kOut; e += S::kThreads) {
            const int row  = e / S::kColsPerCta;
            const int cl   = e % S::kColsPerCta;
            const int ocol = static_cast<int>(blockIdx.x) * S::kColsPerCta + cl;
            if (row < t && ocol < n) {
                float v = 0.0f;
#pragma unroll
                for (int w = 0; w < S::kWarps; ++w) { v += cs[w][e]; }
                out[static_cast<std::int64_t>(row) * out_ld + ocol] = __float2bfloat16(v);
            }
        }
    } else {
        static_assert(kTiles >= 1 && kTiles <= 4,
                      "the DFlash2 fused top-K route supports T <= 32 (kTiles 1..4)");
        // The QPN CTA already owns exactly 32 vocabulary rows. Reduce its four K partitions,
        // reproduce the materialized BF16-logit rounding boundary, sort the 32 order keys in a
        // warp, and emit the best 16 directly. The ordinary merge kernels combine CTA winners.
        const int producer = static_cast<int>(blockIdx.x);
        const int ocol     = producer * S::kColsPerCta + lane;
        const int gid      = ocol < n ? row_to_global_ids[ocol] : 0;
        for (int row = warp; row < t; row += S::kWarps) {
            float v = 0.0f;
#pragma unroll
            for (int w = 0; w < S::kWarps; ++w) {
                v += cs[w][row * S::kColsPerCta + lane];
            }
            std::uint64_t key = 0ull;
            if (ocol < n) {
                const float rounded = __bfloat162float(__float2bfloat16(v));
                key                 = score_id_order_key(rounded, gid);
            }
            const std::uint64_t sorted = warp_bitonic_sort32_desc(key, lane);
            if (lane < 16) {
                const std::int64_t offset =
                    (static_cast<std::int64_t>(row) * producer_groups + producer) * 16 + lane;
                partial_keys[offset] = sorted;
            }
        }
    }
}

// Volta WMMA 16x16x16 fused top-16 kernel for the prepacked Q4 proposal head at T=9..64.
// Each 128-thread CTA decodes 64 rows x 64 K once into double-buffered shared memory (2 register
// prefetch stages), multiplies across all T<=16*TT columns in a single weight pass, and emits
// exact top-16 keys via a register-only 64->16 warp bitonic network (halving producer groups from
// 4,096 to 2,048).
template <int TT, class Activation>
__global__ __launch_bounds__(128, 2) void q4_volta_tc_topk_kernel(
    const std::uint8_t* __restrict__ codes, const std::uint8_t* __restrict__ scales,
    const Activation* __restrict__ x, int n, int k, int t, int x_tiles, int padded_groups,
    const std::int32_t* __restrict__ row_to_global_ids, std::uint64_t* __restrict__ partial_keys,
    int producer_groups) {
    using namespace nvcuda;
    constexpr int ROWS    = 64;
    constexpr int KC      = 64;
    constexpr int LDS     = KC + 8; // 72 halves per row
    constexpr int TCOLS   = 16 * TT;
    constexpr int THREADS = 128;
    constexpr int B_TOTAL = TCOLS * (KC / 8);
    constexpr int B_VEC   = (B_TOTAL + THREADS - 1) / THREADS;

    extern __shared__ __align__(32) unsigned char q4_tc_smem[];
    half* a_s = reinterpret_cast<half*>(q4_tc_smem);
    half* b_s = a_s + 2 * ROWS * LDS;

    const int tid          = static_cast<int>(threadIdx.x);
    const int warp         = tid >> 5;
    const int lane         = tid & 31;
    const int local_tile   = tid >> 6;      // 0 or 1
    const int half_idx     = (tid >> 5) & 1; // 0 or 1
    const int lane_w       = tid & 31;
    const int tile         = static_cast<int>(blockIdx.x) * 2 + local_tile;
    const int w_row_in_cta = local_tile * 32 + volta_tc_lane_row(lane_w);
    const int chunks       = k / KC;
    const half2 bias       = __half2half2(__ushort_as_half(0x6408)); // 1032.0

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[TT];
#pragma unroll
    for (int j = 0; j < TT; ++j) { wmma::fill_fragment(acc[j], 0.0f); }

    uint4 q0, q1;
    std::uint16_t s0, s1;
    uint4 bx0[B_VEC], bx1[B_VEC];

    const auto load = [&](int chunk, uint4& q, std::uint16_t& s, uint4 (&bx)[B_VEC]) {
        const std::int64_t tg = static_cast<std::int64_t>(tile) * padded_groups + chunk;
        q = __ldg(reinterpret_cast<const uint4*>(codes) + tg * 64 + half_idx * 32 + lane_w);
        s = __ldg(reinterpret_cast<const std::uint16_t*>(scales) + tg * 32 + lane_w);
#pragma unroll
        for (int v = 0; v < B_VEC; ++v) {
            const int index = v * THREADS + tid;
            const int col   = index / (KC / 8);
            const int part  = index % (KC / 8);
            bx[v]           = (index < B_TOTAL && col < t)
                                  ? volta_tc_load_activation(x, k, col, chunk * KC + part * 8, x_tiles)
                                  : make_uint4(0u, 0u, 0u, 0u);
        }
    };

    const auto store = [&](int buf, const uint4& q, std::uint16_t s, const uint4 (&bx)[B_VEC]) {
        const half2 sc2            = __half2half2(__ushort_as_half(s));
        const std::uint32_t w4[4]  = {q.x, q.y, q.z, q.w};
        half* dst_row              = a_s + buf * ROWS * LDS + w_row_in_cta * LDS + half_idx * 32;
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            const std::uint32_t p4 = w4[w] ^ 0x88888888u;
            half2 b[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const std::uint32_t byte = (p4 >> (8 * j)) & 0xffu;
                std::uint32_t bits = ((byte & 0x0fu) | ((byte & 0xf0u) << 12)) | 0x64006400u;
                b[j] = __hmul2(__hsub2(*reinterpret_cast<half2*>(&bits), bias), sc2);
            }
            *reinterpret_cast<uint4*>(dst_row + w * 8) = *reinterpret_cast<const uint4*>(b);
        }
#pragma unroll
        for (int v = 0; v < B_VEC; ++v) {
            const int index = v * THREADS + tid;
            if (index < B_TOTAL) {
                const int col  = index / (KC / 8);
                const int part = index % (KC / 8);
                *reinterpret_cast<uint4*>(b_s + buf * TCOLS * LDS + col * LDS + part * 8) = bx[v];
            }
        }
    };

    const auto compute = [&](int buf) {
        const half* a_buf = a_s + buf * ROWS * LDS + warp * 16 * LDS;
        const half* b_buf = b_s + buf * TCOLS * LDS;
#pragma unroll
        for (int kk = 0; kk < KC / 16; ++kk) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
            wmma::load_matrix_sync(a, a_buf + kk * 16, LDS);
#pragma unroll
            for (int j = 0; j < TT; ++j) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b;
                wmma::load_matrix_sync(b, b_buf + j * 16 * LDS + kk * 16, LDS);
                wmma::mma_sync(acc[j], a, b, acc[j]);
            }
        }
    };

    if (chunks > 0) {
        load(0, q0, s0, bx0);
        store(0, q0, s0, bx0);
    }
    if (chunks > 1) { load(1, q0, s0, bx0); }
    if (chunks > 2) { load(2, q1, s1, bx1); }
    __syncthreads();
    for (int c = 0; c < chunks; c += 2) {
        compute(0);
        if (c + 1 < chunks) { store(1, q0, s0, bx0); }
        if (c + 3 < chunks) { load(c + 3, q0, s0, bx0); }
        __syncthreads();
        if (c + 1 >= chunks) { break; }
        compute(1);
        if (c + 2 < chunks) { store(0, q1, s1, bx1); }
        if (c + 4 < chunks) { load(c + 4, q1, s1, bx1); }
        __syncthreads();
    }

    // Reuse smem as float c_s[TCOLS][64] so each column's 64 rows are contiguous.
    float* c_s = reinterpret_cast<float*>(q4_tc_smem);
#pragma unroll
    for (int j = 0; j < TT; ++j) {
        wmma::store_matrix_sync(c_s + j * 16 * ROWS + warp * 16, acc[j], ROWS, wmma::mem_col_major);
    }
    __syncthreads();

    const int producer = static_cast<int>(blockIdx.x);
    const int row0     = producer * ROWS;
    const int gid0     = (row0 + lane < n) ? row_to_global_ids[row0 + lane] : 0;
    const int gid1     = (row0 + lane + 32 < n) ? row_to_global_ids[row0 + lane + 32] : 0;
    for (int col = warp; col < t; col += 4) {
        const float v0       = c_s[col * ROWS + lane];
        const float v1       = c_s[col * ROWS + lane + 32];
        const std::uint64_t k0 = (row0 + lane < n)
                                     ? score_id_order_key(__bfloat162float(__float2bfloat16(v0)), gid0)
                                     : 0ull;
        const std::uint64_t k1 =
            (row0 + lane + 32 < n)
                ? score_id_order_key(__bfloat162float(__float2bfloat16(v1)), gid1)
                : 0ull;
        const std::uint64_t top = warp_bitonic_top16_of_64(k0, k1, lane);
        if (lane < 16) {
            const std::int64_t offset =
                (static_cast<std::int64_t>(col) * producer_groups + producer) * 16 + lane;
            partial_keys[offset] = top;
        }
    }
}

#endif // sm_70

} // namespace ninfer::ops::detail
