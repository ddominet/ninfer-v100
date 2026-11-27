#pragma once

// Coalesced RowSplit Volta (sm_70) streaming Tensor-Core (mma.sync.m8n8k4) GEMM for Q4, Q5, and Q6.
//
// Key Volta microarchitectural invariants:
// 1. ZERO local-memory stack frames: software-pipelined scalar `cur`/`nxt` (or ping-pong `gate`/`up`
//    for SwiGLU) keeps every weight carry in registers with 100% overlapped HBM2 prefetch.
// 2. ZERO L1 cache replay storms: activations X[T, 128] for each 128-K step are cooperatively
//    loaded with 100% coalesced 256-byte row transactions into bank-conflict-free shared memory
//    (`stride = 136 halfs = 68 banks`, 0 bank conflicts across r=0..7) and pre-converted to FP16.
// 3. 100% coalesced weight loads: every 4 consecutive lanes read 64 contiguous code bytes (2 full
//    32B HBM2/L2 sectors = 2 RowSplit groups = 128 K), transposed to Volta quadpair layout via a
//    register-only warp shuffle (`src_lane = (r << 2) | qp`).
// 4. Exact FP16 Tensor Core arithmetic with FP32 accumulation (zero activation quantization noise).

#include "ops/common/volta_mma.cuh"
#include "ops/linear/q4/q4_rowsplit_storage.cuh"
#include "ops/linear/q5/q5_rowsplit_storage.cuh"
#include "ops/linear/q6/q6_rowsplit_storage.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <type_traits>

namespace ninfer::ops::detail {

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ == 700

enum class RowSplitStreamFormat : int {
    Q4 = 4,
    Q5 = 5,
    Q6 = 6,
};

constexpr int kStreamXStride = 136; // 128 + 8 halfs: 272 B = 68 banks -> 0 conflicts across r=0..7

__device__ __forceinline__ unsigned long long stream_bf16_argmax_key(float value,
                                                                     std::int32_t row) {
    const __nv_bfloat16 bf = __float2bfloat16_rn(value);
    std::uint16_t bits     = __bfloat16_as_ushort(bf);
    if ((bits & 0x7FFFu) > 0x7F80u) { return 0ULL; }
    if (bits == 0x8000u) { bits = 0x0000u; }
    const std::uint32_t ordered = (bits & 0x8000u) == 0
                                      ? (static_cast<std::uint32_t>(bits) ^ 0x8000u)
                                      : (static_cast<std::uint32_t>(~bits) & 0xFFFFu);
    return (static_cast<unsigned long long>(ordered) << 32) |
           static_cast<unsigned long long>(~static_cast<std::uint32_t>(row));
}

__device__ __forceinline__ half2 stream_bf16_pair_to_half2(std::uint32_t bf16x2_bits) {
    const float f0 = __uint_as_float(bf16x2_bits << 16);
    const float f1 = __uint_as_float(bf16x2_bits & 0xffff0000u);
    return __floats2half2_rn(f0, f1);
}

template <typename Activation>
__device__ __forceinline__ uint4 stream_load_act8_as_fp16(const Activation* __restrict__ src) {
    const uint4 raw = __ldg(reinterpret_cast<const uint4*>(src));
    if constexpr (std::is_same_v<Activation, half>) {
        return raw;
    } else {
        half2 h[4];
        h[0] = stream_bf16_pair_to_half2(raw.x);
        h[1] = stream_bf16_pair_to_half2(raw.y);
        h[2] = stream_bf16_pair_to_half2(raw.z);
        h[3] = stream_bf16_pair_to_half2(raw.w);
        return *reinterpret_cast<const uint4*>(h);
    }
}

// 128-thread CTA cooperative load of [kTiles*8, 128] activations into x_sh (1 uint4/thread for T<=8).
template <int kTiles, typename Activation>
__device__ __forceinline__ void stream_stage_x_cta(const Activation* __restrict__ x,
                                                   half (&x_sh)[kTiles * 8][kStreamXStride], int k,
                                                   int t, int col_offset, int step128, int tid) {
    constexpr int kVecs = kTiles * 8 * 16; // 16 uint4s (128 elems) per token row
#pragma unroll
    for (int idx = tid; idx < kVecs; idx += 128) {
        const int local_t = idx >> 4;
        const int vec     = idx & 15;
        const int tok     = col_offset + local_t;
        const uint4 v     = (tok < t)
                                ? stream_load_act8_as_fp16(x + static_cast<std::int64_t>(tok) * k +
                                                           step128 * 128 + vec * 8)
                                : make_uint4(0u, 0u, 0u, 0u);
        *reinterpret_cast<uint4*>(&x_sh[local_t][vec * 8]) = v;
    }
}

// 32-thread warp cooperative load of [kTiles*8, 128] activations into per-warp x_sh (0 CTA barriers).
template <int kTiles, typename Activation>
__device__ __forceinline__ void stream_stage_x_warp(const Activation* __restrict__ x,
                                                    half (&x_sh)[kTiles * 8][kStreamXStride], int k,
                                                    int t, int col_offset, int step128, int lane) {
    constexpr int kVecs = kTiles * 8 * 16;
#pragma unroll
    for (int idx = lane; idx < kVecs; idx += 32) {
        const int local_t = idx >> 4;
        const int vec     = idx & 15;
        const int tok     = col_offset + local_t;
        const uint4 v     = (tok < t)
                                ? stream_load_act8_as_fp16(x + static_cast<std::int64_t>(tok) * k +
                                                           step128 * 128 + vec * 8)
                                : make_uint4(0u, 0u, 0u, 0u);
        *reinterpret_cast<uint4*>(&x_sh[local_t][vec * 8]) = v;
    }
    __syncwarp();
}

template <RowSplitStreamFormat Fmt>
struct RowSplitStreamDecoder;

template <>
struct RowSplitStreamDecoder<RowSplitStreamFormat::Q4> {
    struct Carry {
        uint4 code;
        std::uint16_t scale;
    };

    __device__ __forceinline__ static void load(const std::uint8_t* __restrict__ codes,
                                                const std::uint8_t* __restrict__,
                                                const std::uint16_t* __restrict__ scales,
                                                std::int64_t row_idx, bool row_valid,
                                                int groups_total, int step128, int load_chunk,
                                                Carry& out) {
        if (row_valid) {
            const int g = step128 * 2;
            const std::int64_t code_off =
                (row_idx * groups_total + g) * Q4RowSplitStorage::kCodeBytesPerGroup +
                load_chunk * 16;
            out.code  = __ldg(reinterpret_cast<const uint4*>(codes + code_off));
            out.scale = __ldg(scales + row_idx * groups_total + g + (load_chunk >> 1));
        } else {
            out.code  = make_uint4(0u, 0u, 0u, 0u);
            out.scale = 0u;
        }
    }

    __device__ __forceinline__ static void shuffle_to_mma(const Carry& in, int src_lane,
                                                          Carry& out) {
        out.code.x = __shfl_sync(0xffffffffu, in.code.x, src_lane);
        out.code.y = __shfl_sync(0xffffffffu, in.code.y, src_lane);
        out.code.z = __shfl_sync(0xffffffffu, in.code.z, src_lane);
        out.code.w = __shfl_sync(0xffffffffu, in.code.w, src_lane);
        out.scale  = static_cast<std::uint16_t>(
            __shfl_sync(0xffffffffu, static_cast<std::uint32_t>(in.scale), src_lane));
    }

    __device__ __forceinline__ static void decode_word(const Carry& mma_w, int w_idx, half2 sc2,
                                                       half2 (&b)[4]) {
        const std::uint32_t raw = w_idx == 0 ? mma_w.code.x
                                  : (w_idx == 1 ? mma_w.code.y
                                                : (w_idx == 2 ? mma_w.code.z : mma_w.code.w));
        const std::uint32_t p4  = raw ^ 0x88888888u;
        const half2 bias        = __half2half2(__ushort_as_half(0x6408)); // 1032.0
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const std::uint32_t lb = (p4 >> (8 * j)) & 0xffu;
            std::uint32_t bits     = ((lb & 0x0fu) | ((lb & 0xf0u) << 12)) | 0x64006400u;
            b[j]                   = __hmul2(__hsub2(*reinterpret_cast<half2*>(&bits), bias), sc2);
        }
    }
};

template <>
struct RowSplitStreamDecoder<RowSplitStreamFormat::Q5> {
    struct Carry {
        uint4 code;
        std::uint32_t high;
        std::uint16_t scale;
    };

    __device__ __forceinline__ static void load(const std::uint8_t* __restrict__ codes,
                                                const std::uint8_t* __restrict__ high,
                                                const std::uint16_t* __restrict__ scales,
                                                std::int64_t row_idx, bool row_valid,
                                                int groups_total, int step128, int load_chunk,
                                                Carry& out) {
        if (row_valid) {
            const int g = step128 * 2;
            const std::int64_t code_off =
                (row_idx * groups_total + g) * Q5RowSplitStorage::kCodeBytesPerGroup +
                load_chunk * 16;
            const std::int64_t high_off =
                (row_idx * groups_total + g) * Q5RowSplitStorage::kHighBytesPerGroup +
                load_chunk * 4;
            out.code  = __ldg(reinterpret_cast<const uint4*>(codes + code_off));
            out.high  = __ldg(reinterpret_cast<const std::uint32_t*>(high + high_off));
            out.scale = __ldg(scales + row_idx * groups_total + g + (load_chunk >> 1));
        } else {
            out.code  = make_uint4(0u, 0u, 0u, 0u);
            out.high  = 0u;
            out.scale = 0u;
        }
    }

    __device__ __forceinline__ static void shuffle_to_mma(const Carry& in, int src_lane,
                                                          Carry& out) {
        out.code.x = __shfl_sync(0xffffffffu, in.code.x, src_lane);
        out.code.y = __shfl_sync(0xffffffffu, in.code.y, src_lane);
        out.code.z = __shfl_sync(0xffffffffu, in.code.z, src_lane);
        out.code.w = __shfl_sync(0xffffffffu, in.code.w, src_lane);
        out.high   = __shfl_sync(0xffffffffu, in.high, src_lane);
        out.scale  = static_cast<std::uint16_t>(
            __shfl_sync(0xffffffffu, static_cast<std::uint32_t>(in.scale), src_lane));
    }

    __device__ __forceinline__ static void decode_word(const Carry& mma_w, int w_idx, half2 sc2,
                                                       half2 (&b)[4]) {
        const std::uint32_t raw = w_idx == 0 ? mma_w.code.x
                                  : (w_idx == 1 ? mma_w.code.y
                                                : (w_idx == 2 ? mma_w.code.z : mma_w.code.w));
        const std::uint32_t hb  = ((mma_w.high >> (8 * w_idx)) & 0xffu) ^ 0xffu;
        const half2 bias        = __half2half2(__ushort_as_half(0x6410)); // 1040.0
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const std::uint32_t lb = (raw >> (8 * j)) & 0xffu;
            std::uint32_t bits     = ((lb & 0x0fu) | ((lb & 0xf0u) << 12)) | 0x64006400u;
            bits |= (((hb >> (2 * j)) & 1u) << 4) | (((hb >> (2 * j + 1)) & 1u) << 20);
            b[j] = __hmul2(__hsub2(*reinterpret_cast<half2*>(&bits), bias), sc2);
        }
    }
};

template <>
struct RowSplitStreamDecoder<RowSplitStreamFormat::Q6> {
    struct Carry {
        uint4 code;
        uint2 high;
        std::uint16_t scale;
    };

    __device__ __forceinline__ static void load(const std::uint8_t* __restrict__ codes,
                                                const std::uint8_t* __restrict__ high,
                                                const std::uint16_t* __restrict__ scales,
                                                std::int64_t row_idx, bool row_valid,
                                                int groups_total, int step128, int load_chunk,
                                                Carry& out) {
        if (row_valid) {
            const int g = step128 * 2;
            const std::int64_t code_off =
                (row_idx * groups_total + g) * Q6RowSplitStorage::kCodeBytesPerGroup +
                load_chunk * 16;
            const std::int64_t high_off =
                (row_idx * groups_total + g) * Q6RowSplitStorage::kHighBytesPerGroup +
                load_chunk * 8;
            out.code  = __ldg(reinterpret_cast<const uint4*>(codes + code_off));
            out.high  = __ldg(reinterpret_cast<const uint2*>(high + high_off));
            out.scale = __ldg(scales + row_idx * groups_total + g + (load_chunk >> 1));
        } else {
            out.code  = make_uint4(0u, 0u, 0u, 0u);
            out.high  = make_uint2(0u, 0u);
            out.scale = 0u;
        }
    }

    __device__ __forceinline__ static void shuffle_to_mma(const Carry& in, int src_lane,
                                                          Carry& out) {
        out.code.x = __shfl_sync(0xffffffffu, in.code.x, src_lane);
        out.code.y = __shfl_sync(0xffffffffu, in.code.y, src_lane);
        out.code.z = __shfl_sync(0xffffffffu, in.code.z, src_lane);
        out.code.w = __shfl_sync(0xffffffffu, in.code.w, src_lane);
        out.high.x = __shfl_sync(0xffffffffu, in.high.x, src_lane);
        out.high.y = __shfl_sync(0xffffffffu, in.high.y, src_lane);
        out.scale  = static_cast<std::uint16_t>(
            __shfl_sync(0xffffffffu, static_cast<std::uint32_t>(in.scale), src_lane));
    }

    __device__ __forceinline__ static void decode_word(const Carry& mma_w, int w_idx, half2 sc2,
                                                       half2 (&b)[4]) {
        const std::uint32_t raw = w_idx == 0 ? mma_w.code.x
                                  : (w_idx == 1 ? mma_w.code.y
                                                : (w_idx == 2 ? mma_w.code.z : mma_w.code.w));
        const std::uint32_t h32 = (w_idx < 2) ? mma_w.high.x : mma_w.high.y;
        const std::uint32_t h16 = ((h32 >> (16 * (w_idx & 1))) & 0xffffu) ^ 0xaaaau;
        const half2 bias        = __half2half2(__ushort_as_half(0x6420)); // 1056.0
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const std::uint32_t lb = (raw >> (8 * j)) & 0xffu;
            std::uint32_t bits     = ((lb & 0x0fu) | ((lb & 0xf0u) << 12)) | 0x64006400u;
            bits |= (((h16 >> (4 * j)) & 3u) << 4) | (((h16 >> (4 * j + 2)) & 3u) << 20);
            b[j] = __hmul2(__hsub2(*reinterpret_cast<half2*>(&bits), bias), sc2);
        }
    }
};

template <RowSplitStreamFormat Fmt, int kTiles>
__device__ __forceinline__ void stream_compute_step(
    const typename RowSplitStreamDecoder<Fmt>::Carry& cur_load, int src_lane, int qp, int r,
    const half (&x_sh)[kTiles * 8][kStreamXStride], float (&c)[kTiles][8]) {
    using Decoder = RowSplitStreamDecoder<Fmt>;
    typename Decoder::Carry mma_w;
    Decoder::shuffle_to_mma(cur_load, src_lane, mma_w);
    const half2 sc2 = __half2half2(__ushort_as_half(mma_w.scale));
#pragma unroll
    for (int w = 0; w < 4; ++w) {
        half2 b[4];
        Decoder::decode_word(mma_w, w, sc2, b);
        const unsigned* B = reinterpret_cast<const unsigned*>(b);
        const int x_col   = qp * 32 + w * 8;
#pragma unroll
        for (int tile = 0; tile < kTiles; ++tile) {
            const uint4 araw  = *reinterpret_cast<const uint4*>(&x_sh[tile * 8 + r][x_col]);
            const unsigned* A = reinterpret_cast<const unsigned*>(&araw);
            volta_mma_qp_n(c[tile], A[0], A[1], B[0], B[1]);
            volta_mma_qp_n(c[tile], A[2], A[3], B[2], B[3]);
        }
    }
}

struct StreamContiguousEpilogue {
    __nv_bfloat16* out;
    std::int32_t out_ld;

    __device__ __forceinline__ void store(int row, int tok, float value) const {
        out[static_cast<std::int64_t>(tok) * out_ld + row] = __float2bfloat16(value);
    }
};

struct StreamResidualAddEpilogue {
    __nv_bfloat16* out;
    std::int32_t out_ld;

    __device__ __forceinline__ void store(int row, int tok, float value) const {
        const std::int64_t idx = static_cast<std::int64_t>(tok) * out_ld + row;
        out[idx]               = __float2bfloat16(value + __bfloat162float(out[idx]));
    }
};

struct StreamSplitRowEpilogue {
    __nv_bfloat16* out0;
    __nv_bfloat16* out1;
    std::int32_t out0_ld;
    std::int32_t out1_ld;
    std::int32_t split_row;

    __device__ __forceinline__ void store(int row, int tok, float value) const {
        const __nv_bfloat16 bf = __float2bfloat16(value);
        if (row < split_row) {
            out0[static_cast<std::int64_t>(tok) * out0_ld + row] = bf;
        } else {
            out1[static_cast<std::int64_t>(tok) * out1_ld + (row - split_row)] = bf;
        }
    }
};

// Split-K (4 warps split K over 8 rows/CTA): zero CTA barriers inside the K loop, per-warp
// coalesced FP16 x_sh, scalar software-pipelined cur/nxt prefetch (guaranteed 0 stack frame).
template <RowSplitStreamFormat Fmt, int kTiles, typename Activation, typename Epilogue>
__global__ __launch_bounds__(128, kTiles == 1 ? 8 : 4) void volta_rowsplit_stream_hmma_splitk_kernel(
    const std::uint8_t* __restrict__ codes, const std::uint8_t* __restrict__ high,
    const std::uint16_t* __restrict__ scales, const Activation* __restrict__ x, int n, int k, int t,
    int col_offset, Epilogue epilogue) {
    using Decoder        = RowSplitStreamDecoder<Fmt>;
    constexpr int kWarps = 4;

    __shared__ __align__(16) half x_sh[kWarps][kTiles * 8][kStreamXStride];
    __shared__ float cs[kWarps][kTiles * 64];

    const int lane       = static_cast<int>(threadIdx.x) & 31;
    const int warp       = static_cast<int>(threadIdx.x) >> 5;
    const int row0       = static_cast<int>(blockIdx.x) * 8;
    const int load_row   = lane >> 2;
    const int load_chunk = lane & 3;
    const int row_idx    = row0 + load_row;
    const bool row_valid = row_idx < n;

    const int qp       = (lane >> 2) & 3;
    const int r        = (lane & 3) + ((lane & 16) != 0 ? 4 : 0);
    const int src_lane = (r << 2) | qp;

    const int groups_total = k / 64;
    const int steps_total  = k / 128;
    const int steps_per_w  = steps_total / kWarps;
    const int s0           = warp * steps_per_w;
    const int s_end        = (warp == kWarps - 1) ? steps_total : (s0 + steps_per_w);

    float c[kTiles][8];
#pragma unroll
    for (int tile = 0; tile < kTiles; ++tile) {
#pragma unroll
        for (int i = 0; i < 8; ++i) { c[tile][i] = 0.0f; }
    }

    typename Decoder::Carry cur, nxt;
    if (s0 < s_end) {
        Decoder::load(codes, high, scales, static_cast<std::int64_t>(row_idx), row_valid,
                      groups_total, s0, load_chunk, cur);
    }

    for (int step = s0; step < s_end; ++step) {
        // Stage X[step] into per-warp shared memory while cur (or nxt) weight loads are in flight.
        stream_stage_x_warp<kTiles>(x, x_sh[warp], k, t, col_offset, step, lane);
        if (step + 1 < s_end) {
            Decoder::load(codes, high, scales, static_cast<std::int64_t>(row_idx), row_valid,
                          groups_total, step + 1, load_chunk, nxt);
        }
        stream_compute_step<Fmt, kTiles>(cur, src_lane, qp, r, x_sh[warp], c);
        cur = nxt;
    }

#pragma unroll
    for (int tile = 0; tile < kTiles; ++tile) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            c[tile][i] += __shfl_xor_sync(0xffffffffu, c[tile][i], 4);
            c[tile][i] += __shfl_xor_sync(0xffffffffu, c[tile][i], 8);
        }
    }

    if (qp == 0) {
#pragma unroll
        for (int tile = 0; tile < kTiles; ++tile) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int row_t = (i & 2) | ((lane & 16) != 0 ? 4 : 0) | (lane & 1);
                const int col_n = (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
                cs[warp][(tile * 8 + row_t) * 8 + col_n] = c[tile][i];
            }
        }
    }
    __syncthreads();

    constexpr int kOut = kTiles * 64;
    for (int e = static_cast<int>(threadIdx.x); e < kOut; e += 128) {
        const int local_t = e >> 3;
        const int local_n = e & 7;
        const int tok     = col_offset + local_t;
        const int row     = row0 + local_n;
        if (tok < t && row < n) {
            const float v = cs[0][e] + cs[1][e] + cs[2][e] + cs[3][e];
            epilogue.store(row, tok, v);
        }
    }
}

// Split-N (4 warps x 8 rows = 32 rows/CTA): all 4 warps share one double-buffered x_sh[2] (only
// 1 uint4 activation load/thread for T<=8) with scalar software-pipelined weight prefetch.
template <RowSplitStreamFormat Fmt, int kTiles, bool kArgmax, typename Activation>
__global__ __launch_bounds__(128, kTiles == 1 ? 8 : 6) void volta_rowsplit_stream_hmma_splitn_kernel(
    const std::uint8_t* __restrict__ codes, const std::uint8_t* __restrict__ high,
    const std::uint16_t* __restrict__ scales, const Activation* __restrict__ x,
    __nv_bfloat16* __restrict__ out, unsigned long long* __restrict__ packed_max, int n,
    int valid_rows, int k, int t, int col_offset, int out_ld) {
    using Decoder        = RowSplitStreamDecoder<Fmt>;
    constexpr int kWarps = 4;

    __shared__ __align__(16) half x_sh[2][kTiles * 8][kStreamXStride];
    __shared__ float cs[kWarps][kTiles * 64];

    const int tid        = static_cast<int>(threadIdx.x);
    const int lane       = tid & 31;
    const int warp       = tid >> 5;
    const int cta_row0   = static_cast<int>(blockIdx.x) * 32;
    const int row0       = cta_row0 + warp * 8;
    const int load_row   = lane >> 2;
    const int load_chunk = lane & 3;
    const int row_idx    = row0 + load_row;
    const bool row_valid = row_idx < n;

    const int qp       = (lane >> 2) & 3;
    const int r        = (lane & 3) + ((lane & 16) != 0 ? 4 : 0);
    const int src_lane = (r << 2) | qp;

    const int groups_total = k / 64;
    const int steps_total  = k / 128;

    float c[kTiles][8];
#pragma unroll
    for (int tile = 0; tile < kTiles; ++tile) {
#pragma unroll
        for (int i = 0; i < 8; ++i) { c[tile][i] = 0.0f; }
    }

    typename Decoder::Carry cur, nxt;
    if (steps_total > 0) {
        Decoder::load(codes, high, scales, static_cast<std::int64_t>(row_idx), row_valid,
                      groups_total, 0, load_chunk, cur);
        stream_stage_x_cta<kTiles>(x, x_sh[0], k, t, col_offset, 0, tid);
    }
    int buf = 0;
    for (int step = 0; step < steps_total; ++step) {
        __syncthreads();
        const int next_step = step + 1;
        if (next_step < steps_total) {
            Decoder::load(codes, high, scales, static_cast<std::int64_t>(row_idx), row_valid,
                          groups_total, next_step, load_chunk, nxt);
            stream_stage_x_cta<kTiles>(x, x_sh[buf ^ 1], k, t, col_offset, next_step, tid);
        }
        stream_compute_step<Fmt, kTiles>(cur, src_lane, qp, r, x_sh[buf], c);
        cur = nxt;
        buf ^= 1;
    }

#pragma unroll
    for (int tile = 0; tile < kTiles; ++tile) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            c[tile][i] += __shfl_xor_sync(0xffffffffu, c[tile][i], 4);
            c[tile][i] += __shfl_xor_sync(0xffffffffu, c[tile][i], 8);
        }
    }

    if (qp == 0) {
#pragma unroll
        for (int tile = 0; tile < kTiles; ++tile) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int row_t = (i & 2) | ((lane & 16) != 0 ? 4 : 0) | (lane & 1);
                const int col_n = (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
                cs[warp][(tile * 8 + row_t) * 8 + col_n] = c[tile][i];
            }
        }
    }

    if constexpr (!kArgmax) {
        __syncwarp();
        constexpr int kWarpOut = kTiles * 64;
        for (int e = lane; e < kWarpOut; e += 32) {
            const int local_t = e >> 3;
            const int local_n = e & 7;
            const int tok     = col_offset + local_t;
            const int row     = row0 + local_n;
            if (tok < t && row < n) {
                out[static_cast<std::int64_t>(tok) * out_ld + row] = __float2bfloat16(cs[warp][e]);
            }
        }
    } else {
        __syncthreads();
        const int active_t  = min(kTiles * 8, t - col_offset);
        const int vocab_row = cta_row0 + lane;
        for (int local_t = warp; local_t < active_t; local_t += kWarps) {
            const float val = cs[lane >> 3][local_t * 8 + (lane & 7)];
            unsigned long long key =
                (vocab_row < valid_rows) ? stream_bf16_argmax_key(val, vocab_row) : 0ULL;
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                const unsigned long long other = __shfl_xor_sync(0xffffffffu, key, offset);
                if (other > key) { key = other; }
            }
            const int tok = col_offset + local_t;
            if (lane == 0 &&
                key > *reinterpret_cast<const volatile unsigned long long*>(packed_max + tok)) {
                atomicMax(packed_max + tok, key);
            }
        }
    }
}

// Fused Q4 SwiGLU (Split-N: 4 warps x 8 pairs = 32 output rows/CTA, 544 CTAs for N=17408).
// Shares one double-buffered x_sh[2] across all 4 warps AND both gate and up (8x activation
// reuse!), with ping-pong gate/up weight prefetch (0 extra carry regs, 0 stack frame, 0 workspace).
template <int kTiles, typename Activation>
__global__ __launch_bounds__(128, kTiles == 1 ? 8 : 4) void volta_q4_stream_hmma_swiglu_kernel(
    const std::uint8_t* __restrict__ codes, const std::uint16_t* __restrict__ scales,
    const Activation* __restrict__ x, __nv_bfloat16* __restrict__ out, int half_n, int k, int t,
    int out_ld) {
    using Decoder        = RowSplitStreamDecoder<RowSplitStreamFormat::Q4>;
    constexpr int kWarps = 4;

    __shared__ __align__(16) half x_sh[2][kTiles * 8][kStreamXStride];
    __shared__ float cs_gate[kWarps][kTiles * 64];
    __shared__ float cs_up[kWarps][kTiles * 64];

    const int tid        = static_cast<int>(threadIdx.x);
    const int lane       = tid & 31;
    const int warp       = tid >> 5;
    const int row0       = (static_cast<int>(blockIdx.x) * kWarps + warp) * 8;
    const int load_row   = lane >> 2;
    const int load_chunk = lane & 3;
    const int row_idx    = row0 + load_row;
    const bool row_valid = row_idx < half_n;

    const int qp       = (lane >> 2) & 3;
    const int r        = (lane & 3) + ((lane & 16) != 0 ? 4 : 0);
    const int src_lane = (r << 2) | qp;

    const int groups_total = k / 64;
    const int steps_total  = k / 128;

    float c_gate[kTiles][8];
    float c_up[kTiles][8];
#pragma unroll
    for (int tile = 0; tile < kTiles; ++tile) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            c_gate[tile][i] = 0.0f;
            c_up[tile][i]   = 0.0f;
        }
    }

    Decoder::Carry g_cur, u_cur;
    if (steps_total > 0) {
        Decoder::load(codes, nullptr, scales, static_cast<std::int64_t>(row_idx), row_valid,
                      groups_total, 0, load_chunk, g_cur);
        stream_stage_x_cta<kTiles>(x, x_sh[0], k, t, 0, 0, tid);
    }
    int buf = 0;
    for (int step = 0; step < steps_total; ++step) {
        __syncthreads();
        Decoder::load(codes, nullptr, scales, static_cast<std::int64_t>(row_idx + half_n),
                      row_valid, groups_total, step, load_chunk, u_cur);
        const int next_step = step + 1;
        if (next_step < steps_total) {
            stream_stage_x_cta<kTiles>(x, x_sh[buf ^ 1], k, t, 0, next_step, tid);
        }
        stream_compute_step<RowSplitStreamFormat::Q4, kTiles>(g_cur, src_lane, qp, r, x_sh[buf],
                                                              c_gate);
        if (next_step < steps_total) {
            Decoder::load(codes, nullptr, scales, static_cast<std::int64_t>(row_idx), row_valid,
                          groups_total, next_step, load_chunk, g_cur);
        }
        stream_compute_step<RowSplitStreamFormat::Q4, kTiles>(u_cur, src_lane, qp, r, x_sh[buf],
                                                              c_up);
        buf ^= 1;
    }

#pragma unroll
    for (int tile = 0; tile < kTiles; ++tile) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            c_gate[tile][i] += __shfl_xor_sync(0xffffffffu, c_gate[tile][i], 4);
            c_gate[tile][i] += __shfl_xor_sync(0xffffffffu, c_gate[tile][i], 8);
            c_up[tile][i] += __shfl_xor_sync(0xffffffffu, c_up[tile][i], 4);
            c_up[tile][i] += __shfl_xor_sync(0xffffffffu, c_up[tile][i], 8);
        }
    }

    if (qp == 0) {
#pragma unroll
        for (int tile = 0; tile < kTiles; ++tile) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int row_t    = (i & 2) | ((lane & 16) != 0 ? 4 : 0) | (lane & 1);
                const int col_n    = (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
                const int idx      = (tile * 8 + row_t) * 8 + col_n;
                cs_gate[warp][idx] = c_gate[tile][i];
                cs_up[warp][idx]   = c_up[tile][i];
            }
        }
    }
    __syncwarp();

    constexpr int kWarpOut = kTiles * 64;
    for (int e = lane; e < kWarpOut; e += 32) {
        const int tok     = e >> 3;
        const int local_n = e & 7;
        const int row     = row0 + local_n;
        if (tok < t && row < half_n) {
            const float g      = cs_gate[warp][e];
            const float u      = cs_up[warp][e];
            const float silu_g = g / (1.0f + __expf(-g));
            out[static_cast<std::int64_t>(tok) * out_ld + row] = __float2bfloat16(silu_g * u);
        }
    }
}

__global__ void stream_argmax_extract_kernel(const unsigned long long* __restrict__ packed_max,
                                             std::int32_t* __restrict__ out_ids, int t) {
    const int tok = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) +
                    static_cast<int>(threadIdx.x);
    if (tok < t) {
        const std::uint32_t inv_row = static_cast<std::uint32_t>(packed_max[tok] & 0xffffffffULL);
        out_ids[tok]                = static_cast<std::int32_t>(~inv_row);
    }
}

#endif // !defined(__CUDA_ARCH__) || __CUDA_ARCH__ == 700

} // namespace ninfer::ops::detail
