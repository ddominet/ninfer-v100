#pragma once

// Tensor-core skinny GEMM for 9..64 token columns on sm_70, shared by the prepacked W8, FP8 and
// NVFP4 weight formats.
//
// The quadpair kernels (*_volta_qpn_gemm.cuh) are the right shape for one to eight tokens: every
// lane streams its own weight row and the activation tile rides along in registers. Past eight
// tokens they repeat the whole decode-and-multiply per extra 8-row tile, so a projection costs
// roughly 1 ms per extra column on the 27B target and the verify block of a speculative round
// stops being weight-bound. Here a CTA decodes a ROWS x 64 weight slice into shared memory once
// and every warp multiplies its 16 rows against all T columns (WMMA 16x16x16, FP32 accumulate),
// with the activation slice staged in shared memory beside it. The weight is read once whatever T
// is, so the cost stays close to the weight-read floor up to 64 columns.
//
// A format supplies a Decoder (load one 16-byte slice of a prepacked 32-row tile for one 64-K
// chunk; write its FP16 values into the tile; finish one output value), and the caller's output
// policy receives finished values exactly as the quadpair kernels hand them over, so every fused
// epilogue keeps working unchanged.
//
// Narrow projections (5120 rows) are split along K across gridDim.y CTAs. The splits of one row
// block meet in a serial fix-up: split s waits for its turn on a per-row-block semaphore, adds
// its partial to the FP32 plane in split order, and the last split finishes and stores. The order
// is fixed, so results are reproducible run to run, unlike an atomic reduction.

#include "core/device.h"
#include "ops/linear/volta_fp16_stage.h"
#include "ops/linear/volta_tc_skinny.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <mma.h>

#include <cstdint>
#include <type_traits>

namespace ninfer::ops::detail {

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ == 700

inline constexpr int kVoltaTcChunkK = 64;
inline constexpr int kVoltaTcLds    = kVoltaTcChunkK + 8; // padded shared row, in halves

// The quadpair lane -> row map within a 32-row prepacked tile (w8/fp8/nvfp4 prepack kernels).
__device__ __forceinline__ int volta_tc_lane_row(int lane) {
    return ((lane >> 2) & 3) * 8 + (lane & 3) + ((lane & 16) ? 4 : 0);
}

// Eight consecutive-K activations of token `col` starting at `k0`, as FP16.
template <class Activation>
__device__ __forceinline__ uint4 volta_tc_load_activation(const Activation* x, int k, int col,
                                                          int k0, int x_tiles) {
    if constexpr (std::is_same_v<Activation, VoltaGroupMajorHalf>) {
        // [K/16 group][row tile][k half][8 rows][8 k] (volta_stage_fp16_group_major).
        const std::int64_t offset =
            ((static_cast<std::int64_t>(k0 >> 4) * x_tiles + (col >> 3)) * 16 +
             ((k0 >> 3) & 1) * 8 + (col & 7)) *
            8;
        return __ldg(reinterpret_cast<const uint4*>(reinterpret_cast<const half*>(x) + offset));
    } else if constexpr (std::is_same_v<Activation, half>) {
        return __ldg(reinterpret_cast<const uint4*>(x + static_cast<std::int64_t>(col) * k + k0));
    } else {
        static_assert(std::is_same_v<Activation, __nv_bfloat16>, "BF16, FP16 or group-major FP16");
        const uint4 raw =
            __ldg(reinterpret_cast<const uint4*>(x + static_cast<std::int64_t>(col) * k + k0));
        const auto* source = reinterpret_cast<const __nv_bfloat162*>(&raw);
        uint4 out;
        auto* destination = reinterpret_cast<half2*>(&out);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            destination[i] = __float22half2_rn(__bfloat1622float2(source[i]));
        }
        return out;
    }
}

// grid (n / ROWS, splits), ROWS / 16 warps. Dynamic shared memory:
// A [2][ROWS][kVoltaTcLds] + B [2][16 * TT][kVoltaTcLds] halves.
template <class Decoder, int ROWS, int TT, class Activation, class OutputPolicy>
__global__ __launch_bounds__(ROWS / 16 * 32) void volta_tc_skinny_kernel(
    Decoder decoder, const Activation* __restrict__ x, int n, int k, int t, int x_tiles,
    OutputPolicy output, float* __restrict__ partial, unsigned* __restrict__ semaphores) {
    using namespace nvcuda;
    constexpr int KC = kVoltaTcChunkK, LDS = kVoltaTcLds;
    constexpr int WARPS = ROWS / 16, THREADS = WARPS * 32, TCOLS = 16 * TT, TILES = ROWS / 32;
    constexpr int A_ITEMS = TILES * Decoder::kItemsPerTileChunk;
    constexpr int A_VEC   = A_ITEMS / THREADS;
    static_assert(A_VEC * THREADS == A_ITEMS, "decoder items must split evenly over the CTA");
    constexpr int B_TOTAL = TCOLS * (KC / 8);
    constexpr int B_VEC   = (B_TOTAL + THREADS - 1) / THREADS;

    extern __shared__ __align__(32) unsigned char volta_tc_smem[];
    half* a_s = reinterpret_cast<half*>(volta_tc_smem);
    half* b_s = a_s + 2 * ROWS * LDS;

    const int warp   = static_cast<int>(threadIdx.x) / 32;
    const int lane   = static_cast<int>(threadIdx.x) % 32;
    const int tile0  = static_cast<int>(blockIdx.x) * TILES;
    const int chunks = k / KC;
    const int split  = static_cast<int>(blockIdx.y);
    const int splits = static_cast<int>(gridDim.y);
    const int c0     = static_cast<int>(static_cast<long long>(split) * chunks / splits);
    const int c1     = static_cast<int>(static_cast<long long>(split + 1) * chunks / splits);

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[TT];
#pragma unroll
    for (int j = 0; j < TT; ++j) { wmma::fill_fragment(acc[j], 0.0f); }

    typename Decoder::Raw raw[A_VEC];
    uint4 bx[B_VEC];
    auto load = [&](int chunk) {
#pragma unroll
        for (int v = 0; v < A_VEC; ++v) {
            const int item = v * THREADS + static_cast<int>(threadIdx.x);
            raw[v] = decoder.load(tile0 + item / Decoder::kItemsPerTileChunk, chunk,
                                  item % Decoder::kItemsPerTileChunk);
        }
#pragma unroll
        for (int v = 0; v < B_VEC; ++v) {
            const int index = v * THREADS + static_cast<int>(threadIdx.x);
            const int col   = index / (KC / 8);
            const int part  = index % (KC / 8);
            bx[v]           = index < B_TOTAL && col < t
                                  ? volta_tc_load_activation(x, k, col, chunk * KC + part * 8, x_tiles)
                                  : make_uint4(0, 0, 0, 0);
        }
    };
    auto store = [&](int buffer) {
        half* a_buffer = a_s + buffer * ROWS * LDS;
#pragma unroll
        for (int v = 0; v < A_VEC; ++v) {
            const int item = v * THREADS + static_cast<int>(threadIdx.x);
            decoder.store(raw[v], item % Decoder::kItemsPerTileChunk,
                          a_buffer + (item / Decoder::kItemsPerTileChunk) * 32 * LDS, LDS);
        }
#pragma unroll
        for (int v = 0; v < B_VEC; ++v) {
            const int index = v * THREADS + static_cast<int>(threadIdx.x);
            if (index < B_TOTAL) {
                const int col  = index / (KC / 8);
                const int part = index % (KC / 8);
                *reinterpret_cast<uint4*>(b_s + buffer * TCOLS * LDS + col * LDS + part * 8) =
                    bx[v];
            }
        }
    };

    if (c0 < c1) {
        load(c0);
        store(0);
    }
    __syncthreads();
    for (int c = c0; c < c1; ++c) {
        const int buffer = (c - c0) & 1;
        if (c + 1 < c1) { load(c + 1); }
        const half* a_buffer = a_s + buffer * ROWS * LDS + warp * 16 * LDS;
        const half* b_buffer = b_s + buffer * TCOLS * LDS;
#pragma unroll
        for (int kk = 0; kk < KC / 16; ++kk) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
            wmma::load_matrix_sync(a, a_buffer + kk * 16, LDS);
#pragma unroll
            for (int j = 0; j < TT; ++j) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b;
                wmma::load_matrix_sync(b, b_buffer + j * 16 * LDS + kk * 16, LDS);
                wmma::mma_sync(acc[j], a, b, acc[j]);
            }
        }
        if (c + 1 < c1) { store(buffer ^ 1); }
        __syncthreads();
    }

    // Accumulators -> per-warp 16x16 staging (column-major: element e is row e % 16, token
    // e / 16), reusing the A buffers.
    float* c_s       = reinterpret_cast<float*>(volta_tc_smem) + warp * 256;
    const int row0   = static_cast<int>(blockIdx.x) * ROWS + warp * 16;
    const bool first = split == 0;
    const bool last  = split == splits - 1;
    if (splits > 1) {
        // Serial fix-up: wait until the earlier splits of this row block have added theirs.
        __shared__ unsigned turn;
        if (threadIdx.x == 0) {
            const volatile unsigned* semaphore = semaphores + blockIdx.x;
            while (*semaphore != static_cast<unsigned>(split)) {}
            __threadfence();
            turn = 1;
        }
        __syncthreads();
        (void)turn;
    }
#pragma unroll
    for (int j = 0; j < TT; ++j) {
        wmma::store_matrix_sync(c_s, acc[j], 16, wmma::mem_col_major);
        __syncwarp();
        for (int e = lane; e < 256; e += 32) {
            const int row   = row0 + (e & 15);
            const int token = j * 16 + (e >> 4);
            if (row < n && token < t) {
                float value = c_s[e];
                if (splits > 1) {
                    float* slot = partial + static_cast<std::int64_t>(token) * n + row;
                    if (!first) { value += __ldcg(slot); }
                    if (!last) {
                        __stcg(slot, value);
                        continue;
                    }
                }
                output.store(row, token, decoder.finish(row, value));
            }
        }
        __syncwarp();
    }
    if (splits > 1) {
        __syncthreads();
        if (threadIdx.x == 0) {
            __threadfence();
            // The last split leaves the semaphore at zero for the next launch.
            atomicExch(semaphores + blockIdx.x, last ? 0U : static_cast<unsigned>(split + 1));
        }
    }
}

// Shapes and schedules ---------------------------------------------------------------------------

struct VoltaTcSchedule {
    int rows   = 0; // 64 or 128
    int splits = 1;
};

// Measured on the DFlash2 draft and 27B target shapes (volta_tc_skinny_bench): 128-row CTAs while
// the activation slice is at most 32 tokens, 64-row CTAs past it (the 48- and 64-token forms are
// shared-memory bound), and K split until about a thousand CTAs are in flight.
inline VoltaTcSchedule volta_tc_schedule(int n, int k, int t) {
    const VoltaTcControl& control = volta_tc_control();
    VoltaTcSchedule schedule;
    schedule.rows = control.rows != 0 ? control.rows : (t <= 32 && n % 128 == 0 ? 128 : 64);
    const int blocks = n / schedule.rows;
    const int chunks = k / kVoltaTcChunkK;
    int splits       = control.splits;
    if (splits == 0) {
        splits = 1;
        while (splits < 8 && blocks * splits * 2 <= 1100 && chunks / (splits * 2) >= 8) {
            splits *= 2;
        }
    }
    schedule.splits = splits;
    return schedule;
}

template <class Decoder, int ROWS, int TT, class Activation, class OutputPolicy>
void launch_volta_tc_skinny_shape(const Decoder& decoder, const Activation* x, int n, int k, int t,
                                  int x_tiles, OutputPolicy output, int splits,
                                  const VoltaTcSplitWorkspace& workspace, cudaStream_t stream) {
    constexpr int kSmem = (2 * ROWS + 2 * 16 * TT) * kVoltaTcLds * static_cast<int>(sizeof(half));
    auto* kernel        = volta_tc_skinny_kernel<Decoder, ROWS, TT, Activation, OutputPolicy>;
    static const bool configured = [&] {
        CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem));
        return true;
    }();
    (void)configured;
    kernel<<<dim3(static_cast<unsigned>(n / ROWS), static_cast<unsigned>(splits)), ROWS / 16 * 32,
             kSmem, stream>>>(decoder, x, n, k, t, x_tiles, output, workspace.partial,
                              workspace.semaphores);
    CUDA_CHECK(cudaGetLastError());
}

template <class Decoder, int ROWS, class Activation, class OutputPolicy>
void launch_volta_tc_skinny_rows(const Decoder& decoder, const Activation* x, int n, int k, int t,
                                 int x_tiles, OutputPolicy output, int splits,
                                 const VoltaTcSplitWorkspace& workspace, cudaStream_t stream) {
    if (t <= 16) {
        launch_volta_tc_skinny_shape<Decoder, ROWS, 1>(decoder, x, n, k, t, x_tiles, output, splits,
                                                       workspace, stream);
    } else if (t <= 32) {
        launch_volta_tc_skinny_shape<Decoder, ROWS, 2>(decoder, x, n, k, t, x_tiles, output, splits,
                                                       workspace, stream);
    } else if (t <= 48) {
        launch_volta_tc_skinny_shape<Decoder, ROWS, 3>(decoder, x, n, k, t, x_tiles, output, splits,
                                                       workspace, stream);
    } else {
        launch_volta_tc_skinny_shape<Decoder, ROWS, 4>(decoder, x, n, k, t, x_tiles, output, splits,
                                                       workspace, stream);
    }
}

// Runs the tensor-core route when it applies and returns true; returns false (nothing launched)
// when the caller should keep its quadpair route. `x_tiles` is the group-major staging's row-tile
// count (ignored for row-major activations).
template <class Decoder, class Activation, class OutputPolicy>
bool launch_volta_tc_skinny(const Decoder& decoder, const Activation* x, int n, int k, int t,
                            int x_tiles, OutputPolicy output, cudaStream_t stream) {
    if (!volta_tc_applies(n, k, t)) { return false; }
    if (reinterpret_cast<std::uintptr_t>(x) % 16 != 0) { return false; }
    const VoltaTcSchedule schedule = volta_tc_schedule(n, k, t);
    if (schedule.rows != 64 && schedule.rows != 128) { return false; }
    if (n % schedule.rows != 0) { return false; }
    VoltaTcSplitWorkspace workspace{};
    int splits = schedule.splits;
    if (splits > 1) {
        workspace = volta_tc_split_workspace(stream, static_cast<std::size_t>(n) * t,
                                             static_cast<std::size_t>(n / schedule.rows));
        if (workspace.partial == nullptr) { splits = 1; }
    }
    if (schedule.rows == 128) {
        launch_volta_tc_skinny_rows<Decoder, 128>(decoder, x, n, k, t, x_tiles, output, splits,
                                                  workspace, stream);
    } else {
        launch_volta_tc_skinny_rows<Decoder, 64>(decoder, x, n, k, t, x_tiles, output, splits,
                                                 workspace, stream);
    }
    return true;
}

#endif // sm_70

} // namespace ninfer::ops::detail
