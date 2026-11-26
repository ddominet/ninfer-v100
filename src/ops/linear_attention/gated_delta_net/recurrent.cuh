#pragma once

#include "ops/common/bf16_vector.cuh"
#include "ops/linear_attention/gated_delta_net/common.cuh"
#include "ops/linear_attention/gated_delta_net/launch.h"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail::gated_delta_net {

// Geometry is compile-time overridable for single-TU microbench sweeps;
// production TUs never set the macros, so the ODR is stable. Finer value tiles trade redundant
// per-warp normalize work for more resident chains, which matters at prefill width where the
// serial delta-rule chain is latency-bound, not issue-bound.
#ifndef NINFER_GDN_DV_PER_WARP
#define NINFER_GDN_DV_PER_WARP 4
#endif
#ifndef NINFER_GDN_NUM_WARPS
#define NINFER_GDN_NUM_WARPS 4
#endif
inline constexpr int kDvPerWarp = NINFER_GDN_DV_PER_WARP;
inline constexpr int kNumWarps  = NINFER_GDN_NUM_WARPS;
inline constexpr int kBlockDv   = kNumWarps * kDvPerWarp;
inline constexpr int kQkPerLane = kStateDim / kWarpSize;

static_assert(kStateDim % kWarpSize == 0);
static_assert(kQkPerLane == 4);
static_assert(kStateDim % kBlockDv == 0);

__device__ __forceinline__ void load_qk_lane(float (&reg)[kQkPerLane], const float* base,
                                             std::uint32_t dqk_base) {
    // State tiles are touched once per decode step: stream through L2, keep L1 for q/k/v.
    float4 v;
    asm volatile("ld.global.cs.v4.f32 {%0, %1, %2, %3}, [%4];"
                 : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
                 : "l"(base + dqk_base));
    store_vec(reg, v);
}

__device__ __forceinline__ void store_qk_lane(const float (&reg)[kQkPerLane], float* base,
                                              std::uint32_t dqk_base) {
    float4 v = load_vec<float4>(reg);
    asm volatile("st.global.cs.v4.f32 [%0], {%1, %2, %3, %4};" ::"l"(base + dqk_base), "f"(v.x),
                 "f"(v.y), "f"(v.z), "f"(v.w));
}

inline constexpr float kQkL2NormEps = 1.0e-6f;

struct RawQkLane {
    Bf16x4Pack bits;
    float value[kQkPerLane];
};

struct RawValueLane {
    __nv_bfloat16 bits;
    float value;
};

struct RawGatePair {
    uint2 bits;
    float g;
    float beta;
};

__device__ __forceinline__ RawQkLane load_raw_qk_lane(const __nv_bfloat16* base,
                                                      std::uint32_t dqk_base) {
    RawQkLane out;
    out.bits        = load_vec<Bf16x4Pack>(base + dqk_base);
    const float2 lo = bf16x2_to_float2(out.bits.pair[0]);
    const float2 hi = bf16x2_to_float2(out.bits.pair[1]);
    out.value[0]    = lo.x;
    out.value[1]    = lo.y;
    out.value[2]    = hi.x;
    out.value[3]    = hi.y;
    return out;
}

template <bool Normalize>
__device__ __forceinline__ void normalize_qk_lane(float (&value)[kQkPerLane], int lane) {
    if constexpr (Normalize) {
        float sum = 0.0f;
#pragma unroll
        for (int i = 0; i < kQkPerLane; ++i) { sum += value[i] * value[i]; }
        sum       = warp_reduce_sum(sum);
        float inv = lane == 0 ? rsqrtf(sum + kQkL2NormEps) : 0.0f;
        inv       = __shfl_sync(kFullWarpMask, inv, 0);
#pragma unroll
        for (int i = 0; i < kQkPerLane; ++i) { value[i] *= inv; }
    }
}

__device__ __forceinline__ RawValueLane load_value_lane(const __nv_bfloat16* base, int lane,
                                                        std::uint32_t dv_base) {
    RawValueLane out{__float2bfloat16(0.0f), 0.0f};
    if (lane < kDvPerWarp) {
        out.bits  = base[dv_base + lane];
        out.value = __bfloat162float(out.bits);
    }
    return out;
}

__device__ __forceinline__ RawGatePair load_source_gate(const float* g, const float* beta,
                                                        std::int64_t offset) {
    const float g_value    = g[offset];
    const float beta_value = beta[offset];
    return {make_uint2(__float_as_uint(g_value), __float_as_uint(beta_value)), g_value, beta_value};
}

__device__ __forceinline__ RawGatePair load_record_gate(const uint2* gate, std::int64_t offset) {
    const uint2 bits = load_vec<uint2>(gate + offset);
    return {bits, __uint_as_float(bits.x), __uint_as_float(bits.y)};
}

// Batched-reduce switch: the per-token delta-rule step runs kDvPerWarp row reductions through
// warp_sum; issuing their butterflies interleaved (mask outer, row inner) collapses kDvPerWarp
// dependent 5-step chains into one, with identical per-value xor-butterfly order (bit-identical
// sums). The serial chain, not issue throughput, is what paces the prefill recurrence.
#ifndef NINFER_GDN_BATCH_REDUCE
#ifdef NINFER_VOLTA_BUILD
#define NINFER_GDN_BATCH_REDUCE 1
#else
#define NINFER_GDN_BATCH_REDUCE 0
#endif
#endif

__device__ __forceinline__ void apply_gdn_transition(float (&state)[kDvPerWarp][kQkPerLane],
                                                     const float (&key)[kQkPerLane], float v_local,
                                                     float g, float beta) {
    const float alpha = expf(g);

#if NINFER_GDN_BATCH_REDUCE
    float partial[kDvPerWarp];
#pragma unroll
    for (int r = 0; r < kDvPerWarp; ++r) {
        float p = 0.0f;
#pragma unroll
        for (int c = 0; c < kQkPerLane; ++c) { p += state[r][c] * key[c]; }
        partial[r] = p;
    }
#pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
#pragma unroll
        for (int r = 0; r < kDvPerWarp; ++r) {
            partial[r] += __shfl_xor_sync(kFullWarpMask, partial[r], offset, kWarpSize);
        }
    }
#pragma unroll
    for (int r = 0; r < kDvPerWarp; ++r) {
        const float v_r   = __shfl_sync(kFullWarpMask, v_local, r, kWarpSize);
        const float delta = beta * (v_r - alpha * partial[r]);
#pragma unroll
        for (int c = 0; c < kQkPerLane; ++c) { state[r][c] = alpha * state[r][c] + delta * key[c]; }
    }
#else
#pragma unroll
    for (int r = 0; r < kDvPerWarp; ++r) {
        float partial = 0.0f;
#pragma unroll
        for (int c = 0; c < kQkPerLane; ++c) { partial += state[r][c] * key[c]; }
        partial = warp_sum<kWarpSize>(partial);

        const float v_r   = __shfl_sync(0xffffffff, v_local, r, kWarpSize);
        const float delta = beta * (v_r - alpha * partial);

#pragma unroll
        for (int c = 0; c < kQkPerLane; ++c) { state[r][c] = alpha * state[r][c] + delta * key[c]; }
    }
#endif
}

__device__ __forceinline__ void readout_normalized_and_store(
    float (&state)[kDvPerWarp][kQkPerLane], const float (&q_norm)[kQkPerLane],
    __nv_bfloat16* output, std::uint32_t dv_base, int lane, float scale) {
#if NINFER_GDN_BATCH_REDUCE
    float partial[kDvPerWarp];
#pragma unroll
    for (int r = 0; r < kDvPerWarp; ++r) {
        float p = 0.0f;
#pragma unroll
        for (int c = 0; c < kQkPerLane; ++c) { p += state[r][c] * q_norm[c]; }
        partial[r] = p;
    }
#pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
#pragma unroll
        for (int r = 0; r < kDvPerWarp; ++r) {
            partial[r] += __shfl_xor_sync(kFullWarpMask, partial[r], offset, kWarpSize);
        }
    }
    float attn_val = 0.0f;
#pragma unroll
    for (int r = 0; r < kDvPerWarp; ++r) {
        if (lane == r) { attn_val = partial[r]; }
    }
    if (lane < kDvPerWarp) { output[dv_base + lane] = __float2bfloat16(attn_val * scale); }
#else
    float attn_val = 0.0f;
#pragma unroll
    for (int r = 0; r < kDvPerWarp; ++r) {
        float partial = 0.0f;
#pragma unroll
        for (int c = 0; c < kQkPerLane; ++c) { partial += state[r][c] * q_norm[c]; }
        partial = warp_sum<kWarpSize>(partial);
        if (lane == r) { attn_val = partial; }
    }
    if (lane < kDvPerWarp) { output[dv_base + lane] = __float2bfloat16(attn_val * scale); }
#endif
}

template <bool Normalize>
__device__ __forceinline__ void readout_and_store(float (&state)[kDvPerWarp][kQkPerLane],
                                                  const __nv_bfloat16* query, __nv_bfloat16* output,
                                                  std::uint32_t dqk_base, std::uint32_t dv_base,
                                                  int lane, float scale) {
    RawQkLane q = load_raw_qk_lane(query, dqk_base);
    normalize_qk_lane<Normalize>(q.value, lane);
    readout_normalized_and_store(state, q.value, output, dv_base, lane, scale);
}

struct RecurrentCoordinates {
    int lane;
    int warp;
    std::int32_t batch;
    std::int32_t layer;
    std::int32_t state_tile;
    std::uint32_t value_head;
    std::uint32_t qk_head;
    std::uint32_t dv_base;
    std::uint32_t dqk_base;
};

__device__ __forceinline__ RecurrentCoordinates make_coordinates(std::int32_t batch,
                                                                 std::int32_t layer,
                                                                 std::int32_t state_tile,
                                                                 head_map heads) {
    const int lane                 = threadIdx.x;
    const int warp                 = threadIdx.y;
    const std::uint32_t value_head = static_cast<std::uint32_t>(blockIdx.x);
    const std::uint32_t qk_head =
        static_cast<std::uint32_t>(heads.qk_head(static_cast<int>(value_head)));
    const std::uint32_t dv_base =
        static_cast<std::uint32_t>(state_tile * kBlockDv + warp * kDvPerWarp);
    return {lane,    warp,       batch,
            layer,   state_tile, value_head,
            qk_head, dv_base,    static_cast<std::uint32_t>(lane * kQkPerLane)};
}

struct OutputEffects {
    template <class Access>
    __device__ __forceinline__ static void observe_key(const Access&, const RecurrentCoordinates&,
                                                       std::int32_t, const RawQkLane&) {}

    template <class Access>
    __device__ __forceinline__ static void
    observe_value_gate(const Access&, const RecurrentCoordinates&, std::int32_t,
                       const RawValueLane&, const RawGatePair&) {}

    template <bool NormalizeInputs, class Access>
    __device__ __forceinline__ static void
    publish_output(float (&state)[kDvPerWarp][kQkPerLane], const Access& access,
                   const RecurrentCoordinates& coord, std::int32_t token) {
        readout_and_store<NormalizeInputs>(state, access.query_ptr(coord, token),
                                           access.output_ptr(coord, token), coord.dqk_base,
                                           coord.dv_base, coord.lane, access.scale);
    }
};

struct RecordEffects {
    template <class Access>
    __device__ __forceinline__ static void observe_key(const Access& access,
                                                       const RecurrentCoordinates& coord,
                                                       std::int32_t token, const RawQkLane& key) {
        access.store_key(coord, token, key);
    }

    template <class Access>
    __device__ __forceinline__ static void
    observe_value_gate(const Access& access, const RecurrentCoordinates& coord, std::int32_t token,
                       const RawValueLane& value, const RawGatePair& gate) {
        access.store_value(coord, token, value);
        access.store_gate(coord, token, gate);
    }

    template <bool NormalizeInputs, class Access>
    __device__ __forceinline__ static void
    publish_output(float (&state)[kDvPerWarp][kQkPerLane], const Access& access,
                   const RecurrentCoordinates& coord, std::int32_t token) {
        OutputEffects::publish_output<NormalizeInputs>(state, access, coord, token);
    }
};

struct FoldEffects {
    template <class Access>
    __device__ __forceinline__ static void observe_key(const Access&, const RecurrentCoordinates&,
                                                       std::int32_t, const RawQkLane&) {}

    template <class Access>
    __device__ __forceinline__ static void
    observe_value_gate(const Access&, const RecurrentCoordinates&, std::int32_t,
                       const RawValueLane&, const RawGatePair&) {}

    template <bool NormalizeInputs, class Access>
    __device__ __forceinline__ static void
    publish_output(float (&)[kDvPerWarp][kQkPerLane], const Access&, const RecurrentCoordinates&,
                   std::int32_t) {}
};

struct DirectAccess {
    const __nv_bfloat16* q;
    const __nv_bfloat16* k;
    const __nv_bfloat16* v;
    const float* g;
    const float* beta;
    const float* state_read;
    float* state_write;
    __nv_bfloat16* out;
    head_map heads;
    std::int32_t width;
    float scale;

    __device__ __forceinline__ RecurrentCoordinates coordinates() const {
        return make_coordinates(0, 0, static_cast<std::int32_t>(blockIdx.z), heads);
    }

    __device__ __forceinline__ std::int64_t column(const RecurrentCoordinates& coord,
                                                   std::int32_t token) const {
        (void)coord;
        return token;
    }

    __device__ __forceinline__ const float*
    state_read_base(const RecurrentCoordinates& coord) const {
        return state_read + static_cast<std::int64_t>(coord.value_head) * kStateDim * kStateDim;
    }

    __device__ __forceinline__ float* state_write_base(const RecurrentCoordinates& coord) const {
        return state_write + static_cast<std::int64_t>(coord.value_head) * kStateDim * kStateDim;
    }

    __device__ __forceinline__ const __nv_bfloat16* key_ptr(const RecurrentCoordinates& coord,
                                                            std::int32_t token) const {
        return k + (column(coord, token) * heads.H_qk + coord.qk_head) * kStateDim;
    }

    __device__ __forceinline__ const __nv_bfloat16* value_ptr(const RecurrentCoordinates& coord,
                                                              std::int32_t token) const {
        return v + (column(coord, token) * heads.H_v + coord.value_head) * kStateDim;
    }

    __device__ __forceinline__ RawGatePair load_gate(const RecurrentCoordinates& coord,
                                                     std::int32_t token) const {
        return load_source_gate(g, beta, column(coord, token) * heads.H_v + coord.value_head);
    }

    __device__ __forceinline__ const __nv_bfloat16* query_ptr(const RecurrentCoordinates& coord,
                                                              std::int32_t token) const {
        return q + (column(coord, token) * heads.H_qk + coord.qk_head) * kStateDim;
    }

    __device__ __forceinline__ __nv_bfloat16* output_ptr(const RecurrentCoordinates& coord,
                                                         std::int32_t token) const {
        return out + (column(coord, token) * heads.H_v + coord.value_head) * kStateDim;
    }
};

struct BatchUpdateAccess {
    const __nv_bfloat16* q;
    const __nv_bfloat16* k;
    const __nv_bfloat16* v;
    const float* g;
    const float* beta;
    const float* states_read;
    float* states_write;
    const std::int32_t* source_state_slots;
    const std::int32_t* destination_state_slots;
    __nv_bfloat16* out;
    head_map heads;
    std::int64_t state_slot_stride;
    float scale;

    __device__ __forceinline__ RecurrentCoordinates coordinates() const {
        return make_coordinates(static_cast<std::int32_t>(blockIdx.y), 0,
                                static_cast<std::int32_t>(blockIdx.z), heads);
    }

    __device__ __forceinline__ std::int64_t column(const RecurrentCoordinates& coord,
                                                   std::int32_t token) const {
        (void)token;
        return coord.batch;
    }

    __device__ __forceinline__ const float*
    state_read_base(const RecurrentCoordinates& coord) const {
        return states_read +
               static_cast<std::int64_t>(source_state_slots[coord.batch]) * state_slot_stride +
               static_cast<std::int64_t>(coord.value_head) * kStateDim * kStateDim;
    }

    __device__ __forceinline__ float* state_write_base(const RecurrentCoordinates& coord) const {
        return states_write +
               static_cast<std::int64_t>(destination_state_slots[coord.batch]) * state_slot_stride +
               static_cast<std::int64_t>(coord.value_head) * kStateDim * kStateDim;
    }

    __device__ __forceinline__ const __nv_bfloat16* key_ptr(const RecurrentCoordinates& coord,
                                                            std::int32_t token) const {
        return k + (column(coord, token) * heads.H_qk + coord.qk_head) * kStateDim;
    }

    __device__ __forceinline__ const __nv_bfloat16* value_ptr(const RecurrentCoordinates& coord,
                                                              std::int32_t token) const {
        return v + (column(coord, token) * heads.H_v + coord.value_head) * kStateDim;
    }

    __device__ __forceinline__ RawGatePair load_gate(const RecurrentCoordinates& coord,
                                                     std::int32_t token) const {
        return load_source_gate(g, beta, column(coord, token) * heads.H_v + coord.value_head);
    }

    __device__ __forceinline__ const __nv_bfloat16* query_ptr(const RecurrentCoordinates& coord,
                                                              std::int32_t token) const {
        return q + (column(coord, token) * heads.H_qk + coord.qk_head) * kStateDim;
    }

    __device__ __forceinline__ __nv_bfloat16* output_ptr(const RecurrentCoordinates& coord,
                                                         std::int32_t token) const {
        return out + (column(coord, token) * heads.H_v + coord.value_head) * kStateDim;
    }
};

template <bool Masked>
struct RecordAccess {
    const __nv_bfloat16* q;
    const __nv_bfloat16* k;
    const __nv_bfloat16* v;
    const float* g;
    const float* beta;
    const float* states;
    const std::int32_t* valid_columns;
    const std::int32_t* initial_slots;
    __nv_bfloat16* key_record;
    __nv_bfloat16* value_record;
    uint2* gate_record;
    __nv_bfloat16* out;
    head_map heads;
    std::int32_t width;
    std::int64_t state_slot_stride;
    float scale;

    __device__ __forceinline__ RecurrentCoordinates coordinates() const {
        return make_coordinates(static_cast<std::int32_t>(blockIdx.y), 0,
                                static_cast<std::int32_t>(blockIdx.z), heads);
    }

    __device__ __forceinline__ std::int32_t
    active_columns(const RecurrentCoordinates& coord) const {
        if constexpr (Masked) { return valid_columns[coord.batch]; }
        return width;
    }

    __device__ __forceinline__ std::int64_t column(const RecurrentCoordinates& coord,
                                                   std::int32_t token) const {
        return static_cast<std::int64_t>(coord.batch) * width + token;
    }

    __device__ __forceinline__ const float*
    state_read_base(const RecurrentCoordinates& coord) const {
        return states + static_cast<std::int64_t>(initial_slots[coord.batch]) * state_slot_stride +
               static_cast<std::int64_t>(coord.value_head) * kStateDim * kStateDim;
    }

    __device__ __forceinline__ const __nv_bfloat16* key_ptr(const RecurrentCoordinates& coord,
                                                            std::int32_t token) const {
        return k + (column(coord, token) * heads.H_qk + coord.qk_head) * kStateDim;
    }

    __device__ __forceinline__ const __nv_bfloat16* value_ptr(const RecurrentCoordinates& coord,
                                                              std::int32_t token) const {
        return v + (column(coord, token) * heads.H_v + coord.value_head) * kStateDim;
    }

    __device__ __forceinline__ RawGatePair load_gate(const RecurrentCoordinates& coord,
                                                     std::int32_t token) const {
        return load_source_gate(g, beta, column(coord, token) * heads.H_v + coord.value_head);
    }

    __device__ __forceinline__ const __nv_bfloat16* query_ptr(const RecurrentCoordinates& coord,
                                                              std::int32_t token) const {
        return q + (column(coord, token) * heads.H_qk + coord.qk_head) * kStateDim;
    }

    __device__ __forceinline__ __nv_bfloat16* output_ptr(const RecurrentCoordinates& coord,
                                                         std::int32_t token) const {
        return out + (column(coord, token) * heads.H_v + coord.value_head) * kStateDim;
    }

    __device__ __forceinline__ void store_key(const RecurrentCoordinates& coord, std::int32_t token,
                                              const RawQkLane& raw) const {
        if (coord.state_tile == 0 && coord.warp == 0 &&
            static_cast<int>(coord.value_head) % heads.group_size() == 0) {
            __nv_bfloat16* destination =
                key_record + (column(coord, token) * heads.H_qk + coord.qk_head) * kStateDim;
            store_vec(destination + coord.dqk_base, raw.bits);
        }
    }

    __device__ __forceinline__ void store_value(const RecurrentCoordinates& coord,
                                                std::int32_t token, const RawValueLane& raw) const {
        if (coord.lane < kDvPerWarp) {
            __nv_bfloat16* destination =
                value_record + (column(coord, token) * heads.H_v + coord.value_head) * kStateDim;
            destination[coord.dv_base + coord.lane] = raw.bits;
        }
    }

    __device__ __forceinline__ void store_gate(const RecurrentCoordinates& coord,
                                               std::int32_t token, const RawGatePair& raw) const {
        if (coord.state_tile == 0 && coord.warp == 0 && coord.lane == 0) {
            gate_record[column(coord, token) * heads.H_v + coord.value_head] = raw.bits;
        }
    }
};

template <int Layers, int QkHeads, int ValueHeads, int ConvChannels>
struct FoldGeometry {
    static constexpr int kLayers       = Layers;
    static constexpr int kQkHeads      = QkHeads;
    static constexpr int kValueHeads   = ValueHeads;
    static constexpr int kConvChannels = ConvChannels;
    static_assert(ValueHeads % QkHeads == 0);
    static_assert(ConvChannels % 128 == 0);
};

using FoldGeometry48x48 = FoldGeometry<48, 16, 48, 10240>;
using FoldGeometry30x32 = FoldGeometry<30, 16, 32, 8192>;

template <class Geometry>
struct FoldAccess {
    const __nv_bfloat16* key_record;
    const __nv_bfloat16* value_record;
    const uint2* gate_record;
    const __nv_bfloat16* conv_record;
    float* recurrent_layer0;
    __nv_bfloat16* conv_layer0;
    std::int64_t recurrent_layer_stride;
    std::int64_t conv_layer_stride;
    std::int32_t record_capacity;
    std::int32_t width;
    GdnReplayFoldKernelRows rows;

    __device__ __forceinline__ RecurrentCoordinates coordinates() const {
        const std::int32_t batch       = static_cast<std::int32_t>(blockIdx.y);
        const std::int32_t layer_tile  = static_cast<std::int32_t>(blockIdx.z);
        const int lane                 = threadIdx.x;
        const int warp                 = threadIdx.y;
        const std::int32_t state_tile  = layer_tile & 7;
        const std::uint32_t value_head = static_cast<std::uint32_t>(blockIdx.x);
        constexpr std::uint32_t kGroup = Geometry::kValueHeads / Geometry::kQkHeads;
        const std::uint32_t qk_head    = value_head / kGroup;
        const std::uint32_t dv_base =
            static_cast<std::uint32_t>(state_tile * kBlockDv + warp * kDvPerWarp);
        return {lane,
                warp,
                batch,
                layer_tile >> 3,
                state_tile,
                value_head,
                qk_head,
                dv_base,
                static_cast<std::uint32_t>(lane * kQkPerLane)};
    }

    __device__ __forceinline__ std::int32_t
    active_columns(const RecurrentCoordinates& coord) const {
        return rows.row[coord.batch].commit_columns;
    }

    __device__ __forceinline__ std::int64_t record_outer(const RecurrentCoordinates& coord) const {
        return static_cast<std::int64_t>(coord.layer) * record_capacity + coord.batch;
    }

    __device__ __forceinline__ const float*
    state_read_base(const RecurrentCoordinates& coord) const {
        const std::int64_t slot_stride =
            static_cast<std::int64_t>(Geometry::kValueHeads) * kStateDim * kStateDim;
        return recurrent_layer0 + static_cast<std::int64_t>(coord.layer) * recurrent_layer_stride +
               static_cast<std::int64_t>(rows.row[coord.batch].source_state_slot) * slot_stride +
               static_cast<std::int64_t>(coord.value_head) * kStateDim * kStateDim;
    }

    __device__ __forceinline__ float* state_write_base(const RecurrentCoordinates& coord) const {
        const std::int64_t slot_stride =
            static_cast<std::int64_t>(Geometry::kValueHeads) * kStateDim * kStateDim;
        return recurrent_layer0 + static_cast<std::int64_t>(coord.layer) * recurrent_layer_stride +
               static_cast<std::int64_t>(rows.row[coord.batch].destination_state_slot) *
                   slot_stride +
               static_cast<std::int64_t>(coord.value_head) * kStateDim * kStateDim;
    }

    __device__ __forceinline__ const __nv_bfloat16* key_ptr(const RecurrentCoordinates& coord,
                                                            std::int32_t token) const {
        const std::int64_t column = record_outer(coord) * width + token;
        return key_record + (column * Geometry::kQkHeads + coord.qk_head) * kStateDim;
    }

    __device__ __forceinline__ const __nv_bfloat16* value_ptr(const RecurrentCoordinates& coord,
                                                              std::int32_t token) const {
        const std::int64_t column = record_outer(coord) * width + token;
        return value_record + (column * Geometry::kValueHeads + coord.value_head) * kStateDim;
    }

    __device__ __forceinline__ RawGatePair load_gate(const RecurrentCoordinates& coord,
                                                     std::int32_t token) const {
        const std::int64_t column = record_outer(coord) * width + token;
        return load_record_gate(gate_record, column * Geometry::kValueHeads + coord.value_head);
    }

    __device__ __forceinline__ void
    store_final_state(const RecurrentCoordinates& coord,
                      const float (&state)[kDvPerWarp][kQkPerLane]) const {
        float* destination = state_write_base(coord);
#pragma unroll
        for (int r = 0; r < kDvPerWarp; ++r) {
            store_qk_lane(state[r],
                          destination + static_cast<std::int64_t>(coord.dv_base + r) * kStateDim,
                          coord.dqk_base);
        }
    }

    __device__ __forceinline__ void publish_final_conv_history(const RecurrentCoordinates& coord,
                                                               std::int32_t commit) const {
        const std::int32_t tile_block =
            static_cast<std::int32_t>(coord.value_head) * 8 + coord.state_tile;
        if (tile_block >= Geometry::kConvChannels / 128) { return; }

        const std::int32_t tid     = coord.warp * kWarpSize + coord.lane;
        const std::int32_t channel = tile_block * 128 + tid;
        const __nv_bfloat16* source_history =
            conv_layer0 + static_cast<std::int64_t>(coord.layer) * conv_layer_stride +
            static_cast<std::int64_t>(rows.row[coord.batch].source_state_slot) *
                (3LL * Geometry::kConvChannels) +
            channel;
        __nv_bfloat16* destination_history =
            conv_layer0 + static_cast<std::int64_t>(coord.layer) * conv_layer_stride +
            static_cast<std::int64_t>(rows.row[coord.batch].destination_state_slot) *
                (3LL * Geometry::kConvChannels) +
            channel;
        const __nv_bfloat16* record =
            conv_record + record_outer(coord) * width * Geometry::kConvChannels + channel;

        __nv_bfloat16 h0;
        __nv_bfloat16 h1;
        __nv_bfloat16 h2;
        if (commit == 1) {
            h0 = source_history[Geometry::kConvChannels];
            h1 = source_history[2LL * Geometry::kConvChannels];
            h2 = record[0];
        } else if (commit == 2) {
            h0 = source_history[2LL * Geometry::kConvChannels];
            h1 = record[0];
            h2 = record[Geometry::kConvChannels];
        } else {
            h0 = record[static_cast<std::int64_t>(commit - 3) * Geometry::kConvChannels];
            h1 = record[static_cast<std::int64_t>(commit - 2) * Geometry::kConvChannels];
            h2 = record[static_cast<std::int64_t>(commit - 1) * Geometry::kConvChannels];
        }
        destination_history[0]                             = h0;
        destination_history[Geometry::kConvChannels]       = h1;
        destination_history[2LL * Geometry::kConvChannels] = h2;
    }
};

__device__ __forceinline__ void load_state_tile(float (&state)[kDvPerWarp][kQkPerLane],
                                                const float* base,
                                                const RecurrentCoordinates& coord) {
#pragma unroll
    for (int r = 0; r < kDvPerWarp; ++r) {
        load_qk_lane(state[r], base + static_cast<std::int64_t>(coord.dv_base + r) * kStateDim,
                     coord.dqk_base);
    }
}

__device__ __forceinline__ void store_state_tile(const float (&state)[kDvPerWarp][kQkPerLane],
                                                 float* base, const RecurrentCoordinates& coord) {
#pragma unroll
    for (int r = 0; r < kDvPerWarp; ++r) {
        store_qk_lane(state[r], base + static_cast<std::int64_t>(coord.dv_base + r) * kStateDim,
                      coord.dqk_base);
    }
}

template <bool NormalizeInputs, class Effects, class Access>
__device__ __forceinline__ void
run_recurrent_sequence(float (&state)[kDvPerWarp][kQkPerLane], const Access& access,
                       const RecurrentCoordinates& coord, std::int32_t valid) {
    if (valid <= 0) { return; }
    RawQkLane key      = load_raw_qk_lane(access.key_ptr(coord, 0), coord.dqk_base);
    RawGatePair gate   = access.load_gate(coord, 0);
    RawValueLane value = load_value_lane(access.value_ptr(coord, 0), coord.lane, coord.dv_base);
    Effects::observe_key(access, coord, 0, key);
    Effects::observe_value_gate(access, coord, 0, value, gate);
    normalize_qk_lane<NormalizeInputs>(key.value, coord.lane);

    for (std::int32_t token = 0; token < valid; ++token) {
        Bf16x4Pack next_key_bits{};
        RawGatePair next_gate{};
        RawValueLane next_value{};
        if (token + 1 < valid) {
            next_key_bits = load_vec<Bf16x4Pack>(access.key_ptr(coord, token + 1) + coord.dqk_base);
            next_gate     = access.load_gate(coord, token + 1);
            next_value    = load_value_lane(access.value_ptr(coord, token + 1), coord.lane,
                                            coord.dv_base);
        }

        apply_gdn_transition(state, key.value, value.value, gate.g, gate.beta);
        Effects::template publish_output<NormalizeInputs>(state, access, coord, token);

        if (token + 1 < valid) {
            key.bits        = next_key_bits;
            const float2 lo = bf16x2_to_float2(next_key_bits.pair[0]);
            const float2 hi = bf16x2_to_float2(next_key_bits.pair[1]);
            key.value[0]    = lo.x;
            key.value[1]    = lo.y;
            key.value[2]    = hi.x;
            key.value[3]    = hi.y;
            gate            = next_gate;
            value           = next_value;
            Effects::observe_key(access, coord, token + 1, key);
            Effects::observe_value_gate(access, coord, token + 1, value, gate);
            normalize_qk_lane<NormalizeInputs>(key.value, coord.lane);
        }
    }
}

template <class Access>
__device__ __forceinline__ void zero_output_suffix(const Access& access,
                                                   const RecurrentCoordinates& coord,
                                                   std::int32_t valid, std::int32_t width) {
    if (coord.lane < kDvPerWarp) {
        for (std::int32_t token = valid; token < width; ++token) {
            access.output_ptr(coord, token)[coord.dv_base + coord.lane] = __float2bfloat16(0.0f);
        }
    }
}

template <bool NormalizeInputs>
__global__ void __launch_bounds__(kWarpSize* kNumWarps, 2)
    recurrent_bf16_direct_kernel(const __nv_bfloat16* __restrict__ q,
                                 const __nv_bfloat16* __restrict__ k,
                                 const __nv_bfloat16* __restrict__ v, const float* __restrict__ g,
                                 const float* __restrict__ beta,
                                 const float* __restrict__ state_read,
                                 float* __restrict__ state_write, __nv_bfloat16* __restrict__ out,
                                 std::int32_t width, head_map heads, float scale) {
    const DirectAccess access{q, k, v, g, beta, state_read, state_write, out, heads, width, scale};
    const RecurrentCoordinates coord = access.coordinates();
    __align__(16) float state[kDvPerWarp][kQkPerLane];
    load_state_tile(state, access.state_read_base(coord), coord);
    run_recurrent_sequence<NormalizeInputs, OutputEffects>(state, access, coord, width);
    store_state_tile(state, access.state_write_base(coord), coord);
}

// Software-pipelined form of recurrent_bf16_direct_kernel for long widths: identical lane mapping
// and arithmetic (bit-identical output), but each token's q/k/v/gate bits are loaded
// kDirectPipelineDepth tokens ahead and only converted when consumed. The plain form loads the
// gate and value at the top of each iteration and uses them immediately, so every token pays a
// global-memory round trip on the state chain; that, not the shuffle reductions, is what makes it
// ~2 us/token regardless of how many heads share the GPU.
#ifndef NINFER_GDN_DIRECT_MIN_BLOCKS
#define NINFER_GDN_DIRECT_MIN_BLOCKS 6
#endif
#ifndef NINFER_GDN_DIRECT_PIPELINE_DEPTH
#define NINFER_GDN_DIRECT_PIPELINE_DEPTH 2
#endif
inline constexpr int kDirectPipelineDepth = NINFER_GDN_DIRECT_PIPELINE_DEPTH;

struct DirectTokenBits {
    Bf16x4Pack key;
    Bf16x4Pack query;
    __nv_bfloat16 value;
    float g;
    float beta;
};

__device__ __forceinline__ DirectTokenBits load_direct_token(const DirectAccess& access,
                                                             const RecurrentCoordinates& coord,
                                                             std::int32_t token) {
    DirectTokenBits bits;
    bits.key   = load_vec<Bf16x4Pack>(access.key_ptr(coord, token) + coord.dqk_base);
    bits.query = load_vec<Bf16x4Pack>(access.query_ptr(coord, token) + coord.dqk_base);
    bits.value = coord.lane < kDvPerWarp
                     ? access.value_ptr(coord, token)[coord.dv_base + coord.lane]
                     : __float2bfloat16(0.0f);
    const std::int64_t gate =
        static_cast<std::int64_t>(token) * access.heads.H_v + coord.value_head;
    bits.g    = access.g[gate];
    bits.beta = access.beta[gate];
    return bits;
}

__device__ __forceinline__ void unpack_qk_lane(float (&out)[kQkPerLane], const Bf16x4Pack& bits) {
    const float2 lo = bf16x2_to_float2(bits.pair[0]);
    const float2 hi = bf16x2_to_float2(bits.pair[1]);
    out[0]          = lo.x;
    out[1]          = lo.y;
    out[2]          = hi.x;
    out[3]          = hi.y;
}

template <bool NormalizeInputs>
__device__ __forceinline__ void consume_direct_token(float (&state)[kDvPerWarp][kQkPerLane],
                                                     const DirectTokenBits& bits,
                                                     const DirectAccess& access,
                                                     const RecurrentCoordinates& coord,
                                                     std::int32_t token) {
    float key[kQkPerLane];
    unpack_qk_lane(key, bits.key);
    normalize_qk_lane<NormalizeInputs>(key, coord.lane);
    const float value = coord.lane < kDvPerWarp ? __bfloat162float(bits.value) : 0.0f;
    apply_gdn_transition(state, key, value, bits.g, bits.beta);

    float query[kQkPerLane];
    unpack_qk_lane(query, bits.query);
    normalize_qk_lane<NormalizeInputs>(query, coord.lane);
#if NINFER_GDN_BATCH_REDUCE
    float partial[kDvPerWarp];
#pragma unroll
    for (int r = 0; r < kDvPerWarp; ++r) {
        float p = 0.0f;
#pragma unroll
        for (int c = 0; c < kQkPerLane; ++c) { p += state[r][c] * query[c]; }
        partial[r] = p;
    }
#pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
#pragma unroll
        for (int r = 0; r < kDvPerWarp; ++r) {
            partial[r] += __shfl_xor_sync(kFullWarpMask, partial[r], offset, kWarpSize);
        }
    }
    float attn_val = 0.0f;
#pragma unroll
    for (int r = 0; r < kDvPerWarp; ++r) {
        if (coord.lane == r) { attn_val = partial[r]; }
    }
    if (coord.lane < kDvPerWarp) {
        access.output_ptr(coord, token)[coord.dv_base + coord.lane] =
            __float2bfloat16(attn_val * access.scale);
    }
#else
    float attn_val = 0.0f;
#pragma unroll
    for (int r = 0; r < kDvPerWarp; ++r) {
        float partial = 0.0f;
#pragma unroll
        for (int c = 0; c < kQkPerLane; ++c) { partial += state[r][c] * query[c]; }
        partial = warp_sum<kWarpSize>(partial);
        if (coord.lane == r) { attn_val = partial; }
    }
    if (coord.lane < kDvPerWarp) {
        access.output_ptr(coord, token)[coord.dv_base + coord.lane] =
            __float2bfloat16(attn_val * access.scale);
    }
#endif
}

template <bool NormalizeInputs>
__global__ void __launch_bounds__(kWarpSize* kNumWarps, NINFER_GDN_DIRECT_MIN_BLOCKS)
    recurrent_bf16_direct_pipelined_kernel(
        const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k,
        const __nv_bfloat16* __restrict__ v, const float* __restrict__ g,
        const float* __restrict__ beta, const float* __restrict__ state_read,
        float* __restrict__ state_write, __nv_bfloat16* __restrict__ out, std::int32_t width,
        head_map heads, float scale) {
    const DirectAccess access{q, k, v, g, beta, state_read, state_write, out, heads, width, scale};
    const RecurrentCoordinates coord = access.coordinates();
    __align__(16) float state[kDvPerWarp][kQkPerLane];
    load_state_tile(state, access.state_read_base(coord), coord);

    DirectTokenBits ring[kDirectPipelineDepth];
#pragma unroll
    for (int i = 0; i < kDirectPipelineDepth; ++i) {
        if (i < width) { ring[i] = load_direct_token(access, coord, i); }
    }
    for (std::int32_t base = 0; base < width; base += kDirectPipelineDepth) {
#pragma unroll
        for (int i = 0; i < kDirectPipelineDepth; ++i) {
            const std::int32_t token = base + i;
            if (token < width) {
                const DirectTokenBits current = ring[i];
                if (token + kDirectPipelineDepth < width) {
                    ring[i] = load_direct_token(access, coord, token + kDirectPipelineDepth);
                }
                consume_direct_token<NormalizeInputs>(state, current, access, coord, token);
            }
        }
    }
    store_state_tile(state, access.state_write_base(coord), coord);
}

// ---- Prepared form of the direct recurrence for prefill widths --------------------------------
// The direct kernels are bound by instruction count (384 CTAs x 4 warps, all resident, ~2 warp
// instructions per SM cycle), and a fifth of that is every warp re-deriving values that depend
// only on (token, head): the L2-normalized key and query, exp(g), and their bf16 unpacking. A
// prepass computes those once; the recurrence then reads FP32 planes.
//
// It also takes both state contractions against the pre-update state, using
//     M_t q = alpha (M_{t-1} q) + delta (k . q),
// so the readout no longer waits on the state update and k . q comes from the prepass, and it
// reduces the four rows of a warp with a transposed butterfly (6 shuffles for all four rows; lane
// L ends holding row L & 3) instead of four separate 5-level butterflies. The summation order
// differs from the direct kernels, so results agree to rounding, not bit for bit.
template <bool Normalize>
__global__ void recurrent_prepare_qk_kernel(const __nv_bfloat16* __restrict__ q,
                                            const __nv_bfloat16* __restrict__ k,
                                            float* __restrict__ qn, float* __restrict__ kn,
                                            float* __restrict__ kq, std::int64_t units) {
    const int lane          = static_cast<int>(threadIdx.x) & 31;
    const std::int64_t unit = static_cast<std::int64_t>(blockIdx.x) * (blockDim.x / 32) +
                              (static_cast<int>(threadIdx.x) >> 5);
    if (unit >= units) { return; }
    const std::int64_t base = unit * kStateDim + lane * kQkPerLane;
    float key[kQkPerLane];
    float query[kQkPerLane];
    unpack_qk_lane(key, load_vec<Bf16x4Pack>(k + base));
    normalize_qk_lane<Normalize>(key, lane);
    unpack_qk_lane(query, load_vec<Bf16x4Pack>(q + base));
    normalize_qk_lane<Normalize>(query, lane);
    store_vec(kn + base, load_vec<float4>(key));
    store_vec(qn + base, load_vec<float4>(query));
    float dot = 0.0f;
#pragma unroll
    for (int c = 0; c < kQkPerLane; ++c) { dot += key[c] * query[c]; }
    dot = warp_reduce_sum(dot);
    if (lane == 0) { kq[unit] = dot; }
}

__global__ void recurrent_prepare_gate_kernel(const float* __restrict__ g,
                                              float* __restrict__ alpha, std::int64_t count) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) { alpha[i] = expf(g[i]); }
}

// Four values per lane -> lane L holds the warp total of value[L & 3].
__device__ __forceinline__ float transposed_reduce4(const float (&value)[4], int lane) {
    const bool b0    = (lane & 1) != 0;
    const bool b1    = (lane & 2) != 0;
    const float lo   = (b0 ? value[1] : value[0]) +
                       __shfl_xor_sync(kFullWarpMask, b0 ? value[0] : value[1], 1, kWarpSize);
    const float hi   = (b0 ? value[3] : value[2]) +
                       __shfl_xor_sync(kFullWarpMask, b0 ? value[2] : value[3], 1, kWarpSize);
    float total      = (b1 ? hi : lo) + __shfl_xor_sync(kFullWarpMask, b1 ? lo : hi, 2, kWarpSize);
    total += __shfl_xor_sync(kFullWarpMask, total, 4, kWarpSize);
    total += __shfl_xor_sync(kFullWarpMask, total, 8, kWarpSize);
    total += __shfl_xor_sync(kFullWarpMask, total, 16, kWarpSize);
    return total;
}

struct PreparedTokenBits {
    float4 key;
    float4 query;
    float value;
    float alpha;
    float beta;
    float kq;
};

__global__ void __launch_bounds__(kWarpSize* kNumWarps, NINFER_GDN_DIRECT_MIN_BLOCKS)
    recurrent_bf16_direct_prepared_kernel(
        const float* __restrict__ qn, const float* __restrict__ kn, const float* __restrict__ kq,
        const float* __restrict__ alpha, const __nv_bfloat16* __restrict__ v,
        const float* __restrict__ beta, const float* __restrict__ state_read,
        float* __restrict__ state_write, __nv_bfloat16* __restrict__ out, std::int32_t width,
        head_map heads, float scale) {
    static_assert(kDvPerWarp == 4 && kQkPerLane == 4);
    const DirectAccess access{nullptr, nullptr, v,     nullptr, beta, state_read,
                              state_write, out, heads, width,   scale};
    const RecurrentCoordinates coord = access.coordinates();
    __align__(16) float state[kDvPerWarp][kQkPerLane];
    load_state_tile(state, access.state_read_base(coord), coord);

    const auto load_token = [&](std::int32_t token) {
        PreparedTokenBits bits;
        const std::int64_t qk_unit = static_cast<std::int64_t>(token) * heads.H_qk + coord.qk_head;
        const std::int64_t base    = qk_unit * kStateDim + coord.dqk_base;
        bits.key                   = load_vec<float4>(kn + base);
        bits.query                 = load_vec<float4>(qn + base);
        bits.value = coord.lane < kDvPerWarp
                         ? __bfloat162float(
                               access.value_ptr(coord, token)[coord.dv_base + coord.lane])
                         : 0.0f;
        const std::int64_t gate = static_cast<std::int64_t>(token) * heads.H_v + coord.value_head;
        bits.alpha              = alpha[gate];
        bits.beta               = beta[gate];
        bits.kq                 = kq[qk_unit];
        return bits;
    };
    const auto consume_token = [&](const PreparedTokenBits& bits, std::int32_t token) {
        const float key[4]   = {bits.key.x, bits.key.y, bits.key.z, bits.key.w};
        const float query[4] = {bits.query.x, bits.query.y, bits.query.z, bits.query.w};
        float pk[4];
        float pq[4];
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            float a = 0.0f;
            float b = 0.0f;
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                a = fmaf(state[r][c], key[c], a);
                b = fmaf(state[r][c], query[c], b);
            }
            pk[r] = a;
            pq[r] = b;
        }
        const float sk = transposed_reduce4(pk, coord.lane);
        const float sq = transposed_reduce4(pq, coord.lane);
        // Lanes 0..3 own rows 0..3 and their value element.
        const float delta_own = bits.beta * (bits.value - bits.alpha * sk);
        if (coord.lane < kDvPerWarp) {
            access.output_ptr(coord, token)[coord.dv_base + coord.lane] =
                __float2bfloat16((bits.alpha * sq + delta_own * bits.kq) * access.scale);
        }
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            const float delta = __shfl_sync(kFullWarpMask, delta_own, r, kWarpSize);
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                state[r][c] = fmaf(delta, key[c], bits.alpha * state[r][c]);
            }
        }
    };

    PreparedTokenBits ring[kDirectPipelineDepth];
#pragma unroll
    for (int i = 0; i < kDirectPipelineDepth; ++i) {
        if (i < width) { ring[i] = load_token(i); }
    }
    for (std::int32_t base = 0; base < width; base += kDirectPipelineDepth) {
#pragma unroll
        for (int i = 0; i < kDirectPipelineDepth; ++i) {
            const std::int32_t token = base + i;
            if (token < width) {
                const PreparedTokenBits current = ring[i];
                if (token + kDirectPipelineDepth < width) {
                    ring[i] = load_token(token + kDirectPipelineDepth);
                }
                consume_token(current, token);
            }
        }
    }
    store_state_tile(state, access.state_write_base(coord), coord);
}

template <bool NormalizeInputs>
__global__ void __launch_bounds__(kWarpSize* kNumWarps, 2)
    recurrent_batch_update_kernel(BatchUpdateAccess access) {
    const RecurrentCoordinates coord = access.coordinates();
    __align__(16) float state[kDvPerWarp][kQkPerLane];
    load_state_tile(state, access.state_read_base(coord), coord);
    run_recurrent_sequence<NormalizeInputs, OutputEffects>(state, access, coord, 1);
    store_state_tile(state, access.state_write_base(coord), coord);
}

template <bool Masked>
__global__ void __launch_bounds__(kWarpSize* kNumWarps, NINFER_GDN_DIRECT_MIN_BLOCKS)
    recurrent_record_kernel(RecordAccess<Masked> access) {
    const RecurrentCoordinates coord = access.coordinates();
    const std::int32_t valid         = access.active_columns(coord);
    if (valid > 0) {
        // Overlap token-0 input loads with state-tile HBM reads and prefetch query alongside
        // key/value/gate so readout never stalls on query DRAM latency (bit-exact).
        Bf16x4Pack cur_key_bits = load_vec<Bf16x4Pack>(access.key_ptr(coord, 0) + coord.dqk_base);
        Bf16x4Pack cur_q_bits   = load_vec<Bf16x4Pack>(access.query_ptr(coord, 0) + coord.dqk_base);
        RawGatePair gate        = access.load_gate(coord, 0);
        RawValueLane value      = load_value_lane(access.value_ptr(coord, 0), coord.lane,
                                                  coord.dv_base);

        __align__(16) float state[kDvPerWarp][kQkPerLane];
        load_state_tile(state, access.state_read_base(coord), coord);

        RawQkLane key;
        key.bits = cur_key_bits;
        unpack_qk_lane(key.value, cur_key_bits);
        access.store_key(coord, 0, key);
        access.store_value(coord, 0, value);
        access.store_gate(coord, 0, gate);
        normalize_qk_lane<true>(key.value, coord.lane);

        float query[kQkPerLane];
        unpack_qk_lane(query, cur_q_bits);
        normalize_qk_lane<true>(query, coord.lane);

        for (std::int32_t token = 0; token < valid; ++token) {
            Bf16x4Pack next_key_bits{};
            Bf16x4Pack next_q_bits{};
            RawGatePair next_gate{};
            RawValueLane next_value{};
            if (token + 1 < valid) {
                next_key_bits =
                    load_vec<Bf16x4Pack>(access.key_ptr(coord, token + 1) + coord.dqk_base);
                next_q_bits =
                    load_vec<Bf16x4Pack>(access.query_ptr(coord, token + 1) + coord.dqk_base);
                next_gate  = access.load_gate(coord, token + 1);
                next_value = load_value_lane(access.value_ptr(coord, token + 1), coord.lane,
                                             coord.dv_base);
            }

            apply_gdn_transition(state, key.value, value.value, gate.g, gate.beta);
            readout_normalized_and_store(state, query, access.output_ptr(coord, token),
                                         coord.dv_base, coord.lane, access.scale);

            if (token + 1 < valid) {
                key.bits = next_key_bits;
                unpack_qk_lane(key.value, next_key_bits);
                gate  = next_gate;
                value = next_value;
                access.store_key(coord, token + 1, key);
                access.store_value(coord, token + 1, value);
                access.store_gate(coord, token + 1, gate);
                normalize_qk_lane<true>(key.value, coord.lane);
                unpack_qk_lane(query, next_q_bits);
                normalize_qk_lane<true>(query, coord.lane);
            }
        }
    }
    zero_output_suffix(access, coord, valid, access.width);
}

template <class Geometry>
__global__ void __launch_bounds__(kWarpSize* kNumWarps, NINFER_GDN_DIRECT_MIN_BLOCKS)
    recurrent_fold_kernel(const __grid_constant__ FoldAccess<Geometry> access) {
    const RecurrentCoordinates coord = access.coordinates();
    const std::int32_t valid         = access.active_columns(coord);
    if (valid == 0) { return; }
    __align__(16) float state[kDvPerWarp][kQkPerLane];
    load_state_tile(state, access.state_read_base(coord), coord);
    run_recurrent_sequence<true, FoldEffects>(state, access, coord, valid);
    access.store_final_state(coord, state);
    access.publish_final_conv_history(coord, valid);
}

} // namespace ninfer::ops::detail::gated_delta_net
