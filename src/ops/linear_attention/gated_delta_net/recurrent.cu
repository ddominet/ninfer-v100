#include "ops/linear_attention/gated_delta_net/launch.h"

#include "core/device.h"
#include "ops/linear_attention/gated_delta_net/recurrent.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <cstdlib>
#include <new>
#include <stdexcept>
#include <type_traits>

namespace ninfer::ops::detail::gated_delta_net {
namespace {

static_assert(sizeof(GdnReplayFoldKernelRow) == 16);
static_assert(alignof(GdnReplayFoldKernelRow) == 16);
static_assert(sizeof(GdnReplayFoldKernelRows) == 128);
static_assert(alignof(GdnReplayFoldKernelRows) == 16);
static_assert(std::is_trivially_copyable_v<GdnReplayFoldKernelRows>);

template <bool NormalizeQK, class StateT>
void launch_recurrent_direct_fixed(const Tensor& q, const Tensor& k, const Tensor& v,
                                   const Tensor& g, const Tensor& beta, float scale,
                                   const Tensor& state_read, Tensor& state_write, Tensor& out,
                                   float* prepared, cudaStream_t stream) {
    const auto heads = head_map::of(q.ne[1], v.ne[1]);
    const dim3 grid(static_cast<unsigned>(v.ne[1]), 1, static_cast<unsigned>(kStateDim / kBlockDv));
    const dim3 block(kWarpSize, kNumWarps, 1);
    // Pipelined form hides q/k/v/gate latency on long sequences. At decode width=1 the ring
    // has nothing to hide and only burns registers; use the plain kernel so more CTAs fit
    // (Qwen3.6-27B: 48 value heads × 8 state tiles = 384 CTAs on an 80-SM V100).
    static const bool pipelined_env = [] {
        const char* value = std::getenv("NINFER_GDN_PIPELINED");
        return value == nullptr || value[0] != '0';
    }();
    // Prefill widths: hoist the per-(token, head) work into a prepass and run the leaner
    // recurrence (recurrent.cuh, "Prepared form"): 3.45 -> 2.24 ms per layer-chunk at T=2048 on a
    // V100 including the prepass, outputs within one bf16 step of the direct kernel.
    if (prepared != nullptr) {
        const std::int64_t width      = q.ne[2];
        const std::int64_t qk_units   = width * q.ne[1];
        const std::int64_t gate_count = width * v.ne[1];
        float* qn    = prepared;
        float* kn    = qn + qk_units * kStateDim;
        float* kq    = kn + qk_units * kStateDim;
        float* alpha = kq + qk_units;
        recurrent_prepare_qk_kernel<NormalizeQK>
            <<<static_cast<unsigned>((qk_units + 7) / 8), 256, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(q.data),
                static_cast<const __nv_bfloat16*>(k.data), qn, kn, kq, qk_units);
        recurrent_prepare_gate_kernel
            <<<static_cast<unsigned>((gate_count + 255) / 256), 256, 0, stream>>>(
                static_cast<const float*>(g.data), alpha, gate_count);
        recurrent_bf16_direct_prepared_kernel<StateT><<<grid, block, 0, stream>>>(
            qn, kn, kq, alpha, static_cast<const __nv_bfloat16*>(v.data),
            static_cast<const float*>(beta.data), static_cast<const StateT*>(state_read.data),
            static_cast<StateT*>(state_write.data), static_cast<__nv_bfloat16*>(out.data), q.ne[2],
            heads, scale);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    const bool pipelined = pipelined_env && q.ne[2] > 1;
    auto* kernel = pipelined ? recurrent_bf16_direct_pipelined_kernel<NormalizeQK, StateT>
                             : recurrent_bf16_direct_kernel<NormalizeQK, StateT>;
    kernel<<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(q.data), static_cast<const __nv_bfloat16*>(k.data),
        static_cast<const __nv_bfloat16*>(v.data), static_cast<const float*>(g.data),
        static_cast<const float*>(beta.data), static_cast<const StateT*>(state_read.data),
        static_cast<StateT*>(state_write.data), static_cast<__nv_bfloat16*>(out.data), q.ne[2],
        heads, scale);
    CUDA_CHECK(cudaGetLastError());
}

template <bool NormalizeInputs, class StateT>
void launch_recurrent_batch_update_fixed(const Tensor& q, const Tensor& k, const Tensor& v,
                                         const Tensor& g, const Tensor& beta, float scale,
                                         Tensor& ssm_states, const Tensor& source_state_slots,
                                         const Tensor& destination_state_slots, Tensor& out,
                                         cudaStream_t stream) {
    const auto heads = head_map::of(q.ne[1], v.ne[1]);
    const dim3 grid(static_cast<unsigned>(v.ne[1]), static_cast<unsigned>(q.ne[3]),
                    static_cast<unsigned>(kStateDim / kBlockDv));
    const dim3 block(kWarpSize, kNumWarps, 1);
    const std::int64_t state_slot_stride =
        static_cast<std::int64_t>(kStateDim) * kStateDim * ssm_states.ne[2];
    const BatchUpdateAccess<StateT> access{
        static_cast<const __nv_bfloat16*>(q.data),
        static_cast<const __nv_bfloat16*>(k.data),
        static_cast<const __nv_bfloat16*>(v.data),
        static_cast<const float*>(g.data),
        static_cast<const float*>(beta.data),
        static_cast<const StateT*>(ssm_states.data),
        static_cast<StateT*>(ssm_states.data),
        static_cast<const std::int32_t*>(source_state_slots.data),
        static_cast<const std::int32_t*>(destination_state_slots.data),
        static_cast<__nv_bfloat16*>(out.data),
        heads,
        state_slot_stride,
        scale,
    };
    recurrent_batch_update_kernel<NormalizeInputs, StateT><<<grid, block, 0, stream>>>(access);
    CUDA_CHECK(cudaGetLastError());
}

template <bool Masked, class StateT>
void launch_recurrent_record_fixed(const Tensor& q, const Tensor& k, const Tensor& v,
                                   const Tensor& g, const Tensor& beta, float scale,
                                   const Tensor& ssm_states, const Tensor& valid_columns,
                                   const Tensor& initial_state_slots, Tensor& key_record,
                                   Tensor& value_record, Tensor& gate_record, Tensor& out,
                                   cudaStream_t stream) {
    const auto heads = head_map::of(q.ne[1], v.ne[1]);
    const dim3 grid(static_cast<unsigned>(v.ne[1]), static_cast<unsigned>(q.ne[3]),
                    static_cast<unsigned>(kStateDim / kBlockDv));
    const dim3 block(kWarpSize, kNumWarps, 1);
    const std::int64_t state_slot_stride =
        static_cast<std::int64_t>(kStateDim) * kStateDim * ssm_states.ne[2];
    const RecordAccess<Masked, StateT> access{
        static_cast<const __nv_bfloat16*>(q.data),
        static_cast<const __nv_bfloat16*>(k.data),
        static_cast<const __nv_bfloat16*>(v.data),
        static_cast<const float*>(g.data),
        static_cast<const float*>(beta.data),
        static_cast<const StateT*>(ssm_states.data),
        Masked ? static_cast<const std::int32_t*>(valid_columns.data) : nullptr,
        static_cast<const std::int32_t*>(initial_state_slots.data),
        static_cast<__nv_bfloat16*>(key_record.data),
        static_cast<__nv_bfloat16*>(value_record.data),
        reinterpret_cast<uint2*>(gate_record.data),
        static_cast<__nv_bfloat16*>(out.data),
        heads,
        q.ne[2],
        state_slot_stride,
        scale,
    };
    recurrent_record_kernel<Masked, StateT><<<grid, block, 0, stream>>>(access);
    CUDA_CHECK(cudaGetLastError());
}

template <class Geometry, class StateT>
void launch_replay_fold_fixed(const GdnReplayRecords& records,
                              LinearAttentionStateAllLayersView states,
                              const GdnReplayFoldKernelRows& rows, std::int32_t active_rows,
                              cudaStream_t stream) {
    const FoldAccess<Geometry, StateT> access{
        static_cast<const __nv_bfloat16*>(records.key.data),
        static_cast<const __nv_bfloat16*>(records.value.data),
        reinterpret_cast<const uint2*>(records.gate.data),
        static_cast<const __nv_bfloat16*>(records.conv.data),
        static_cast<StateT*>(states.recurrent_layer0.data),
        static_cast<__nv_bfloat16*>(states.conv_layer0.data),
        states.recurrent_layer_stride_bytes / static_cast<std::int64_t>(sizeof(StateT)),
        states.conv_layer_stride_bytes / static_cast<std::int64_t>(sizeof(__nv_bfloat16)),
        records.spec.record_capacity,
        records.spec.width,
        rows,
    };
    const dim3 grid(static_cast<unsigned>(Geometry::kValueHeads),
                    static_cast<unsigned>(active_rows),
                    static_cast<unsigned>(Geometry::kLayers * (kStateDim / kBlockDv)));
    const dim3 block(kWarpSize, kNumWarps, 1);
    recurrent_fold_kernel<Geometry, StateT><<<grid, block, 0, stream>>>(access);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void launch_recurrent(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                      const Tensor& beta, float scale, bool normalize_qk, Tensor& ssm_state,
                      Tensor& out, cudaStream_t stream) {
    const bool fp16 = ssm_state.dtype == DType::FP16;
    if (normalize_qk) {
        if (fp16) {
            launch_recurrent_direct_fixed<true, __half>(q, k, v, g, beta, scale, ssm_state,
                                                        ssm_state, out, nullptr, stream);
        } else {
            launch_recurrent_direct_fixed<true, float>(q, k, v, g, beta, scale, ssm_state,
                                                       ssm_state, out, nullptr, stream);
        }
    } else {
        if (fp16) {
            launch_recurrent_direct_fixed<false, __half>(q, k, v, g, beta, scale, ssm_state,
                                                         ssm_state, out, nullptr, stream);
        } else {
            launch_recurrent_direct_fixed<false, float>(q, k, v, g, beta, scale, ssm_state,
                                                        ssm_state, out, nullptr, stream);
        }
    }
}

std::size_t recurrent_prepared_bytes(std::int32_t qk_heads, std::int32_t value_heads,
                                     std::int32_t tokens) {
#ifdef NINFER_VOLTA_BUILD
    // Below 64 tokens the prepass costs more than it saves; decode keeps the plain kernel.
    if (tokens < 64) { return 0; }
    const std::size_t qk_units = static_cast<std::size_t>(tokens) * qk_heads;
    return (2 * qk_units * kStateDim + qk_units + static_cast<std::size_t>(tokens) * value_heads) *
           sizeof(float);
#else
    (void)qk_heads;
    (void)value_heads;
    (void)tokens;
    return 0;
#endif
}

void launch_recurrent_inout(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                            const Tensor& beta, float scale, bool normalize_qk,
                            const Tensor& ssm_state_in, Tensor& ssm_state_out, Tensor& out,
                            DeviceSpan prepared, cudaStream_t stream) {
    float* planes = nullptr;
    if (const std::size_t need = recurrent_prepared_bytes(q.ne[1], v.ne[1], q.ne[2]); need > 0) {
        if (prepared.data == nullptr || prepared.bytes < need) { throw std::bad_alloc(); }
        planes = static_cast<float*>(prepared.data);
    }
    const bool fp16 = ssm_state_in.dtype == DType::FP16;
    if (normalize_qk) {
        if (fp16) {
            launch_recurrent_direct_fixed<true, __half>(q, k, v, g, beta, scale, ssm_state_in,
                                                        ssm_state_out, out, planes, stream);
        } else {
            launch_recurrent_direct_fixed<true, float>(q, k, v, g, beta, scale, ssm_state_in,
                                                       ssm_state_out, out, planes, stream);
        }
    } else {
        if (fp16) {
            launch_recurrent_direct_fixed<false, __half>(q, k, v, g, beta, scale, ssm_state_in,
                                                         ssm_state_out, out, planes, stream);
        } else {
            launch_recurrent_direct_fixed<false, float>(q, k, v, g, beta, scale, ssm_state_in,
                                                        ssm_state_out, out, planes, stream);
        }
    }
}

void launch_recurrent_batch_update(const Tensor& q, const Tensor& k, const Tensor& v,
                                   const Tensor& g, const Tensor& beta, float scale,
                                   bool normalize_qk, Tensor& ssm_states,
                                   const Tensor& source_state_slots,
                                   const Tensor& destination_state_slots, Tensor& out,
                                   cudaStream_t stream) {
    const bool fp16 = ssm_states.dtype == DType::FP16;
    if (normalize_qk) {
        if (fp16) {
            launch_recurrent_batch_update_fixed<true, __half>(
                q, k, v, g, beta, scale, ssm_states, source_state_slots, destination_state_slots,
                out, stream);
        } else {
            launch_recurrent_batch_update_fixed<true, float>(
                q, k, v, g, beta, scale, ssm_states, source_state_slots, destination_state_slots,
                out, stream);
        }
    } else {
        if (fp16) {
            launch_recurrent_batch_update_fixed<false, __half>(
                q, k, v, g, beta, scale, ssm_states, source_state_slots, destination_state_slots,
                out, stream);
        } else {
            launch_recurrent_batch_update_fixed<false, float>(
                q, k, v, g, beta, scale, ssm_states, source_state_slots, destination_state_slots,
                out, stream);
        }
    }
}

void launch_recurrent_record(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                             const Tensor& beta, float scale, const Tensor& ssm_states,
                             const Tensor& valid_columns, const Tensor& initial_state_slots,
                             Tensor& key_record, Tensor& value_record, Tensor& gate_record,
                             Tensor& out, cudaStream_t stream) {
    const bool fp16 = ssm_states.dtype == DType::FP16;
    if (valid_columns.data == nullptr) {
        if (fp16) {
            launch_recurrent_record_fixed<false, __half>(
                q, k, v, g, beta, scale, ssm_states, valid_columns, initial_state_slots, key_record,
                value_record, gate_record, out, stream);
        } else {
            launch_recurrent_record_fixed<false, float>(
                q, k, v, g, beta, scale, ssm_states, valid_columns, initial_state_slots, key_record,
                value_record, gate_record, out, stream);
        }
    } else {
        if (fp16) {
            launch_recurrent_record_fixed<true, __half>(
                q, k, v, g, beta, scale, ssm_states, valid_columns, initial_state_slots, key_record,
                value_record, gate_record, out, stream);
        } else {
            launch_recurrent_record_fixed<true, float>(
                q, k, v, g, beta, scale, ssm_states, valid_columns, initial_state_slots, key_record,
                value_record, gate_record, out, stream);
        }
    }
}

void launch_replay_fold(const GdnReplayRecords& records, LinearAttentionStateAllLayersView states,
                        const GdnReplayFoldKernelRows& rows, std::int32_t active_rows,
                        cudaStream_t stream) {
    const bool fp16 = states.recurrent_layer0.dtype == DType::FP16;
    if (records.spec.layers == FoldGeometry48x48::kLayers &&
        records.spec.qk_heads == FoldGeometry48x48::kQkHeads &&
        records.spec.value_heads == FoldGeometry48x48::kValueHeads &&
        records.spec.conv_channels == FoldGeometry48x48::kConvChannels) {
        if (fp16) {
            launch_replay_fold_fixed<FoldGeometry48x48, __half>(records, states, rows, active_rows,
                                                                stream);
        } else {
            launch_replay_fold_fixed<FoldGeometry48x48, float>(records, states, rows, active_rows,
                                                               stream);
        }
        return;
    }
    if (records.spec.layers == FoldGeometry30x32::kLayers &&
        records.spec.qk_heads == FoldGeometry30x32::kQkHeads &&
        records.spec.value_heads == FoldGeometry30x32::kValueHeads &&
        records.spec.conv_channels == FoldGeometry30x32::kConvChannels) {
        if (fp16) {
            launch_replay_fold_fixed<FoldGeometry30x32, __half>(records, states, rows, active_rows,
                                                                stream);
        } else {
            launch_replay_fold_fixed<FoldGeometry30x32, float>(records, states, rows, active_rows,
                                                               stream);
        }
        return;
    }
    throw std::invalid_argument("GDN replay fold launcher received an unregistered geometry");
}

} // namespace ninfer::ops::detail::gated_delta_net
