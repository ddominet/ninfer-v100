#include "targets/qwen3_6_27b/impl/variant.h"

#include "ninfer/ops/attn_input_proj.h"
#include "ninfer/ops/gdn_gating_proj.h"
#include "ninfer/ops/gdn_input_proj.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_pair.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ninfer/ops/mtp_pack.h"
#include "ninfer/ops/residual_add.h"
#include "ninfer/ops/silu_mul.h"
#ifdef NINFER_VOLTA_BUILD
#include "ops/attn_input_proj/fp8/fp8_attn_input_plan.h"
#include "ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_kernels.h"
#include "ops/gdn_input_proj/fp8/fp8_gdn_input_plan.h"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_plan.h"
#include "ops/launcher/rmsnorm.h"
#include "ops/launcher/sigmoid_gate_mul.h"
#include "ops/linear/common/volta_qpn_groupwise.h"
#include "ops/linear/fp8/fp8_launch.h"
#include "ops/linear/nvfp4/nvfp4_launch.h"
#include "ops/linear/volta_fp16_stage.h"
#include "ops/linear/volta_qpn_width.h"
#include "ops/linear/volta_tc_skinny.h"
#include "ops/linear_add/fp8/fp8_linear_add_plan.h"
#include "ops/linear_add/nvfp4/nvfp4_linear_add_plan.h"
#include "ops/linear_swiglu/fp8/fp8_linear_swiglu_plan.h"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.h"
#include <cstdlib>
#endif

#include <algorithm>
#include <stdexcept>

#define NINFER_QWEN36_VARIANT    ::ninfer::targets::qwen3_6_27b::detail::Variant
#define NINFER_QWEN36_RUNTIME_NS qwen3_6_27b_runtime
#include "targets/qwen3_6/impl/runtime/instantiate.h"

namespace ninfer::targets::qwen3_6_27b::detail {
namespace {

std::vector<GraphExecutionProfile>
graph_profiles_through(std::uint32_t max_frontier,
                       const std::vector<std::uint32_t>& preferred_ends) {
    std::vector<GraphExecutionProfile> out;
    std::uint32_t begin = 0;
    for (const std::uint32_t preferred_end : preferred_ends) {
        if (begin > max_frontier) { break; }
        const std::uint32_t end = std::min(preferred_end, max_frontier);
        out.push_back({begin, end});
        if (end == max_frontier) { return out; }
        begin = end + 1;
    }
    if (begin <= max_frontier) { out.push_back({begin, max_frontier}); }
    return out;
}

void validate_token_interval(std::int32_t first, std::int32_t last) {
    if (first <= 0 || last < first) {
        throw std::invalid_argument("invalid target leaf token interval");
    }
}

#ifdef NINFER_VOLTA_BUILD
constexpr ops::LinearPolicy kNvfp4TextPolicy = ops::LinearPolicy::A16Only;
constexpr ops::LinearPolicy kFp8TextPolicy   = ops::LinearPolicy::A16Only;
#else
constexpr ops::LinearPolicy kNvfp4TextPolicy = ops::LinearPolicy::AllowA4;
constexpr ops::LinearPolicy kFp8TextPolicy   = ops::LinearPolicy::AllowA8;
#endif

ops::LinearPolicy text_policy(const Weight& weight) {
    switch (weight.qtype) {
    case QType::NVFP4:
        return kNvfp4TextPolicy;
    case QType::FP8_E4M3FN_ROW_BF16S:
        return kFp8TextPolicy;
    default:
        return ops::LinearPolicy::A16Only;
    }
}

constexpr std::size_t kMinimumLeafWorkspaceBytes = 1;

std::size_t gdn_snapshot_workspace_bytes(const Tensor& hidden,
                                         const Variant::GdnProjectionWeights& weights) {
    const std::int32_t batch = hidden.ne[2];
    const std::int32_t width = hidden.ne[1];
    if (std::holds_alternative<SplitGdnInputProjectionPayload>(weights.input_projection)) {
        return std::max(kMinimumLeafWorkspaceBytes,
                        ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
                            TextConfig::key_dim, TextConfig::key_dim, TextConfig::value_dim, batch,
                            width, width));
    }
    const Weight& parent =
        std::get<FusedGdnInputProjectionPayload>(weights.input_projection).query_key_value_z;
    return std::max(
        kMinimumLeafWorkspaceBytes,
        ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
            parent.qtype, parent.n, parent.k, text_policy(parent), batch, width, width));
}

std::size_t gdn_record_workspace_bytes(const Tensor& hidden,
                                       const Variant::GdnProjectionWeights& weights) {
    const std::int32_t batch = hidden.ne[2];
    const std::int32_t width = hidden.ne[1];
    if (std::holds_alternative<SplitGdnInputProjectionPayload>(weights.input_projection)) {
        return std::max(kMinimumLeafWorkspaceBytes,
                        ops::gdn_input_proj_conv_record_workspace_capacity_bytes(
                            TextConfig::key_dim, TextConfig::key_dim, TextConfig::value_dim, batch,
                            width, width));
    }
    const Weight& parent =
        std::get<FusedGdnInputProjectionPayload>(weights.input_projection).query_key_value_z;
    return std::max(
        kMinimumLeafWorkspaceBytes,
        ops::gdn_input_proj_conv_record_workspace_capacity_bytes(
            parent.qtype, parent.n, parent.k, text_policy(parent), batch, width, width));
}

std::size_t post_mixer_workspace_bytes(QType gate_up_qtype, QType down_qtype,
                                       ops::LinearPolicy policy, std::int32_t first,
                                       std::int32_t last) {
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::BF16, {TextConfig::intermediate, last});
    {
        auto scope = layout.scope();
        (void)layout.alloc_bytes(ops::linear_swiglu_workspace_capacity_bytes(
            gate_up_qtype, 2 * TextConfig::intermediate, TextConfig::hidden, policy, first, last));
    }
    {
        auto scope = layout.scope();
        (void)layout.alloc_bytes(ops::linear_add_workspace_capacity_bytes(
            down_qtype, TextConfig::hidden, TextConfig::intermediate, policy, first, last));
    }
    std::size_t peak = layout.peak_bytes(1);
#ifdef NINFER_VOLTA_BUILD
    const bool volta_fused_mlp =
        policy == ops::LinearPolicy::A16Only && first <= ops::detail::kVoltaQpnMaxTokens &&
        (((gate_up_qtype == QType::NVFP4 || gate_up_qtype == QType::FP8_E4M3FN_ROW_BF16S) &&
          down_qtype == gate_up_qtype) ||
         (gate_up_qtype == QType::Q4G64_F16S &&
          (down_qtype == QType::Q4G64_F16S || down_qtype == QType::Q5G64_F16S)));
    if (volta_fused_mlp) {
        const std::int32_t fused_t = std::min(last, ops::detail::kVoltaQpnMaxTokens);
        WorkspaceLayoutBuilder fused;
        if (gate_up_qtype == QType::NVFP4 || gate_up_qtype == QType::Q4G64_F16S) {
            const int tiles = ops::detail::volta_group_major_tiles(fused_t);
            (void)fused.alloc_bytes(static_cast<std::size_t>(TextConfig::hidden) * tiles * 8 *
                                        sizeof(std::uint16_t),
                                    256);
            (void)fused.alloc_bytes(static_cast<std::size_t>(TextConfig::intermediate) * fused_t *
                                        sizeof(float),
                                    256);
            (void)fused.alloc_bytes(static_cast<std::size_t>(TextConfig::intermediate) * tiles * 8 *
                                        sizeof(std::uint16_t),
                                    256);
        } else {
            (void)fused.alloc_bytes(static_cast<std::size_t>(TextConfig::hidden) * fused_t *
                                        sizeof(std::uint16_t),
                                    256);
            (void)fused.alloc_bytes(static_cast<std::size_t>(TextConfig::intermediate) * fused_t *
                                        sizeof(float),
                                    256);
            (void)fused.alloc_bytes(static_cast<std::size_t>(TextConfig::intermediate) * fused_t *
                                        sizeof(std::uint16_t),
                                    256);
        }
        peak = std::max(peak, fused.peak_bytes(1));
    }
#endif
    return peak;
}

} // namespace

std::vector<GraphExecutionProfile> Variant::ordinary_graph_profiles(std::uint32_t capacity) {
    // E+1 is the one-token visible window. Early ranges limit empty producer CTAs; later ranges
    // follow measured split-policy transitions until the producer grid reaches its fixed cap.
    return graph_profiles_through(capacity - 1, {127, 511, 2047, 4095, 8197, 16389, 32767});
}

std::vector<GraphExecutionProfile> Variant::mtp_graph_profiles(std::uint32_t capacity,
                                                               std::uint32_t draft_window) {
    if (draft_window == 0 || capacity == 0) { return {}; }
    // Bound the final AR window E+2K at split-policy transitions until the grid reaches its cap.
    std::vector<std::uint32_t> ends;
    const auto add_shifted = [&](std::uint32_t visible_end, std::uint32_t offset) {
        if (visible_end >= offset) { ends.push_back(visible_end - offset); }
    };
    for (const std::uint32_t visible_end : {128U, 512U, 2048U, 4096U, 8198U, 16390U, 32768U}) {
        add_shifted(visible_end, 2 * draft_window);
    }
    // Target verify and MTP batch both have T=K+1 and W=E+K+1. Preserve one concrete INT8
    // implementation per range at the T=4/5/6 launch boundaries.
    if (draft_window == 3) {
        add_shifted(1029, draft_window + 1);
    } else if (draft_window == 4) {
        for (const std::uint32_t visible_end : {128U, 512U, 1029U}) {
            add_shifted(visible_end, draft_window + 1);
        }
    } else if (draft_window == 5) {
        for (const std::uint32_t visible_end : {128U, 160U, 2054U, 8198U}) {
            add_shifted(visible_end, draft_window + 1);
        }
    }
    std::sort(ends.begin(), ends.end());
    ends.erase(std::unique(ends.begin(), ends.end()), ends.end());
    return graph_profiles_through(capacity - 1, ends);
}

std::vector<GraphExecutionProfile>
Variant::dflash_graph_profiles(std::uint32_t capacity, std::uint32_t draft_window, std::uint32_t) {
    if (capacity == 0 || draft_window == 0 || draft_window > maximum_dflash_draft_tokens) {
        throw std::invalid_argument("invalid DFlash2 graph dimensions");
    }
    // Bounded attention envelopes; each tier owns its topology and can change kernel decomposition.
    auto profiles = graph_profiles_through(capacity - 1, {96, 511, 2047, 8191, 32767});
    for (std::size_t i = 0; i < profiles.size(); ++i) {
        profiles[i].topology_class = static_cast<std::uint32_t>(i);
    }
    return profiles;
}

void Variant::attention_projection(const Tensor& hidden,
                                   const FullAttentionProjectionWeights& weights, Tensor& query,
                                   Tensor& gate, Tensor& key, Tensor& value, qwen3_6::TextPhase,
                                   WorkspaceArena& workspace, cudaStream_t stream) {
    if (const auto* split = std::get_if<SplitAttentionProjectionPayload>(&weights)) {
        ops::attn_input_proj(hidden, split->query_key, split->gate_value, query, gate, key, value,
                             workspace, stream);
        return;
    }
    const Weight& fused = std::get<FusedAttentionProjectionPayload>(weights).query_key_gate_value;
    ops::attn_input_proj(hidden, fused, query, gate, key, value, text_policy(fused), workspace,
                         stream);
}

#ifdef NINFER_VOLTA_BUILD
bool Variant::attention_projection_from_residual(const Tensor& residual, const Tensor& input_norm,
                                                 float eps,
                                                 const FullAttentionProjectionWeights& weights,
                                                 Tensor& query, Tensor& gate, Tensor& key,
                                                 Tensor& value, qwen3_6::TextPhase,
                                                 WorkspaceArena& workspace, cudaStream_t stream) {
    const std::int32_t t = residual.ne[1];
    if (residual.ne[0] != TextConfig::hidden || t <= 0 || t > ops::detail::kVoltaQpnMaxTokens ||
        !ops::detail::volta_stage_fp16_enabled()) {
        return false;
    }
    if (const auto* split = std::get_if<SplitAttentionProjectionPayload>(&weights)) {
        if (split->query_key.qtype != QType::Q4G64_F16S ||
            split->gate_value.qtype != QType::Q5G64_F16S ||
            !ops::detail::volta_group_major_enabled() ||
            !ops::detail::volta_qpn_groupwise_supported(split->query_key) ||
            !ops::detail::volta_qpn_groupwise_supported(split->gate_value)) {
            return false;
        }
        auto scope      = workspace.scope();
        const int tiles = ops::detail::volta_group_major_tiles(t);
        auto* staged_gm = static_cast<ops::detail::VoltaGroupMajorHalf*>(
            workspace
                .alloc_bytes(static_cast<std::size_t>(TextConfig::hidden) * tiles * 8 *
                                 sizeof(std::uint16_t),
                             256)
                .data);
        ops::detail::volta_rmsnorm_offset_d5120_to_group_major(residual, input_norm, eps, staged_gm,
                                                               stream);
        ops::detail::launch_q4_prepacked_volta_split_gm(residual, staged_gm, split->query_key,
                                                        query, key, query.ne[0], stream);
        ops::detail::launch_q5_prepacked_volta_split_gm(residual, staged_gm, split->gate_value,
                                                        gate, value, gate.ne[0], stream);
        return true;
    }
    const auto* payload = std::get_if<FusedAttentionProjectionPayload>(&weights);
    if (payload == nullptr) { return false; }
    const Weight& fused = payload->query_key_gate_value;
    if (fused.qtype != QType::FP8_E4M3FN_ROW_BF16S ||
        text_policy(fused) != ops::LinearPolicy::A16Only ||
        t > ops::detail::kFp8VoltaQpnMaxTokens ||
        !ops::detail::fp8_volta_qpn_supported(fused.n, fused.k, t)) {
        return false;
    }
    auto scope            = workspace.scope();
    DeviceSpan activation = workspace.alloc_bytes(
        static_cast<std::size_t>(TextConfig::hidden) * t * sizeof(std::uint16_t), 256);
    ops::detail::volta_rmsnorm_offset_d5120_to_fp16(
        residual, input_norm, eps, static_cast<half*>(activation.data), stream);
    ops::detail::launch_fp8_attn_input_volta_qpn(residual, fused, query, activation.data, gate, key,
                                                 value, stream);
    return true;
}
#endif

void Variant::attention_output_projection(const Tensor& attention, const Weight& weight,
                                          Tensor& residual, qwen3_6::TextPhase,
                                          WorkspaceArena& workspace, cudaStream_t stream) {
    ops::linear_add(attention, weight, residual, text_policy(weight), workspace, stream);
}

#ifdef NINFER_VOLTA_BUILD
bool Variant::attention_output_projection_from_gated(const Tensor& gate, Tensor& attention,
                                                     const Weight& weight, Tensor& residual,
                                                     qwen3_6::TextPhase, WorkspaceArena&,
                                                     cudaStream_t stream) {
    const std::int32_t t = attention.ne[2];
    if (t <= 0 || t > ops::detail::kVoltaQpnMaxTokens || !ops::detail::volta_stage_fp16_enabled()) {
        return false;
    }
    if ((weight.qtype == QType::Q4G64_F16S || weight.qtype == QType::Q5G64_F16S) &&
        ops::detail::volta_qpn_groupwise_supported(weight)) {
        if (!ops::detail::sigmoid_gate_mul_to_fp16_launch(gate, attention, stream)) {
            return false;
        }
        Tensor flat = attention.view({TextConfig::query_size, t});
        if (weight.qtype == QType::Q4G64_F16S) {
            ops::detail::launch_q4_prepacked_volta_residual_fp16(
                flat, static_cast<const half*>(attention.data), weight, residual, stream);
        } else {
            ops::detail::launch_q5_prepacked_volta_residual_fp16(
                flat, static_cast<const half*>(attention.data), weight, residual, stream);
        }
        return true;
    }
    if (weight.qtype != QType::FP8_E4M3FN_ROW_BF16S ||
        text_policy(weight) != ops::LinearPolicy::A16Only ||
        t > ops::detail::kFp8VoltaQpnMaxTokens ||
        !ops::detail::fp8_volta_qpn_supported(weight.n, weight.k, t)) {
        return false;
    }
    if (!ops::detail::sigmoid_gate_mul_to_fp16_launch(gate, attention, stream)) { return false; }
    Tensor flat = attention.view({TextConfig::query_size, t});
    ops::detail::fp8_linear_add_qpn_fp16_launch(flat, static_cast<const half*>(attention.data),
                                                weight, residual, stream);
    return true;
}
#endif

void Variant::mtp_attention_projection(const Tensor& hidden,
                                       const MtpAttentionProjectionWeights& weights, Tensor& query,
                                       Tensor& gate, Tensor& key, Tensor& value,
                                       WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope     = workspace.scope();
    const int cols = hidden.ne[1];
    Tensor packed  = workspace.alloc(DType::BF16, {TextConfig::mtp_attention_input_rows, cols});
    ops::linear(hidden, weights.packed, packed, stream);
    Tensor query_heads = query.view({TextConfig::head_dim, TextConfig::query_heads, cols});
    Tensor key_heads   = key.view({TextConfig::head_dim, TextConfig::kv_heads, cols});
    Tensor gate_heads  = gate.view({TextConfig::head_dim, TextConfig::query_heads, cols});
    Tensor value_heads = value.view({TextConfig::head_dim, TextConfig::kv_heads, cols});
    ops::mtp_split_attn_in(packed, query_heads, key_heads, gate_heads, value_heads, stream);
}

void Variant::mtp_kv_projection(const Tensor& hidden, const MtpAttentionProjectionWeights& weights,
                                Tensor& key, Tensor& value, WorkspaceArena&, cudaStream_t stream) {
    ops::linear_pair(hidden, weights.key, weights.value, key, value, stream);
}

void Variant::mtp_q_gate_projection(const Tensor& hidden,
                                    const MtpAttentionProjectionWeights& weights, Tensor& query,
                                    Tensor& gate, WorkspaceArena&, cudaStream_t stream) {
    ops::linear(hidden, weights.query, query, stream);
    ops::linear(hidden, weights.output_gate, gate, stream);
}

bool Variant::gdn_input_projection_scatter(const Tensor& hidden,
                                           const GdnProjectionWeights& weights,
                                           Tensor& qkv_padded, Tensor& output_gate,
                                           const Tensor& qkv_columns, cudaStream_t stream) {
    const auto* fused = std::get_if<FusedGdnInputProjectionPayload>(&weights.input_projection);
    if (fused == nullptr) { return false; }
    Tensor output_gate_flat =
        output_gate.view({TextConfig::value_dim, static_cast<int>(hidden.ne[1])});
    return ops::gdn_input_proj_scatter_qkv(hidden, fused->query_key_value_z, qkv_padded,
                                           output_gate_flat, qkv_columns, stream);
}

void Variant::gdn_input_projection(const Tensor& hidden, const GdnProjectionWeights& weights,
                                   Tensor& qkv, Tensor& output_gate, qwen3_6::TextPhase,
                                   WorkspaceArena& workspace, cudaStream_t stream) {
    Tensor output_gate_flat =
        output_gate.view({TextConfig::value_dim, static_cast<int>(hidden.ne[1])});
    if (const auto* split =
            std::get_if<SplitGdnInputProjectionPayload>(&weights.input_projection)) {
        ops::gdn_input_proj(hidden, split->query_key, split->value_z, qkv, output_gate_flat,
                            workspace, stream);
        return;
    }
    const Weight& fused =
        std::get<FusedGdnInputProjectionPayload>(weights.input_projection).query_key_value_z;
    ops::gdn_input_proj(hidden, fused, qkv, output_gate_flat, text_policy(fused), workspace,
                        stream);
}

void Variant::gdn_input_projection_snapshot(
    const Tensor& hidden, const GdnProjectionWeights& weights, const Tensor& conv_weight,
    Tensor& conv_states, const Tensor& valid_columns, const Tensor& initial_slot,
    const Tensor& snapshot_base_slot, Tensor& query, Tensor& key, Tensor& value,
    Tensor& output_gate, qwen3_6::TextPhase, WorkspaceArena& workspace, cudaStream_t stream) {
    auto workspace_scope     = workspace.scope();
    const DeviceSpan storage = workspace.alloc_bytes(gdn_snapshot_workspace_bytes(hidden, weights));
    WorkspaceArena leaf_workspace(storage);
    Tensor output_gate_view = output_gate.view({TextConfig::value_dim, hidden.ne[1], hidden.ne[2]});
    if (const auto* split =
            std::get_if<SplitGdnInputProjectionPayload>(&weights.input_projection)) {
        ops::gdn_input_proj_conv_snapshot(hidden, split->query_key, split->value_z, conv_weight,
                                          conv_states, valid_columns, initial_slot,
                                          snapshot_base_slot, query, key, value, output_gate_view,
                                          leaf_workspace, stream);
        return;
    }
    const Weight& fused =
        std::get<FusedGdnInputProjectionPayload>(weights.input_projection).query_key_value_z;
    ops::gdn_input_proj_conv_snapshot(hidden, fused, conv_weight, conv_states, valid_columns,
                                      initial_slot, snapshot_base_slot, query, key, value,
                                      output_gate_view, text_policy(fused), leaf_workspace, stream);
}

void Variant::gdn_input_projection_record(const Tensor& hidden, const GdnProjectionWeights& weights,
                                          const Tensor& conv_weight, const Tensor& conv_states,
                                          const Tensor& valid_columns, const Tensor& initial_slots,
                                          Tensor& conv_record, Tensor& query, Tensor& key,
                                          Tensor& value, Tensor& output_gate, qwen3_6::TextPhase,
                                          WorkspaceArena& workspace, cudaStream_t stream) {
    auto workspace_scope     = workspace.scope();
    const DeviceSpan storage = workspace.alloc_bytes(gdn_record_workspace_bytes(hidden, weights));
    WorkspaceArena leaf_workspace(storage);
    Tensor output_gate_view = output_gate.view({TextConfig::value_dim, hidden.ne[1], hidden.ne[2]});
    if (const auto* split =
            std::get_if<SplitGdnInputProjectionPayload>(&weights.input_projection)) {
        ops::gdn_input_proj_conv_record(hidden, split->query_key, split->value_z, conv_weight,
                                        conv_states, valid_columns, initial_slots, conv_record,
                                        query, key, value, output_gate_view, leaf_workspace,
                                        stream);
        return;
    }
    const Weight& fused =
        std::get<FusedGdnInputProjectionPayload>(weights.input_projection).query_key_value_z;
    ops::gdn_input_proj_conv_record(hidden, fused, conv_weight, conv_states, valid_columns,
                                    initial_slots, conv_record, query, key, value, output_gate_view,
                                    text_policy(fused), leaf_workspace, stream);
}

void Variant::gdn_output_projection(const Tensor& hidden, const Weight& weight, Tensor& residual,
                                    qwen3_6::TextPhase, WorkspaceArena& workspace,
                                    cudaStream_t stream) {
    ops::linear_add(hidden, weight, residual, text_policy(weight), workspace, stream);
}

void Variant::gdn_norm_control_projection(const Tensor& residual, const Tensor& norm_weight,
                                          float eps, const GdnProjectionWeights& weights,
                                          Tensor& hidden, Tensor& g, Tensor& beta,
                                          WorkspaceArena& workspace,
                                          DeviceExecutionView execution) {
    if (const auto* split =
            std::get_if<SplitGdnControlProjectionPayload>(&weights.control_projection)) {
        ops::gdn_norm_gating_proj(residual, norm_weight, eps, split->a_projection,
                                  split->b_projection, weights.a_log, weights.dt_bias, workspace,
                                  hidden, g, beta, execution);
        return;
    }
    const Weight& fused =
        std::get<FusedGdnControlProjectionPayload>(weights.control_projection).a_b_projection;
    ops::gdn_norm_gating_proj(residual, norm_weight, eps, fused, weights.a_log, weights.dt_bias,
                              workspace, hidden, g, beta, execution);
}

bool Variant::gdn_norm_control_projection_scatter(const Tensor& residual, const Tensor& norm_weight,
                                                  float eps, const GdnProjectionWeights& weights,
                                                  const Tensor& scatter_columns, Tensor& hidden,
                                                  Tensor& g_padded, Tensor& beta_padded,
                                                  cudaStream_t stream) {
    const auto* fused =
        std::get_if<FusedGdnControlProjectionPayload>(&weights.control_projection);
    if (fused == nullptr) { return false; }
    return ops::gdn_norm_gating_proj_scatter(residual, norm_weight, eps, fused->a_b_projection,
                                             weights.a_log, weights.dt_bias, scatter_columns,
                                             hidden, g_padded, beta_padded, stream);
}

#ifdef NINFER_VOLTA_BUILD
bool Variant::gdn_norm_control_projection_fp16(const Tensor& residual, const Tensor& norm_weight,
                                               float eps, const GdnProjectionWeights& weights,
                                               const Tensor* scatter_columns, Tensor& hidden_fp16,
                                               Tensor& g, Tensor& beta, cudaStream_t stream) {
    const std::int32_t t = residual.ne[1];
    if (t <= 0 || t > ops::detail::kVoltaQpnMaxTokens || !ops::detail::volta_stage_fp16_enabled()) {
        return false;
    }
    if (scatter_columns != nullptr &&
        (scatter_columns->dtype != DType::I32 || scatter_columns->ne[0] != t)) {
        return false;
    }
    const auto* split_ctrl =
        std::get_if<SplitGdnControlProjectionPayload>(&weights.control_projection);
    const auto* split_in =
        std::get_if<SplitGdnInputProjectionPayload>(&weights.input_projection);
    if (split_ctrl != nullptr && split_in != nullptr) {
        if (split_ctrl->a_projection.n != 48 || split_ctrl->a_projection.k != 5120 ||
            split_ctrl->b_projection.n != 48 || split_ctrl->b_projection.k != 5120 ||
            split_in->query_key.qtype != QType::Q4G64_F16S ||
            split_in->value_z.qtype != QType::Q5G64_F16S ||
            !ops::detail::volta_qpn_groupwise_supported(split_in->query_key) ||
            !ops::detail::volta_qpn_groupwise_supported(split_in->value_z)) {
            return false;
        }
        ops::detail::bf16_gdn_norm_gating_proj_27_launch(
            residual, norm_weight, eps, hidden_fp16, split_ctrl->a_projection,
            split_ctrl->b_projection, weights.a_log, weights.dt_bias, g, beta, stream,
            scatter_columns != nullptr ? static_cast<const std::int32_t*>(scatter_columns->data)
                                       : nullptr,
            /*write_fp16=*/true);
        return true;
    }
    const auto* in_proj =
        std::get_if<FusedGdnInputProjectionPayload>(&weights.input_projection);
    if (in_proj == nullptr) { return false; }
    const Weight& qkvz = in_proj->query_key_value_z;
    const bool fp8_ok =
        qkvz.qtype == QType::FP8_E4M3FN_ROW_BF16S &&
        text_policy(qkvz) == ops::LinearPolicy::A16Only &&
        t <= ops::detail::kFp8VoltaQpnMaxTokens &&
        ops::detail::fp8_volta_qpn_supported(qkvz.n, qkvz.k, t);
    const bool nvfp4_ok =
        qkvz.qtype == QType::NVFP4 &&
        text_policy(qkvz) == ops::LinearPolicy::A16Only &&
        t <= ops::detail::kNvfp4VoltaQpnMaxTokens &&
        ops::detail::nvfp4_volta_qpn_supported(qkvz.n, qkvz.k, t);
    if (!fp8_ok && !nvfp4_ok) { return false; }
    if (split_ctrl != nullptr) {
        if (split_ctrl->a_projection.n != 48 || split_ctrl->a_projection.k != 5120 ||
            split_ctrl->b_projection.n != 48 || split_ctrl->b_projection.k != 5120) {
            return false;
        }
        ops::detail::bf16_gdn_norm_gating_proj_27_launch(
            residual, norm_weight, eps, hidden_fp16, split_ctrl->a_projection,
            split_ctrl->b_projection, weights.a_log, weights.dt_bias, g, beta, stream,
            scatter_columns != nullptr ? static_cast<const std::int32_t*>(scatter_columns->data)
                                       : nullptr,
            /*write_fp16=*/true);
        return true;
    }
    const auto* ctrl =
        std::get_if<FusedGdnControlProjectionPayload>(&weights.control_projection);
    if (ctrl == nullptr) { return false; }
    const Weight& ab = ctrl->a_b_projection;
    if (ab.n != 96 || ab.k != 5120) { return false; }
    Weight a_w          = ab;
    a_w.n               = 48;
    a_w.shape[0]        = 48;
    a_w.padded_shape[0] = 48;
    Weight b_w          = a_w;
    b_w.qdata           = static_cast<const std::uint8_t*>(ab.qdata) +
                static_cast<std::int64_t>(48) * 5120 * sizeof(std::uint16_t);
    ops::detail::bf16_gdn_norm_gating_proj_27_launch(
        residual, norm_weight, eps, hidden_fp16, a_w, b_w, weights.a_log, weights.dt_bias, g, beta,
        stream,
        scatter_columns != nullptr ? static_cast<const std::int32_t*>(scatter_columns->data)
                                   : nullptr,
        /*write_fp16=*/true);
    return true;
}

bool Variant::gdn_input_projection_scatter_fp16(const Tensor& hidden_fp16,
                                                const GdnProjectionWeights& weights,
                                                Tensor& qkv_padded, Tensor& output_gate,
                                                const Tensor& qkv_columns, cudaStream_t stream) {
    Tensor output_gate_flat =
        output_gate.view({TextConfig::value_dim, static_cast<int>(hidden_fp16.ne[1])});
    if (const auto* split =
            std::get_if<SplitGdnInputProjectionPayload>(&weights.input_projection)) {
        if (split->query_key.qtype != QType::Q4G64_F16S ||
            split->value_z.qtype != QType::Q5G64_F16S ||
            !ops::detail::volta_qpn_groupwise_supported(split->query_key) ||
            !ops::detail::volta_qpn_groupwise_supported(split->value_z)) {
            return false;
        }
        Tensor qk_padded    = qkv_padded.slice(0, 0, split->query_key.n);
        Tensor value_padded = qkv_padded.slice(0, split->query_key.n, TextConfig::value_dim);
        const auto* x_fp16  = static_cast<const half*>(hidden_fp16.data);
        const auto* cols    = static_cast<const std::int32_t*>(qkv_columns.data);
        ops::detail::launch_q4_prepacked_volta_scatter_fp16(hidden_fp16, x_fp16, split->query_key,
                                                            qk_padded, cols, stream);
        ops::detail::launch_q5_prepacked_volta_split_scatter_fp16(
            hidden_fp16, x_fp16, split->value_z, value_padded, output_gate_flat,
            TextConfig::value_dim, cols, stream);
        return true;
    }
    const auto* fused = std::get_if<FusedGdnInputProjectionPayload>(&weights.input_projection);
    if (fused == nullptr) { return false; }
    if (fused->query_key_value_z.qtype == QType::NVFP4) {
        return ops::detail::launch_nvfp4_gdn_input_volta_qpn_scatter(
            hidden_fp16, fused->query_key_value_z, qkv_padded, output_gate_flat, qkv_columns,
            stream, hidden_fp16.data);
    }
    return ops::detail::launch_fp8_gdn_input_volta_qpn_scatter(
        hidden_fp16, fused->query_key_value_z, qkv_padded, output_gate_flat, qkv_columns, stream,
        hidden_fp16.data);
}

void Variant::gdn_input_projection_record_fp16(
    const Tensor& hidden_fp16, const GdnProjectionWeights& weights, const Tensor& conv_weight,
    const Tensor& conv_states, const Tensor& valid_columns, const Tensor& initial_slots,
    Tensor& conv_record, Tensor& query, Tensor& key, Tensor& value, Tensor& output_gate,
    cudaStream_t stream) {
    const std::int32_t cols = hidden_fp16.ne[1] * hidden_fp16.ne[2];
    Tensor x_flat(hidden_fp16.data, DType::BF16, {TextConfig::hidden, cols});
    Tensor record_flat(conv_record.data, DType::BF16, {conv_record.ne[0], cols});
    Tensor gate_flat(output_gate.data, DType::BF16, {TextConfig::value_dim, cols});
    if (const auto* split =
            std::get_if<SplitGdnInputProjectionPayload>(&weights.input_projection)) {
        Tensor qk_slice    = record_flat.slice(0, 0, split->query_key.n);
        Tensor value_slice = record_flat.slice(0, split->query_key.n, TextConfig::value_dim);
        const auto* x_fp16 = static_cast<const half*>(hidden_fp16.data);
        ops::detail::launch_q4_prepacked_volta_fp16(x_flat, x_fp16, split->query_key, qk_slice,
                                                    stream);
        ops::detail::launch_q5_prepacked_volta_split_fp16(x_flat, x_fp16, split->value_z,
                                                          value_slice, gate_flat,
                                                          TextConfig::value_dim, stream);
    } else {
        const Weight& fused =
            std::get<FusedGdnInputProjectionPayload>(weights.input_projection).query_key_value_z;
        if (fused.qtype == QType::NVFP4) {
            ops::detail::launch_nvfp4_gdn_input_volta_qpn(x_flat, fused, record_flat, gate_flat,
                                                          hidden_fp16.data, stream);
        } else {
            ops::detail::launch_fp8_gdn_input_volta_qpn(x_flat, fused, record_flat, gate_flat,
                                                        hidden_fp16.data, stream);
        }
    }
    ops::gdn_projected_conv_record(conv_record, conv_weight, conv_states, valid_columns,
                                   initial_slots, query, key, value, stream);
}

bool Variant::gdn_output_projection_from_gated_norm(
    const Tensor& o, const Tensor& gdn_norm, const Tensor& z, float eps,
    const Tensor* live_columns, Tensor& on_scratch, const Weight& weight, Tensor& residual,
    qwen3_6::TextPhase, WorkspaceArena&, cudaStream_t stream) {
    const std::int32_t d = on_scratch.ne[0];
    const std::int32_t t = on_scratch.ne[2];
    if (t <= 0 || t > ops::detail::kVoltaQpnMaxTokens || !ops::detail::volta_stage_fp16_enabled() ||
        d < 64 || d > 256 || (d % 64) != 0) {
        return false;
    }
    const bool is_groupwise =
        (weight.qtype == QType::Q4G64_F16S || weight.qtype == QType::Q5G64_F16S) &&
        ops::detail::volta_qpn_groupwise_supported(weight);
    const bool is_fp8 = weight.qtype == QType::FP8_E4M3FN_ROW_BF16S &&
                        text_policy(weight) == ops::LinearPolicy::A16Only &&
                        t <= ops::detail::kFp8VoltaQpnMaxTokens &&
                        ops::detail::fp8_volta_qpn_supported(weight.n, weight.k, t);
    const bool is_nvfp4 = weight.qtype == QType::NVFP4 &&
                          text_policy(weight) == ops::LinearPolicy::A16Only &&
                          t <= ops::detail::kNvfp4VoltaQpnMaxTokens &&
                          ops::detail::nvfp4_volta_qpn_supported(weight.n, weight.k, t);
    if (!is_groupwise && !is_fp8 && !is_nvfp4) { return false; }
    const auto addr = reinterpret_cast<std::uintptr_t>(o.data) |
                      reinterpret_cast<std::uintptr_t>(gdn_norm.data) |
                      reinterpret_cast<std::uintptr_t>(z.data) |
                      reinterpret_cast<std::uintptr_t>(on_scratch.data);
    if ((addr & 3U) != 0) { return false; }
    if (live_columns != nullptr) {
        ops::detail::gated_rmsnorm_columns_fp16_launch(o, gdn_norm, z, eps, *live_columns,
                                                       on_scratch, stream);
    } else {
        ops::detail::gated_rmsnorm_fp16_launch(o, gdn_norm, z, eps, on_scratch, stream);
    }
    Tensor flat = on_scratch.view({TextConfig::value_dim, t});
    if (is_groupwise) {
        if (weight.qtype == QType::Q4G64_F16S) {
            ops::detail::launch_q4_prepacked_volta_residual_fp16(
                flat, static_cast<const half*>(on_scratch.data), weight, residual, stream);
        } else {
            ops::detail::launch_q5_prepacked_volta_residual_fp16(
                flat, static_cast<const half*>(on_scratch.data), weight, residual, stream);
        }
    } else if (is_nvfp4) {
        ops::detail::nvfp4_linear_add_qpn_fp16_launch(
            flat, static_cast<const half*>(on_scratch.data), weight, residual, stream);
    } else {
        ops::detail::fp8_linear_add_qpn_fp16_launch(flat, static_cast<const half*>(on_scratch.data),
                                                    weight, residual, stream);
    }
    return true;
}
#endif

void Variant::post_mixer(const Tensor& hidden, const PostMixerWeights& weights, Tensor& residual,
                         qwen3_6::TextPhase, const ::ninfer::ops::SparseMoeHints&,
                         WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope        = workspace.scope();
    Tensor activation = workspace.alloc(DType::BF16, {TextConfig::intermediate, hidden.ne[1]});
    ops::linear_swiglu(hidden, weights.gate_up, activation, text_policy(weights.gate_up), workspace,
                       stream);
    ops::linear_add(activation, weights.down, residual, text_policy(weights.down), workspace,
                    stream);
}

#ifdef NINFER_VOLTA_BUILD
bool Variant::post_mixer_from_residual(Tensor& residual, const Tensor& post_norm, float eps,
                                       const PostMixerWeights& weights, qwen3_6::TextPhase,
                                       const ::ninfer::ops::SparseMoeHints&,
                                       WorkspaceArena& workspace, cudaStream_t stream) {
    const std::int32_t t = residual.ne[1];
    if (residual.ne[0] != TextConfig::hidden || t <= 0 || t > ops::detail::kVoltaQpnMaxTokens) {
        return false;
    }
    if (weights.gate_up.qtype == QType::NVFP4 && weights.down.qtype == QType::NVFP4 &&
        weights.gate_up.layout == QuantLayout::VoltaQpnPrepacked &&
        weights.down.layout == QuantLayout::VoltaQpnPrepacked &&
        text_policy(weights.gate_up) == ops::LinearPolicy::A16Only &&
        text_policy(weights.down) == ops::LinearPolicy::A16Only &&
        ops::detail::volta_group_major_enabled() &&
        !ops::detail::volta_tc_applies(TextConfig::intermediate, TextConfig::hidden, t,
                                       ops::detail::kVoltaTcNvfp4MinTokens) &&
        !ops::detail::volta_tc_applies(TextConfig::hidden, TextConfig::intermediate, t,
                                       ops::detail::kVoltaTcNvfp4MinTokens) &&
        ops::detail::nvfp4_linear_swiglu_qpn_split_supported(TextConfig::hidden, t) &&
        ops::detail::nvfp4_volta_qpn_supported(TextConfig::hidden, TextConfig::intermediate, t)) {
        static const bool any_t1_gemv = [] {
            const char* a = std::getenv("NINFER_NVFP4_SWIGLU_T1_GEMV");
            const char* b = std::getenv("NINFER_NVFP4_DOWN_T1_GEMV");
            return (a != nullptr && a[0] == '1') || (b != nullptr && b[0] == '1');
        }();
        if (t == 1 && any_t1_gemv) { return false; }

        auto scope      = workspace.scope();
        const int tiles = ops::detail::volta_group_major_tiles(t);
        auto* staged_h  = static_cast<ops::detail::VoltaGroupMajorHalf*>(
            workspace
                .alloc_bytes(static_cast<std::size_t>(TextConfig::hidden) * tiles * 8 *
                                 sizeof(std::uint16_t),
                             256)
                .data);
        auto* gate_scratch = static_cast<float*>(
            workspace
                .alloc_bytes(static_cast<std::size_t>(TextConfig::intermediate) * t * sizeof(float),
                             256)
                .data);
        auto* inter_gm = static_cast<ops::detail::VoltaGroupMajorHalf*>(
            workspace
                .alloc_bytes(static_cast<std::size_t>(TextConfig::intermediate) * tiles * 8 *
                                 sizeof(std::uint16_t),
                             256)
                .data);

        ops::detail::volta_rmsnorm_offset_d5120_to_group_major(residual, post_norm, eps, staged_h,
                                                               stream);
        ops::detail::nvfp4_linear_swiglu_qpn_split_group_major_io_launch(
            residual, staged_h, weights.gate_up, inter_gm, gate_scratch, stream);
        Tensor inter_shape(inter_gm, DType::BF16, {TextConfig::intermediate, t});
        ops::detail::nvfp4_linear_add_qpn_group_major_launch(inter_shape, inter_gm, weights.down,
                                                             residual, stream);
        return true;
    }
    if (weights.gate_up.qtype == QType::FP8_E4M3FN_ROW_BF16S &&
        weights.down.qtype == QType::FP8_E4M3FN_ROW_BF16S &&
        text_policy(weights.gate_up) == ops::LinearPolicy::A16Only &&
        text_policy(weights.down) == ops::LinearPolicy::A16Only &&
        ops::detail::volta_stage_fp16_enabled() &&
        ops::detail::fp8_linear_swiglu_qpn_split_supported(TextConfig::hidden, t) &&
        ops::detail::fp8_volta_qpn_supported(TextConfig::hidden, TextConfig::intermediate, t)) {
        auto scope     = workspace.scope();
        auto* staged_h = static_cast<half*>(
            workspace
                .alloc_bytes(static_cast<std::size_t>(TextConfig::hidden) * t *
                                 sizeof(std::uint16_t),
                             256)
                .data);
        auto* gate_scratch = static_cast<float*>(
            workspace
                .alloc_bytes(static_cast<std::size_t>(TextConfig::intermediate) * t * sizeof(float),
                             256)
                .data);
        auto* inter_fp16 = static_cast<half*>(
            workspace
                .alloc_bytes(static_cast<std::size_t>(TextConfig::intermediate) * t *
                                 sizeof(std::uint16_t),
                             256)
                .data);

        ops::detail::volta_rmsnorm_offset_d5120_to_fp16(residual, post_norm, eps, staged_h, stream);
        ops::detail::fp8_linear_swiglu_qpn_split_fp16_io_launch(
            residual, staged_h, weights.gate_up, inter_fp16, gate_scratch, stream);
        Tensor inter_shape(inter_fp16, DType::BF16, {TextConfig::intermediate, t});
        ops::detail::fp8_linear_add_qpn_fp16_launch(inter_shape, inter_fp16, weights.down, residual,
                                                    stream);
        return true;
    }
    if (weights.gate_up.qtype == QType::Q4G64_F16S &&
        (weights.down.qtype == QType::Q4G64_F16S || weights.down.qtype == QType::Q5G64_F16S) &&
        ops::detail::volta_qpn_groupwise_supported(weights.gate_up) &&
        ops::detail::volta_qpn_groupwise_supported(weights.down) &&
        ops::detail::volta_group_major_enabled()) {
        auto scope      = workspace.scope();
        const int tiles = ops::detail::volta_group_major_tiles(t);
        auto* staged_gm = static_cast<ops::detail::VoltaGroupMajorHalf*>(
            workspace
                .alloc_bytes(static_cast<std::size_t>(TextConfig::hidden) * tiles * 8 *
                                 sizeof(std::uint16_t),
                             256)
                .data);
        auto* gate_scratch = static_cast<float*>(
            workspace
                .alloc_bytes(static_cast<std::size_t>(TextConfig::intermediate) * t * sizeof(float),
                             256)
                .data);
        auto* inter_gm = static_cast<ops::detail::VoltaGroupMajorHalf*>(
            workspace
                .alloc_bytes(static_cast<std::size_t>(TextConfig::intermediate) * tiles * 8 *
                                 sizeof(std::uint16_t),
                             256)
                .data);

        ops::detail::volta_rmsnorm_offset_d5120_to_group_major(residual, post_norm, eps, staged_gm,
                                                               stream);
        ops::detail::launch_q4_prepacked_volta_swiglu_gm_io(residual, staged_gm, weights.gate_up,
                                                            inter_gm, gate_scratch, stream);
        Tensor inter_shape(inter_gm, DType::BF16, {TextConfig::intermediate, t});
        if (weights.down.qtype == QType::Q4G64_F16S) {
            ops::detail::launch_q4_prepacked_volta_residual_gm(inter_shape, inter_gm, weights.down,
                                                               residual, stream);
        } else {
            ops::detail::launch_q5_prepacked_volta_residual_gm(inter_shape, inter_gm, weights.down,
                                                               residual, stream);
        }
        return true;
    }
    return false;
}
#endif

void Variant::mtp_post_mixer(const Tensor& hidden, const MtpPostMixerWeights& weights,
                             Tensor& residual, WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope     = workspace.scope();
    const int cols = hidden.ne[1];
    Tensor gate_up = workspace.alloc(DType::BF16, {TextConfig::mtp_mlp_gate_up_rows, cols});
    ops::linear(hidden, weights.gate_up, gate_up, stream);
    Tensor activation = workspace.alloc(DType::BF16, {TextConfig::intermediate, cols});
    ops::silu_mul(gate_up.slice(0, 0, TextConfig::intermediate),
                  gate_up.slice(0, TextConfig::intermediate, TextConfig::intermediate), activation,
                  stream);
    Tensor delta = workspace.alloc(DType::BF16, {TextConfig::hidden, cols});
    ops::linear(activation, weights.down, delta, stream);
    ops::residual_add(delta, residual, stream);
}

std::size_t Variant::mtp_attention_projection_workspace_capacity_bytes(std::int32_t first,
                                                                       std::int32_t last) {
    validate_token_interval(first, last);
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::BF16, {TextConfig::mtp_attention_input_rows, last});
    return layout.peak_bytes(1);
}

std::size_t Variant::mtp_kv_projection_workspace_capacity_bytes(std::int32_t first,
                                                                std::int32_t last) {
    validate_token_interval(first, last);
    return 0;
}

std::size_t Variant::mtp_q_gate_projection_workspace_capacity_bytes(std::int32_t first,
                                                                    std::int32_t last) {
    validate_token_interval(first, last);
    return 0;
}

std::size_t Variant::attention_projection_workspace_capacity_bytes(WeightsProfile weights_profile,
                                                                   qwen3_6::TextPhase,
                                                                   std::int32_t first,
                                                                   std::int32_t last) {
    validate_token_interval(first, last);
    switch (weights_profile) {
    case WeightsProfile::Qwen36GroupwiseInt:
    case WeightsProfile::Qwen38GroupwiseInt: {
        std::size_t peak = ops::q4_q5_attn_input_proj_workspace_capacity_bytes(first, last);
#ifdef NINFER_VOLTA_BUILD
        if (first <= ops::detail::kVoltaQpnMaxTokens) {
            const std::int32_t fused_t = std::min(last, ops::detail::kVoltaQpnMaxTokens);
            const int tiles            = ops::detail::volta_group_major_tiles(fused_t);
            WorkspaceLayoutBuilder fused;
            (void)fused.alloc_bytes(static_cast<std::size_t>(TextConfig::hidden) * tiles * 8 *
                                        sizeof(std::uint16_t),
                                    256);
            peak = std::max(peak, fused.peak_bytes(1));
        }
#endif
        return peak;
    }
    case WeightsProfile::Qwen36Nvfp4:
        return ops::attn_input_proj_workspace_capacity_bytes(
            QType::NVFP4, 14336, TextConfig::hidden, kNvfp4TextPolicy, first, last);
    case WeightsProfile::Qwen38Nvfp4:
        return ops::attn_input_proj_workspace_capacity_bytes(
            QType::FP8_E4M3FN_ROW_BF16S, 14336, TextConfig::hidden, kFp8TextPolicy, first, last);
    }
    throw std::logic_error("invalid 27B weights profile");
}

std::size_t Variant::attention_output_projection_workspace_capacity_bytes(
    WeightsProfile weights_profile, qwen3_6::TextPhase, std::int32_t first, std::int32_t last) {
    validate_token_interval(first, last);
    switch (weights_profile) {
    case WeightsProfile::Qwen36GroupwiseInt:
    case WeightsProfile::Qwen38GroupwiseInt:
        return std::max(
            ops::linear_add_workspace_capacity_bytes(QType::Q4G64_F16S, TextConfig::hidden,
                                                     TextConfig::query_size,
                                                     ops::LinearPolicy::A16Only, first, last),
            ops::linear_add_workspace_capacity_bytes(QType::Q5G64_F16S, TextConfig::hidden,
                                                     TextConfig::query_size,
                                                     ops::LinearPolicy::A16Only, first, last));
    case WeightsProfile::Qwen36Nvfp4:
        return ops::linear_add_workspace_capacity_bytes(QType::NVFP4, TextConfig::hidden,
                                                        TextConfig::query_size, kNvfp4TextPolicy,
                                                        first, last);
    case WeightsProfile::Qwen38Nvfp4:
        return ops::linear_add_workspace_capacity_bytes(QType::FP8_E4M3FN_ROW_BF16S,
                                                        TextConfig::hidden, TextConfig::query_size,
                                                        kFp8TextPolicy, first, last);
    }
    throw std::logic_error("invalid 27B weights profile");
}

std::size_t Variant::gdn_input_projection_workspace_capacity_bytes(WeightsProfile weights_profile,
                                                                   qwen3_6::TextPhase,
                                                                   std::int32_t first,
                                                                   std::int32_t last) {
    validate_token_interval(first, last);
    switch (weights_profile) {
    case WeightsProfile::Qwen36GroupwiseInt:
    case WeightsProfile::Qwen38GroupwiseInt:
        return ops::q4_q5_gdn_input_proj_workspace_capacity_bytes(first, last);
    case WeightsProfile::Qwen36Nvfp4:
        return ops::gdn_input_proj_workspace_capacity_bytes(QType::NVFP4, 16384, TextConfig::hidden,
                                                            kNvfp4TextPolicy, first, last);
    case WeightsProfile::Qwen38Nvfp4:
        return std::max(
            ops::gdn_input_proj_workspace_capacity_bytes(
                QType::FP8_E4M3FN_ROW_BF16S, 16384, TextConfig::hidden, kFp8TextPolicy, first, last),
            ops::gdn_input_proj_workspace_capacity_bytes(QType::NVFP4, 16384, TextConfig::hidden,
                                                         kNvfp4TextPolicy, first, last));
    }
    throw std::logic_error("invalid 27B weights profile");
}

std::size_t Variant::gdn_input_projection_snapshot_workspace_capacity_bytes(
    WeightsProfile weights_profile, qwen3_6::TextPhase, std::int32_t batch_size, std::int32_t first,
    std::int32_t last) {
    validate_token_interval(first, last);
    switch (weights_profile) {
    case WeightsProfile::Qwen36GroupwiseInt:
    case WeightsProfile::Qwen38GroupwiseInt:
        return std::max(kMinimumLeafWorkspaceBytes,
                        ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
                            TextConfig::key_dim, TextConfig::key_dim, TextConfig::value_dim,
                            batch_size, first, last));
    case WeightsProfile::Qwen36Nvfp4:
        return std::max(kMinimumLeafWorkspaceBytes,
                        ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
                            QType::NVFP4, 16384, TextConfig::hidden, kNvfp4TextPolicy, batch_size,
                            first, last));
    case WeightsProfile::Qwen38Nvfp4:
        return std::max({
            kMinimumLeafWorkspaceBytes,
            ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
                QType::FP8_E4M3FN_ROW_BF16S, 16384, TextConfig::hidden, kFp8TextPolicy, batch_size,
                first, last),
            ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
                QType::NVFP4, 16384, TextConfig::hidden, kNvfp4TextPolicy, batch_size, first, last),
        });
    }
    throw std::logic_error("invalid 27B weights profile");
}

std::size_t Variant::gdn_input_projection_record_workspace_capacity_bytes(
    WeightsProfile weights_profile, qwen3_6::TextPhase, std::int32_t batch_size, std::int32_t first,
    std::int32_t last) {
    validate_token_interval(first, last);
    switch (weights_profile) {
    case WeightsProfile::Qwen36GroupwiseInt:
    case WeightsProfile::Qwen38GroupwiseInt:
        return std::max(kMinimumLeafWorkspaceBytes,
                        ops::gdn_input_proj_conv_record_workspace_capacity_bytes(
                            TextConfig::key_dim, TextConfig::key_dim, TextConfig::value_dim,
                            batch_size, first, last));
    case WeightsProfile::Qwen36Nvfp4:
        return std::max(kMinimumLeafWorkspaceBytes,
                        ops::gdn_input_proj_conv_record_workspace_capacity_bytes(
                            QType::NVFP4, 16384, TextConfig::hidden, kNvfp4TextPolicy, batch_size,
                            first, last));
    case WeightsProfile::Qwen38Nvfp4:
        return std::max({
            kMinimumLeafWorkspaceBytes,
            ops::gdn_input_proj_conv_record_workspace_capacity_bytes(
                QType::FP8_E4M3FN_ROW_BF16S, 16384, TextConfig::hidden, kFp8TextPolicy, batch_size,
                first, last),
            ops::gdn_input_proj_conv_record_workspace_capacity_bytes(
                QType::NVFP4, 16384, TextConfig::hidden, kNvfp4TextPolicy, batch_size, first, last),
        });
    }
    throw std::logic_error("invalid 27B weights profile");
}

std::size_t Variant::gdn_output_projection_workspace_capacity_bytes(WeightsProfile weights_profile,
                                                                    qwen3_6::TextPhase,
                                                                    std::int32_t first,
                                                                    std::int32_t last) {
    validate_token_interval(first, last);
    switch (weights_profile) {
    case WeightsProfile::Qwen36GroupwiseInt:
    case WeightsProfile::Qwen38GroupwiseInt:
        return std::max(
            ops::linear_add_workspace_capacity_bytes(QType::Q4G64_F16S, TextConfig::hidden,
                                                     TextConfig::value_dim,
                                                     ops::LinearPolicy::A16Only, first, last),
            ops::linear_add_workspace_capacity_bytes(QType::Q5G64_F16S, TextConfig::hidden,
                                                     TextConfig::value_dim,
                                                     ops::LinearPolicy::A16Only, first, last));
    case WeightsProfile::Qwen36Nvfp4:
        return ops::linear_add_workspace_capacity_bytes(
            QType::NVFP4, TextConfig::hidden, TextConfig::value_dim, kNvfp4TextPolicy, first, last);
    case WeightsProfile::Qwen38Nvfp4:
        return std::max(
            ops::linear_add_workspace_capacity_bytes(QType::FP8_E4M3FN_ROW_BF16S,
                                                     TextConfig::hidden, TextConfig::value_dim,
                                                     kFp8TextPolicy, first, last),
            ops::linear_add_workspace_capacity_bytes(QType::NVFP4, TextConfig::hidden,
                                                     TextConfig::value_dim, kNvfp4TextPolicy, first,
                                                     last));
    }
    throw std::logic_error("invalid 27B weights profile");
}

std::size_t Variant::gdn_norm_control_projection_workspace_capacity_bytes(std::int32_t first,
                                                                          std::int32_t last) {
    return ops::gdn_norm_gating_proj_workspace_capacity_bytes(TextConfig::gdn_value_heads,
                                                              TextConfig::hidden, first, last);
}

std::size_t Variant::post_mixer_workspace_capacity_bytes(WeightsProfile weights_profile,
                                                         qwen3_6::TextPhase, std::int32_t first,
                                                         std::int32_t last) {
    validate_token_interval(first, last);
    switch (weights_profile) {
    case WeightsProfile::Qwen36GroupwiseInt:
    case WeightsProfile::Qwen38GroupwiseInt:
        return std::max(post_mixer_workspace_bytes(QType::Q4G64_F16S, QType::Q5G64_F16S,
                                                   ops::LinearPolicy::A16Only, first, last),
                        post_mixer_workspace_bytes(QType::Q4G64_F16S, QType::Q4G64_F16S,
                                                   ops::LinearPolicy::A16Only, first, last));
    case WeightsProfile::Qwen36Nvfp4:
        return post_mixer_workspace_bytes(QType::NVFP4, QType::NVFP4, kNvfp4TextPolicy, first,
                                          last);
    case WeightsProfile::Qwen38Nvfp4: {
        const std::size_t nvfp4 =
            post_mixer_workspace_bytes(QType::NVFP4, QType::NVFP4, kNvfp4TextPolicy, first, last);
        const std::size_t fp8 = post_mixer_workspace_bytes(
            QType::FP8_E4M3FN_ROW_BF16S, QType::FP8_E4M3FN_ROW_BF16S, kFp8TextPolicy, first, last);
        return std::max(nvfp4, fp8);
    }
    }
    throw std::invalid_argument("qwen3_6_27b: invalid weights profile");
}

std::size_t Variant::mtp_post_mixer_workspace_capacity_bytes(std::int32_t first,
                                                             std::int32_t last) {
    validate_token_interval(first, last);
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::BF16, {TextConfig::mtp_mlp_gate_up_rows, last});
    (void)layout.alloc(DType::BF16, {TextConfig::intermediate, last});
    (void)layout.alloc(DType::BF16, {TextConfig::hidden, last});
    return layout.peak_bytes(1);
}

} // namespace ninfer::targets::qwen3_6_27b::detail
