#pragma once

#include "core/layout.h"
#include "core/tensor.h"
#include "ninfer/ops/sampling.h"
#include "ninfer/types.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>

namespace ninfer::targets::qwen3_6 {

inline constexpr std::uint32_t kMtpDecodeMaximumDrafts    = 7;
inline constexpr std::uint32_t kMtpDecodeMaximumWidth     = kMtpDecodeMaximumDrafts + 1;
inline constexpr std::uint32_t kMtpLookupMaximumDrafts    = 15;
inline constexpr std::uint32_t kMtpLookupMaximumWidth     = kMtpLookupMaximumDrafts + 1;
inline constexpr std::uint32_t kDFlashDecodeMaximumDrafts = 15;
inline constexpr std::uint32_t kDFlashDecodeMaximumWidth  = kDFlashDecodeMaximumDrafts + 1;

struct RoundStateSpec {
    std::int32_t hidden             = 0;
    std::int32_t output_rows        = 0;
    std::uint32_t batch_capacity    = 1;
    std::uint32_t draft_window      = 0;
    std::uint32_t copy_draft_window = 0;
    SpeculativeBackend backend      = SpeculativeBackend::None;
};

// Stable pinned/device transfer format for ordinary decode. The full fixed-size object is copied
// once per round; only its exact-B prefixes are consumed by the model schedule.
struct OrdinaryDecodeIngress {
    std::array<TokenId, kMaximumConcurrency> tokens{};
    std::array<std::int32_t, kMaximumConcurrency> cache_positions{};
    std::array<std::int32_t, kMaximumConcurrency> rope_positions{};
    std::array<std::int32_t, kMaximumConcurrency> text_kv_table_rows{};
    std::array<std::int32_t, kMaximumConcurrency> state_source_slots{};
    std::array<std::int32_t, kMaximumConcurrency> state_destination_slots{};
    std::array<ops::SamplingConfig, kMaximumConcurrency> sampling{};
};

struct OrdinaryDecodeEgress {
    std::array<TokenId, kMaximumConcurrency> sampled_tokens{};
};

// Stable pinned/device transfer formats for concurrent MTP decode. The arrays use the maximum
// product domain; each decode frame binds its exact verification width and the configured
// seven-token-or-smaller learned proposal width.
struct MtpDecodeIngress {
    std::array<TokenId, kMaximumConcurrency> anchors{};
    std::array<std::int32_t, kMaximumConcurrency> base_frontiers{};
    std::array<std::int32_t, kMaximumConcurrency> remaining_budgets{};
    std::array<std::int32_t, kMaximumConcurrency> current_extents{};
    std::array<std::int32_t, kMaximumConcurrency> target_valid_columns{};
    std::array<TokenId, kMaximumConcurrency * kMtpLookupMaximumDrafts> current_drafts{};
    std::array<std::int32_t, kMaximumConcurrency * kMtpLookupMaximumWidth> target_rope_positions{};
    std::array<std::int32_t, kMaximumConcurrency> text_kv_table_rows{};
    std::array<std::int32_t, kMaximumConcurrency> mtp_kv_table_rows{};
    std::array<std::int32_t, kMaximumConcurrency> state_source_slots{};
    std::array<std::int32_t, kMaximumConcurrency> state_destination_slots{};
    std::array<std::int32_t, kMaximumConcurrency> rope_deltas{};
    std::array<ops::SamplingConfig, kMaximumConcurrency> sampling{};
};

struct MtpDecodeEgress {
    std::array<TokenId, kMaximumConcurrency * kMtpLookupMaximumWidth> licensed_tokens{};
    std::array<std::int32_t, kMaximumConcurrency> licensed_counts{};
    std::array<std::int32_t, kMaximumConcurrency> accepted_drafts{};
    // Step-major: all B rows for proposal step 0, followed by all B rows for step 1, etc.
    std::array<TokenId, kMaximumConcurrency * kMtpDecodeMaximumDrafts> next_drafts{};
    std::array<std::int32_t, kMaximumConcurrency> next_extents{};
};

// Stable pinned/device transfer formats for one exact-B DFlash transaction. The proposal is
// produced and verified in the same round, so no draft state crosses the round boundary.
struct DFlashDecodeIngress {
    std::array<TokenId, kMaximumConcurrency> anchors{};
    std::array<std::int32_t, kMaximumConcurrency> execution_frontiers{};
    std::array<std::int32_t, kMaximumConcurrency> context_frontiers{};
    std::array<std::int32_t, kMaximumConcurrency> proposal_extents{};
    std::array<std::int32_t, kMaximumConcurrency> target_valid_columns{};
    std::array<std::int32_t, kMaximumConcurrency> proposal_valid_columns{};
    // DFlash uses logical positions for its own attention. Target verification carries a separate
    // continuation RoPE position so multimodal rows retain their per-sequence rope_delta.
    std::array<std::int32_t, kMaximumConcurrency * kDFlashDecodeMaximumWidth>
        target_rope_positions{};
    std::array<std::int32_t, kMaximumConcurrency> text_kv_table_rows{};
    std::array<std::int32_t, kMaximumConcurrency> dflash_kv_table_rows{};
    std::array<std::int32_t, kMaximumConcurrency> active_lanes{};
    std::array<std::int32_t, kMaximumConcurrency> state_source_slots{};
    std::array<std::int32_t, kMaximumConcurrency> state_destination_slots{};
    std::array<ops::SamplingConfig, kMaximumConcurrency> sampling{};
    // Kept-column verification (TextContext VerifyCompaction): verify_live > 0 verifies only the
    // listed padded columns (j + W*b, ascending); verify_padded_slots maps every padded column to
    // its kept index or -1. Zero verifies the whole padded block.
    std::int32_t verify_live = 0;
    std::array<std::int32_t, kMaximumConcurrency * kDFlashDecodeMaximumWidth> verify_live_columns{};
    std::array<std::int32_t, kMaximumConcurrency * kDFlashDecodeMaximumWidth> verify_padded_slots{};
    // The same for the proposal head over the K draft positions (i + K*b): proposal_live > 0
    // scores only the listed positions; the rest get zero candidates the round never accepts.
    std::int32_t proposal_live = 0;
    // Columns of the context-append block: the most target features any request appends this
    // round (0 = the full verify width). Each request appends only its last round's commits.
    std::int32_t append_width = 0;
    // Compacted context-append columns for eager DFlash2 rounds: -1 skips empty append (0 live
    // tokens), 0 uses the legacy ragged append path, >0 packs only the listed live (lane, col)
    // pairs so feature_projection and 5x context_kv_materialize process zero padded columns.
    std::int32_t append_live = 0;
    std::array<std::int32_t, kMaximumConcurrency * kDFlashDecodeMaximumWidth> append_src_lanes{};
    std::array<std::int32_t, kMaximumConcurrency * kDFlashDecodeMaximumWidth> append_src_cols{};
    std::array<std::int32_t, kMaximumConcurrency * kDFlashDecodeMaximumWidth> append_dst_positions{};
    std::array<std::int32_t, kMaximumConcurrency * kDFlashDecodeMaximumWidth> append_dst_slots{};
    // Draft block width this round (0 = the full K + 1). A narrower block drafts only as many
    // positions as any request keeps; draft_expand maps each K-wide draft column i + K*b to its
    // narrow column i + k'*b, or -1.
    std::int32_t draft_width = 0;
    // Attention-core layout of kept-column verification (TextContext VerifyCompaction): width
    // attention_width (0 = the verify width) with its slot, kept and source-column maps.
    std::int32_t attention_width = 0;
    std::array<std::int32_t, kMaximumConcurrency * kDFlashDecodeMaximumWidth> attention_slots{};
    std::array<std::int32_t, kMaximumConcurrency * kDFlashDecodeMaximumWidth> attention_live{};
    std::array<std::int32_t, kMaximumConcurrency * kDFlashDecodeMaximumWidth> attention_source{};
    std::array<std::int32_t, kMaximumConcurrency * kDFlashDecodeMaximumDrafts> draft_expand{};
    std::array<std::int32_t, kMaximumConcurrency * kDFlashDecodeMaximumDrafts> proposal_live_columns{};
    std::array<std::int32_t, kMaximumConcurrency * kDFlashDecodeMaximumDrafts> proposal_slots{};
};

struct DFlashDecodeEgress {
    std::array<TokenId, kMaximumConcurrency * kDFlashDecodeMaximumWidth> licensed_tokens{};
    std::array<std::int32_t, kMaximumConcurrency> licensed_counts{};
    std::array<std::int32_t, kMaximumConcurrency> accepted_drafts{};
    std::array<float, kMaximumConcurrency * kDFlashDecodeMaximumDrafts> draft_confidence{};
};

struct OrdinaryDecodeStateLayout {
    LayoutRegion ingress;
    LayoutRegion egress;
    TensorRegion logits;
    TensorRegion hidden;
};

struct MtpPrefillStateLayout {
    TensorRegion position;
    TensorRegion ar_hidden;
    TensorRegion draft_tokens;
    TensorRegion target_input_ids;
    TensorRegion target_positions;
};

struct DFlashPrefillStateLayout {
    TensorRegion produced_count;
};

struct MtpDecodeStateLayout {
    LayoutRegion ingress;
    LayoutRegion egress;
    TensorRegion verify_ids;
    TensorRegion target_positions;
    TensorRegion target_argmax;
    TensorRegion target_logits;
    TensorRegion target_hidden;
    TensorRegion target_continuation_hidden;
    TensorRegion proposal_logits;
    TensorRegion alignment_ids;
    TensorRegion alignment_hidden;
    TensorRegion ar_hidden;
    TensorRegion next_hidden;
    TensorRegion ar_positions;
    TensorRegion ar_rope_positions;
    TensorRegion ar_valid_columns;
};

struct DFlashDecodeStateLayout {
    LayoutRegion ingress;
    LayoutRegion egress;
    TensorRegion proposal_ids;
    TensorRegion proposal_positions;
    TensorRegion verify_positions;
    std::optional<TensorRegion> candidate_ids;
    std::optional<TensorRegion> proposal_q;
    TensorRegion append_positions;
    TensorRegion append_counts;
    TensorRegion draft_tokens;
    TensorRegion verify_ids;
    TensorRegion target_argmax;
    TensorRegion target_logits;
    TensorRegion target_hidden;
    TensorRegion target_continuation_hidden;
};

struct RoundStateLayout {
    RoundStateSpec spec;
    std::optional<OrdinaryDecodeStateLayout> ordinary;
    TensorRegion token;
    TensorRegion pos;
    TensorRegion rope_pos;
    TensorRegion rope_delta;
    TensorRegion logits;
    TensorRegion text_kv_table_row;
    TensorRegion backend_kv_table_row;
    std::optional<MtpPrefillStateLayout> mtp;
    std::optional<DFlashPrefillStateLayout> dflash_prefill;
    std::optional<MtpDecodeStateLayout> mtp_decode;
    std::optional<MtpDecodeStateLayout> mtp_lookup_decode;
    std::optional<DFlashDecodeStateLayout> dflash_decode;
    std::optional<DFlashDecodeStateLayout> dflash_copy_decode;
    bool complete = false;
};

struct OrdinaryDecodeState {
    DeviceSpan ingress;
    DeviceSpan egress;
    Tensor tokens;
    Tensor cache_positions;
    Tensor rope_positions;
    Tensor text_kv_table_rows;
    Tensor state_source_slots;
    Tensor state_destination_slots;
    const ops::SamplingConfig* sampling = nullptr;
    Tensor sampled_tokens;
    Tensor logits;
    Tensor hidden;

    OrdinaryDecodeState() = default;
    OrdinaryDecodeState(DeviceSpan backing, const OrdinaryDecodeStateLayout& layout,
                        std::uint32_t batch_capacity);
};

// The two planning calls expose one deliberate exact-target extension seam after scalar logits.
// This lets a target retain its schedule-sized prefill activation at the established physical
// address without making that activation part of the family round contract.
[[nodiscard]] RoundStateLayout begin_round_state_layout(LayoutBuilder& builder,
                                                        const RoundStateSpec& spec);
void complete_round_state_layout(LayoutBuilder& builder, RoundStateLayout& layout);

struct MtpPrefillState {
    Tensor position;
    Tensor ar_hidden;
    Tensor draft_tokens;
    Tensor target_input_ids;
    Tensor target_positions;

    MtpPrefillState() = default;
    MtpPrefillState(DeviceSpan backing, const MtpPrefillStateLayout& layout);
};

struct DFlashPrefillState {
    Tensor produced_count;

    DFlashPrefillState() = default;
    DFlashPrefillState(DeviceSpan backing, const DFlashPrefillStateLayout& layout);
};

struct MtpDecodeState {
    DeviceSpan ingress;
    DeviceSpan egress;
    Tensor anchors;
    Tensor base_frontiers;
    Tensor remaining_budgets;
    Tensor current_extents;
    Tensor target_valid_columns;
    Tensor current_drafts;
    Tensor target_rope_positions;
    Tensor text_kv_table_rows;
    Tensor mtp_kv_table_rows;
    Tensor state_source_slots;
    Tensor state_destination_slots;
    Tensor rope_deltas;
    const ops::SamplingConfig* sampling = nullptr;
    Tensor licensed_tokens;
    Tensor licensed_counts;
    Tensor accepted_drafts;
    Tensor next_drafts;
    Tensor next_extents;
    Tensor verify_ids;
    Tensor target_positions;
    Tensor target_argmax;
    Tensor target_logits;
    Tensor target_hidden;
    Tensor target_continuation_hidden;
    Tensor proposal_logits;
    Tensor alignment_ids;
    Tensor alignment_hidden;
    Tensor ar_hidden;
    Tensor next_hidden;
    Tensor ar_positions;
    Tensor ar_rope_positions;
    Tensor ar_valid_columns;

    MtpDecodeState() = default;
    MtpDecodeState(DeviceSpan backing, const MtpDecodeStateLayout& layout,
                   std::uint32_t batch_capacity, std::uint32_t verify_window,
                   std::uint32_t proposal_window);
};

struct DFlashDecodeState {
    DeviceSpan ingress;
    DeviceSpan egress;
    Tensor anchors;
    Tensor execution_frontiers;
    Tensor context_frontiers;
    Tensor proposal_extents;
    Tensor target_valid_columns;
    Tensor proposal_valid_columns;
    Tensor target_rope_positions;
    Tensor text_kv_table_rows;
    Tensor dflash_kv_table_rows;
    Tensor active_lanes;
    Tensor state_source_slots;
    Tensor state_destination_slots;
    const ops::SamplingConfig* sampling = nullptr;
    Tensor licensed_tokens;
    Tensor licensed_counts;
    Tensor accepted_drafts;
    Tensor draft_confidence;
    Tensor proposal_ids;
    Tensor proposal_positions;
    Tensor verify_positions;
    Tensor candidate_ids;
    Tensor proposal_q;
    Tensor append_positions;
    Tensor append_counts;
    Tensor draft_tokens;
    Tensor verify_ids;
    Tensor target_argmax;
    Tensor target_logits;
    Tensor target_hidden;
    Tensor target_continuation_hidden;

    DFlashDecodeState() = default;
    DFlashDecodeState(DeviceSpan backing, const DFlashDecodeStateLayout& layout,
                      std::uint32_t batch_capacity, std::uint32_t draft_window);
};

struct RoundState {
    std::optional<OrdinaryDecodeState> ordinary;
    Tensor token;
    Tensor pos;
    Tensor rope_pos;
    Tensor rope_delta;
    Tensor logits;
    Tensor text_kv_table_row;
    Tensor backend_kv_table_row;
    std::optional<MtpPrefillState> mtp;
    std::optional<DFlashPrefillState> dflash_prefill;
    std::optional<MtpDecodeState> mtp_decode;
    std::optional<MtpDecodeState> mtp_lookup_decode;
    std::optional<DFlashDecodeState> dflash_decode;
    std::optional<DFlashDecodeState> dflash_copy_decode;

    RoundState() = default;
    RoundState(DeviceSpan backing, const RoundStateLayout& layout);
};

} // namespace ninfer::targets::qwen3_6
