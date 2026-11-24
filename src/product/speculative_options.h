#pragma once

#include "ninfer/types.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>

namespace ninfer::product {

[[nodiscard]] inline SpeculativeBackend parse_speculative_backend(std::string_view value) {
    if (value == "mtp") { return SpeculativeBackend::Mtp; }
    if (value == "dflash") { return SpeculativeBackend::DFlash; }
    if (value == "dflash2") { return SpeculativeBackend::DFlash2; }
    throw std::invalid_argument("invalid speculative backend: " + std::string(value));
}

[[nodiscard]] inline const char* speculative_backend_name(SpeculativeBackend backend) noexcept {
    switch (backend) {
    case SpeculativeBackend::None:
        return "none";
    case SpeculativeBackend::Mtp:
        return "mtp";
    case SpeculativeBackend::DFlash:
        return "dflash";
    case SpeculativeBackend::DFlash2:
        return "dflash2";
    }
    return "unknown";
}

// "5,3,2" -> {5,3,2,2,2,2,2,2}: one draft length per batch size, the last repeated to the end.
[[nodiscard]] inline std::array<std::uint8_t, kMaximumConcurrency>
parse_draft_tokens_by_batch(std::string_view value) {
    std::array<std::uint8_t, kMaximumConcurrency> out{};
    std::size_t count = 0;
    while (!value.empty()) {
        const std::size_t comma = value.find(',');
        const std::string_view item = value.substr(0, comma);
        if (item.empty() || item.size() > 2 || count == out.size() ||
            !std::all_of(item.begin(), item.end(), [](char c) { return c >= '0' && c <= '9'; })) {
            throw std::invalid_argument(
                "--draft-tokens-by-batch takes 1-8 comma-separated draft lengths");
        }
        const int k = std::stoi(std::string(item));
        if (k < 1 || k > 15) {
            throw std::invalid_argument("--draft-tokens-by-batch lengths must be positive");
        }
        out[count++] = static_cast<std::uint8_t>(k);
        if (comma == std::string_view::npos) { break; }
        value.remove_prefix(comma + 1);
        if (value.empty()) {
            throw std::invalid_argument("--draft-tokens-by-batch has a trailing comma");
        }
    }
    if (count == 0) { throw std::invalid_argument("--draft-tokens-by-batch is empty"); }
    for (std::size_t i = count; i < out.size(); ++i) { out[i] = out[count - 1]; }
    return out;
}

[[nodiscard]] inline bool has_draft_tokens_by_batch(const SpeculativeOptions& options) noexcept {
    return options.draft_tokens_by_batch[0] != 0;
}

inline void validate_speculative_cli_options(const SpeculativeOptions& options) {
    if (has_draft_tokens_by_batch(options)) {
        if (options.backend != SpeculativeBackend::Mtp) {
            throw std::invalid_argument("--draft-tokens-by-batch requires --spec mtp");
        }
        for (const std::uint8_t k : options.draft_tokens_by_batch) {
            if (k == 0 || k > options.draft_tokens) {
                throw std::invalid_argument(
                    "--draft-tokens-by-batch lengths must be in [1, --draft-tokens]");
            }
        }
    }
    switch (options.backend) {
    case SpeculativeBackend::None:
        if (options.draft_tokens != 0 || options.proposal_head != ProposalHead::Full) {
            throw std::invalid_argument(
                "--draft-tokens and --lm-head-draft require --spec mtp|dflash|dflash2");
        }
        return;
    case SpeculativeBackend::Mtp:
        // The sm_70 width-6+ target-verify regression (draft window >= 5 drifting off the
        // greedy argmax) tracked back to the dedicated Volta small_t_i8/bf16 verify kernels
        // dropped during the DFlash2 merge; both are restored (small_t_i8_volta.cuh,
        // small_t_bf16_volta.cuh) and every draft window through 7 is bit-exact against
        // --spec none again, so sm_70 no longer needs a narrower cap than upstream.
        if (options.draft_tokens == 0 || options.draft_tokens > 7) {
            throw std::invalid_argument("--spec mtp requires --draft-tokens in [1,7]");
        }
        return;
    case SpeculativeBackend::DFlash:
        if (options.draft_tokens == 0 || options.draft_tokens > 15) {
            throw std::invalid_argument("--spec dflash requires --draft-tokens in [1,15]");
        }
        return;
    case SpeculativeBackend::DFlash2:
        if (options.draft_tokens == 0 || options.draft_tokens > 15) {
            throw std::invalid_argument("--spec dflash2 requires --draft-tokens in [1,15]");
        }
        return;
    }
    throw std::invalid_argument("invalid speculative backend");
}

} // namespace ninfer::product
