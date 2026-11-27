// N-gram copy proposer and tool-result de-numbering (CPU only).
#include "targets/qwen3_6/impl/frontend/ngram_sources.h"
#include "targets/qwen3_6/impl/runtime/ngram_proposer.h"

#include <algorithm>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using ninfer::targets::qwen3_6::detail::NgramCopyBackoff;
using ninfer::targets::qwen3_6::detail::NgramPartnerBalance;
using ninfer::targets::qwen3_6::detail::NgramProposer;
using ninfer::targets::qwen3_6::detail::NgramRequestIndex;
using ninfer::targets::qwen3_6::detail::ngram_index_capacity;
using ninfer::targets::qwen3_6::frontend_internal::ngram_numbered_sources;
using Token = ninfer::TokenId;

namespace {

void require(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}

void test_proposer() {
    std::vector<Token> source(128);
    std::iota(source.begin(), source.end(), 100);
    NgramProposer proposer(256, 1024);
    proposer.ingest(source);
    require(proposer.propose(source, 0).tokens.empty(), "zero budget cannot propose tokens");
    for (unsigned n : {4, 8, 12, 16, 32, 64}) {
        const auto match = proposer.propose(std::span<const Token>(source).first(n), 15, 4);
        require(match.matched == n, "backward match extension");
        require(match.tokens == std::vector<Token>(source.begin() + n, source.begin() + n + 15),
                "contiguous source proposal");
    }
    auto match = proposer.propose(std::span<const Token>(source).first(126), 15, 12);
    require(match.tokens.size() == 2, "stop at source boundary");
    require(proposer.propose(source, 15, 12).tokens.empty(), "no future source tokens");
    require(proposer.propose(std::span<const Token>(source).first(8), 15, 12).tokens.empty(),
            "admission threshold");

    for (std::uint32_t available = 0; available <= 32; ++available) {
        std::vector<Token> bounded_source(64 + available);
        std::iota(bounded_source.begin(), bounded_source.end(), 1000);
        NgramProposer bounded(256, 1024);
        bounded.ingest(bounded_source);
        const auto history = std::span<const Token>(bounded_source).first(64);
        const auto draft   = bounded.propose_for_round(history, 15, 7);
        if (available >= 17) {
            require(draft.tokens.size() == 15, "full copy window preserved");
        } else if (available > 9) {
            require(draft.tokens.size() == available - 2, "partial copy beats neural drafts");
        } else {
            require(draft.tokens.empty(), "short copy dropped in favor of neural draft");
        }
    }

    NgramProposer special(256, 1024);
    special.set_boundaries({120});
    special.ingest(source);
    require(special.propose(std::span<const Token>(source).first(18), 15, 12).tokens.size() == 2,
            "stop before control token");
}

void test_capacity_and_request_index() {
    require(ngram_index_capacity(2'048).tokens == 4'096, "min ring size");
    require(ngram_index_capacity(262'144).tokens == (1U << 18), "256K context ring");
    require(ngram_index_capacity(std::numeric_limits<std::uint32_t>::max()).tokens == (1U << 20),
            "max ring size");

    std::vector<Token> prompt(400);
    std::iota(prompt.begin(), prompt.end(), 500);
    NgramRequestIndex index(ngram_index_capacity(4'096), {248044});
    index.index_prompt(prompt, 0, {});
    std::vector<Token> ledger = prompt;
    ledger.push_back(9999);
    ledger.insert(ledger.end(), prompt.begin() + 50, prompt.begin() + 70);
    const auto copy = index.propose_for_round(ledger, 15, 7, 12);
    require(copy.tokens == std::vector<Token>(prompt.begin() + 70, prompt.begin() + 85),
            "lazy catch-up proposed exact continuation");
}

void test_backoff_and_partner() {
    NgramCopyBackoff backoff;
    require(backoff.required_match(12) == 12, "initial required match");
    backoff.record(15, 2);
    require(backoff.required_match(12) == 24, "miss doubles match");
    backoff.record(15, 0);
    require(backoff.required_match(12) == 36, "second miss triples match");
    backoff.record(15, 15);
    require(backoff.required_match(12) == 24, "hit lowers backoff");

    NgramPartnerBalance partner;
    require(!partner.can_widen(), "new partner cannot be widened immediately");
    for (int i = 0; i < 20; ++i) { partner.neural_round(); }
    require(partner.can_widen(), "partner accumulates allowance");
    partner.widened();
    require(!partner.can_widen(), "widening consumes allowance");
}

void test_numbered_sources() {
    require(ngram_numbered_sources("     1\xe2\x86\x92#include <a>\n     2\xe2\x86\x92\n"
                                   "     3\xe2\x86\x92int x;\n") ==
                std::vector<std::string>{"#include <a>\n\nint x;\n"},
            "Read-style arrow numbering");
    require(ngram_numbered_sources("L7: a\nL8: b\nL9: c\n") == std::vector<std::string>{"a\nb\nc\n"},
            "L-prefixed numbering");
    require(ngram_numbered_sources("1: a\n2: b\n").empty(), "fewer than three lines ignored");
}

} // namespace

int main() {
    try {
        test_proposer();
        test_capacity_and_request_index();
        test_backoff_and_partner();
        test_numbered_sources();
        std::cout << "ngram proposer tests passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
