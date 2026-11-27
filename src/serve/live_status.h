#pragma once

// Live scheduler status served by GET /status: the request phase derived from one RuntimeStats
// snapshot, plus prefill and decode token rates over a short trailing window.

#include "ninfer/types.h"

#include <nlohmann/json.hpp>

#include <chrono>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string_view>

namespace ninfer::serve {

// "reading" while a lane prefills, "generating" while an admitted request is past its prefill,
// "waiting" while requests are only queued for admission, otherwise "idle".
[[nodiscard]] std::string_view live_status_state(const ninfer::RuntimeStats& stats) noexcept;

struct LiveStatusRates {
    double prefill_tok_s = 0.0;
    double decode_tok_s  = 0.0;
};

// The prefill and decode counters advance only at prefill-chunk and decode-round boundaries, so the
// rate between two adjacent polls alternates between zero and a burst. The window keeps the samples
// of the trailing kWindow and reports the rate since the oldest one once they span kMinimumSpan.
class LiveStatusRateWindow {
public:
    using Clock = std::chrono::steady_clock;

    static constexpr std::chrono::seconds kMinimumSpan{3};
    static constexpr std::chrono::seconds kWindow{10};

    [[nodiscard]] LiveStatusRates observe(Clock::time_point now, const ninfer::RuntimeStats& stats);

private:
    struct Sample {
        Clock::time_point time;
        std::uint64_t prefill_tokens = 0;
        std::uint64_t decode_tokens  = 0;
    };

    std::mutex mutex_;
    std::deque<Sample> samples_;
};

[[nodiscard]] nlohmann::json render_live_status(bool available, std::string_view model_id,
                                                std::uint32_t max_concurrency,
                                                const ninfer::RuntimeStats& stats,
                                                const LiveStatusRates& rates);

} // namespace ninfer::serve
