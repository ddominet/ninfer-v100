#include "serve/live_status.h"

#include <string>

namespace ninfer::serve {

std::string_view live_status_state(const ninfer::RuntimeStats& stats) noexcept {
    if (stats.prefilling_requests > 0) { return "reading"; }
    if (stats.running_requests > 0) { return "generating"; }
    if (stats.waiting_requests > 0) { return "waiting"; }
    return "idle";
}

namespace {

double counter_rate(std::uint64_t current, std::uint64_t first, double seconds) noexcept {
    // The counters are monotonic within one Engine; a smaller value cannot produce a rate.
    if (current <= first || seconds <= 0.0) { return 0.0; }
    return static_cast<double>(current - first) / seconds;
}

} // namespace

LiveStatusRates LiveStatusRateWindow::observe(Clock::time_point now,
                                              const ninfer::RuntimeStats& stats) {
    std::lock_guard lock(mutex_);
    samples_.push_back({now, stats.computed_prefill_tokens, stats.committed_decode_tokens});
    // Keep the newest sample at or beyond the window start so the reported span stays >= kWindow
    // once enough history exists.
    while (samples_.size() > 2 && now - samples_[1].time >= kWindow) { samples_.pop_front(); }

    LiveStatusRates rates;
    const Sample& first = samples_.front();
    if (now - first.time < kMinimumSpan) { return rates; }
    const double seconds = std::chrono::duration<double>(now - first.time).count();
    rates.prefill_tok_s =
        counter_rate(stats.computed_prefill_tokens, first.prefill_tokens, seconds);
    rates.decode_tok_s = counter_rate(stats.committed_decode_tokens, first.decode_tokens, seconds);
    return rates;
}

nlohmann::json render_live_status(bool available, std::string_view model_id,
                                  std::uint32_t max_concurrency, const ninfer::RuntimeStats& stats,
                                  const LiveStatusRates& rates) {
    return nlohmann::json{
        {"status", available ? "ok" : "unavailable"},
        {"model", std::string(model_id)},
        {"state", std::string(live_status_state(stats))},
        {"running", stats.running_requests},
        {"prefilling", stats.prefilling_requests},
        {"decoding", stats.decode_ready_requests},
        {"waiting", stats.waiting_requests},
        {"max_concurrency", max_concurrency},
        {"prefill_tok_s", rates.prefill_tok_s},
        {"decode_tok_s", rates.decode_tok_s},
    };
}

} // namespace ninfer::serve
