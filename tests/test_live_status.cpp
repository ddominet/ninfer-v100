#include "serve/live_status.h"

#include <nlohmann/json.hpp>

#include <chrono>
#include <cmath>
#include <iostream>
#include <string>

namespace {

using ninfer::RuntimeStats;
using ninfer::serve::LiveStatusRates;
using ninfer::serve::LiveStatusRateWindow;
using ninfer::serve::live_status_state;
using ninfer::serve::render_live_status;
using std::chrono::seconds;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

bool near(double actual, double expected) { return std::abs(actual - expected) <= 1e-9; }

RuntimeStats counters(std::uint64_t prefill, std::uint64_t decode) {
    RuntimeStats stats;
    stats.computed_prefill_tokens = prefill;
    stats.committed_decode_tokens = decode;
    return stats;
}

int test_state() {
    int failures = 0;
    RuntimeStats stats;
    failures += check(live_status_state(stats) == "idle", "empty Engine is not idle");

    stats.waiting_requests = 2;
    failures += check(live_status_state(stats) == "waiting", "queued-only Engine is not waiting");

    stats.running_requests      = 1;
    stats.decode_ready_requests = 1;
    failures += check(live_status_state(stats) == "generating",
                      "admitted decode request is not generating");

    stats.running_requests    = 2;
    stats.prefilling_requests = 1;
    failures += check(live_status_state(stats) == "reading",
                      "a prefilling lane does not take precedence over decoding lanes");
    return failures;
}

int test_rate_window() {
    int failures = 0;
    LiveStatusRateWindow window;
    const auto t0 = LiveStatusRateWindow::Clock::time_point{} + seconds(1000);

    LiveStatusRates rates = window.observe(t0, counters(0, 0));
    failures += check(rates.prefill_tok_s == 0.0 && rates.decode_tok_s == 0.0,
                      "first sample reported a rate");

    rates = window.observe(t0 + seconds(2), counters(2048, 0));
    failures +=
        check(rates.prefill_tok_s == 0.0, "rate reported before the window spans the minimum");

    rates = window.observe(t0 + seconds(4), counters(4096, 100));
    failures += check(near(rates.prefill_tok_s, 1024.0) && near(rates.decode_tok_s, 25.0),
                      "rate over the first 4 s is not counter delta / elapsed time");

    // A poll between two chunk boundaries sees no counter change but still reports the window rate.
    rates = window.observe(t0 + seconds(5), counters(4096, 140));
    failures += check(near(rates.prefill_tok_s, 4096.0 / 5.0) && near(rates.decode_tok_s, 28.0),
                      "poll between counter updates dropped the window rate");

    // After 11 s of no progress every retained sample is newer than the work, so the rate is zero.
    rates = window.observe(t0 + seconds(16), counters(4096, 140));
    failures += check(rates.prefill_tok_s == 0.0 && rates.decode_tok_s == 0.0,
                      "stale work outside the window still produced a rate");

    // A smaller counter cannot occur within one Engine; it must not wrap into a huge rate.
    rates = window.observe(t0 + seconds(20), counters(10, 10));
    failures += check(rates.prefill_tok_s == 0.0 && rates.decode_tok_s == 0.0,
                      "decreasing counter produced a rate");
    return failures;
}

int test_render() {
    int failures = 0;
    RuntimeStats stats;
    stats.running_requests      = 2;
    stats.prefilling_requests   = 1;
    stats.decode_ready_requests = 1;
    stats.waiting_requests      = 3;
    const nlohmann::json body =
        render_live_status(true, "qwen3.8-27b", 4, stats, LiveStatusRates{1200.5, 41.25});
    failures += check(body.at("status") == "ok" && body.at("model") == "qwen3.8-27b" &&
                          body.at("state") == "reading",
                      "status identity fields are wrong");
    failures +=
        check(body.at("running") == 2 && body.at("prefilling") == 1 && body.at("decoding") == 1 &&
                  body.at("waiting") == 3 && body.at("max_concurrency") == 4,
              "status counts are wrong");
    failures += check(body.at("prefill_tok_s") == 1200.5 && body.at("decode_tok_s") == 41.25,
                      "status rates are wrong");
    failures += check(body.size() == 10, "status body has unexpected fields");

    const nlohmann::json down = render_live_status(false, "m", 1, RuntimeStats{}, {});
    failures += check(down.at("status") == "unavailable" && down.at("state") == "idle",
                      "unavailable Engine is not reported");
    return failures;
}

} // namespace

int main() {
    const int failures = test_state() + test_rate_window() + test_render();
    if (failures != 0) {
        std::cerr << failures << " live status check(s) failed\n";
        return 1;
    }
    return 0;
}
