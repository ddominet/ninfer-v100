// DFlash2 context K/V pair (two [1024, 5120] W8 weights) through ops::linear_pair against two
// quadpair launches, at append widths. Not a test: build target only.

#include "core/arena.h"
#include "core/device.h"
#include "ninfer/ops/linear_pair.h"
#include "ops/linear/linear_test_common.h"
#include "ops/linear/w8/w8_launch.h"
#include "ops/quantized_weight.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

namespace {
using namespace ninfer;
template <class F>
float time_us(F&& launch, int repeats = 50) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    for (int i = 0; i < 3; ++i) { launch(); }
    cudaEventRecord(start);
    for (int i = 0; i < repeats; ++i) { launch(); }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    return 1000.0F * ms / static_cast<float>(repeats);
}
std::vector<float> host(const DeviceBuffer& b, std::size_t n) {
    std::vector<__nv_bfloat16> v(n);
    CUDA_CHECK(cudaMemcpy(v.data(), b.p, n * 2, cudaMemcpyDeviceToHost));
    std::vector<float> f(n);
    for (std::size_t i = 0; i < n; ++i) { f[i] = __bfloat162float(v[i]); }
    return f;
}
} // namespace

int main() {
    constexpr int n = 1024, k = 5120;
    const auto first_host  = test::linear::make_w8g32_f16s_weight(n, k, 31);
    const auto second_host = test::linear::make_w8g32_f16s_weight(n, k, 32);
    DeviceBuffer first_payload(first_host.payload.size()), second_payload(second_host.payload.size());
    first_payload.copy_from_host(first_host.payload.data(), first_payload.bytes);
    second_payload.copy_from_host(second_host.payload.data(), second_payload.bytes);
    const Weight first = first_host.device_weight(first_payload.p);
    const Weight second = second_host.device_weight(second_payload.p);
    std::vector<__nv_bfloat16> activation(static_cast<std::size_t>(k) * 64);
    for (std::size_t i = 0; i < activation.size(); ++i) {
        activation[i] = __float2bfloat16(static_cast<float>(static_cast<int>((i * 7919U) % 255U) - 127) / 256.0F);
    }
    DeviceBuffer x(activation.size() * 2);
    x.copy_from_host(activation.data(), x.bytes);
    DeviceBuffer a1(n * 64 * 2), a2(n * 64 * 2), b1(n * 64 * 2), b2(n * 64 * 2);
    for (int t : {2, 4, 8, 12, 16, 24, 32, 48, 64}) {
        Tensor input(x.p, DType::BF16, {k, t});
        Tensor o1(a1.p, DType::BF16, {n, t}), o2(a2.p, DType::BF16, {n, t});
        Tensor q1(b1.p, DType::BF16, {n, t}), q2(b2.p, DType::BF16, {n, t});
        const float pair = time_us([&] { ops::linear_pair(input, first, second, o1, o2, nullptr); });
        const float qpn  = time_us([&] {
            ops::detail::launch_w8_volta_qpn(input, first, q1, nullptr);
            ops::detail::launch_w8_volta_qpn(input, second, q2, nullptr);
        });
        CUDA_CHECK(cudaDeviceSynchronize());
        double worst = 0, scale = 0;
        for (auto [p, q] : {std::pair{&a1, &b1}, std::pair{&a2, &b2}}) {
            const auto u = host(*p, static_cast<std::size_t>(n) * t), v = host(*q, static_cast<std::size_t>(n) * t);
            for (std::size_t i = 0; i < u.size(); ++i) {
                worst = std::max(worst, std::fabs(static_cast<double>(u[i]) - v[i]));
                scale = std::max(scale, std::fabs(static_cast<double>(u[i])));
            }
        }
        std::printf("T=%2d  linear_pair %7.1f us | two quadpair launches %7.1f us | max|diff| %.3g of %.3g\n",
                    t, pair, qpn, worst, scale);
    }
    return 0;
}
