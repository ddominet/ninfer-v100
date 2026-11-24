// Tensor-core skinny route vs the quadpair kernels through the real ops::linear dispatch, on the
// 27B target's prepacked FP8 and NVFP4 projection shapes at speculative verify widths. For each
// width: quadpair time, the measured-default tensor-core schedule, the best swept schedule, and the
// largest output difference between the two routes. Not a test: build target only.

#include "core/arena.h"
#include "core/device.h"
#include "ninfer/ops/linear.h"
#include "ops/linear/fp8/fp8_prepack_sm70.h"
#include "ops/linear/linear_test_common.h"
#include "ops/linear/nvfp4/nvfp4_prepack_sm70.h"
#include "ops/linear/volta_tc_skinny.h"
#include "ops/quantized_weight.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

using namespace ninfer;

template <class F>
float time_us(F&& launch, int repeats = 30) {
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
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 1000.0F * ms / static_cast<float>(repeats);
}

std::vector<float> to_host(const DeviceBuffer& buffer, std::size_t count) {
    std::vector<__nv_bfloat16> bits(count);
    CUDA_CHECK(cudaMemcpy(bits.data(), buffer.p, count * 2, cudaMemcpyDeviceToHost));
    std::vector<float> values(count);
    for (std::size_t i = 0; i < count; ++i) { values[i] = __bfloat162float(bits[i]); }
    return values;
}

struct Case {
    const char* label;
    QType qtype;
    int n;
    int k;
};

} // namespace

int main(int argc, char** argv) {
    const bool sweep = argc > 1 && std::strcmp(argv[1], "--sweep") == 0;
    const Case cases[] = {
        {"nvfp4 gate|up half 17408x5120", QType::NVFP4, 17408, 5120},
        {"nvfp4 down 5120x17408", QType::NVFP4, 5120, 17408},
        {"fp8 gdn qkvz 16384x5120", QType::FP8_E4M3FN_ROW_BF16S, 16384, 5120},
        {"fp8 attn qkgv 14336x5120", QType::FP8_E4M3FN_ROW_BF16S, 14336, 5120},
        {"fp8 out 5120x6144", QType::FP8_E4M3FN_ROW_BF16S, 5120, 6144},
    };
    auto& control = ops::detail::volta_tc_control();
    for (const Case& c : cases) {
        const auto host = c.qtype == QType::NVFP4 ? test::linear::make_nvfp4_weight(c.n, c.k, 91)
                                                  : test::linear::make_fp8_weight(c.n, c.k, 91);
        DeviceBuffer payload(host.payload.size());
        payload.copy_from_host(host.payload.data(), payload.bytes);
        Weight weight = host.device_weight(payload.p);
        if (c.qtype == QType::NVFP4) {
            ops::detail::nvfp4_prepack_qpn_sm70(weight);
        } else {
            ops::detail::fp8_prepack_qpn_sm70(weight);
        }
        std::vector<__nv_bfloat16> activation(static_cast<std::size_t>(c.k) * 64);
        for (std::size_t i = 0; i < activation.size(); ++i) {
            activation[i] = __float2bfloat16(static_cast<float>(static_cast<int>((i * 7919U) % 255U) - 127) / 256.0F);
        }
        DeviceBuffer x(activation.size() * 2);
        x.copy_from_host(activation.data(), x.bytes);
        DeviceBuffer out_qpn(static_cast<std::size_t>(c.n) * 64 * 2);
        DeviceBuffer out_tc(static_cast<std::size_t>(c.n) * 64 * 2);
        const double bytes = static_cast<double>(host.payload.size());
        for (int t : {8, 9, 16, 24, 32, 40, 48, 64}) {
            Tensor input(x.p, DType::BF16, {c.k, t});
            Tensor qpn_out(out_qpn.p, DType::BF16, {c.n, t});
            Tensor tc_out(out_tc.p, DType::BF16, {c.n, t});
            control.enabled = 0;
            const float qpn = time_us([&] { ops::linear(input, weight, qpn_out, nullptr); });
            control.enabled = 1;
            control.rows = 0;
            control.splits = 0;
            const float tc = time_us([&] { ops::linear(input, weight, tc_out, nullptr); });
            CUDA_CHECK(cudaDeviceSynchronize());
            const auto a = to_host(out_qpn, static_cast<std::size_t>(c.n) * t);
            const auto b = to_host(out_tc, static_cast<std::size_t>(c.n) * t);
            double worst = 0, scale = 0;
            for (std::size_t i = 0; i < a.size(); ++i) {
                worst = std::max(worst, std::fabs(static_cast<double>(a[i]) - b[i]));
                scale = std::max(scale, std::fabs(static_cast<double>(a[i])));
            }
            char best_desc[32] = "-";
            float best = tc;
            if (sweep && t > 8) {
                for (int rows : {64, 128}) {
                    if (c.n % rows != 0) { continue; }
                    for (int splits : {1, 2, 4, 8}) {
                        control.rows   = rows;
                        control.splits = splits;
                        const float v = time_us([&] { ops::linear(input, weight, tc_out, nullptr); });
                        if (v < best) {
                            best = v;
                            std::snprintf(best_desc, sizeof(best_desc), "%dr s%d", rows, splits);
                        }
                    }
                }
                control.rows   = 0;
                control.splits = 0;
            }
            std::printf("%-30s T=%2d  quadpair %7.1f us | tensor-core %7.1f us (%4.0f GB/s, %.2fx)"
                        " | best %7.1f [%s] | max|diff| %.3g of %.3g\n",
                        c.label, t, qpn, tc, bytes / tc / 1e3, qpn / tc, best, best_desc, worst,
                        scale);
        }
    }
    control.enabled = -1;
    return 0;
}
