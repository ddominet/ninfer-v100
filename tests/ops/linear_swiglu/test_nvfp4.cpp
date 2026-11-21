#include "ops/linear_swiglu/linear_swiglu_test_common.h"

#include <array>
#include <exception>
#include <iostream>

int main() {
    using namespace ninfer;
    using namespace ninfer::test::linear_swiglu;

    try {
#ifdef NINFER_VOLTA_BUILD
        // One width per QPN tile bucket (1, 2, 3, 4, 6, 8 tiles), multi-pass widths, then the
        // CUTLASS route in one and in two column passes.
        constexpr std::array<std::int32_t, 16> kA16Cases{1,  4,  8,   16,  17,  32,  33,  49,
                                                         64, 65, 100, 128, 192, 193, 256, 1027};
#else
        constexpr std::array<std::int32_t, 4> kA16Cases{1, 4, 8, 16};
#endif
        constexpr std::array<std::int32_t, 6> kA4Cases{5, 48, 49, 128, 256, 1024};
        int failures = 0;
        failures += run_profile("LinearSwiGLU NVFP4_A16",
                                {QType::NVFP4, 34816, 5120, 17408, 1801U, ActivationCompute::A16},
                                kA16Cases);
#ifndef NINFER_VOLTA_BUILD
        failures +=
            run_profile("LinearSwiGLU NVFP4_A4",
                        {QType::NVFP4, 34816, 5120, 17408, 1803U, ActivationCompute::A4}, kA4Cases);
#endif
        std::cout << (failures == 0 ? "OK" : "FAIL") << " LinearSwiGLU NVFP4 correctness\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "LinearSwiGLU NVFP4 test failed: " << error.what() << '\n';
        return 1;
    }
}
