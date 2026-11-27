#include "core/arena.h"
#include "ninfer/ops/attn_input_proj.h"
#include "ninfer/ops/gdn_input_proj.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ops/linear/linear_test_common.h"
#include "ops/op_tester.h"
#include "ops/quantized_weight.h"

#ifdef NINFER_VOLTA_BUILD
#include "ops/linear/common/volta_qpn_groupwise.h"
#include "ops/linear/q4/q4_prepack_sm70.h"
#include "ops/linear/q5/q5_prepack_sm70.h"
#include "ops/linear/q6/q6_prepack_sm70.h"
#include "ops/linear/volta_fp16_stage.h"
#endif

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::test;

#ifdef NINFER_VOLTA_BUILD

constexpr double kBf16UnitRoundoff = 1.0 / 256.0;
constexpr ReductionCriterion kA16Criterion{
    kBf16UnitRoundoff, kBf16UnitRoundoff, 2.0 * kBf16UnitRoundoff};

struct PrepackedPair {
    DeviceBuffer row_split_buf;
    DeviceBuffer prepacked_buf;
    Weight row_split;
    Weight prepacked;
};

PrepackedPair make_pair(QType qtype, std::int32_t n, std::int32_t k, std::uint32_t seed) {
    quantized_weight::PackedWeight host = quantized_weight::make_patterned_weight(
        qtype, n, k, seed,
        {quantized_weight::RowSplitScalePattern::Small,
         quantized_weight::RowSplitCodePattern::Hashed});
    DeviceBuffer d_rs(host.payload.size());
    d_rs.copy_from_host(host.payload.data(), host.payload.size());
    DeviceBuffer d_pp(host.payload.size());
    d_pp.copy_from_host(host.payload.data(), host.payload.size());
    Weight w_rs = host.device_weight(d_rs.p);
    Weight w_pp = host.device_weight(d_pp.p);
    switch (qtype) {
    case QType::Q4G64_F16S: ops::detail::q4_prepack_qpn_sm70(w_pp, nullptr); break;
    case QType::Q5G64_F16S: ops::detail::q5_prepack_qpn_sm70(w_pp, nullptr); break;
    case QType::Q6G64_F16S: ops::detail::q6_prepack_qpn_sm70(w_pp, nullptr); break;
    default: throw std::invalid_argument("unsupported qtype");
    }
    cuda_synchronize();
    return {std::move(d_rs), std::move(d_pp), w_rs, w_pp};
}

DeviceBuffer make_bf16_input(std::size_t elems, std::uint32_t seed) {
    std::vector<float> h(elems);
    fill_uniform(h, seed, -0.5f, 0.5f);
    round_to_bf16(h);
    return to_device_bf16(h);
}

int test_linear_qtype(const char* name, QType qtype, std::int32_t n, std::int32_t k,
                      std::uint32_t seed) {
    int failures                    = 0;
    constexpr std::array<int, 11> kTs{1, 7, 8, 9, 16, 24, 32, 48, 64, 65, 128};
    PrepackedPair weights           = make_pair(qtype, n, k, seed);
    const int max_t                 = kTs.back();
    DeviceBuffer d_x                = make_bf16_input(static_cast<std::size_t>(k) * max_t, seed + 1);
    const std::size_t ws_cap = ops::linear_workspace_capacity_bytes(
        qtype, n, k, ops::LinearPolicy::A16Only, 1, max_t);
    DeviceArena ws(std::max<std::size_t>(ws_cap, 1024));

    for (int t : kTs) {
        const std::size_t out_elems = static_cast<std::size_t>(n) * t;
        DeviceBuffer d_ref(out_elems * sizeof(std::uint16_t));
        DeviceBuffer d_act(out_elems * sizeof(std::uint16_t));
        Tensor x(d_x.p, DType::BF16, {k, t});
        Tensor out_ref(d_ref.p, DType::BF16, {n, t});
        Tensor out_act(d_act.p, DType::BF16, {n, t});
        ops::linear(x, weights.row_split, out_ref, ops::LinearPolicy::A16Only, ws, nullptr);
        ops::linear(x, weights.prepacked, out_act, ops::LinearPolicy::A16Only, ws, nullptr);
        cuda_synchronize();

        const std::string lbl = std::string("linear ") + name + " [" + std::to_string(n) + "," +
                                std::to_string(k) + "] T=" + std::to_string(t);
        failures += verify_reduction(lbl, from_device_bf16(d_act, out_elems),
                                     from_device_bf16(d_ref, out_elems), kA16Criterion);
    }
    return failures;
}

int test_swiglu_q4() {
    int failures          = 0;
    constexpr int n       = 34816;
    constexpr int k       = 5120;
    constexpr int out_n   = n / 2;
    constexpr std::array<int, 8> kTs{1, 8, 16, 24, 32, 48, 64, 128};
    PrepackedPair weights = make_pair(QType::Q4G64_F16S, n, k, 301U);
    const int max_t       = kTs.back();
    DeviceBuffer d_x      = make_bf16_input(static_cast<std::size_t>(k) * max_t, 302U);
    const std::size_t ws_cap = ops::linear_swiglu_workspace_capacity_bytes(
        QType::Q4G64_F16S, n, k, ops::LinearPolicy::A16Only, 1, max_t);
    DeviceArena ws(std::max<std::size_t>(ws_cap, 1024));

    for (int t : kTs) {
        const std::size_t out_elems = static_cast<std::size_t>(out_n) * t;
        DeviceBuffer d_ref(out_elems * sizeof(std::uint16_t));
        DeviceBuffer d_act(out_elems * sizeof(std::uint16_t));
        Tensor x(d_x.p, DType::BF16, {k, t});
        Tensor out_ref(d_ref.p, DType::BF16, {out_n, t});
        Tensor out_act(d_act.p, DType::BF16, {out_n, t});
        ops::linear_swiglu(x, weights.row_split, out_ref, ops::LinearPolicy::A16Only, ws, nullptr);
        ops::linear_swiglu(x, weights.prepacked, out_act, ops::LinearPolicy::A16Only, ws, nullptr);
        cuda_synchronize();

        const std::string lbl = "linear_swiglu Q4 T=" + std::to_string(t);
        failures += verify_reduction(lbl, from_device_bf16(d_act, out_elems),
                                     from_device_bf16(d_ref, out_elems), kA16Criterion);
    }

    // Also validate fused GM-in -> GM-out SwiGLU + GM residual add pipeline (T = 1..64)
    PrepackedPair down_w = make_pair(QType::Q5G64_F16S, 5120, 17408, 305U);
    DeviceBuffer d_res_init = make_bf16_input(static_cast<std::size_t>(5120) * 64, 306U);
    DeviceBuffer d_gate_scratch(static_cast<std::size_t>(17408) * 64 * sizeof(float));
    DeviceBuffer d_mid_gm(static_cast<std::size_t>(17408) * 64 * sizeof(half));
    const std::size_t add_ws_cap = ops::linear_add_workspace_capacity_bytes(
        QType::Q5G64_F16S, 5120, 17408, ops::LinearPolicy::A16Only, 1, 64);
    DeviceArena add_ws(std::max<std::size_t>(add_ws_cap, 1024));

    for (int t : std::array<int, 6>{1, 8, 16, 24, 32, 64}) {
        const std::size_t mid_elems = static_cast<std::size_t>(17408) * t;
        const std::size_t res_elems = static_cast<std::size_t>(5120) * t;
        DeviceBuffer d_mid_ref(mid_elems * sizeof(std::uint16_t));
        DeviceBuffer d_res_ref(res_elems * sizeof(std::uint16_t));
        DeviceBuffer d_res_act(res_elems * sizeof(std::uint16_t));
        cuda_check(cudaMemcpy(d_res_ref.p, d_res_init.p, d_res_ref.bytes, cudaMemcpyDeviceToDevice),
                   "copy res_ref");
        cuda_check(cudaMemcpy(d_res_act.p, d_res_init.p, d_res_act.bytes, cudaMemcpyDeviceToDevice),
                   "copy res_act");

        Tensor x(d_x.p, DType::BF16, {k, t});
        Tensor mid_ref(d_mid_ref.p, DType::BF16, {17408, t});
        Tensor res_ref(d_res_ref.p, DType::BF16, {5120, t});
        Tensor res_act(d_res_act.p, DType::BF16, {5120, t});

        ops::linear_swiglu(x, weights.row_split, mid_ref, ops::LinearPolicy::A16Only, ws, nullptr);
        ops::linear_add(mid_ref, down_w.row_split, res_ref, ops::LinearPolicy::A16Only, add_ws,
                        nullptr);

        const ops::detail::VoltaGroupMajorHalf* x_gm =
            ops::detail::volta_stage_fp16_group_major(x, nullptr);
        if (x_gm == nullptr) {
            std::cerr << "volta_stage_fp16_group_major returned nullptr at T=" << t << "\n";
            ++failures;
            continue;
        }
        ops::detail::launch_q4_prepacked_volta_swiglu_gm_io(
            x, x_gm, weights.prepacked,
            static_cast<ops::detail::VoltaGroupMajorHalf*>(d_mid_gm.p),
            static_cast<float*>(d_gate_scratch.p), nullptr);
        Tensor mid_shape(d_mid_gm.p, DType::BF16, {17408, t});
        ops::detail::launch_q5_prepacked_volta_residual_gm(
            mid_shape, static_cast<const ops::detail::VoltaGroupMajorHalf*>(d_mid_gm.p),
            down_w.prepacked, res_act, nullptr);
        cuda_synchronize();

        const std::string lbl = "mlp_gm_io Q4+Q5 T=" + std::to_string(t);
        // Composes two projections where prepacked keeps FP16 intermediate vs RowSplit's BF16.
        constexpr ReductionCriterion kTwoOpCriterion{
            2.0 * kBf16UnitRoundoff, 2.0 * kBf16UnitRoundoff, 4.0 * kBf16UnitRoundoff};
        failures += verify_reduction(lbl, from_device_bf16(d_res_act, res_elems),
                                     from_device_bf16(d_res_ref, res_elems), kTwoOpCriterion);
    }
    return failures;
}

int test_linear_add(const char* name, QType qtype, std::int32_t n, std::int32_t k,
                    std::uint32_t seed) {
    int failures          = 0;
    constexpr std::array<int, 7> kTs{1, 8, 16, 32, 48, 64, 128};
    PrepackedPair weights = make_pair(qtype, n, k, seed);
    const int max_t       = kTs.back();
    DeviceBuffer d_x      = make_bf16_input(static_cast<std::size_t>(k) * max_t, seed + 1);
    DeviceBuffer d_r0     = make_bf16_input(static_cast<std::size_t>(n) * max_t, seed + 2);
    const std::size_t ws_cap = ops::linear_add_workspace_capacity_bytes(
        qtype, n, k, ops::LinearPolicy::A16Only, 1, max_t);
    DeviceArena ws(std::max<std::size_t>(ws_cap, 1024));

    for (int t : kTs) {
        const std::size_t elems = static_cast<std::size_t>(n) * t;
        DeviceBuffer d_ref(elems * sizeof(std::uint16_t));
        DeviceBuffer d_act(elems * sizeof(std::uint16_t));
        cuda_check(cudaMemcpy(d_ref.p, d_r0.p, d_ref.bytes, cudaMemcpyDeviceToDevice), "cp ref");
        cuda_check(cudaMemcpy(d_act.p, d_r0.p, d_act.bytes, cudaMemcpyDeviceToDevice), "cp act");
        Tensor x(d_x.p, DType::BF16, {k, t});
        Tensor r_ref(d_ref.p, DType::BF16, {n, t});
        Tensor r_act(d_act.p, DType::BF16, {n, t});
        ops::linear_add(x, weights.row_split, r_ref, ops::LinearPolicy::A16Only, ws, nullptr);
        ops::linear_add(x, weights.prepacked, r_act, ops::LinearPolicy::A16Only, ws, nullptr);
        cuda_synchronize();

        const std::string lbl = std::string("linear_add ") + name + " [" + std::to_string(n) + "," +
                                std::to_string(k) + "] T=" + std::to_string(t);
        failures += verify_reduction(lbl, from_device_bf16(d_act, elems),
                                     from_device_bf16(d_ref, elems), kA16Criterion);
    }
    return failures;
}

int test_attn_and_gdn_input() {
    int failures = 0;
    constexpr std::array<int, 7> kTs{1, 8, 16, 32, 48, 64, 128};
    const int max_t = kTs.back();
    DeviceBuffer d_x = make_bf16_input(static_cast<std::size_t>(5120) * max_t, 501U);

    // AttnInputProj: Q4 [7168,5120] + Q5 [7168,5120]
    {
        PrepackedPair qk_w = make_pair(QType::Q4G64_F16S, 7168, 5120, 502U);
        PrepackedPair gv_w = make_pair(QType::Q5G64_F16S, 7168, 5120, 503U);
        const std::size_t ws_cap = ops::q4_q5_attn_input_proj_workspace_capacity_bytes(1, max_t);
        DeviceArena ws(std::max<std::size_t>(ws_cap, 1024));
        for (int t : kTs) {
            const std::size_t qg_elems = static_cast<std::size_t>(6144) * t;
            const std::size_t kv_elems = static_cast<std::size_t>(1024) * t;
            DeviceBuffer q_ref(qg_elems * 2), g_ref(qg_elems * 2), k_ref(kv_elems * 2),
                v_ref(kv_elems * 2);
            DeviceBuffer q_act(qg_elems * 2), g_act(qg_elems * 2), k_act(kv_elems * 2),
                v_act(kv_elems * 2);
            Tensor x(d_x.p, DType::BF16, {5120, t});
            Tensor tq_r(q_ref.p, DType::BF16, {6144, t}), tg_r(g_ref.p, DType::BF16, {6144, t}),
                tk_r(k_ref.p, DType::BF16, {1024, t}), tv_r(v_ref.p, DType::BF16, {1024, t});
            Tensor tq_a(q_act.p, DType::BF16, {6144, t}), tg_a(g_act.p, DType::BF16, {6144, t}),
                tk_a(k_act.p, DType::BF16, {1024, t}), tv_a(v_act.p, DType::BF16, {1024, t});
            ops::attn_input_proj(x, qk_w.row_split, gv_w.row_split, tq_r, tg_r, tk_r, tv_r, ws,
                                 nullptr);
            ops::attn_input_proj(x, qk_w.prepacked, gv_w.prepacked, tq_a, tg_a, tk_a, tv_a, ws,
                                 nullptr);
            cuda_synchronize();

            const std::string pfx = "attn_input_proj T=" + std::to_string(t);
            failures += verify_reduction(pfx + " q", from_device_bf16(q_act, qg_elems),
                                         from_device_bf16(q_ref, qg_elems), kA16Criterion);
            failures += verify_reduction(pfx + " gate", from_device_bf16(g_act, qg_elems),
                                         from_device_bf16(g_ref, qg_elems), kA16Criterion);
            failures += verify_reduction(pfx + " k", from_device_bf16(k_act, kv_elems),
                                         from_device_bf16(k_ref, kv_elems), kA16Criterion);
            failures += verify_reduction(pfx + " v", from_device_bf16(v_act, kv_elems),
                                         from_device_bf16(v_ref, kv_elems), kA16Criterion);
        }
    }

    // GdnInputProj: Q4 [4096,5120] + Q5 [12288,5120]
    {
        PrepackedPair qk_w = make_pair(QType::Q4G64_F16S, 4096, 5120, 504U);
        PrepackedPair vz_w = make_pair(QType::Q5G64_F16S, 12288, 5120, 505U);
        const std::size_t ws_cap = ops::q4_q5_gdn_input_proj_workspace_capacity_bytes(1, max_t);
        DeviceArena ws(std::max<std::size_t>(ws_cap, 1024));
        for (int t : kTs) {
            const std::size_t qkv_elems = static_cast<std::size_t>(10240) * t;
            const std::size_t z_elems   = static_cast<std::size_t>(6144) * t;
            DeviceBuffer qkv_ref(qkv_elems * 2), z_ref(z_elems * 2);
            DeviceBuffer qkv_act(qkv_elems * 2), z_act(z_elems * 2);
            Tensor x(d_x.p, DType::BF16, {5120, t});
            Tensor tqkv_r(qkv_ref.p, DType::BF16, {10240, t}),
                tz_r(z_ref.p, DType::BF16, {6144, t});
            Tensor tqkv_a(qkv_act.p, DType::BF16, {10240, t}),
                tz_a(z_act.p, DType::BF16, {6144, t});
            ops::gdn_input_proj(x, qk_w.row_split, vz_w.row_split, tqkv_r, tz_r, ws, nullptr);
            ops::gdn_input_proj(x, qk_w.prepacked, vz_w.prepacked, tqkv_a, tz_a, ws, nullptr);
            cuda_synchronize();

            const std::string pfx = "gdn_input_proj T=" + std::to_string(t);
            failures += verify_reduction(pfx + " qkv", from_device_bf16(qkv_act, qkv_elems),
                                         from_device_bf16(qkv_ref, qkv_elems), kA16Criterion);
            failures += verify_reduction(pfx + " z", from_device_bf16(z_act, z_elems),
                                         from_device_bf16(z_ref, z_elems), kA16Criterion);
        }
    }
    return failures;
}

int test_linear_q6_head() {
    int failures                    = 0;
    constexpr int n                 = 1152;
    constexpr int k                 = 1536;
    constexpr std::array<int, 9> kTs{4, 8, 12, 16, 24, 32, 48, 64, 96};
    PrepackedPair weights           = make_pair(QType::Q6G64_F16S, n, k, 107U);
    const int max_t                 = kTs.back();
    DeviceBuffer d_x                = make_bf16_input(static_cast<std::size_t>(k) * max_t, 108U);
    const std::size_t ws_cap = ops::linear_workspace_capacity_bytes(
        QType::Q6G64_F16S, n, k, ops::LinearPolicy::A16Only, 4, max_t);
    DeviceArena ws(std::max<std::size_t>(ws_cap, 1024));
    DeviceBuffer d_scratch(static_cast<std::size_t>(max_t) * 2048 * sizeof(std::uint64_t));

    std::vector<double> col0_ref;
    for (int t : kTs) {
        const std::size_t out_elems = static_cast<std::size_t>(n) * t;
        DeviceBuffer d_ref(out_elems * sizeof(std::uint16_t));
        DeviceBuffer d_act(out_elems * sizeof(std::uint16_t));
        Tensor x(d_x.p, DType::BF16, {k, t});
        Tensor out_ref(d_ref.p, DType::BF16, {n, t});
        Tensor out_act(d_act.p, DType::BF16, {n, t});
        ops::linear(x, weights.row_split, out_ref, ops::LinearPolicy::A16Only, ws, nullptr);
        ops::linear(x, weights.prepacked, out_act, ops::LinearPolicy::A16Only, ws, nullptr);
        cuda_synchronize();

        const std::vector<double> act_h = from_device_bf16(d_act, out_elems);
        const std::vector<double> ref_h = from_device_bf16(d_ref, out_elems);
        if (t == 4) { col0_ref.assign(ref_h.begin(), ref_h.begin() + n); }
        const std::string lbl =
            "linear Q6 [1152,1536] T=" + std::to_string(t);
        failures += verify_reduction(lbl, act_h, ref_h, kA16Criterion);

        // Verify fused Q6 greedy argmax for T = 4, 8, 16, 32, 64.
        if (t == 4 || t == 8 || t == 16 || t == 32 || t == 64) {
            DeviceBuffer d_ids(static_cast<std::size_t>(t) * sizeof(std::int32_t));
            Tensor out_ids(d_ids.p, DType::I32, {t});
            const half* x_fp16 = ops::detail::volta_stage_fp16_activation(x, nullptr);
            if (x_fp16 == nullptr) {
                std::cerr << "volta_stage_fp16_activation returned nullptr at T=" << t << "\n";
                ++failures;
                continue;
            }
            const int valid_rows = n - 17;
            ops::detail::launch_q6_prepacked_volta_argmax_fp16(x, x_fp16, weights.prepacked,
                                                               out_ids, valid_rows, d_scratch.p,
                                                               nullptr);
            cuda_synchronize();
            const std::vector<std::int32_t> ids = from_device<std::int32_t>(d_ids, t);
            for (int col = 0; col < t; ++col) {
                int best_idx    = 0;
                double best_val = -std::numeric_limits<double>::infinity();
                for (int r = 0; r < valid_rows; ++r) {
                    const double v = act_h[static_cast<std::size_t>(col) * n + r];
                    if (v > best_val) {
                        best_val = v;
                        best_idx = r;
                    }
                }
                const double chosen_val =
                    (ids[col] >= 0 && ids[col] < valid_rows)
                        ? act_h[static_cast<std::size_t>(col) * n + ids[col]]
                        : -std::numeric_limits<double>::infinity();
                if (ids[col] < 0 || ids[col] >= valid_rows ||
                    std::abs(chosen_val - best_val) > std::max(1e-3, std::abs(best_val) * 0.01)) {
                    std::cerr << "q6_argmax T=" << t << " col=" << col << " got id=" << ids[col]
                              << " (" << chosen_val << ") expected id=" << best_idx << " ("
                              << best_val << ")\n";
                    ++failures;
                }
            }
        }
    }

    // Also verify Q6 direct launch + argmax at T = 1 against col 0 of T = 4 RowSplit ref.
    {
        DeviceBuffer d_act(static_cast<std::size_t>(n) * sizeof(std::uint16_t));
        DeviceBuffer d_ids(sizeof(std::int32_t));
        Tensor x1(d_x.p, DType::BF16, {k, 1});
        Tensor out1(d_act.p, DType::BF16, {n, 1});
        Tensor ids1(d_ids.p, DType::I32, {1});
        ops::detail::launch_q6_prepacked_volta(x1, weights.prepacked, out1, nullptr);
        const half* x_fp16 = ops::detail::volta_stage_fp16_activation(x1, nullptr);
        const int valid_rows = n - 17;
        ops::detail::launch_q6_prepacked_volta_argmax_fp16(x1, x_fp16, weights.prepacked, ids1,
                                                           valid_rows, d_scratch.p, nullptr);
        cuda_synchronize();
        const std::vector<double> act1 = from_device_bf16(d_act, n);
        failures += verify_reduction("linear Q6 [1152,1536] T=1", act1, col0_ref, kA16Criterion);
        const std::vector<std::int32_t> id = from_device<std::int32_t>(d_ids, 1);
        double best_val = -std::numeric_limits<double>::infinity();
        for (int r = 0; r < valid_rows; ++r) { best_val = std::max(best_val, act1[r]); }
        if (id[0] < 0 || id[0] >= valid_rows ||
            std::abs(act1[id[0]] - best_val) > std::max(1e-3, std::abs(best_val) * 0.01)) {
            std::cerr << "q6_argmax T=1 mismatch: got id=" << id[0] << "\n";
            ++failures;
        }
    }
    return failures;
}

#endif // NINFER_VOLTA_BUILD

} // namespace

int main() {
#ifndef NINFER_VOLTA_BUILD
    std::cout << "SKIP: VoltaQpnPrepacked test requires NINFER_VOLTA_BUILD\n";
    return 77;
#else
    if (!ninfer::test::linear::cuda_available()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    try {
        int failures = 0;
        failures += test_linear_qtype("Q4", QType::Q4G64_F16S, 4096, 5120, 101U);
        failures += test_linear_qtype("Q5", QType::Q5G64_F16S, 5120, 6144, 103U);
        failures += test_linear_q6_head();
        failures += test_swiglu_q4();
        failures += test_linear_add("Q4", QType::Q4G64_F16S, 5120, 6144, 401U);
        failures += test_linear_add("Q5", QType::Q5G64_F16S, 5120, 6144, 403U);
        failures += test_linear_add("Q5", QType::Q5G64_F16S, 5120, 17408, 405U);
        failures += test_attn_and_gdn_input();
        std::cout << (failures == 0 ? "OK" : "FAIL") << " VoltaQpnPrepacked Q4/Q5/Q6\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "VoltaQpnPrepacked test failed: " << error.what() << '\n';
        return 1;
    }
#endif
}
