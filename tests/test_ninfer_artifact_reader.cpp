#include "artifact/reader.h"
#include "artifact_fixture.h"

#include <nlohmann/json.hpp>

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using ninfer::artifact::NumericFormat;
using ninfer::artifact::ObjectDescriptor;
using ninfer::artifact::Reader;
using ninfer::artifact::ResourceDescriptor;
using ninfer::artifact::StorageLayout;
using ninfer::artifact::TensorDescriptor;
using Json = nlohmann::json;
using ninfer::test::artifact_fixture::write_fixture;

Json normative_directory() {
    return {
        {"identity", {{"model_id", "fixture-model"}, {"weights_id", "fixture-weights"}}},
        {"objects", Json::array({
                        {{"name", "resource"},
                         {"kind", "resource"},
                         {"encoding", "raw-bytes-v1"},
                         {"offset", 0},
                         {"bytes", 3}},
                        {{"name", "bf16"},
                         {"kind", "tensor"},
                         {"shape", {2, 3}},
                         {"format", "BF16"},
                         {"layout", "contiguous-le-v1"},
                         {"offset", 256},
                         {"bytes", 12}},
                        {{"name", "fp32_scalar"},
                         {"kind", "tensor"},
                         {"shape", Json::array()},
                         {"format", "FP32"},
                         {"layout", "contiguous-le-v1"},
                         {"offset", 512},
                         {"bytes", 4}},
                        {{"name", "i32"},
                         {"kind", "tensor"},
                         {"shape", {2}},
                         {"format", "I32"},
                         {"layout", "contiguous-le-v1"},
                         {"offset", 768},
                         {"bytes", 8}},
                        {{"name", "q4"},
                         {"kind", "tensor"},
                         {"shape", {1, 1}},
                         {"format", "Q4G64_F16S"},
                         {"layout", "row-split-k128-v1"},
                         {"offset", 1024},
                         {"bytes", 260}},
                        {{"name", "q5"},
                         {"kind", "tensor"},
                         {"shape", {2, 130}},
                         {"format", "Q5G64_F16S"},
                         {"layout", "row-split-k128-v1"},
                         {"offset", 1536},
                         {"bytes", 528}},
                        {{"name", "q6"},
                         {"kind", "tensor"},
                         {"shape", {1, 64}},
                         {"format", "Q6G64_F16S"},
                         {"layout", "row-split-k128-v1"},
                         {"offset", 2304},
                         {"bytes", 516}},
                        {{"name", "w8"},
                         {"kind", "tensor"},
                         {"shape", {1, 33}},
                         {"format", "W8G32_F16S"},
                         {"layout", "row-split-k128-v1"},
                         {"offset", 3072},
                         {"bytes", 264}},
                        {{"name", "fp8_row"},
                         {"kind", "tensor"},
                         {"shape", {2, 4}},
                         {"format", "FP8_E4M3FN_ROW_BF16S"},
                         {"layout", "row-scale-v1"},
                         {"offset", 3584},
                         {"bytes", 260}},
                    })},
    };
}

template <typename Function>
void expect_artifact_error(Function&& function, std::string_view label) {
    try {
        function();
    } catch (const ninfer::artifact::ArtifactError&) { return; }
    throw std::runtime_error(std::string(label) + " was accepted");
}

void test_registered_sizes() {
    using ninfer::artifact::tensor_encoded_size;
    constexpr StorageLayout direct   = StorageLayout::ContiguousLeV1;
    constexpr StorageLayout rows     = StorageLayout::RowSplitK128V1;
    constexpr StorageLayout fp8_rows = StorageLayout::RowScaleV1;

    const std::array<std::uint64_t, 2> shape_2x3 = {2, 3};
    const std::array<std::uint64_t, 1> shape_2   = {2};
    const std::array<std::uint64_t, 2> q4_shape  = {1, 1};
    const std::array<std::uint64_t, 2> q5_shape  = {2, 130};
    const std::array<std::uint64_t, 2> q6_shape  = {1, 64};
    const std::array<std::uint64_t, 2> w8_shape  = {1, 33};
    const std::array<std::uint64_t, 2> fp8_shape = {2, 4};

    if (tensor_encoded_size(direct, NumericFormat::BF16, shape_2x3) != 12 ||
        tensor_encoded_size(direct, NumericFormat::FP32, {}) != 4 ||
        tensor_encoded_size(direct, NumericFormat::I32, shape_2) != 8 ||
        tensor_encoded_size(rows, NumericFormat::Q4G64_F16S, q4_shape) != 260 ||
        tensor_encoded_size(rows, NumericFormat::Q5G64_F16S, q5_shape) != 528 ||
        tensor_encoded_size(rows, NumericFormat::Q6G64_F16S, q6_shape) != 516 ||
        tensor_encoded_size(rows, NumericFormat::W8G32_F16S, w8_shape) != 264 ||
        tensor_encoded_size(fp8_rows, NumericFormat::FP8_E4M3FN_ROW_BF16S, fp8_shape) != 260) {
        throw std::runtime_error("registered encoded-size calculation is wrong");
    }
    expect_artifact_error([&] { tensor_encoded_size(fp8_rows, NumericFormat::NVFP4, fp8_shape); },
                          "row-scale format mismatch");
    expect_artifact_error(
        [&] { tensor_encoded_size(fp8_rows, NumericFormat::FP8_E4M3FN_ROW_BF16S, shape_2); },
        "row-scale rank mismatch");
}

void test_normative_fixture() {
    auto fixture = write_fixture(normative_directory(), "valid");
    Reader reader(fixture.path);
    if (reader.identity().model_id != "fixture-model" ||
        reader.identity().weights_id != "fixture-weights" || reader.objects().size() != 9 ||
        reader.payload_offset() != 4096) {
        throw std::runtime_error("fixture root descriptor mismatch");
    }

    const std::array<std::string_view, 9> expected_names = {
        "resource", "bf16", "fp32_scalar", "i32", "q4", "q5", "q6", "w8", "fp8_row",
    };
    for (std::size_t i = 0; i < expected_names.size(); ++i) {
        const auto& object = reader.objects()[i];
        if (ninfer::artifact::object_name(object) != expected_names[i] ||
            reader.find(expected_names[i]) != &object) {
            throw std::runtime_error("fixture name index mismatch");
        }
        const auto payload = reader.payload(object);
        if (payload.absolute_offset !=
                reader.payload_offset() + ninfer::artifact::object_offset(object) ||
            payload.data.size() != ninfer::artifact::object_bytes(object) ||
            payload.data.front() != std::byte(i + 1) || payload.data.back() != std::byte(i + 1)) {
            throw std::runtime_error("fixture payload span mismatch");
        }
    }
    if (reader.find("missing") != nullptr) {
        throw std::runtime_error("missing object unexpectedly resolved");
    }

    const auto* resource = std::get_if<ResourceDescriptor>(&reader.objects().front());
    const auto* q5       = std::get_if<TensorDescriptor>(reader.find("q5"));
    const auto* fp8      = std::get_if<TensorDescriptor>(reader.find("fp8_row"));
    if (resource == nullptr || q5 == nullptr || q5->shape != std::vector<std::uint64_t>({2, 130}) ||
        q5->format != NumericFormat::Q5G64_F16S || q5->layout != StorageLayout::RowSplitK128V1 ||
        fp8 == nullptr || fp8->shape != std::vector<std::uint64_t>({2, 4}) ||
        fp8->format != NumericFormat::FP8_E4M3FN_ROW_BF16S ||
        fp8->layout != StorageLayout::RowScaleV1) {
        throw std::runtime_error("fixture object signature mismatch");
    }
}

void test_common_validation() {
    {
        auto directory                   = normative_directory();
        directory["objects"][5]["bytes"] = 527;
        auto fixture                     = write_fixture(directory, "wrong_encoded_size");
        expect_artifact_error([&] { Reader reader(fixture.path); }, "wrong encoded size");
    }
    {
        auto directory                    = normative_directory();
        directory["objects"][1]["offset"] = 257;
        auto fixture                      = write_fixture(directory, "misaligned_offset");
        expect_artifact_error([&] { Reader reader(fixture.path); }, "misaligned offset");
    }
    {
        constexpr std::array<std::uint8_t, 8> invalid_magic = {
            'I', 'N', 'V', 'A', 'L', 'I', 'D', '!',
        };
        auto fixture = write_fixture(normative_directory(), "invalid_magic", invalid_magic);
        expect_artifact_error([&] { Reader reader(fixture.path); }, "invalid magic");
    }
}

void test_v3_fixture() {
    const Json directory = {
        {"metadata", {{"name", "qwen3.8-27b"}}},
        {"files", Json::array({{{"path", nullptr}, {"payload_bytes", 1024}}})},
        {"components",
         {{"text", {{"resources", {{"tokenizer.json", "resource/text/tokenizer.json"}}}}}}},
        {"objects",
         Json::array({
             {{"id", "resource/text/tokenizer.json"},
              {"kind", "resource"},
              {"encoding", "raw_bytes_v1"},
              {"offset", 0},
              {"bytes", 4}},
             {{"id", "weight/0"},
              {"kind", "tensor"},
              {"shape", {2, 128}},
              {"format", "q4_g64_fp16"},
              {"layout", "row_split_k128_v1"},
              {"offset", 256},
              {"bytes", 264}},
             {{"id", "weight/1"},
              {"kind", "tensor"},
              {"shape", {4}},
              {"format", "bf16"},
              {"layout", "contiguous_le_v1"},
              {"offset", 768},
              {"bytes", 8}},
         })},
        {"bindings",
         {
             {"text/layers/0/mlp/gate",
              {{"parts", Json::array({{{"object", "weight/0"}, {"range", {0, 128}}}})}}},
             {"text/layers/0/mlp/up",
              {{"parts", Json::array({{{"object", "weight/0"}, {"range", {128, 256}}}})}}},
             {"proposal/head", {{"object", "weight/1"}}},
         }},
    };

    const std::string json_text = directory.dump();
    const std::uint64_t jlen    = json_text.size();
    const std::size_t payload_start = ((32 + jlen + 4095) / 4096) * 4096;
    std::vector<std::uint8_t> image(payload_start + 1024, 0);
    const std::array<std::uint8_t, 8> magic = {'N', 'I', 'N', 'F', 'E', 'R', 0, 3};
    std::copy(magic.begin(), magic.end(), image.begin());
    for (int i = 0; i < 8; ++i) { image[8 + i] = static_cast<std::uint8_t>((jlen >> (8 * i)) & 0xffU); }
    std::copy(json_text.begin(), json_text.end(), image.begin() + 32);
    for (std::size_t i = 0; i < 4; ++i) { image[payload_start + i] = 0x11; }
    for (std::size_t i = 0; i < 264; ++i) { image[payload_start + 256 + i] = 0x22; }
    for (std::size_t i = 0; i < 8; ++i) { image[payload_start + 768 + i] = 0x33; }

    const auto path = std::filesystem::temp_directory_path() / "ninfer_test_v3_fixture.ninfer";
    {
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        out.write(reinterpret_cast<const char*>(image.data()), static_cast<std::streamsize>(image.size()));
    }

    Reader reader(path);
    std::filesystem::remove(path);
    if (reader.identity().model_id != "qwen3.8-27b" ||
        reader.identity().weights_id != "groupwise-int" || reader.objects().size() != 3) {
        throw std::runtime_error("v3 fixture identity or object count mismatch");
    }
    if (reader.find("frontend/tokenizer.json") == nullptr ||
        reader.find("text/layers/0/mlp/gate_up") == nullptr ||
        reader.find("text/draft_head") == nullptr) {
        throw std::runtime_error("v3 fixture name remapping mismatch");
    }
    const auto* gate_up = std::get_if<TensorDescriptor>(reader.find("text/layers/0/mlp/gate_up"));
    if (gate_up == nullptr || gate_up->format != NumericFormat::Q4G64_F16S ||
        gate_up->layout != StorageLayout::RowSplitK128V1) {
        throw std::runtime_error("v3 fixture tensor descriptor mismatch");
    }
}

void test_v3_nvfp4_qad_fixture() {
    // Minimal v3 NVFP4-QAD fixture verifying metadata normalization, mixed NVFP4/FP8 detection,
    // and uses[].auxiliaries.activation_input_divisor mapping for output_head, MLP, and GDN.
    // NVFP4 tensor shape [128, 64]: codes=4096B (aligned to 256 -> 4096), scales=512B, divisor=4B => 4612B.
    const Json directory = {
        {"metadata", {{"name", "qwen3.8-27b-nvfp4-qad"}}},
        {"files", Json::array({{{"path", nullptr}}})},
        {"components",
         {
             {"text", {{"resources", {{"tokenizer.json", "res/tok"}}}}},
         }},
        {"objects",
         Json::array({
             {{"id", "res/tok"},
              {"kind", "resource"},
              {"encoding", "raw_bytes_v1"},
              {"offset", 0},
              {"bytes", 4}},
             {{"id", "weight/head"},
              {"kind", "tensor"},
              {"shape", {128, 64}},
              {"format", "nvfp4"},
              {"layout", "block_scale_k16_m128x4_v1"},
              {"offset", 256},
              {"bytes", 4612}},
             {{"id", "aux/head_text"},
              {"kind", "tensor"},
              {"shape", Json::array()},
              {"format", "fp32"},
              {"layout", "contiguous_le_v1"},
              {"offset", 5120},
              {"bytes", 4}},
             {{"id", "aux/head_mtp"},
              {"kind", "tensor"},
              {"shape", Json::array()},
              {"format", "fp32"},
              {"layout", "contiguous_le_v1"},
              {"offset", 5376},
              {"bytes", 4}},
             {{"id", "aux/gdn_q"},
              {"kind", "tensor"},
              {"shape", Json::array()},
              {"format", "fp32"},
              {"layout", "contiguous_le_v1"},
              {"offset", 5632},
              {"bytes", 4}},
             {{"id", "aux/gdn_out"},
              {"kind", "tensor"},
              {"shape", Json::array()},
              {"format", "fp32"},
              {"layout", "contiguous_le_v1"},
              {"offset", 5888},
              {"bytes", 4}},
             {{"id", "aux/mlp_gate"},
              {"kind", "tensor"},
              {"shape", Json::array()},
              {"format", "fp32"},
              {"layout", "contiguous_le_v1"},
              {"offset", 6144},
              {"bytes", 4}},
         })},
        {"bindings",
         {
             {"text/output_head", {{"object", "weight/head"}}},
         }},
        {"uses",
         Json::array({
             {{"parameter", "text/output_head"},
              {"input", "text/final_hidden"},
              {"activation_policy", "AllowA4"},
              {"auxiliaries", {{"activation_input_divisor", {{"object", "aux/head_text"}}}}}},
             {{"parameter", "text/output_head"},
              {"input", "mtp/final_hidden"},
              {"activation_policy", "AllowA4"},
              {"auxiliaries", {{"activation_input_divisor", {{"object", "aux/head_mtp"}}}}}},
             {{"parameter", "text/layers/6/gdn/query"},
              {"input", "text/layers/6/mixer_input"},
              {"activation_policy", "AllowA4"},
              {"auxiliaries", {{"activation_input_divisor", {{"object", "aux/gdn_q"}}}}}},
             {{"parameter", "text/layers/6/gdn/output"},
              {"input", "text/layers/6/gdn/gated_output"},
              {"activation_policy", "AllowA4"},
              {"auxiliaries", {{"activation_input_divisor", {{"object", "aux/gdn_out"}}}}}},
             {{"parameter", "text/layers/0/mlp/gate"},
              {"input", "text/layers/0/ffn_input"},
              {"activation_policy", "AllowA4"},
              {"auxiliaries", {{"activation_input_divisor", {{"object", "aux/mlp_gate"}}}}}},
         })},
    };

    const std::string json_text     = directory.dump();
    const std::uint64_t jlen        = json_text.size();
    const std::size_t payload_start = ((32 + jlen + 4095) / 4096) * 4096;
    std::vector<std::uint8_t> image(payload_start + 8192, 0);
    const std::array<std::uint8_t, 8> magic = {'N', 'I', 'N', 'F', 'E', 'R', 0, 3};
    std::copy(magic.begin(), magic.end(), image.begin());
    for (int i = 0; i < 8; ++i) { image[8 + i] = static_cast<std::uint8_t>((jlen >> (8 * i)) & 0xffU); }
    std::copy(json_text.begin(), json_text.end(), image.begin() + 32);

    const auto path = std::filesystem::temp_directory_path() / "ninfer_test_v3_nvfp4_qad.ninfer";
    {
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        out.write(reinterpret_cast<const char*>(image.data()), static_cast<std::streamsize>(image.size()));
    }
    Reader reader(path);
    std::filesystem::remove(path);
    if (reader.identity().model_id != "qwen3.8-27b" || reader.identity().weights_id != "nvfp4" ||
        reader.objects().size() != 6) {
        throw std::runtime_error("v3 NVFP4-QAD fixture identity or object count mismatch");
    }
    const std::array<const char*, 5> required = {
        "text/output_head",
        "text/output_head/input_scale_divisor",
        "text/layers/6/gdn/input_projection/input_scale_divisor",
        "text/layers/6/gdn/output_projection/input_scale_divisor",
        "text/layers/0/mlp/gate_up_projection/input_scale_divisor",
    };
    for (const char* name : required) {
        if (reader.find(name) == nullptr) {
            throw std::runtime_error(std::string("v3 NVFP4-QAD fixture missing ") + name);
        }
    }
    if (reader.find("mtp/output_head/input_scale_divisor") != nullptr) {
        throw std::runtime_error("v3 NVFP4-QAD fixture should deduplicate mtp/output_head divisor");
    }
}

} // namespace

int main() {
    try {
        test_registered_sizes();
        test_normative_fixture();
        test_common_validation();
        test_v3_fixture();
        test_v3_nvfp4_qad_fixture();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
