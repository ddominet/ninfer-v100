#include "artifact/reader.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstring>
#include <functional>
#include <initializer_list>
#include <limits>
#include <span>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <unordered_map>
#include <utility>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace ninfer::artifact {
namespace {

using Json = nlohmann::json;

constexpr std::array<std::byte, 8> kMagicV2 = {
    std::byte{'N'}, std::byte{'I'}, std::byte{'N'}, std::byte{'F'},
    std::byte{'E'}, std::byte{'R'}, std::byte{0},   std::byte{2},
};
constexpr std::array<std::byte, 8> kMagicV3 = {
    std::byte{'N'}, std::byte{'I'}, std::byte{'N'}, std::byte{'F'},
    std::byte{'E'}, std::byte{'R'}, std::byte{0},   std::byte{3},
};
constexpr std::uint64_t kPrefixBytesV2    = 16;
constexpr std::uint64_t kPrefixBytesV3    = 32;
constexpr std::uint64_t kPayloadAlignment = 4096;

std::uint64_t checked_add(std::uint64_t a, std::uint64_t b, std::string_view label) {
    if (b > std::numeric_limits<std::uint64_t>::max() - a) {
        throw ArtifactError(std::string(label) + " overflows u64");
    }
    return a + b;
}

std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment, std::string_view label) {
    const auto biased = checked_add(value, alignment - 1, label);
    return biased / alignment * alignment;
}

std::uint64_t read_u64_le(const std::byte* data) noexcept {
    std::uint64_t value = 0;
    for (unsigned i = 0; i < 8; ++i) {
        value |= std::uint64_t(std::to_integer<unsigned char>(data[i])) << (i * 8);
    }
    return value;
}

template <std::size_t N>
void require_members(const Json& value, const std::array<const char*, N>& members,
                     std::string_view label) {
    if (!value.is_object() || value.size() != N) {
        throw ArtifactError(std::string(label) + " has missing or extra members");
    }
    for (const char* member : members) {
        if (!value.contains(member)) {
            throw ArtifactError(std::string(label) + " has missing or extra members");
        }
    }
}

const std::string& require_string(const Json& value, std::string_view label) {
    if (!value.is_string()) {
        throw ArtifactError(std::string(label) + " must be a nonempty string");
    }
    const auto& result = value.get_ref<const std::string&>();
    if (result.empty()) { throw ArtifactError(std::string(label) + " must be a nonempty string"); }
    return result;
}

std::uint64_t require_unsigned(const Json& value, std::string_view label, bool positive) {
    if (!value.is_number_unsigned()) {
        throw ArtifactError(std::string(label) + " must be an integer");
    }
    const auto result = value.get<std::uint64_t>();
    if (positive && result == 0) { throw ArtifactError(std::string(label) + " must be positive"); }
    return result;
}

NumericFormat parse_format(std::string_view name) {
    if (name == "BF16") { return NumericFormat::BF16; }
    if (name == "FP32") { return NumericFormat::FP32; }
    if (name == "I32") { return NumericFormat::I32; }
    if (name == "Q4G64_F16S") { return NumericFormat::Q4G64_F16S; }
    if (name == "Q5G64_F16S") { return NumericFormat::Q5G64_F16S; }
    if (name == "Q6G64_F16S") { return NumericFormat::Q6G64_F16S; }
    if (name == "W8G32_F16S") { return NumericFormat::W8G32_F16S; }
    if (name == "NVFP4") { return NumericFormat::NVFP4; }
    if (name == "FP8_E4M3FN_ROW_BF16S") { return NumericFormat::FP8_E4M3FN_ROW_BF16S; }
    throw ArtifactError("unknown tensor format: " + std::string(name));
}

StorageLayout parse_layout(std::string_view name) {
    if (name == "contiguous-le-v1") { return StorageLayout::ContiguousLeV1; }
    if (name == "row-split-k128-v1") { return StorageLayout::RowSplitK128V1; }
    if (name == "blockscale-k16-m128x4-v1") { return StorageLayout::BlockScaleK16M128x4V1; }
    if (name == "row-scale-v1") { return StorageLayout::RowScaleV1; }
    throw ArtifactError("unknown tensor layout: " + std::string(name));
}

ResourceEncoding parse_encoding(std::string_view name) {
    if (name == "raw-bytes-v1") { return ResourceEncoding::RawBytesV1; }
    throw ArtifactError("unknown resource encoding: " + std::string(name));
}

NumericFormat parse_v3_format(std::string_view name) {
    if (name == "bf16") { return NumericFormat::BF16; }
    if (name == "fp32") { return NumericFormat::FP32; }
    if (name == "int32" || name == "i32") { return NumericFormat::I32; }
    if (name == "q4_g64_fp16") { return NumericFormat::Q4G64_F16S; }
    if (name == "q5_g64_fp16") { return NumericFormat::Q5G64_F16S; }
    if (name == "q6_g64_fp16") { return NumericFormat::Q6G64_F16S; }
    if (name == "q8_g32_fp16" || name == "w8_g32_fp16") { return NumericFormat::W8G32_F16S; }
    if (name == "nvfp4") { return NumericFormat::NVFP4; }
    if (name == "fp8_e4m3fn_row_bf16") { return NumericFormat::FP8_E4M3FN_ROW_BF16S; }
    return parse_format(name);
}

StorageLayout parse_v3_layout(std::string_view name) {
    if (name == "contiguous_le_v1") { return StorageLayout::ContiguousLeV1; }
    if (name == "row_split_k128_v1") { return StorageLayout::RowSplitK128V1; }
    if (name == "block_scale_k16_m128x4_v1" || name == "blockscale_k16_m128x4_v1") {
        return StorageLayout::BlockScaleK16M128x4V1;
    }
    if (name == "row_scale_v1") { return StorageLayout::RowScaleV1; }
    return parse_layout(name);
}

ResourceEncoding parse_v3_encoding(std::string_view name) {
    if (name == "raw_bytes_v1") { return ResourceEncoding::RawBytesV1; }
    return parse_encoding(name);
}

std::string remap_v3_single_name(std::string_view name) {
    if (name == "proposal/head") { return "text/draft_head"; }
    if (name == "proposal/token_ids") { return "text/draft_head_token_ids"; }
    if (name == "vision/merger/norm_weight") { return "vision/merger/norm/weight"; }
    if (name == "vision/merger/norm_bias") { return "vision/merger/norm/bias"; }
    if (name.starts_with("mtp/layers/0/")) {
        return "mtp/layer/" + std::string(name.substr(std::string_view("mtp/layers/0/").size()));
    }
    if (name.starts_with("vision/layers/")) {
        static constexpr std::array<std::pair<std::string_view, std::string_view>, 4> kNorms = {{
            {"/norm1_weight", "/norm1/weight"},
            {"/norm1_bias", "/norm1/bias"},
            {"/norm2_weight", "/norm2/weight"},
            {"/norm2_bias", "/norm2/bias"},
        }};
        for (const auto& [from, to] : kNorms) {
            if (name.ends_with(from)) {
                return std::string(name.substr(0, name.size() - from.size())) + std::string(to);
            }
        }
    }
    return std::string(name);
}

std::string remap_v3_fused_name(std::vector<std::pair<std::string, std::uint64_t>> parts) {
    std::sort(parts.begin(), parts.end(), [](const auto& lhs, const auto& rhs) {
        if (lhs.second != rhs.second) { return lhs.second < rhs.second; }
        return lhs.first < rhs.first;
    });
    const std::size_t slash = parts.front().first.rfind('/');
    if (slash == std::string::npos) {
        throw ArtifactError("v3 fused binding has no namespace prefix: " + parts.front().first);
    }
    const std::string_view prefix(parts.front().first.data(), slash + 1);
    std::vector<std::string_view> leaves;
    leaves.reserve(parts.size());
    for (const auto& [full_name, _] : parts) {
        if (!std::string_view(full_name).starts_with(prefix)) {
            throw ArtifactError("v3 fused binding parts disagree on prefix: " + full_name);
        }
        leaves.push_back(std::string_view(full_name).substr(prefix.size()));
    }

    const auto matches_ordered = [&](std::initializer_list<std::string_view> expected) {
        return leaves.size() == expected.size() &&
               std::equal(leaves.begin(), leaves.end(), expected.begin());
    };

    std::string suffix;
    if (matches_ordered({"query", "key"})) {
        suffix = "query_key";
    } else if (matches_ordered({"value", "z"})) {
        suffix = "value_z";
    } else if (matches_ordered({"gate", "value"})) {
        suffix = "gate_value";
    } else if (matches_ordered({"gate", "up"})) {
        suffix = "gate_up";
    } else if (matches_ordered({"a_projection", "b_projection"})) {
        suffix = "a_b_projection";
    } else if (matches_ordered({"query", "key", "value"})) {
        suffix = "qkv";
    } else if (matches_ordered({"query_bias", "key_bias", "value_bias"})) {
        suffix = "qkv_bias";
    } else if (matches_ordered({"query", "key", "gate", "value"})) {
        suffix = "query_key_gate_value";
    } else if (matches_ordered({"query", "key", "value", "z"})) {
        suffix = "query_key_value_z";
    } else if (leaves.size() == 5) {
        std::vector<std::string_view> sorted = leaves;
        std::sort(sorted.begin(), sorted.end());
        static constexpr std::array<std::string_view, 5> kDFlashQkv = {
            "context_key", "context_value", "key", "query", "value",
        };
        if (std::equal(sorted.begin(), sorted.end(), kDFlashQkv.begin())) {
            suffix = "query_key_value";
        }
    }
    if (suffix.empty()) {
        throw ArtifactError("unrecognized v3 fused binding group under " + std::string(prefix));
    }
    return remap_v3_single_name(std::string(prefix) + suffix);
}

TensorDescriptor parse_v3_tensor(const Json& value, std::string logical_name) {
    const auto format = parse_v3_format(require_string(value.at("format"), "v3 tensor format"));
    const auto layout = parse_v3_layout(require_string(value.at("layout"), "v3 tensor layout"));
    const auto offset = require_unsigned(value.at("offset"), "v3 tensor offset", false);
    const auto stored_size = require_unsigned(value.at("bytes"), "v3 tensor bytes", true);

    const auto& raw_shape = value.at("shape");
    if (!raw_shape.is_array()) { throw ArtifactError("v3 tensor shape must be an array"); }
    std::vector<std::uint64_t> shape;
    shape.reserve(raw_shape.size());
    for (const auto& dim : raw_shape) {
        shape.push_back(require_unsigned(dim, "v3 shape dimension", true));
    }

    const auto expected_size = tensor_encoded_size(layout, format, shape);
    if (stored_size != expected_size) {
        throw ArtifactError("v3 tensor " + logical_name + " stores " + std::to_string(stored_size) +
                            " bytes; layout requires " + std::to_string(expected_size));
    }
    return {std::move(logical_name), std::move(shape), format, layout, offset, stored_size};
}

ResourceDescriptor parse_v3_resource(const Json& value, std::string logical_name) {
    return {
        std::move(logical_name),
        parse_v3_encoding(require_string(value.at("encoding"), "v3 resource encoding")),
        require_unsigned(value.at("offset"), "v3 resource offset", false),
        require_unsigned(value.at("bytes"), "v3 resource bytes", true),
    };
}

TensorDescriptor parse_tensor(const Json& value) {
    static constexpr std::array members = {
        "name", "kind", "shape", "format", "layout", "offset", "bytes",
    };
    require_members(value, members, "tensor entry");

    const auto name        = require_string(value.at("name"), "tensor name");
    const auto format      = parse_format(require_string(value.at("format"), "tensor format"));
    const auto layout      = parse_layout(require_string(value.at("layout"), "tensor layout"));
    const auto offset      = require_unsigned(value.at("offset"), "tensor offset", false);
    const auto stored_size = require_unsigned(value.at("bytes"), "tensor bytes", true);

    const auto& raw_shape = value.at("shape");
    if (!raw_shape.is_array()) { throw ArtifactError("tensor shape must be an array"); }
    std::vector<std::uint64_t> shape;
    shape.reserve(raw_shape.size());
    for (const auto& dim : raw_shape) {
        shape.push_back(require_unsigned(dim, "shape dimension", true));
    }

    const auto expected_size = tensor_encoded_size(layout, format, shape);
    if (stored_size != expected_size) {
        throw ArtifactError("tensor " + name + " stores " + std::to_string(stored_size) +
                            " bytes; layout requires " + std::to_string(expected_size));
    }
    return {name, std::move(shape), format, layout, offset, stored_size};
}

ResourceDescriptor parse_resource(const Json& value) {
    static constexpr std::array members = {
        "name", "kind", "encoding", "offset", "bytes",
    };
    require_members(value, members, "resource entry");
    return {
        require_string(value.at("name"), "resource name"),
        parse_encoding(require_string(value.at("encoding"), "resource encoding")),
        require_unsigned(value.at("offset"), "resource offset", false),
        require_unsigned(value.at("bytes"), "resource bytes", true),
    };
}

ObjectDescriptor parse_object(const Json& value) {
    if (!value.is_object()) { throw ArtifactError("each object entry must be a JSON object"); }
    const auto it = value.find("kind");
    if (it == value.end() || !it->is_string()) {
        throw ArtifactError("object kind must be 'tensor' or 'resource'");
    }
    const auto& kind = it->get_ref<const std::string&>();
    if (kind == "tensor") { return parse_tensor(value); }
    if (kind == "resource") { return parse_resource(value); }
    throw ArtifactError("object kind must be 'tensor' or 'resource'");
}

struct TransparentStringHash {
    using is_transparent = void;

    std::size_t operator()(std::string_view value) const noexcept {
        return std::hash<std::string_view>{}(value);
    }

    std::size_t operator()(const std::string& value) const noexcept {
        return (*this)(std::string_view(value));
    }
};

class MappedFile {
public:
    explicit MappedFile(const std::filesystem::path& path) {
        const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT);
        if (fd < 0) {
            throw std::system_error(errno, std::generic_category(), "open " + path.string());
        }

        struct stat status {};

        if (::fstat(fd, &status) != 0) {
            const int error = errno;
            ::close(fd);
            throw std::system_error(error, std::generic_category(), "fstat " + path.string());
        }
        if (status.st_size < 0 ||
            static_cast<std::uintmax_t>(status.st_size) > std::numeric_limits<std::size_t>::max()) {
            ::close(fd);
            throw ArtifactError("artifact size does not fit the process address space");
        }

        const auto size = static_cast<std::size_t>(status.st_size);
        void* mapping   = nullptr;
        if (size != 0) {
            mapping = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
            if (mapping == MAP_FAILED) {
                const int error = errno;
                ::close(fd);
                throw std::system_error(error, std::generic_category(), "mmap " + path.string());
            }
        }
        fd_   = fd;
        data_ = static_cast<const std::byte*>(mapping);
        size_ = size;
    }

    ~MappedFile() {
        if (data_ != nullptr) { ::munmap(const_cast<std::byte*>(data_), size_); }
        if (fd_ >= 0) { ::close(fd_); }
    }

    MappedFile(const MappedFile&)            = delete;
    MappedFile& operator=(const MappedFile&) = delete;

    const std::byte* data() const noexcept { return data_; }

    std::size_t size() const noexcept { return size_; }

    std::size_t read_direct(std::uint64_t absolute_offset, std::span<std::byte> destination) const {
        constexpr std::size_t alignment = Reader::direct_io_alignment;
        if (absolute_offset % alignment != 0 || destination.size() % alignment != 0 ||
            reinterpret_cast<std::uintptr_t>(destination.data()) % alignment != 0) {
            throw ArtifactError("direct artifact read is not 4096-byte aligned");
        }
        if (absolute_offset > static_cast<std::uint64_t>(std::numeric_limits<off_t>::max()) ||
            destination.size() > static_cast<std::size_t>(std::numeric_limits<ssize_t>::max())) {
            throw ArtifactError("direct artifact read exceeds platform I/O limits");
        }

        ssize_t bytes = -1;
        do {
            bytes = ::pread(fd_, destination.data(), destination.size(),
                            static_cast<off_t>(absolute_offset));
        } while (bytes < 0 && errno == EINTR);
        if (bytes < 0) {
            throw std::system_error(errno, std::generic_category(), "direct artifact read");
        }
        return static_cast<std::size_t>(bytes);
    }

private:
    int fd_                = -1;
    const std::byte* data_ = nullptr;
    std::size_t size_      = 0;
};

} // namespace

std::string_view object_name(const ObjectDescriptor& object) noexcept {
    return std::visit([](const auto& descriptor) -> std::string_view { return descriptor.name; },
                      object);
}

std::uint64_t object_offset(const ObjectDescriptor& object) noexcept {
    return std::visit([](const auto& descriptor) { return descriptor.offset; }, object);
}

std::uint64_t object_bytes(const ObjectDescriptor& object) noexcept {
    return std::visit([](const auto& descriptor) { return descriptor.bytes; }, object);
}

struct Reader::Impl {
    explicit Impl(const std::filesystem::path& path) : file(path) {
        if (file.size() < kPrefixBytesV2) {
            throw ArtifactError("artifact is shorter than the NInfer header prefix");
        }
        const bool is_v2 = std::equal(kMagicV2.begin(), kMagicV2.end(), file.data());
        const bool is_v3 = std::equal(kMagicV3.begin(), kMagicV3.end(), file.data());
        if (!is_v2 && !is_v3) {
            throw ArtifactError("artifact magic is not NInfer v2 or v3");
        }
        if (is_v3) {
            parse_v3();
            return;
        }
        parse_v2();
    }

    void parse_v2() {
        const auto json_bytes = read_u64_le(file.data() + 8);
        if (json_bytes == 0) { throw ArtifactError("json_bytes must be positive"); }
        const auto metadata_end = checked_add(kPrefixBytesV2, json_bytes, "JSON range");
        payload_start           = align_up(metadata_end, kPayloadAlignment, "payload offset");
        if (metadata_end > file.size() || payload_start > file.size()) {
            throw ArtifactError("declared JSON or payload start extends beyond the file");
        }

        Json directory;
        try {
            const auto* begin = reinterpret_cast<const char*>(file.data() + kPrefixBytesV2);
            directory         = Json::parse(begin, begin + json_bytes);
        } catch (const Json::exception& error) {
            throw ArtifactError(std::string("invalid JSON directory: ") + error.what());
        }

        static constexpr std::array root_members = {"identity", "objects"};
        require_members(directory, root_members, "directory root");
        const auto& raw_identity                     = directory.at("identity");
        static constexpr std::array identity_members = {"model_id", "weights_id"};
        require_members(raw_identity, identity_members, "artifact identity");
        identity.model_id   = require_string(raw_identity.at("model_id"), "model_id");
        identity.weights_id = require_string(raw_identity.at("weights_id"), "weights_id");

        const auto& raw_objects = directory.at("objects");
        if (!raw_objects.is_array() || raw_objects.empty()) {
            throw ArtifactError("objects must be a nonempty array");
        }
        entries.reserve(raw_objects.size());
        index.reserve(raw_objects.size());

        const auto payload_bytes = static_cast<std::uint64_t>(file.size()) - payload_start;
        std::uint64_t cursor     = 0;
        for (const auto& raw_object : raw_objects) {
            register_object(parse_object(raw_object), payload_bytes, cursor);
        }
    }

    void parse_v3() {
        if (file.size() < kPrefixBytesV3) {
            throw ArtifactError("artifact is shorter than the v3 prefix");
        }
        const auto json_bytes = read_u64_le(file.data() + 8);
        if (json_bytes == 0) { throw ArtifactError("v3 json_bytes must be positive"); }
        const auto metadata_end = checked_add(kPrefixBytesV3, json_bytes, "v3 JSON range");
        payload_start           = align_up(metadata_end, kPayloadAlignment, "v3 payload offset");
        if (metadata_end > file.size() || payload_start > file.size()) {
            throw ArtifactError("declared v3 JSON or payload start extends beyond the file");
        }

        Json directory;
        try {
            const auto* begin = reinterpret_cast<const char*>(file.data() + kPrefixBytesV3);
            directory         = Json::parse(begin, begin + json_bytes);
        } catch (const Json::exception& error) {
            throw ArtifactError(std::string("invalid v3 JSON directory: ") + error.what());
        }
        if (!directory.is_object()) {
            throw ArtifactError("v3 directory root must be a JSON object");
        }
        if (const auto files_it = directory.find("files"); files_it != directory.end()) {
            if (!files_it->is_array() || files_it->size() != 1 ||
                !files_it->at(0).is_object() ||
                (files_it->at(0).contains("path") && !files_it->at(0).at("path").is_null())) {
                throw ArtifactError("multi-file v3 artifacts are not supported");
            }
        }
        if (!directory.contains("metadata") || !directory.at("metadata").is_object() ||
            !directory.at("metadata").contains("name")) {
            throw ArtifactError("v3 artifact metadata.name is required");
        }
        std::string raw_model_id =
            require_string(directory.at("metadata").at("name"), "v3 metadata.name");
        if (raw_model_id == "qwen3.8-27b-nvfp4-qad" || raw_model_id == "qwen3.8-27b-nvfp4") {
            identity.model_id = "qwen3.8-27b";
        } else if (raw_model_id == "qwen3.6-27b-nvfp4-qad" || raw_model_id == "qwen3.6-27b-nvfp4") {
            identity.model_id = "qwen3.6-27b";
        } else {
            identity.model_id = std::move(raw_model_id);
        }

        std::unordered_map<std::string, std::string> resource_names;
        if (const auto comp_it = directory.find("components"); comp_it != directory.end()) {
            if (!comp_it->is_object()) {
                throw ArtifactError("v3 components must be a JSON object");
            }
            for (const auto& [_, component] : comp_it->items()) {
                if (!component.is_object()) { continue; }
                const auto res_it = component.find("resources");
                if (res_it == component.end() || !res_it->is_object()) { continue; }
                for (const auto& [filename, obj_id_val] : res_it->items()) {
                    const std::string& obj_id = require_string(obj_id_val, "v3 resource id");
                    resource_names.emplace(obj_id, "frontend/" + filename);
                }
            }
        }

        if (!directory.contains("bindings") || !directory.at("bindings").is_object()) {
            throw ArtifactError("v3 bindings must be a JSON object");
        }
        std::unordered_map<std::string, std::vector<std::pair<std::string, std::uint64_t>>>
            bindings_by_object;
        for (const auto& [binding_name, binding_val] : directory.at("bindings").items()) {
            if (!binding_val.is_object()) {
                throw ArtifactError("v3 binding entry must be a JSON object: " + binding_name);
            }
            if (const auto obj_it = binding_val.find("object"); obj_it != binding_val.end()) {
                const std::string& obj_id = require_string(*obj_it, "v3 binding object");
                bindings_by_object[obj_id].emplace_back(binding_name, 0);
            } else if (const auto parts_it = binding_val.find("parts");
                       parts_it != binding_val.end() && parts_it->is_array() &&
                       parts_it->size() == 1 && parts_it->at(0).is_object()) {
                const auto& part          = parts_it->at(0);
                const std::string& obj_id = require_string(part.at("object"), "v3 part object");
                std::uint64_t range_begin = 0;
                if (const auto range_it = part.find("range");
                    range_it != part.end() && range_it->is_array() && !range_it->empty()) {
                    range_begin = require_unsigned(range_it->at(0), "v3 part range", false);
                }
                bindings_by_object[obj_id].emplace_back(binding_name, range_begin);
            } else {
                throw ArtifactError("unsupported v3 binding shape: " + binding_name);
            }
        }

        std::unordered_map<std::string, std::optional<std::string>> auxiliary_names;
        if (const auto uses_it = directory.find("uses"); uses_it != directory.end()) {
            if (!uses_it->is_array()) { throw ArtifactError("v3 uses must be a JSON array"); }
            const auto map_v3_input_divisor_name =
                [](std::string_view parameter, std::string_view input) -> std::optional<std::string> {
                if (parameter == "text/output_head") {
                    if (input == "text/final_hidden") {
                        return "text/output_head/input_scale_divisor";
                    }
                    if (input == "mtp/final_hidden") { return std::nullopt; }
                }
                const auto replace_suffix = [&](std::string_view from,
                                                std::string_view to) -> std::optional<std::string> {
                    if (parameter.ends_with(from)) {
                        return std::string(parameter.substr(0, parameter.size() - from.size())) +
                               std::string(to);
                    }
                    return std::nullopt;
                };
                if (auto mapped =
                        replace_suffix("/mlp/gate", "/mlp/gate_up_projection/input_scale_divisor")) {
                    return *mapped;
                }
                if (parameter.ends_with("/mlp/up")) { return std::nullopt; }
                if (auto mapped =
                        replace_suffix("/mlp/down", "/mlp/down_projection/input_scale_divisor")) {
                    return *mapped;
                }
                if (auto mapped =
                        replace_suffix("/gdn/query", "/gdn/input_projection/input_scale_divisor")) {
                    return *mapped;
                }
                if (parameter.ends_with("/gdn/key") || parameter.ends_with("/gdn/value") ||
                    parameter.ends_with("/gdn/z")) {
                    return std::nullopt;
                }
                if (auto mapped = replace_suffix("/gdn/output",
                                                 "/gdn/output_projection/input_scale_divisor")) {
                    return *mapped;
                }
                if (auto mapped = replace_suffix(
                        "/attention/query", "/attention/input_projection/input_scale_divisor")) {
                    return *mapped;
                }
                if (parameter.ends_with("/attention/key") || parameter.ends_with("/attention/gate") ||
                    parameter.ends_with("/attention/value")) {
                    return std::nullopt;
                }
                if (auto mapped = replace_suffix(
                        "/attention/output", "/attention/output_projection/input_scale_divisor")) {
                    return *mapped;
                }
                return std::string(parameter) + "/" + std::string(input) + "/input_scale_divisor";
            };
            for (const auto& use : *uses_it) {
                if (!use.is_object()) { continue; }
                const auto aux_it = use.find("auxiliaries");
                if (aux_it == use.end() || !aux_it->is_object()) { continue; }
                const auto div_it = aux_it->find("activation_input_divisor");
                if (div_it == aux_it->end() || !div_it->is_object() ||
                    !div_it->contains("object")) {
                    continue;
                }
                const std::string& obj_id =
                    require_string(div_it->at("object"), "v3 auxiliary object");
                const std::string& parameter =
                    require_string(use.at("parameter"), "v3 use parameter");
                const std::string& input = require_string(use.at("input"), "v3 use input");
                auxiliary_names.emplace(obj_id, map_v3_input_divisor_name(parameter, input));
            }
        }

        if (!directory.contains("objects") || !directory.at("objects").is_array() ||
            directory.at("objects").empty()) {
            throw ArtifactError("v3 objects must be a nonempty array");
        }
        const auto& raw_objects = directory.at("objects");
        entries.reserve(raw_objects.size());
        index.reserve(raw_objects.size());

        const auto payload_bytes = static_cast<std::uint64_t>(file.size()) - payload_start;
        std::uint64_t cursor     = 0;
        bool has_nvfp4_or_fp8    = false;
        for (const auto& raw_object : raw_objects) {
            if (!raw_object.is_object()) {
                throw ArtifactError("each v3 object entry must be a JSON object");
            }
            const std::string& obj_id = require_string(raw_object.at("id"), "v3 object id");
            const std::string& kind   = require_string(raw_object.at("kind"), "v3 object kind");
            if (kind == "resource") {
                const auto it = resource_names.find(obj_id);
                if (it == resource_names.end()) {
                    throw ArtifactError("unreferenced v3 resource object: " + obj_id);
                }
                register_object(parse_v3_resource(raw_object, it->second), payload_bytes, cursor);
            } else if (kind == "tensor") {
                std::string logical_name;
                bool skip_registration = false;
                if (const auto it = bindings_by_object.find(obj_id);
                    it != bindings_by_object.end() && !it->second.empty()) {
                    logical_name = it->second.size() == 1
                                       ? remap_v3_single_name(it->second.front().first)
                                       : remap_v3_fused_name(it->second);
                } else if (const auto aux_it = auxiliary_names.find(obj_id);
                           aux_it != auxiliary_names.end()) {
                    if (aux_it->second.has_value()) {
                        logical_name = *aux_it->second;
                    } else {
                        // Redundant sibling activation_input_divisor (e.g. mlp/up, gdn/key|value|z,
                        // or mtp/final_hidden sharing text/output_head): validate payload layout
                        // and advance cursor without exposing an unconsumed duplicate object.
                        logical_name      = obj_id;
                        skip_registration = true;
                    }
                } else {
                    throw ArtifactError("unreferenced v3 tensor object: " + obj_id);
                }
                TensorDescriptor desc = parse_v3_tensor(raw_object, std::move(logical_name));
                if (skip_registration) {
                    const auto alignment = tensor_alignment(desc.layout);
                    if (desc.offset < cursor) {
                        throw ArtifactError("object " + desc.name +
                                            " overlaps or is out of order");
                    }
                    if (desc.offset % alignment != 0) {
                        throw ArtifactError("object " + desc.name + " is not " +
                                            std::to_string(alignment) + "-byte aligned");
                    }
                    const auto end = checked_add(desc.offset, desc.bytes, "object payload range");
                    if (end > payload_bytes) {
                        throw ArtifactError("object " + desc.name + " extends beyond the file");
                    }
                    cursor = end;
                    continue;
                }
                if (std::string_view(desc.name).starts_with("text/") &&
                    (desc.format == NumericFormat::NVFP4 ||
                     desc.format == NumericFormat::FP8_E4M3FN_ROW_BF16S)) {
                    has_nvfp4_or_fp8 = true;
                }
                register_object(std::move(desc), payload_bytes, cursor);
            } else {
                throw ArtifactError("v3 object kind must be 'tensor' or 'resource'");
            }
        }
        identity.weights_id = has_nvfp4_or_fp8 ? "nvfp4" : "groupwise-int";
    }

    void register_object(ObjectDescriptor object, std::uint64_t payload_bytes,
                         std::uint64_t& cursor) {
        const auto name      = object_name(object);
        const auto offset    = object_offset(object);
        const auto bytes     = object_bytes(object);
        const auto alignment = std::visit(
            [](const auto& descriptor) {
                using Descriptor = std::decay_t<decltype(descriptor)>;
                if constexpr (std::is_same_v<Descriptor, TensorDescriptor>) {
                    return tensor_alignment(descriptor.layout);
                } else {
                    return resource_alignment(descriptor.encoding);
                }
            },
            object);

        if (offset < cursor) {
            throw ArtifactError("object " + std::string(name) + " overlaps or is out of order");
        }
        if (offset % alignment != 0) {
            throw ArtifactError("object " + std::string(name) + " is not " +
                                std::to_string(alignment) + "-byte aligned");
        }
        const auto end = checked_add(offset, bytes, "object payload range");
        if (end > payload_bytes) {
            throw ArtifactError("object " + std::string(name) + " extends beyond the file");
        }
        const auto object_index = entries.size();
        auto [_, inserted]      = index.emplace(std::string(name), object_index);
        if (!inserted) { throw ArtifactError("duplicate object name: " + std::string(name)); }
        entries.push_back(std::move(object));
        cursor = end;
    }

    MappedFile file;
    ArtifactIdentity identity;
    std::vector<ObjectDescriptor> entries;
    std::unordered_map<std::string, std::size_t, TransparentStringHash, std::equal_to<>> index;
    std::uint64_t payload_start = 0;
};

Reader::Reader(const std::filesystem::path& path) : impl_(std::make_unique<Impl>(path)) {}

Reader::~Reader()                            = default;
Reader::Reader(Reader&&) noexcept            = default;
Reader& Reader::operator=(Reader&&) noexcept = default;

const ArtifactIdentity& Reader::identity() const noexcept { return impl_->identity; }

const std::vector<ObjectDescriptor>& Reader::objects() const noexcept { return impl_->entries; }

const ObjectDescriptor* Reader::find(std::string_view name) const noexcept {
    const auto it = impl_->index.find(name);
    return it == impl_->index.end() ? nullptr : &impl_->entries[it->second];
}

std::uint64_t Reader::file_bytes() const noexcept { return impl_->file.size(); }

std::uint64_t Reader::payload_offset() const noexcept { return impl_->payload_start; }

PayloadSpan Reader::payload(const ObjectDescriptor& object) const {
    const auto absolute =
        checked_add(impl_->payload_start, object_offset(object), "absolute payload offset");
    const auto end = checked_add(absolute, object_bytes(object), "absolute payload range");
    if (end > impl_->file.size()) { throw ArtifactError("object payload extends beyond the file"); }
    return {
        absolute,
        std::span<const std::byte>(impl_->file.data() + absolute,
                                   static_cast<std::size_t>(object_bytes(object))),
    };
}

PayloadSpan Reader::payload(std::string_view name) const {
    const auto* object = find(name);
    if (object == nullptr) { throw ArtifactError("unknown artifact object: " + std::string(name)); }
    return payload(*object);
}

std::size_t Reader::read_direct(std::uint64_t absolute_offset,
                                std::span<std::byte> destination) const {
    return impl_->file.read_direct(absolute_offset, destination);
}

} // namespace ninfer::artifact
