// ninfer::ops - rmsnorm wrapper: public api validation and launcher dispatch.
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/gated_rmsnorm.h"

#include "ops/launcher/rmsnorm.h" // detail::rmsnorm_launch

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

std::int64_t numel_allow_zero(const Tensor& t, const char* label) {
    bool has_zero = false;
    for (int d = 0; d < 4; ++d) {
        if (t.ne[d] < 0) {
            throw std::invalid_argument(std::string("rmsnorm: ") + label +
                                        " dimensions must be nonnegative");
        }
        if (t.ne[d] == 0) { has_zero = true; }
    }
    if (has_zero) { return 0; }

    std::int64_t total = 1;
    for (int d = 0; d < 4; ++d) {
        if (total > std::numeric_limits<std::int64_t>::max() / t.ne[d]) {
            throw std::overflow_error("rmsnorm: tensor size overflows int64");
        }
        total *= t.ne[d];
    }
    return total;
}

void require_same_shape(const Tensor& a, const Tensor& b, const char* b_label) {
    for (int d = 0; d < 4; ++d) {
        if (a.ne[d] != b.ne[d]) {
            throw std::invalid_argument(std::string("rmsnorm: x/") + b_label +
                                        " shapes must match");
        }
    }
}

} // namespace

namespace {

void rmsnorm_impl(const Tensor& x, const Tensor& weight, float eps, bool unit_offset,
                  const Tensor* z, Tensor& out, cudaStream_t stream) {
    if (x.dtype != DType::BF16 || weight.dtype != DType::BF16 || out.dtype != DType::BF16 ||
        (z != nullptr && z->dtype != DType::BF16)) {
        throw std::invalid_argument("rmsnorm: x/weight/z/out must be BF16");
    }
    if (!(eps > 0.0f) || !std::isfinite(eps)) {
        throw std::invalid_argument("rmsnorm: eps must be positive and finite");
    }

    const std::int64_t n = numel_allow_zero(x, "x");
    (void)numel_allow_zero(out, "out");
    (void)numel_allow_zero(weight, "weight");
    require_same_shape(x, out, "out");
    if (z != nullptr) {
        (void)numel_allow_zero(*z, "z");
        require_same_shape(x, *z, "z");
    }
    if (weight.ne[0] != x.ne[0] || weight.ne[1] != 1 || weight.ne[2] != 1 || weight.ne[3] != 1) {
        throw std::invalid_argument("rmsnorm: weight must be 1-D with ne[0] == x.ne[0]");
    }
    if (n == 0) { return; }
    const std::int64_t rows = n / x.ne[0];
    if (rows > std::numeric_limits<int>::max()) {
        throw std::overflow_error("rmsnorm: row count exceeds CUDA grid limit");
    }

    if (!x.is_contiguous() || !weight.is_contiguous() || !out.is_contiguous() ||
        (z != nullptr && !z->is_contiguous())) {
        throw std::invalid_argument("rmsnorm: x/weight/z/out must be contiguous");
    }
    if (x.data == nullptr || weight.data == nullptr || out.data == nullptr ||
        (z != nullptr && z->data == nullptr)) {
        throw std::invalid_argument("rmsnorm: x/weight/z/out data must be non-null");
    }

    detail::rmsnorm_launch(x, weight, eps, unit_offset, z, out, stream);
}

} // namespace

void rmsnorm(const Tensor& x, const Tensor& weight, float eps, bool unit_offset, Tensor& out,
             cudaStream_t stream) {
    rmsnorm_impl(x, weight, eps, unit_offset, nullptr, out, stream);
}

void gated_rmsnorm(const Tensor& x, const Tensor& weight, const Tensor& z, float eps, Tensor& out,
                   cudaStream_t stream) {
    rmsnorm_impl(x, weight, eps, false, &z, out, stream);
}

void gated_rmsnorm_columns(const Tensor& x, const Tensor& weight, const Tensor& z, float eps,
                           const Tensor& x_columns, Tensor& out, cudaStream_t stream) {
    const auto bf16 = [](const Tensor& t) { return t.dtype == DType::BF16 && t.is_contiguous(); };
    const std::int32_t d = out.ne[0];
    if (!bf16(x) || !bf16(weight) || !bf16(z) || !bf16(out) || x_columns.dtype != DType::I32 ||
        !x_columns.is_contiguous() || !(eps > 0.0f) || !std::isfinite(eps) ||
        (d != 64 && d != 128 && d != 192 && d != 256) || x.ne[0] != d || x.ne[1] != out.ne[1] ||
        z.ne[0] != d || z.ne[1] != out.ne[1] || z.ne[2] != out.ne[2] || weight.ne[0] != d ||
        x_columns.ne[0] != out.ne[2] || x.ne[3] != 1 || out.ne[3] != 1 || z.ne[3] != 1 ||
        x.data == nullptr || z.data == nullptr || out.data == nullptr ||
        ((reinterpret_cast<std::uintptr_t>(x.data) | reinterpret_cast<std::uintptr_t>(z.data) |
          reinterpret_cast<std::uintptr_t>(out.data) |
          reinterpret_cast<std::uintptr_t>(weight.data)) & 3u) != 0) {
        throw std::invalid_argument("gated_rmsnorm_columns: invalid operands");
    }
    if (out.ne[2] == 0) { return; }
    detail::gated_rmsnorm_columns_launch(x, weight, z, eps, x_columns, out, stream);
}

} // namespace ninfer::ops
