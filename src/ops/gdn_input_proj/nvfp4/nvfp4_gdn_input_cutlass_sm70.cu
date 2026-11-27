#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_cutlass_sm70.h"

#include "core/device.h"
#include "core/layout.h"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_output.cuh"
#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_cutlass_sm70.h"

#include <cuda_bf16.h>

namespace ninfer::ops::detail {
namespace {

constexpr std::int32_t kRows = Nvfp4GdnInputOutput::kQkvRows + Nvfp4GdnInputOutput::kZRows;

template <class Allocator>
Tensor allocate_projected(Allocator& allocator, int tokens) {
    return allocator.alloc(DType::BF16, {kRows, tokens});
}

__global__ void split_nvfp4_gdn_output(const __nv_bfloat16* __restrict__ projected,
                                       __nv_bfloat16* __restrict__ qkv,
                                       __nv_bfloat16* __restrict__ z, std::int64_t count) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) { return; }
    const int row   = static_cast<int>(i % kRows);
    const int token = static_cast<int>(i / kRows);
    if (row < Nvfp4GdnInputOutput::kQkvRows) {
        qkv[static_cast<std::int64_t>(token) * Nvfp4GdnInputOutput::kQkvRows + row] = projected[i];
    } else {
        z[static_cast<std::int64_t>(token) * Nvfp4GdnInputOutput::kZRows + row -
          Nvfp4GdnInputOutput::kQkvRows] = projected[i];
    }
}

} // namespace

std::size_t nvfp4_gdn_input_cutlass_workspace_bytes(std::int32_t tokens) {
    WorkspaceLayoutBuilder layout;
    (void)allocate_projected(layout, tokens);
    const std::size_t projected_bytes = layout.peak_bytes(1);
    return projected_bytes + nvfp4_cutlass_sm70_workspace_bytes(
                                 kRows, Nvfp4GdnInputGeometry::kInputRows, tokens);
}

void nvfp4_gdn_input_cutlass_sm70_launch(const Tensor& x, const Weight& weight, Tensor& qkv,
                                         Tensor& z, WorkspaceArena& workspace,
                                         cudaStream_t stream) {
    auto scope       = workspace.scope();
    Tensor projected = allocate_projected(workspace, x.ne[1]);
    nvfp4_cutlass_sm70_launch(x, weight, projected, workspace, stream);
    const std::int64_t count = static_cast<std::int64_t>(x.ne[1]) * kRows;
    split_nvfp4_gdn_output<<<static_cast<int>((count + 255) / 256), 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(projected.data), static_cast<__nv_bfloat16*>(qkv.data),
        static_cast<__nv_bfloat16*>(z.data), count);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
