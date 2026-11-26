# NInfer for Tesla V100 (v2.0)

A C++/CUDA inference engine for **Qwen3.8-27B** tuned specifically for a single NVIDIA Tesla V100 (`sm_70`, 32 GB). It serves text, image, and video input through a local CLI or OpenAI- and Anthropic-compatible HTTP APIs, built around one GPU, one resident model, and 1–8 concurrent requests.

> **Scope note:** While the underlying engine retains support for the other NInfer Qwen artifacts, all v1.0 and v2.0 kernel, scheduler, and speculative-decoding optimizations in this repository are **specifically built and tuned for `Qwen3.8-27B` NVFP4 on Tesla V100**. Other models are not tuned or benchmarked here.

This repository is the v2.0 release of [ddominet/ninfer-v100](https://github.com/ddominet/ninfer-v100), originally forked from [geoffwatts/ninfer-v100](https://github.com/geoffwatts/ninfer-v100). Version 1.0 focused on decode attention, prefill, long-context scaling, and multi-session MTP throughput. **The goal of v2.0 is to keep all of those v1.0 strengths—high multi-session throughput and strong long-context performance—with DFlash2 enabled**, so a single serving configuration (`--spec dflash2 --draft-tokens 7`) wins across single-stream decode, prompt processing, long-context depth, and 8-stream aggregate throughput without requantizing weights.

> **Use DFlash2 for best performance:** Run with `--spec dflash2 --draft-tokens 7 --lm-head-draft --kv-dtype int8 --decode-share 50 --no-cuda-graph` (see [Serve](#serve-recommended-v20-production-flags) below). Unlike v1.0—where you had to choose between MTP for multi-session throughput and DFlash2 for single-stream decode—v2.0's DFlash2 path is faster than MTP across both single-stream and 8-stream workloads.

---

## Benchmarks (Tesla V100-PCIe-32GB)

Every server below was measured on the same **Tesla V100-PCIe-32GB** (CUDA 12.8) over HTTP using the same cold natural-text prompts (reference prose, fiction, and source code from `eval/corpora/perplexity-1m`), greedy decoding, thinking off, and 256 output tokens per request (6 requests/cell up to 24K, 3 requests/cell at 90K–180K).

Engines compared:
- **[geoffwatts/ninfer-v100](https://github.com/geoffwatts/ninfer-v100)** — original V100 port (`b37d0dd`), Qwen3.8-27B NVFP4, INT8 KV, MTP K=3 / K=1.
- **[tuxKOH/ninfer-v100xN](https://github.com/tuxKOH/ninfer-v100xN)** — V100 fork, Qwen3.8-27B NVFP4, INT8 KV, MTP K=3 / K=1.
- **[ddominet/ninfer-v100](https://github.com/ddominet/ninfer-v100) (v1.0)** — previous release of this fork, Qwen3.8-27B NVFP4, INT8 KV, MTP K=3 / K=1 and unoptimized DFlash2.
- **[inscora/llama-superfast-v100-BWE-q8-dflash2](https://github.com/inscora/llama-superfast-v100-BWE-q8-dflash2)** — V100 llama.cpp fork with Unsloth `UD-Q4_K_XL`, z-lab DFlash2 `Q4_K_M` (`K=7`), Q8_0 KV.
- **This release (v2.0)** — Qwen3.8-27B NVFP4, INT8 KV, `--spec dflash2 --draft-tokens 7 --lm-head-draft --decode-share 50 --no-cuda-graph`.

### Single-stream decode (tok/s, higher is better)

| Prompt context | geoffwatts (MTP3) | tuxKOH (MTP3) | ddominet v1.0 (MTP3) | ddominet v1.0 (DF7) | llama-superfast (DF7) | **v2.0 (DF7)** |
|---:|---:|---:|---:|---:|---:|---:|
| ~60 | 65.5 | 70.4 | 72.7 | 76.8 | 83.3 | **91.6** |
| 1K | 67.5 | 75.1 | 72.9 | 83.1 | 87.2 | **100.8** |
| 6K | 66.0 | 75.5 | 75.3 | 72.8 | 84.4 | **90.8** |
| 24K | 57.2 | 69.9 | 70.0 | 68.3 | 78.1 | **81.0** |
| 90K | 41.4 | 56.9 | 57.9 | 51.3 | 64.4 | **65.3** |
| 180K | 31.9 | 48.5 | 53.5 | 44.6 | 51.8 | **54.3** |

### Prompt processing / prefill (tok/s, higher is better)

| Prompt context | geoffwatts | tuxKOH | ddominet v1.0 | llama-superfast | **v2.0 (DF7)** |
|---:|---:|---:|---:|---:|---:|
| ~60 | 152.1 | 155.1 | 336.8 | 388.6 | **415.6** |
| 1K | 949.0 | 955.2 | 976.0 | 975.1 | **1046.1** |
| 6K | 1075.4 | 1082.3 | 1160.3 | 1121.9 | **1211.5** |
| 24K | 974.2 | 1009.1 | 1076.3 | 1052.6 | **1118.2** |
| 90K | 666.2 | 779.3 | 820.9 | 814.0 | **834.5** |
| 180K | 454.2 | 595.2 | 622.6 | 620.5 | **627.0** |

### Multi-stream aggregate decode (tok/s, mean of 3 runs, higher is better)

| Streams | geoffwatts (MTP3 / K1) | tuxKOH (MTP3 / K1) | llama-superfast (DF7) | ddominet v1.0 (MTP3 / K1) | ddominet v1.0 (DF7) | v2.0 (MTP3) | **v2.0 (DF7)** |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 57.5 / 45.0 | 64.2 / 50.2 | 76.8 | 66.4 / 53.1 | 65.3 | 71.2 | **82.2** |
| 2 | 92.8 / 80.0 | 101.3 / 88.5 | 120.4 | 113.2 / 95.6 | 96.2 | 122.3 | **137.0** |
| 4 | 109.1 / 125.3 | 122.3 / 135.7 | 135.0 | 157.3 / 160.0 | 95.2 | 160.9 | **176.0** |
| 8 | 129.6 / 155.7 | 145.5 / 175.8 | 151.7 | 211.8 / 243.3 | 121.3 | 245.9 | **252.1** |

### Mixed-arrival Poisson chat (`rate=0.5 req/s`)

| Engine | Makespan | Aggregate throughput | Errors |
|---|---:|---:|---:|
| llama-superfast (DF7) | 152.4 s | 166.3 tok/s | 0 |
| v2.0 (MTP3) | 104.5 s | 242.1 tok/s | 0 |
| **v2.0 (DF7)** | **100.3 s** | **248.2 tok/s** | **0** |

### Context × concurrency grid (total wall time in seconds, lower is better)

End-to-end wall time (cold prefill + 256 decoded tokens per stream) across 14 context/concurrency cells. `v2.0 (DF7)` finishes first in all **14/14 cells**:

| Cell | ddominet v1.0 (DF7) | ddominet v1.0 (MTP3) | llama-superfast (DF7) | **v2.0 (DF7)** |
|---|---:|---:|---:|---:|
| 1K × 1 | 3.725 s | 4.169 s | 4.095 s | **3.248 s** |
| 1K × 2 | 6.989 s | 6.666 s | 6.192 s | **5.281 s** |
| 1K × 4 | 13.560 s | 11.906 s | 12.077 s | **10.982 s** |
| 1K × 8 | 24.936 s | 20.313 s | 23.102 s | **18.373 s** |
| 6K × 1 | 8.002 s | 7.945 s | 7.949 s | **7.268 s** |
| 6K × 2 | 16.113 s | 15.633 s | 15.938 s | **14.603 s** |
| 6K × 4 | 35.215 s | 34.204 s | 32.638 s | **32.369 s** |
| 6K × 8 | 71.494 s | 67.774 s | 65.296 s | **64.742 s** |
| 24K × 1 | 26.879 s | 26.115 s | 26.378 s | **24.973 s** |
| 24K × 2 | 54.054 s | 52.170 s | 53.320 s | **50.184 s** |
| 24K × 4 | 112.694 s | 107.059 s | 108.573 s | **103.786 s** |
| 24K × 8 | 215.074 s | 204.131 s | 213.919 s | **198.966 s** |
| 90K × 1 | 117.292 s | 114.275 s | 113.587 s | **111.117 s** |
| 90K × 2 | 243.859 s | 235.047 s | 247.820 s | **230.085 s** |

---

## What changed

### In v1.0 (from `geoffwatts/ninfer-v100`)

Version 1.0 rebuilt the core V100 attention, linear, prefill, and memory paths:

- **Decode attention:** Replaced per-row KV re-fetches with tensor-core paged INT8 attention kernels (register-direct 1-token kernel plus a staged multi-token kernel that dequantizes each KV tile once), moved Hadamard query rotation to once per round, tuned split-KV granularity by context depth, and batched multiple streams per launch.
- **Linear layers:** Extended Volta QPN decode-width kernels up to 64 rows so batched rounds stop falling back to prefill GEMMs, staged activations as FP16 once per round, and ran FP8 MLP layers in a single pass.
- **Prefill:** Added fused prefill attention reading paged INT8 KV directly (no gather workspace), routed short prompts onto decode-width kernels (over 2× faster on short prompts), kept NVFP4 wide-route SwiGLU in FP32, and software-pipelined the Gated Delta Net (GDN) recurrence.
- **Engine & memory:** Added `--decode-share` time-sliced decode/prefill interleaving, moved GDN scratch into the planned workspace, fixed an FP8 GDN convolution workspace under-allocation, and made full 256K INT8 KV + 2048-token prefill chunks fit reliably within 32 GB with startup capacity checks.

### New in v2.0 (from `ddominet/ninfer-v100` v1.0)

Version 2.0 turns **DFlash2** into the fastest path for both single-stream and 8-stream serving while cutting target-model round latency and kernel-launch overhead across the board (`B=1` round time down to `35.58 ms` with `831` kernels/round and only `0.187 ms` eager launch gap under `--no-cuda-graph`):

- **Full DFlash2 draft stack on Volta:**
  - Load-time QPN prepacking for all DFlash2 W8 draft weights (`q|k|v`, `o_proj`, `gate|up`, `down`, `fc`) with specialized `Split2`/`Split3` small-T QPN kernels and FP16-staged tensor-core skinny GEMMs (`ROWS=128` through `T=48`, warp-aware split-K capped at 4 splits).
  - Coalesced 8-warp double-buffered HMMA dynamic grouped convolution (~2.5× faster).
  - Coalesced half-split Q4 proposal head with register bitonic Top-K and QPN routing through `T=32` (cutting `B=8` head latency from ~1.7 ms to ~0.66 ms).
  - Grouped-query sliding-window draft attention (32 keys/step, 8 warps) and 2-CTA/SM 6–8 token verify attention over paged INT8 KV.
  - Host-mapped token embedding and selector codebooks to keep full 256K INT8 KV + 8 concurrent streams comfortably inside 32 GB VRAM.
- **Adaptive multi-stream DFlash2 runtime:**
  - Online-fitted round-cost budgeting (`DFlashCostModel`) and kept-column verify compaction so DFlash2 scales to 252 tok/s at 8 streams instead of stalling at ~121 tok/s.
  - Sliding-window prefill skip (avoids materializing draft KV for prompt tokens outside the window), in-place kept-column GDN norm/scatter, and compacted DFlash2 context append/materialize.
  - High-priority main stream + low-priority `gdn_fold_stream_` so GDN replay folds overlap completely inside the DFlash2 proposal window.
  - Vectorized penalized argmax, batched token-count updates, and improved short-prompt batching / decode-credit accounting.
- **Target compute & launch-overhead elimination (`-298` kernel launches/round):**
  - SM70 CUTLASS W8/NVFP4/FP8 prefill GEMMs and fused target RMSNorm+RoPE.
  - Fused NVFP4 and FP8 MLP pipeline (`Block=640` staged RMSNorm → staged SwiGLU → fused `ResidualAdd` down projection) and fused FP8 attention input RMSNorm across all 64 layers.
  - High-occupancy pre-unpacked `gdn_norm_gating_27` (`Threads=640/320`, `Tile=2/4`) plus direct FP16 output from `gdn_norm_gating_27`, `gated_rmsnorm`, and `sigmoid_mul` straight into FP8 QPN/TC—cutting `fp8_stage_bf16_activation` from 139 launches/round down to 27.

---

## Quick start

### Requirements

64-bit Linux, a Tesla V100-32GB, **CUDA Toolkit 12.8** (the last toolkit with `sm_70` support), CMake ≥ 3.28, a C++20 compiler, Ninja, `pkg-config`, FFmpeg dev libraries (`libavformat >= 60`, `libavcodec >= 60`, `libavutil >= 58`, `libswscale >= 7`), and `libcurl >= 7.85`.

### Build

```bash
cmake -S . -B build-v100 -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=70
cmake --build build-v100 -j
```

At startup on Volta, NVFP4, FP8, W8, and Q4 weights are repacked in place on the GPU into native Volta QPN fragment layouts. The `.ninfer` file on disk is untouched and temporary repack memory is freed before serving begins.

### Download the model

Pin the v2 artifact revision (later revisions on Hugging Face use a newer container format and will be rejected with `artifact magic is not NInfer v2`):

```bash
hf download neroued/Qwen3.8-27B-nvfp4-NInfer \
  qwen3_8_27b_nvfp4.ninfer \
  --revision 11dbbbbbc33db198afe2f02c9232c771ff7031be \
  --local-dir models
```

Expected SHA-256: `552c374c685dce302603b95fbe940fb04243c0cd44c083efc644ad3d980d462c`.

### Serve (recommended v2.0 production flags)

For the best all-around performance on a V100-32GB (single-stream decode, long context, and up to 8 concurrent sessions), **use DFlash2 with these flags**:

```bash
./build-v100/apps/ninfer-serve models/qwen3_8_27b_nvfp4.ninfer \
  --host 0.0.0.0 --port 8080 \
  --max-context 262144 \
  --kv-capacity 262144 \
  --prefill-chunk 2048 \
  --max-concurrency 8 \
  --device-state-slots 0 \
  --host-kv-mib 6144 \
  --host-state-slots 16 \
  --decode-share 50 \
  --no-cuda-graph \
  --kv-dtype int8 \
  --spec dflash2 --draft-tokens 7 \
  --lm-head-draft
```

Why these options matter:
- `--spec dflash2 --draft-tokens 7 --lm-head-draft`: enables the v2.0 DFlash2 draft stack with the optimized 7-token draft window and fused Q4 proposal head.
- `--kv-dtype int8 --max-context 262144 --kv-capacity 262144`: uses the Volta tensor-core INT8 paged KV cache for full 256K context capacity inside 32 GB VRAM.
- `--max-concurrency 8 --decode-share 50`: supports up to 8 active sessions while interleaving decode rounds with long prefills so active streams stay responsive.
- `--no-cuda-graph`: saves graph capture VRAM for the full 256K KV pool; v2.0's kernel fusions cut eager launch overhead down to ~0.19 ms/round, so there is no throughput penalty.
- `--device-state-slots 0 --host-kv-mib 6144 --host-state-slots 16`: keeps prefix checkpoints in pinned host RAM so GPU memory goes to KV capacity.

Add `--vision` if you need image/video input (note that `--vision` reserves extra VRAM, so use `--kv-capacity auto` when enabling it).

Send a test request:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [{"role": "user", "content": "Reply with one short sentence."}],
    "max_tokens": 64
  }'
```

### One-shot CLI

```bash
./build-v100/apps/ninfer models/qwen3_8_27b_nvfp4.ninfer \
  --prompt "Explain prefill and decode, then give a concise conclusion." \
  --max-context 32768 \
  --max-new 4096 \
  --kv-dtype int8 \
  --spec dflash2 --draft-tokens 7 \
  --lm-head-draft
```

---

## Docker & Kubernetes

### Docker Compose

The image builds for `sm_70` with CUDA 12.8 ([`Dockerfile`](Dockerfile), [`.github/workflows/image.yml`](.github/workflows/image.yml)). With the NVIDIA Container Toolkit installed:

```bash
mkdir -p models
hf download neroued/Qwen3.8-27B-nvfp4-NInfer qwen3_8_27b_nvfp4.ninfer \
  --revision 11dbbbbbc33db198afe2f02c9232c771ff7031be --local-dir models
docker compose up -d            # pull published image
docker compose up -d --build    # or build locally
```

[`docker-compose.yml`](docker-compose.yml) defaults to the v2.0 DFlash2 configuration (`262,144` context, 8 streams, INT8 KV, `--spec dflash2 --draft-tokens 7`). Override via `NINFER_GPU`, `NINFER_PORT`, `NINFER_MODELS`, `NINFER_MAX_CONTEXT`, `NINFER_MAX_CONCURRENCY`, `NINFER_SPEC`, and `NINFER_DRAFT_TOKENS`.

### Kubernetes

[`deploy/kubernetes/ninfer.yaml`](deploy/kubernetes/ninfer.yaml) provides a PVC, single-GPU `Recreate` Deployment, and Service:

```bash
kubectl create namespace ninfer
kubectl -n ninfer apply -f deploy/kubernetes/ninfer.yaml
```

Place `qwen3_8_27b_nvfp4.ninfer` on the `ninfer-models` PVC before starting the pod.

---

## Capabilities & limits

Supported on Qwen3.8-27B:
- Text generation (thinking and non-thinking modes) and multimodal image/video input (`--vision`).
- Chunked prefill, `--decode-share` decode/prefill interleaving, and 1–8 concurrent request lanes.
- DFlash2 masked-block speculative decoding (`--spec dflash2 --draft-tokens 7`, recommended) and MTP (`--spec mtp`).
- INT8, FP8, and BF16 paged KV cache with exact-prefix reuse across Device and pinned Host tiers.
- OpenAI Chat Completions, OpenAI Responses, and Anthropic Messages APIs (streaming, tool calls, token counting, usage accounting).

Intentional scope boundaries:
- Single Tesla V100 and one resident model per process (no multi-GPU or weight offload).
- Fixed startup capacity of 1–8 active requests with bounded FIFO ingress.
- Optimizations are tailored specifically for **Qwen3.8-27B NVFP4**.

---

## Credits

- **[Neroued/ninfer](https://github.com/Neroued/ninfer)** — original NInfer engine, artifact container, serving stack, and published model artifacts.
- **[geoffwatts/ninfer-v100](https://github.com/geoffwatts/ninfer-v100)** — original Tesla V100 (`sm_70`) port, software NVFP4 on Volta tensor cores, quadpair-split-N kernels, load-time prepacking, and INT8 paged KV cache.
- **[dnv2003/v100-skinny](https://github.com/dnv2003/v100-skinny)** — inspiration for Volta QPN prepacking and fragment layouts.
- **[syv-ai/qwen38-27b-rtx3090](https://github.com/syv-ai/qwen38-27b-rtx3090)** — inspiration for context-lookup MTP.

## Documentation

- [Documentation index](docs/README.md)
- [CLI guide](docs/cli.md)
- [HTTP serving](docs/serving.md)
- [V100 performance rework notes](docs/v100-rework.md)
- [V100 port notes](docs/v100.md)
- [Resource scheduling and context cache](docs/maintainer/resource-scheduling-and-context-cache.md)

## License

Apache License 2.0 ([`LICENSE`](LICENSE)). Model artifacts are derived from [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) and [unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4) (Apache-2.0). Vendored dependencies retain their own licenses under `third_party/`.
