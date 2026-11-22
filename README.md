# NInfer for Tesla V100 — performance rework

A from-scratch C++/CUDA inference engine for selected Qwen checkpoints, tuned for a single
NVIDIA Tesla V100 (Volta, `sm_70`). It serves text, image, and video input through a local CLI or
OpenAI-/Anthropic-compatible HTTP APIs. The runtime is intentionally narrow: one GPU, one resident
model, 1–8 active requests.

This repository is a performance-focused fork of
[geoffwatts/ninfer-v100](https://github.com/geoffwatts/ninfer-v100). Both builds were measured
through the HTTP server on one V100-PCIe-32GB with the published Qwen3.8-27B NVFP4 artifact and
natural-text prompts (reference prose, fiction, source code). This fork was faster in every cell
measured:

| Workload | Upstream port | This fork | Gain |
|---|---:|---:|---:|
| Decode, 1.1K-token prompt, MTP K=3 | 67.5 tok/s | 72.9 tok/s | 1.08× |
| Decode, 24K-token prompt, MTP K=3 | 57.2 tok/s | 70.0 tok/s | 1.22× |
| Decode, 89K-token prompt, MTP K=3 | 41.4 tok/s | 57.9 tok/s | 1.40× |
| Decode, 172K-token prompt, MTP K=3 | 31.9 tok/s | 53.5 tok/s | 1.68× |
| Decode, 172K-token prompt, MTP off | 14.1 tok/s | 24.2 tok/s | 1.72× |
| 8 concurrent streams, MTP K=1 | 155.7 tok/s | 243.3 tok/s | 1.56× |
| 8 concurrent streams, MTP K=3 | 129.6 tok/s | 211.8 tok/s | 1.63× |
| Prefill, 52–66 token prompts | 154 tok/s | 339 tok/s | 2.21× |
| Prefill, 172K-token prompt | 455 tok/s (380 s) | 625 tok/s (276 s) | 1.37× |

Decode rows are means over three to six requests; draft acceptance at K=3 was 53–60% on both
builds. Full tables, per-request spread, acceptance rates, derived memory and arithmetic
throughput, and the method are in [V100 performance rework](docs/v100-rework.md).

## Credits

This work stands on other people's engines and ideas:

- **[Neroued/ninfer](https://github.com/Neroued/ninfer)** — the original NInfer engine, its
  artifact format, serving stack, and the published model artifacts. Everything here is that
  engine.
- **[geoffwatts/ninfer-v100](https://github.com/geoffwatts/ninfer-v100)** — the Tesla V100 port
  this fork starts from: the `sm_70` build, software NVFP4 on Volta tensor cores, the
  quadpair-split-N linear kernels, load-time weight prepacking, and the INT8 paged KV cache. The
  rework below tunes that design; it does not replace it.
- **[dnv2003/v100-skinny](https://github.com/dnv2003/v100-skinny)** — inspiration for the Volta
  QPN prepacking and fragment maps used by the port.
- **[syv-ai/qwen38-27b-rtx3090](https://github.com/syv-ai/qwen38-27b-rtx3090)** — inspiration for
  context-lookup MTP.

## What changed from the upstream port

Baseline is upstream `master` at `b37d0dd`. No model artifact, API, or output contract changed;
every change is in the Volta execution path.

**Decode attention**
- New tensor-core attention kernels over the paged INT8 KV cache for rounds of 1–8 tokens: a
  register-direct kernel for one token and a staged kernel that dequantises each key/value tile
  once and shares it across all rows, instead of every row re-fetching it.
- Queries are Hadamard-rotated once per round, not once per kernel block.
- Split sizes follow context length and round width, with fewer, longer splits at long context.
- The kernels handle several streams in one launch.

**Linear layers**
- The decode-width (QPN) kernels now cover rounds up to 64 rows; previously anything above 32
  rows fell through to the prefill GEMM with a per-call weight dequantisation.
- Activations are staged once per round as FP16 in the layout the kernels read, instead of being
  converted inside the weight-streaming loop.
- FP8 MLP layers process the full round in one pass.

**Prefill**
- A fused prefill attention kernel reads the paged INT8 cache directly, with no gather workspace.
- Short prompts no longer pay the wide-prefill route: a 55-token prompt is 2.5× faster.
- The NVFP4 MLP keeps gate and up in FP32 through SwiGLU on the wide route, matching the
  reference more closely.
- The Gated Delta Net recurrence is software-pipelined and reads pre-normalised inputs.

**Engine and memory**
- `--decode-share` interleaves decode rounds with long prefills by time share.
- Gated Delta Net scratch moved into the planned workspace; the remaining activation staging uses
  small fixed per-stream blocks instead of buffers that grow on demand.
- 2,048-token prefill chunks and CUDA Graphs fit alongside a 262,144-token INT8 KV pool with MTP up
  to K=5 and no device state slots; configurations that do not fit are refused at startup with the
  byte counts.
- An FP8 GDN convolution workspace under-allocation that could abort under load is fixed.

## Models

| Model | Weights | Artifact | Download and model card |
|---|---|---|---|
| Qwen3.6-27B | `groupwise-int` | `qwen3_6_27b.ninfer` | [Qwen3.6-27B](https://huggingface.co/neroued/Qwen3.6-27B-NInfer) |
| Qwen3.6-27B | `nvfp4` | `qwen3_6_27b_nvfp4.ninfer` | [Qwen3.6-27B NVFP4](https://huggingface.co/neroued/Qwen3.6-27B-nvfp4-NInfer) |
| Qwen3.8-27B | `groupwise-int` | `qwen3_8_27b.ninfer` | [Qwen3.8-27B](https://huggingface.co/neroued/Qwen3.8-27B-NInfer) |
| Qwen3.8-27B | `nvfp4` | `qwen3_8_27b_nvfp4.ninfer` | [Qwen3.8-27B NVFP4](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer) |
| Qwen3.6-35B-A3B | `groupwise-int` | `qwen3_6_35b_a3b.ninfer` | [Qwen3.6-35B-A3B](https://huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer) |

Artifacts contain the exact model weights, tokenizer, chat template, and required media frontend resources.

## Performance

Numbers are from the HTTP server on a Tesla V100-PCIe-32GB, CUDA 12.8, the published Qwen3.8-27B
NVFP4 artifact at the revision pinned in the quick start, INT8 group-64 KV. Each request is a cold
natural-text prompt of the stated size followed by 256 greedily decoded tokens; cells are means
over six requests (three from 89K).

Single-stream decode, tok/s (upstream port → this fork):

| Prompt tokens | MTP off | K=1 | K=3 | K=4 |
|---:|---|---|---|---|
| 60 | 29.4 → 33.8 | 49.2 → 55.7 | 65.5 → 72.7 | 63.3 → 74.7 |
| 1.1K | 29.2 → 33.6 | 49.1 → 55.5 | 67.5 → 72.9 | 69.1 → 80.7 |
| 5.9K | 28.0 → 33.2 | 47.2 → 54.9 | 66.0 → 75.3 | 67.8 → 81.1 |
| 24K | 24.3 → 32.1 | 41.8 → 53.9 | 57.2 → 70.0 | 56.8 → 75.7 |
| 89K | 18.8 → 27.9 | 31.8 → 47.2 | 41.4 → 57.9 | 41.4 → 60.9 |
| 172K | 14.1 → 24.2 | – | 31.9 → 53.5 | – |

Time per decode round, ms (independent of draft acceptance):

| Prompt tokens | MTP off | K=1 | K=3 | K=4 |
|---:|---|---|---|---|
| 60 | 34.0 → 29.6 | 36.9 → 32.6 | 41.3 → 37.1 | 45.4 → 39.0 |
| 24K | 41.2 → 31.1 | 43.5 → 33.4 | 49.6 → 39.5 | 54.6 → 41.6 |
| 89K | 53.1 → 35.9 | 58.0 → 38.4 | 65.4 → 45.8 | 71.4 → 48.4 |
| 172K | 71.0 → 41.3 | – | 87.6 → 51.7 | – |

Prefill, tok/s:

| Prompt tokens | Upstream port | This fork | Gain |
|---:|---:|---:|---:|
| 60 | 154 | 339 | 2.21× |
| 1,127 | 946 | 978 | 1.03× |
| 5,935 | 1,072 | 1,163 | 1.08× |
| 24,108 | 973 | 1,078 | 1.11× |
| 88,515 | 665 | 822 | 1.24× |
| 171,901 | 455 | 625 | 1.37× |

Draft acceptance in this test is 78–83% at K=1, 53–60% at K=3 and 45–53% at K=4, about the same
on both builds, and it varies with the text: at K=3 reference prose is accepted 59–74% of the
time and fiction 36–55%. Decode throughput depends on it, so every MTP figure in
[the full report](docs/v100-rework.md) is published with its acceptance rate.

The gain is smallest for short prompts on a single stream (8–20%) and largest at long context
and under concurrency (40–72%).

The upstream port's own published figures, measured with the `ninfer_bench` tool on a synthetic
high-acceptance corpus, remain in the [V100 port notes](docs/v100.md).

## Quick start

NInfer requires 64-bit Linux, a Tesla V100 with CUDA Toolkit 12.8, CMake 3.28 or newer, a C++20
host compiler, Ninja, `pkg-config`, FFmpeg
development libraries (`libavformat >= 60`, `libavcodec >= 60`, `libavutil >= 58`, and
`libswscale >= 7`), and `libcurl >= 7.85`. This port builds for `sm_70`.

Select CUDA 12.8 and Volta explicitly:

```bash
cmake -S . -B build-v100 -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=70
cmake --build build-v100 -j
```

The same five `.ninfer` artifacts use the public Engine/CLI/serving routes. See the
[V100 port notes](docs/v100.md) for qualification and the preferred-GPU launcher.

On Volta, loading an NVFP4 artifact repacks each dense MLP gate/up and down payload in place on the
GPU into the QPN fragment order used by the decode kernels. Row-scaled FP8 payloads are likewise
permuted into their native Volta QPN stream. The `.ninfer` file and host-side artifact bytes
are unchanged, the device allocation remains the same size, and temporary repack storage is
released before inference. These device-resident weights are therefore deliberately mutated at
load; the V100 path does not retain checkpoint-native immutable layout for those payloads.

Tests, benchmarks, and maintainer tools are excluded from the default build. There is no install
target or packaged binary distribution; run NInfer from its source build tree.

Download the artifact used by this example with the Hugging Face CLI:

```bash
hf download neroued/Qwen3.8-27B-nvfp4-NInfer \
  qwen3_8_27b_nvfp4.ninfer \
  --revision 11dbbbbbc33db198afe2f02c9232c771ff7031be \
  --local-dir models
```

The revision matters. This engine reads the v2 artifact format; the file currently at the head of
that Hugging Face repository is a later format and is rejected at startup with
`artifact magic is not NInfer v2`. The pinned revision loads in this build
(SHA-256 `552c374c685dce302603b95fbe940fb04243c0cd44c083efc644ad3d980d462c`).
The other model links in the table above have not been checked against this build.

Start a long-running text/vision server with one active-request lane and Device/Host
checkpoint retention:

```bash
./build/apps/ninfer-serve models/qwen3_8_27b_nvfp4.ninfer \
  --max-context 240000 \
  --prefill-chunk 2048 \
  --kv-capacity auto \
  --max-concurrency 1 \
  --kv-dtype int8 \
  --device-state-slots 1 \
  --host-state-slots 8 \
  --host-kv-mib 8192 \
  --spec mtp --draft-tokens 4 \
  --lm-head-draft \
  --preserve-thinking \
  --vision
```

Each request has a 240,000-token logical ceiling. The Device KV pool is sized from the memory left
after weights, workspace, state, Vision, and graph allocations. The cache tiers provide one Device
checkpoint slot, eight pinned Host State slots, and 8 GiB of pinned Host KV.

Send an OpenAI-style request:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [{"role": "user", "content": "Reply with one short sentence."}],
    "max_tokens": 64
  }'
```

Run a one-shot CLI request with a 32,768-token allocation:

```bash
./build/apps/ninfer models/qwen3_8_27b_nvfp4.ninfer \
  --prompt "Explain prefill and decode, then give a concise conclusion." \
  --max-context 32768 \
  --max-new 8192 \
  --kv-dtype fp8 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft
```

Answer content is written to stdout. Structured startup/runtime-error records and the CLI-owned
reasoning, timing, throughput, memory, and speculative-decoding report are written to stderr;
reasoning and the result report remain unprefixed product output. On a terminal, weight
materialization additionally uses one transient progress line. Redirected stderr receives only
persistent structured phase records, including rate-limited progress for long loads. Option and
local input errors remain direct command diagnostics. Use `--messages FILE` and `--vision` for
structured image/video input; see the [CLI guide](docs/cli.md) and
[committed examples](examples/cli/).

## Resource-aware long-context reuse

A reusable prefix checkpoint contains KV and the complete continuation state for its exact prompt
frontier. A Device-resident checkpoint resumes directly. Under pressure, the planner weighs Device
retention, pinned Host State/KV, and eviction by immediate restore work and later reuse cost. Active
requests retain their completion reservations.

See [Resource scheduling and context cache](docs/maintainer/resource-scheduling-and-context-cache.md)
for the algorithm and [Serve TTFT benchmark](tools/bench/ttft/) for public-HTTP coverage of hot
reuse, Host resume, eviction, shared prefixes, scheduling boundaries, and multimodal load.

## Evaluation

This section is about answer quality, not speed. The scores below are the original NInfer
project's published results, carried over unchanged. They were not re-measured for the V100 port
or for this fork, which changes kernels and scheduling but not the model artifacts.

Capability scores were measured through NInfer's OpenAI-compatible serving route with thinking
enabled, MTP3, and EvalScope 1.9.0 (0-shot, rule scoring, one sample per problem):

| Model profile | AIME 2025 | AIME 2026 | GPQA-Diamond | ERQA | RealWorldQA |
|---|---:|---:|---:|---:|---:|
| [Qwen3.6-27B groupwise-int](model-cards/Qwen3.6-27B-NInfer/README.md) | 86.67% | 93.33% | 86.87% | — | — |
| [Qwen3.6-27B NVFP4](model-cards/Qwen3.6-27B-nvfp4-NInfer/README.md) | 93.33% | 93.33% | 84.34% | — | — |
| [Qwen3.6-35B-A3B groupwise-int](model-cards/Qwen3.6-35B-A3B-NInfer/README.md) | 90.00% | 90.00% | 85.35% | — | — |
| [Qwen3.8-27B groupwise-int](model-cards/Qwen3.8-27B-NInfer/README.md) | 96.67% | 96.67% | 87.37% | 66.25% | 82.22% |
| [Qwen3.8-27B NVFP4](model-cards/Qwen3.8-27B-nvfp4-NInfer/README.md) | 96.67% | 96.67% | 90.40% | 66.25% | 83.53% |

The Qwen3.6 rows used temperature 0.6 and presence penalty 1.0; the Qwen3.8 rows used temperature
1.0 and presence penalty 0.0. Multimodal evaluation used `--vision` and an 81,920-token context
limit. Text evaluation used 262,144 tokens except Qwen3.8-27B NVFP4, which used 252,928 tokens for
its measured memory envelope. Each score is one sample per problem; model cards contain the
correct/total counts and evaluation notes.

## Startup notes

GPU residency is fixed at process startup. `--spec` selects speculative decoding residency, and
`--vision` selects Vision residency. DFlash is available for text-only Qwen3.6-35B-A3B execution,
and DFlash2 for Qwen3.8-27B.

## Docker

The image builds for `sm_70` with CUDA 12.8.
[`.github/workflows/image.yml`](.github/workflows/image.yml) builds it on pushes to `master` and
pushes it to GitHub Container Registry.

Run with Docker Compose on a host with the NVIDIA Container Toolkit:

```bash
mkdir -p models
hf download neroued/Qwen3.8-27B-nvfp4-NInfer qwen3_8_27b_nvfp4.ninfer \
  --revision 11dbbbbbc33db198afe2f02c9232c771ff7031be --local-dir models
docker compose up -d            # pulls the published image
docker compose up -d --build    # or builds it locally
```

[`docker-compose.yml`](docker-compose.yml) serves on port 8080 with a 262,144-token context, eight
request lanes, INT8 KV, and MTP with three draft tokens. `NINFER_GPU`, `NINFER_PORT`,
`NINFER_MODELS`, `NINFER_MAX_CONTEXT`, `NINFER_MAX_CONCURRENCY`, and `NINFER_DRAFT_TOKENS` override
the defaults.

## Kubernetes

[`deploy/kubernetes/ninfer.yaml`](deploy/kubernetes/ninfer.yaml) defines a model volume claim, a
one-replica Deployment that requests one `nvidia.com/gpu`, and a Service:

```bash
kubectl create namespace ninfer
kubectl -n ninfer apply -f deploy/kubernetes/ninfer.yaml
```

Place `qwen3_8_27b_nvfp4.ninfer` on the `ninfer-models` claim before the first start. The
Deployment uses the `Recreate` strategy because one V100 cannot hold two copies of the model, and
pulls `latest` on every restart, so `kubectl rollout restart deployment/ninfer` deploys the newest
published image.

## Capabilities and limits

All registered model IDs support:

- text generation with thinking and non-thinking prompt modes;
- image, multi-image, video, and mixed multimodal messages;
- chunked prefill, exact-batch CUDA Graph decode, and startup-bounded batched decode;
- MTP speculative decoding with draft windows from one to seven;
- DFlash2 masked-block speculative decoding for Qwen3.8-27B (from the upstream integration),
  draft windows from one to fifteen;
- BF16, INT8, and FP8 KV storage;
- offline causal-perplexity scoring;
- private and shared exact-prefix reuse with Device/Host State and KV retention;
- model-aware sampling defaults and explicit sampler overrides;
- OpenAI Responses Core, OpenAI Chat Completions, and Anthropic Messages, including streaming,
  tools, local response state, token counting, and usage accounting.

The 35B-A3B target additionally supports text-only DFlash with draft windows from one to fifteen.

The product boundary remains intentionally small:

- one Tesla V100 and one resident model per Engine;
- a startup-fixed capacity of one to eight active requests with bounded FIFO ingress;
- no request preemption, priority/QoS, active-request swapping, weight offload, multi-GPU, or
  distributed serving;
- one shared startup-fixed KV pool across active requests and retained prefixes;
- no runtime model discovery or unregistered checkpoint fallback;
- parsed tool calls are returned to the client; NInfer does not execute tools;
- the in-tree C++ headers are not distributed as an installed SDK.

`--max-context` is each sequence's logical limit. `--kv-capacity` sizes the shared Main Text KV pool
used by active requests and retained prefixes; `auto` resolves the largest legal capacity at
startup from the memory remaining after weights while keeping 1 GiB of sizing headroom. Explicit
capacities remain fixed for the process lifetime.

## Documentation

- [Documentation index](docs/README.md)
- [CLI](docs/cli.md)
- [HTTP serving](docs/serving.md)
- [V100 performance rework and benchmarks](docs/v100-rework.md)
- [V100 port notes](docs/v100.md)
- [Perplexity evaluation](docs/perplexity.md)
- [Resource scheduling and context cache](docs/maintainer/resource-scheduling-and-context-cache.md)
- [Serve TTFT benchmark](tools/bench/ttft/)
- [CLI examples](examples/cli/)
- [Contributing](CONTRIBUTING.md)

Run the relevant `--help` for the exact current option contract.

## License

NInfer is licensed under the [Apache License 2.0](LICENSE).

The published artifacts are derived from
[Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B),
[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), and
[Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B). The Qwen3.6-27B NVFP4 artifact
also uses the fixed packed weights from
[rdtand/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm](https://huggingface.co/rdtand/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm).
The Qwen3.8-27B NVFP4 artifact also uses the fixed mixed FP8/NVFP4 weights from
[unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4). These source
repositories are distributed under Apache-2.0. Vendored dependencies retain their own license files
under `third_party/`.
