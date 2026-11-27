# NInfer for Tesla V100 (v3.0)

A C++/CUDA inference engine for **Qwen3.8-27B** tuned specifically for a single NVIDIA Tesla V100 (`sm_70`, 32 GB). It serves text, image, and video input through a local CLI or OpenAI- and Anthropic-compatible HTTP APIs, built around one GPU, one resident model, and 1–8 concurrent requests.

> **Scope note:** While the underlying engine retains support for the other NInfer Qwen artifacts, all v1.0, v2.0, and v3.0 kernel, scheduler, and speculative-decoding optimizations in this repository are **specifically built and tuned for `Qwen3.8-27B` NVFP4 / NVFP4-QAD on Tesla V100**. Other models are not tuned or benchmarked here.

This repository is the **v3.0** release of [ddominet/ninfer-v100](https://github.com/ddominet/ninfer-v100), originally forked from [geoffwatts/ninfer-v100](https://github.com/geoffwatts/ninfer-v100). Version 1.0 rebuilt the V100 attention, prefill, and long-context paths; version 2.0 accelerated DFlash2 across 1–8 streams. **Version 3.0 adds native NVIDIA NVFP4 QAD and NInfer v3 artifact support (cutting weight VRAM by 3.2 GiB while accelerating decode by 9–19%), eliminates the `~150K` long-context prefix-cache miss stall, enables FP16 GDN recurrent states by default on Volta, adds n-gram copy speculation and fused zero-logit-traffic greedy argmax output heads, and adds Volta QPN prepacking for Q4/Q5/Q6 groupwise weights.**

> **Recommended v3.0 setup:** Use the NVIDIA NVFP4 QAD v3 artifact (`qwen3_8_27b_nvfp4_qad.ninfer`) with `--spec dflash2 --draft-tokens 7 --lm-head-draft --kv-dtype int8 --max-context 262144 --kv-capacity 262144 --device-state-slots 16 --host-kv-mib 6144 --host-state-slots 16 --decode-share 50 --no-cuda-graph` (see [Serve](#serve-recommended-v30-production-flags) below). Existing NInfer v2 artifacts (`qwen3_8_27b_nvfp4.ninfer`) remain fully supported.

---

## Benchmarks (Tesla V100-PCIe-32GB)

### v3.0 NVFP4 QAD vs v2 NVFP4 (same v3.0 binary, DFlash2 `K=7`, INT8 KV, `--no-cuda-graph`)

Measured on the same **Tesla V100-PCIe-32GB** (CUDA 12.8) over HTTP with identical prompts and flags:

| Metric | `qwen3_8_27b_nvfp4.ninfer` (v2) | **`qwen3_8_27b_nvfp4_qad.ninfer` (v3 QAD)** | Delta |
|---|---:|---:|---:|
| **Model Weights VRAM** (`--spec dflash2`) | `20.0 GiB` | **`16.8 GiB`** | **-3.2 GiB (-16%)** |
| **Free VRAM (`64K` KV, `C=4`)** | `7.63 GiB` | **`10.8 GiB`** | **+3.17 GiB** |
| **Free VRAM (`262K` KV, `C=8`, `0` dev states)** | `1.37 GiB` | **`4.55 GiB`** | **+3.18 GiB** |
| **Free VRAM (`262K` KV, `C=8`, `16` dev states)** | does not fit | **`2.75 GiB`** | **16 GPU states fit** |
| **Code Greedy Decode (`56.7%` DFlash2 accept)** | `140.2 tok/s` | **`152.5 tok/s`** | **+8.8%** |
| **Prose Sampled Decode (`temp=0.7`)** | `67.8 tok/s` | **`80.4 tok/s`** | **+18.6%** |
| **17.5K Prefill + Needle Recall** | `1.18k tok/s` (`14.9s`) | **`1.19k tok/s` (`14.8s`)** | **Exact recall** |
| **Warm Multi-turn Code (`6.7K–17K` ctx, `99.7%` cache)** | — | **`211–254 tok/s`** | `181–200 ms` TTFT |
| **4-Stream / 8-Stream Concurrent (`262K` prod config)** | — | **`188.8` / `203.3 tok/s`** | Peak `365.4 tok/s` |

### NVFP4 QAD across decoding modes (V100-PCIe-32GB, `64K` INT8 KV)

Greedy outputs are bit-identical across all three modes:

| Mode | Weights VRAM | Code Greedy | Sampled Prose | Thinking Math |
|---|---:|---:|---:|---:|
| **No-spec (`--no-cuda-graph`)** | `14.6 GiB` | `39.7 tok/s` | `39.6 tok/s` | `39.7 tok/s` |
| **MTP (`--spec mtp --draft-tokens 4`)** | `15.0 GiB` | `98.6 tok/s` | `78.8 tok/s` | `123.9 tok/s` |
| **DFlash2 (`--spec dflash2 --draft-tokens 7`) + CUDA Graph** | `16.8 GiB` | **`153.1 tok/s`** | `77.6 tok/s` | **`199.1 tok/s`** |

### Long-context `150K–203K` prefix-cache replay & Host-KV swap (v3.0, V100-PCIe-32GB)

| Scenario (`--kv-dtype int8`, `--host-kv-mib 6144` / `5120`) | Prompt tokens | Cached tokens | Warm TTFT | Queue / restore |
|---|---:|---:|---:|---:|
| Sequential replay (`185K → 203K`, 6 warm turns `+2.9K`/turn) | `188K–203K` | **`98.5–98.6%`** | `7.6–8.0 s` | `25–30 ms` |
| Resume after competing `15K` session | `202,964` | **`100.0%` (`202,935`)** | **`761 ms`** | `27.7 ms` |
| Tight-KV (`163,840`) Host-KV spill & restore (2 cycles) | `152K / 155K` | **`98.6%`** | `5.4 / 5.6 s` | `178 / 311 ms` |
| Full `152K` (`2,379`-page) Host-KV swap (`5 GiB` Host KV) | `152,272` | **`100.0%` (`152,243`)** | **`1.7 s`** | `1.0 s` (PCIe H2D) |

### v2.0 Cross-Engine Baseline Sweep (Tesla V100-PCIe-32GB, v2 NVFP4 artifact)

Every server below was measured on the same **Tesla V100-PCIe-32GB** (CUDA 12.8) over HTTP using the same cold natural-text prompts (reference prose, fiction, and source code from `eval/corpora/perplexity-1m`), greedy decoding, thinking off, and 256 output tokens per request (6 requests/cell up to 24K, 3 requests/cell at 90K–180K).

Engines compared:
- **[geoffwatts/ninfer-v100](https://github.com/geoffwatts/ninfer-v100)** — original V100 port (`b37d0dd`), Qwen3.8-27B NVFP4, INT8 KV, MTP K=3 / K=1.
- **[tuxKOH/ninfer-v100xN](https://github.com/tuxKOH/ninfer-v100xN)** — V100 fork, Qwen3.8-27B NVFP4, INT8 KV, MTP K=3 / K=1.
- **[ddominet/ninfer-v100](https://github.com/ddominet/ninfer-v100) (v1.0)** — previous release of this fork, Qwen3.8-27B NVFP4, INT8 KV, MTP K=3 / K=1 and unoptimized DFlash2.
- **[inscora/llama-superfast-v100-BWE-q8-dflash2](https://github.com/inscora/llama-superfast-v100-BWE-q8-dflash2)** — V100 llama.cpp fork with Unsloth `UD-Q4_K_XL`, z-lab DFlash2 `Q4_K_M` (`K=7`), Q8_0 KV.
- **ddominet/ninfer-v100 (v2.0)** — Qwen3.8-27B NVFP4, INT8 KV, `--spec dflash2 --draft-tokens 7 --lm-head-draft --decode-share 50 --no-cuda-graph`.

#### Single-stream decode (tok/s, higher is better)

| Prompt context | geoffwatts (MTP3) | tuxKOH (MTP3) | ddominet v1.0 (MTP3) | ddominet v1.0 (DF7) | llama-superfast (DF7) | **v2.0 (DF7)** |
|---:|---:|---:|---:|---:|---:|---:|
| ~60 | 65.5 | 70.4 | 72.7 | 76.8 | 83.3 | **91.6** |
| 1K | 67.5 | 75.1 | 72.9 | 83.1 | 87.2 | **100.8** |
| 6K | 66.0 | 75.5 | 75.3 | 72.8 | 84.4 | **90.8** |
| 24K | 57.2 | 69.9 | 70.0 | 68.3 | 78.1 | **81.0** |
| 90K | 41.4 | 56.9 | 57.9 | 51.3 | 64.4 | **65.3** |
| 180K | 31.9 | 48.5 | 53.5 | 44.6 | 51.8 | **54.3** |

#### Prompt processing / prefill (tok/s, higher is better)

| Prompt context | geoffwatts | tuxKOH | ddominet v1.0 | llama-superfast | **v2.0 (DF7)** |
|---:|---:|---:|---:|---:|---:|
| ~60 | 152.1 | 155.1 | 336.8 | 388.6 | **415.6** |
| 1K | 949.0 | 955.2 | 976.0 | 975.1 | **1046.1** |
| 6K | 1075.4 | 1082.3 | 1160.3 | 1121.9 | **1211.5** |
| 24K | 974.2 | 1009.1 | 1076.3 | 1052.6 | **1118.2** |
| 90K | 666.2 | 779.3 | 820.9 | 814.0 | **834.5** |
| 180K | 454.2 | 595.2 | 622.6 | 620.5 | **627.0** |

#### Multi-stream aggregate decode (tok/s, mean of 3 runs, higher is better)

| Streams | geoffwatts (MTP3 / K1) | tuxKOH (MTP3 / K1) | llama-superfast (DF7) | ddominet v1.0 (MTP3 / K1) | ddominet v1.0 (DF7) | v2.0 (MTP3) | **v2.0 (DF7)** |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 57.5 / 45.0 | 64.2 / 50.2 | 76.8 | 66.4 / 53.1 | 65.3 | 71.2 | **82.2** |
| 2 | 92.8 / 80.0 | 101.3 / 88.5 | 120.4 | 113.2 / 95.6 | 96.2 | 122.3 | **137.0** |
| 4 | 109.1 / 125.3 | 122.3 / 135.7 | 135.0 | 157.3 / 160.0 | 95.2 | 160.9 | **176.0** |
| 8 | 129.6 / 155.7 | 145.5 / 175.8 | 151.7 | 211.8 / 243.3 | 121.3 | 245.9 | **252.1** |

#### Mixed-arrival Poisson chat (`rate=0.5 req/s`)

| Engine | Makespan | Aggregate throughput | Errors |
|---|---:|---:|---:|
| llama-superfast (DF7) | 152.4 s | 166.3 tok/s | 0 |
| v2.0 (MTP3) | 104.5 s | 242.1 tok/s | 0 |
| **v2.0 (DF7)** | **100.3 s** | **248.2 tok/s** | **0** |

#### Context × concurrency grid (total wall time in seconds, lower is better)

End-to-end wall time (cold prefill + 256 decoded tokens per stream) across 14 context/concurrency cells (`v2.0 DF7` wins **14/14 cells**):

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

### New in v3.0 (from `ddominet/ninfer-v100` v2.0)

#### Added
- **NVIDIA NVFP4 QAD & NInfer v3 artifact support:**
  - Full NInfer v3 container reader in `src/artifact/reader.cpp`: parses v3 JSON directories, normalizes `qwen3.8-27b-nvfp4-qad` / `qwen3.8-27b-nvfp4` identities, remaps single and fused v3 parameter bindings, resolves `uses[].auxiliaries.activation_input_divisor` to canonical divisor tensors, and validates/deduplicates redundant sibling slice divisors (`mlp/up`, `gdn/key|value|z`, `attention/key|gate|value`, shared `mtp/final_hidden`).
  - Mixed per-layer `NVFP4`/`FP8` GDN and MLP weight bindings, split (`a_projection`/`b_projection`) and fused (`a_b_projection`) GDN control projections, and `NVFP4` `text/output_head` loading with load-time Volta QPN prepacking (`src/targets/qwen3_6_27b/impl/load/bindings.cpp`, `variant.cpp`).
  - Native Volta `sm_70` NVFP4 GDN input kernels: QPN prepacked decode/small-T & scattered-column projections (`nvfp4_gdn_input_volta_qpn.cu`) and CUTLASS `sm_70` prefill (`nvfp4_gdn_input_cutlass_sm70.cu`).
  - Native NVFP4 `output_head` (`[248320, 5120]`) across `linear`, `linear_add`, and `linear_topk` (`HeadProfile::Nvfp4Full`) with chunked QPN dispatch (avoiding a 2.54 GB FP16 dequant buffer).
  - Support for NInfer v3 Qwen3.8 reasoning-effort `chat_template.jinja` and semantically equivalent `tokenizer_config.json` template validation.
  - ModelOpt NVFP4 QAD converter (`tools/convert/qwen3_8_27b/convert_nvfp4_qad.py`) for converting `nvidia/Qwen3.8-27B-NVFP4-QAD` (exact `1/weight_scale_2` & `1/input_scale` inversion, exact E4M3 rescaling across differing fused halves, scalar FP8 scale broadcast, `W8G32_F16S`/`FP8` embeddings, optional DFlash2 bundle) plus unit tests (`tests/convert/test_qwen38_nvfp4_qad_converter.py`).
- **Fused zero-logit-traffic greedy argmax `lm_head`:**
  - Warp-reduced fused greedy argmax epilogues directly inside Volta QPN and stream-HMMA output-head kernels for **NVFP4**, **FP8**, and **Q6** heads (`launch_*_argmax_fp16`), eliminating full-vocabulary `[248320]` logit round-trips during greedy decode and target verification, paired with a `raw_greedy` speculative verification fast path.
- **N-gram copy speculation (`--ngram-draft-tokens`, `--ngram-min-match`):**
  - Zero-model-overhead prompt and tool-output n-gram copy proposer (`ngram_proposer.h`, `ngram_sources.h`) integrated with DFlash2 (defaults to up to 15 draft tokens when `--spec dflash2` is active), including automatic de-numbering of line-numbered tool outputs (`cat -n`, `nl`, `L12:`), adaptive backoff/partner balancing, and unit tests (`test_ngram_proposer.cpp`).
- **FP16 GDN recurrent state (`--gdn-state-fp16`, default ON on Volta):**
  - Stores Gated Delta Net recurrent states in FP16 with FP32 accumulation (`recurrent.cuh`, `recurrent.cu`, `linear_attention_state.cpp`), halving GDN state memory (~114 MiB/slot) and doubling recurrent memory bandwidth—unlocking `--device-state-slots 16` alongside full 262K INT8 KV on a 32 GB V100.
- **Volta QPN prepacking & stream-HMMA for groupwise Q4 / Q5 / Q6 weights:**
  - Load-time in-place QPN prepacking (`q5_prepack_qpn_sm70`, `q6_prepack_qpn_sm70`), unified multi-tile groupwise QPN (`volta_qpn_groupwise.cu`) and streaming HMMA (`volta_rowsplit_stream_hmma.cu`) kernels, Q4 `linear_add` CUTLASS/QPN dispatch (`src/ops/linear_add/q4/`), Q4 embedding gather (`embed_gather_q4_launch`), and unit tests (`test_volta_qpn_prepacked.cpp`).

#### Fixed
- **`~150K` long-context prefix-cache miss & prefill stall:**
  - Replaced quadratic `O(P²)` overlap validation, retained-run scans, and sorted `vector::insert` shifts in `HostKVArena::can_allocate_after_suballocation_releases` (`src/core/host_kv_arena.cpp`) with `O(P log P + F)` sorting and linear two-pointer merge-coalescing (verified at 150K / 2,344 pages in `<1 ms`).
  - Replaced the fixed `5 ms` planner timeout in `MaterializationPlanner` with a two-phase dynamic budget (`250 ms` seeding cap while promising reuse candidates lack non-maximal seeds, tightening to `5 ms` for refinement), expanded frontier/target arena capacities to `2 * candidates.size() + kTargetBudget`, skipped evicting purely Host-resident shared prefixes when there is no Host deficit in `PressurePlanner::root_maximal_target`, and added a post-search `root_maximal_target` safety net so materializable reuse candidates never fall back to zero-reuse root.
- **Checkpoint divergence and capture reclamation bugs:**
  - Tracked positively proven private checkpoint divergence (`unreachable`) across `ResourceManager`, `ContextPortfolioValue`, `MaterializationPlanner`, and `SharedCapturePlanner` so diverged checkpoints carry zero transition loss.
  - Allowed physically blocked shared captures to enumerate reclamation victims without prior `pressure_evidence`, and routed physically blocked private-only captures through `SharedCapturePlanner` (`private_reclamation_only = true`) to reclaim the lowest-loss idle owner.
  - Added 4 regression tests in `tests/test_resource_manager.cpp` and a 150K-scale arena test in `tests/test_kv_cache.cpp`.
- **Volta NVFP4 GDN `conv_record` workspace bounds:**
  - Fixed `gdn_input_proj_conv_record_workspace_capacity_bytes` passing `std::max(min_width, 4) > max_width` under Volta's `Materialized` NVFP4 schedule for widths `< 4`.

#### Changed
- Updated recommended Qwen3.8-27B artifact and deployment defaults (`README.md`, `docker-compose.yml`, `deploy/kubernetes/ninfer.yaml`) to `qwen3_8_27b_nvfp4_qad.ninfer` with `--device-state-slots 16` (keeping `--max-context 262144 --kv-capacity 262144 --host-kv-mib 6144 --host-state-slots 16`).
- Cleaned up repository `.gitignore` and documentation references.

### In v2.0 (from `ddominet/ninfer-v100` v1.0)

Version 2.0 turned **DFlash2** into the fastest path for both single-stream and 8-stream serving while cutting target-model round latency and kernel-launch overhead across the board (`B=1` round time down to `35.58 ms` with `831` kernels/round and only `0.187 ms` eager launch gap under `--no-cuda-graph`):

- **Full DFlash2 draft stack on Volta:** load-time QPN prepacking for all DFlash2 W8 draft weights, `Split2`/`Split3` QPN kernels, coalesced 8-warp double-buffered HMMA dynamic grouped convolution (~2.5× faster), coalesced half-split Q4 proposal head with register bitonic Top-K, grouped-query sliding-window draft attention, 2-CTA/SM 6–8 token verify attention over paged INT8 KV, and host-mapped token embedding / selector codebooks.
- **Adaptive multi-stream DFlash2 runtime:** online-fitted `DFlashCostModel` and kept-column verify compaction (252 tok/s at 8 streams), sliding-window prefill skip, high-priority main stream + low-priority `gdn_fold_stream_` overlap, vectorized penalized argmax, and improved short-prompt batching / decode-credit accounting.
- **Target compute & launch-overhead elimination (`-298` kernel launches/round):** SM70 CUTLASS W8/NVFP4/FP8 prefill GEMMs, fused target RMSNorm+RoPE, fused NVFP4/FP8 MLP pipeline, high-occupancy `gdn_norm_gating_27`, and direct FP16 outputs from GDN/attention norms into QPN/TC kernels.

### In v1.0 (from `geoffwatts/ninfer-v100`)

Version 1.0 rebuilt the core V100 attention, linear, prefill, and memory paths: tensor-core paged INT8 decode attention, extended Volta QPN decode-width kernels through 64 rows, fused paged INT8 prefill attention, software-pipelined GDN recurrence, `--decode-share` decode/prefill interleaving, and reliable 256K INT8 KV sizing inside 32 GB VRAM.

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

At startup on Volta, NVFP4, FP8, W8, Q4, Q5, and Q6 weights are repacked in place on the GPU into native Volta QPN fragment layouts. The `.ninfer` file on disk is untouched and temporary repack memory is freed before serving begins.

### Download the model

**Recommended (v3.0 — NVIDIA NVFP4 QAD, `16.8 GiB` VRAM):**

```bash
hf download WarlaxZ/Qwen3.8-27B-NVFP4-QAD-NInfer-V100 \
  qwen3_8_27b_nvfp4_qad.ninfer \
  --local-dir models
```

**Or convert directly from NVIDIA ModelOpt (`nvidia/Qwen3.8-27B-NVFP4-QAD`):**

```bash
python3 -m tools.convert.qwen3_8_27b.convert_nvfp4_qad \
  --model /path/to/Qwen3.8-27B-base-bf16 \
  --quantized-model /path/to/Qwen3.8-27B-NVFP4-QAD \
  --dflash2-model /path/to/Qwen3.8-27B-DFlash2 \
  --out models/qwen3_8_27b_nvfp4_qad.ninfer
```

**Alternative (NInfer v2 NVFP4 artifact, `20.0 GiB` VRAM — remains supported):**

```bash
hf download neroued/Qwen3.8-27B-nvfp4-NInfer \
  qwen3_8_27b_nvfp4.ninfer \
  --revision 11dbbbbbc33db198afe2f02c9232c771ff7031be \
  --local-dir models
```

### Serve (recommended v3.0 production flags)

For the best all-around performance on a V100-32GB (single-stream decode, full 256K context, multi-turn GPU state caching, and up to 8 concurrent sessions):

```bash
./build-v100/apps/ninfer-serve models/qwen3_8_27b_nvfp4_qad.ninfer \
  --host 0.0.0.0 --port 8080 \
  --max-context 262144 \
  --kv-capacity 262144 \
  --prefill-chunk 2048 \
  --max-concurrency 8 \
  --device-state-slots 16 \
  --host-kv-mib 6144 \
  --host-state-slots 16 \
  --decode-share 50 \
  --no-cuda-graph \
  --kv-dtype int8 \
  --spec dflash2 --draft-tokens 7 \
  --lm-head-draft
```

Why these options matter:
- `qwen3_8_27b_nvfp4_qad.ninfer`: uses `16.8 GiB` VRAM (`-3.2 GiB` vs v2) and decodes `9–19%` faster on V100 via Volta QPN prepacked NVFP4 GDN and output-head kernels.
- `--spec dflash2 --draft-tokens 7 --lm-head-draft`: enables DFlash2 + automatic n-gram copy speculation (`--ngram-draft-tokens 15` default).
- `--kv-dtype int8 --max-context 262144 --kv-capacity 262144`: full 256K INT8 paged KV pool inside 32 GB VRAM.
- `--device-state-slots 16`: keeps all 16 private continuation GDN states (FP16 by default on Volta) directly in GPU VRAM for zero-copy warm-turn reuse while leaving `2.75 GiB` VRAM headroom. (Use `--device-state-slots 0` if serving the larger `20.0 GiB` v2 artifact at 262K KV.)
- `--host-kv-mib 6144 --host-state-slots 16`: pinned host RAM overflow tier so alternating `150K–256K` sessions restore in `~1–2 s` over PCIe instead of cold-prefilling; set `--host-kv-mib 0 --host-state-slots 0` if you prefer to save host RAM on single-session workloads.
- `--max-concurrency 8 --decode-share 50 --no-cuda-graph`: supports up to 8 concurrent sessions with responsive decode/prefill interleaving without spending `~2 GiB` VRAM on CUDA graph capture.

Add `--vision` if you need image/video input (use `--kv-capacity auto` when enabling `--vision`).

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
./build-v100/apps/ninfer models/qwen3_8_27b_nvfp4_qad.ninfer \
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
hf download WarlaxZ/Qwen3.8-27B-NVFP4-QAD-NInfer-V100 qwen3_8_27b_nvfp4_qad.ninfer \
  --local-dir models
docker compose up -d            # pull published image
docker compose up -d --build    # or build locally
```

[`docker-compose.yml`](docker-compose.yml) defaults to the v3.0 NVFP4 QAD + DFlash2 configuration (`262,144` context, 8 streams, 16 GPU state slots, INT8 KV, `--spec dflash2 --draft-tokens 7`). Override via `NINFER_MODEL`, `NINFER_GPU`, `NINFER_PORT`, `NINFER_MODELS`, `NINFER_MAX_CONTEXT`, `NINFER_MAX_CONCURRENCY`, `NINFER_DEVICE_STATE_SLOTS`, `NINFER_SPEC`, and `NINFER_DRAFT_TOKENS`.

### Kubernetes

[`deploy/kubernetes/ninfer.yaml`](deploy/kubernetes/ninfer.yaml) provides a PVC, single-GPU `Recreate` Deployment, and Service:

```bash
kubectl create namespace ninfer
kubectl -n ninfer apply -f deploy/kubernetes/ninfer.yaml
```

Place `qwen3_8_27b_nvfp4_qad.ninfer` on the `ninfer-models` PVC before starting the pod.

---

## Capabilities & limits

Supported on Qwen3.8-27B:
- NInfer v3 (`qwen3_8_27b_nvfp4_qad.ninfer`) and v2 (`qwen3_8_27b_nvfp4.ninfer`) artifacts, plus ModelOpt NVFP4 QAD conversion.
- Text generation (thinking and non-thinking modes) and multimodal image/video input (`--vision`).
- Chunked prefill, `--decode-share` decode/prefill interleaving, and 1–8 concurrent request lanes.
- DFlash2 masked-block speculative decoding (`--spec dflash2 --draft-tokens 7`, with n-gram copy speculation) and MTP (`--spec mtp`).
- INT8, FP8, and BF16 paged KV cache with exact-prefix reuse across Device and pinned Host tiers (`O(P log P)` arena planning through 256K).
- OpenAI Chat Completions, OpenAI Responses, and Anthropic Messages APIs (streaming, tool calls, token counting, usage accounting).

Intentional scope boundaries:
- Single Tesla V100 and one resident model per process (no multi-GPU or weight offload).
- Fixed startup capacity of 1–8 active requests with bounded FIFO ingress.
- Optimizations are tailored specifically for **Qwen3.8-27B NVFP4 / NVFP4-QAD**.

---

## Credits

- **[Neroued/ninfer](https://github.com/Neroued/ninfer)** — original NInfer engine, artifact container, serving stack, and published model artifacts.
- **[geoffwatts/ninfer-v100](https://github.com/geoffwatts/ninfer-v100)** — original Tesla V100 (`sm_70`) port, software NVFP4 on Volta tensor cores, quadpair-split-N kernels, load-time prepacking, and INT8 paged KV cache.
- **[NVIDIA](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4-QAD)** & **[WarlaxZ](https://huggingface.co/WarlaxZ/Qwen3.8-27B-NVFP4-QAD-NInfer-V100)** — Qwen3.8-27B NVFP4 QAD quantization and V100 NInfer v3 artifact packaging.
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

Apache License 2.0 ([`LICENSE`](LICENSE)). Model artifacts are derived from [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), [nvidia/Qwen3.8-27B-NVFP4-QAD](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4-QAD), and [unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4) (Apache-2.0). Vendored dependencies retain their own licenses under `third_party/`.
