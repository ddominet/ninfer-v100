# V100 performance rework

This document records what this fork changes relative to the upstream Tesla V100 port,
[geoffwatts/ninfer-v100](https://github.com/geoffwatts/ninfer-v100) at `b37d0dd`, and the measured
comparison between the two. The port's own notes and published figures are in
[V100 port](v100.md).

## Method

| Item | Value |
|---|---|
| GPU | Tesla V100-PCIe-32GB, nothing else running on the card |
| Toolkit | CUDA 12.8, `-DCMAKE_CUDA_ARCHITECTURES=70`, Release |
| Model | a Qwen3.8-27B artifact in the registered `qwen3.8-27b/nvfp4` format (not the published file), INT8 group-64 KV |
| Route | `ninfer-serve` over HTTP, OpenAI Chat Completions, greedy sampling |
| Single-stream test | one cold fixed prompt of the stated length, then 256 decoded tokens |
| Prompt | a summarisation instruction followed by synthetic filler tokens; the decoded output is ordinary English |
| Speculation | `--spec mtp --draft-tokens K --lm-head-draft`; "MTP off" omits `--spec` |
| Repetitions | one run per cell |

Both builds were started with the same flags:

```bash
ninfer-serve qwen3_8_27b_nvfp4.ninfer --kv-dtype int8 --max-concurrency 8 \
  --max-context 200704 --kv-capacity 200704 --prefill-chunk 2048 \
  --device-state-slots 0 --host-kv-mib 6144 --host-state-slots 16 --no-cuda-graph
```

The multi-stream rows use `--max-context 65536 --kv-capacity 196608 --device-state-slots 8`.

Reading the results:

- The context is 200,704 tokens, not 262,144. With these flags, `--kv-capacity 262144` and MTP
  enabled, the upstream build stopped at startup reporting 12.93 GB of runtime memory needed
  against 12.56 GB available.
- The prompt body is synthetic filler, not natural text. Draft acceptance on natural-language
  context may differ.
- Each cell is a single run on one prompt. Prefill repeated across the six K settings stayed
  within about 3% (7% at 1K).
- Draft acceptance moves a few points between runs because small rounding differences change the
  generated text. Tok/s at a given K carries that noise; time per round does not.
- The upstream port publishes 219 tok/s at K=1 with 99.2% acceptance. That figure comes from the
  `ninfer_bench` tool on a synthetic corpus and is described there as a ceiling. In this server
  test the same build decodes 33–51 tok/s at K=1 with 89–94% acceptance. The two are different
  workloads and tools; `ninfer_bench` was not run on this fork.

## Single-stream decode

Tokens per second, upstream → this fork.

| Context | MTP off | K=1 | K=2 | K=3 | K=4 | K=5 |
|---|---|---|---|---|---|---|
| 1K | 29.3 → 33.6 | 51.3 → 58.0 | 68.9 → 80.1 | 78.8 → 92.2 | 79.7 → 103.8 | 86.7 → 99.3 |
| 3K | 28.7 → 33.4 | 51.0 → 58.0 | 65.3 → 76.1 | 76.0 → 86.9 | 77.6 → 93.1 | 79.2 → 113.6 |
| 6K | 27.9 → 33.1 | 49.7 → 57.4 | 64.9 → 75.6 | 75.9 → 90.8 | 80.2 → 96.5 | 81.6 → 96.3 |
| 12K | 26.5 → 32.8 | 47.5 → 57.3 | 64.2 → 78.2 | 73.0 → 88.9 | 76.4 → 97.2 | 78.0 → 105.5 |
| 24K | 24.3 → 32.1 | 42.8 → 55.7 | 55.8 → 71.1 | 68.7 → 82.9 | 71.2 → 91.4 | 68.3 → 94.8 |
| 45K | 22.2 → 30.9 | 39.1 → 53.6 | 51.3 → 68.6 | 60.8 → 79.1 | 63.9 → 83.1 | 58.4 → 90.1 |
| 90K | 18.8 → 27.8 | 33.4 → 50.2 | 44.4 → 61.0 | 52.8 → 72.3 | 53.3 → 75.3 | 50.7 → 83.1 |
| 180K | 13.8 → 24.0 | – | – | 35.6 → 64.8 | – | – |

### Draft acceptance

Share of drafted tokens accepted by verification, upstream / this fork.

| Context | K=1 | K=2 | K=3 | K=4 | K=5 |
|---|---|---|---|---|---|
| 1K | 89% / 89% | 85% / 89% | 75% / 80% | 66% / 75% | 63% / 62% |
| 3K | 92% / 92% | 81% / 84% | 73% / 73% | 65% / 65% | 59% / 70% |
| 6K | 92% / 91% | 83% / 83% | 76% / 79% | 71% / 70% | 63% / 60% |
| 12K | 94% / 92% | 89% / 90% | 78% / 79% | 71% / 72% | 68% / 69% |
| 24K | 91% / 91% | 82% / 82% | 81% / 76% | 73% / 69% | 67% / 66% |
| 45K | 90% / 92% | 83% / 83% | 78% / 76% | 70% / 66% | 63% / 65% |
| 90K | 92% / 92% | 86% / 80% | 81% / 75% | 70% / 65% | 68% / 70% |
| 180K | – | – | 72% / 78% | – | – |

### Time per decode round

Milliseconds, upstream → this fork. This is independent of acceptance.

| Context | MTP off | K=1 | K=3 | K=5 |
|---|---|---|---|---|
| 1K | 34.2 → 29.8 | 37.0 → 32.7 | 41.5 → 37.0 | 48.2 → 41.3 |
| 12K | 37.7 → 30.5 | 41.2 → 33.8 | 45.9 → 38.2 | 56.6 → 42.6 |
| 45K | 45.1 → 32.4 | 48.9 → 36.2 | 55.2 → 41.7 | 71.4 → 47.7 |
| 90K | 53.1 → 36.0 | 58.1 → 38.3 | 65.3 → 45.4 | 87.7 → 54.4 |
| 180K | 72.4 → 41.7 | – | 88.7 → 51.9 | – |

### Memory throughput and tensor-core use

Derived, not measured directly: a verification pass reads 19.1 GB of weights and output head plus
the attention window, and touches 24.35 billion matrix elements per token. Draft-head work is not
counted, so these are lower bounds.

| Context | GB/s, MTP off | GB/s, K=3 | TFLOPS, MTP off | TFLOPS, K=3 | TFLOPS, K=5 |
|---|---|---|---|---|---|
| 1K | 560 → 643 | 462 → 517 | 1.51 → 1.74 | 4.99 → 5.58 | 6.43 → 7.51 |
| 12K | 517 → 639 | 424 → 510 | 1.49 → 1.84 | 4.88 → 5.86 | 5.94 → 7.90 |
| 45K | 456 → 635 | 373 → 494 | 1.53 → 2.13 | 5.00 → 6.63 | 5.80 → 8.69 |
| 90K | 415 → 613 | 338 → 486 | 1.63 → 2.41 | 5.31 → 7.64 | 5.93 → 9.56 |
| 180K | 345 → 599 | 282 → 481 | 1.69 → 2.93 | 5.51 → 9.40 | – |

Without MTP this fork holds 600–640 GB/s at every context length, while the upstream port falls
from 560 to 345 GB/s. Decode TFLOPS are low on both because decode is bound by memory, not compute.

## Prefill

Cold prompt, tokens per second.

| Prompt tokens | Upstream | This fork | Gain | Upstream TFLOPS | This fork TFLOPS |
|---:|---:|---:|---:|---:|---:|
| 55 | 146 | 363 | 2.49× | 7.1 | 17.7 |
| 97 | 251 | 426 | 1.69× | 12.2 | 20.8 |
| 153 | 359 | 482 | 1.34× | 17.5 | 23.5 |
| 321 | 632 | 652 | 1.03× | – | – |
| 671 | 806 | 845 | 1.05× | – | – |
| 1,018 | 989 | 1,046 | 1.06× | 48.4 | 51.1 |
| 3K | 1,084 | 1,157 | 1.07× | – | – |
| 6,023 | 1,117 | 1,197 | 1.07× | 55.7 | 59.7 |
| 12K | 1,077 | 1,167 | 1.08× | – | – |
| 24,020 | 997 | 1,099 | 1.10× | 53.3 | 58.7 |
| 45,020 | 869 | 991 | 1.14× | 50.0 | 57.0 |
| 90,023 | 662 | 817 | 1.23× | 43.9 | 54.2 |
| 180,022 | 439 | 607 | 1.38× | 36.9 | 51.0 |

A 180K-token prompt takes 410 s on the upstream port and 297 s here.

## Concurrent streams

Aggregate tokens per second across all streams, eight device state slots.

| Workload | Upstream | This fork | Gain |
|---|---:|---:|---:|
| 4 streams, K=1 | 148.5 | 184.1 | 1.24× |
| 8 streams, K=1 | 198.1 | 297.4 | 1.50× |
| 2 streams, K=3 | 116.9 | 144.4 | 1.24× |
| 4 streams, K=3 | 155.9 | 227.7 | 1.46× |
| 8 streams, K=3 | 183.9 | 285.8 | 1.55× |
| 2 streams × 45K context, K=3 | 92.5 | 127.4 | 1.38× |
| 4 streams × 40K context, K=3 | 94.0 | 154.6 | 1.64× |
| 4 streams, K=4 | 131.9 | 198.0 | 1.50× |
| 8 streams, K=4 | no result (requests returned HTTP 500 in the one run) | 262.6 | – |

## Where the remaining time goes

A kernel profile of this fork with MTP off at 1K context shows kernels occupying 97% of the round,
with the large weight-reading kernels at 700–865 GB/s (bytes read divided by kernel time). Short
context decode is therefore mostly bound by memory throughput, which is consistent with the gain
over the upstream port being smaller at short context (13–20%) than at long context or under
concurrency (30–82%).

At K=3 the same profile attributes about 4.9 ms of a 37.5 ms round to the MTP draft layer, whose
W8 kernels measure 270–400 GB/s in the linear Op benchmark. Its weights are read in checkpoint
row order, not a prepacked fragment order.

Draft windows of four and five tokens were faster than three for a single stream on this one
prompt, and three was faster than four at eight streams.

## Changes

### Decode attention (`src/ops/softmax_attention/dense/causal_cache/`)

- `direct_i8_volta.cuh` adds two tensor-core kernels over the paged INT8 group-64 cache for rounds
  of 1–8 tokens above a 2,048-key window. The one-token kernel moves key and value codes from
  global memory to registers to the `mma` fragment with no shared-memory staging. The 2–8 token
  kernel dequantises each 32-key tile once into a shared FP16 plane that every row tile reads,
  with loads issued one phase ahead. Before, each row tile fetched the keys and values itself: at
  eight tokens the card moved about 950 GB/s to deliver 158 GB/s of useful data.
- Rotated FP16 queries are prepared once per round by a separate kernel.
- `causal_direct_splits` picks keys per split from window length and round width, capped at 176
  splits.
- Both kernels take the stream index from the launch grid, so concurrent streams share one launch.

### Linear layers (`src/ops/linear/`, `linear_add/`, `linear_swiglu/`, input projections)

- `volta_qpn_width.h` defines one width policy: QPN tile variants for 1, 2, 3, 4, 6 and 8 row
  tiles cover up to 64 rows, wider rounds are chunked, and the CUTLASS prefill GEMM starts at 193
  rows. Rounds of 33 rows and more previously paid a weight dequantisation of about 1.1 ms per
  linear layer.
- `volta_fp16_stage.h` stages the round's activations once as FP16, in group-major order for the
  prepacked NVFP4 kernels, so each activation load covers one 128-byte line.
- FP8 MLP layers run the whole round in one chunk instead of four rows at a time.
- On the CUTLASS route the NVFP4 MLP keeps gate and up in FP32 through SwiGLU, in 1,024-column
  passes.
- The prepacked NVFP4 kernels issue a batch of code and scale loads before consuming any.

### Prefill and Gated Delta Net

- A fused prefill attention kernel reads the paged INT8 cache directly and removes the gather
  workspace; the flash query block is 256 tokens so 2,048-token chunks fit beside a 262,144-token
  KV pool.
- Prompts under 193 tokens run on the decode-width kernels in chunks of up to 64 rows.
- The direct GDN recurrence is software-pipelined, and from 64 tokens reads inputs normalised by a
  separate pass. Its scratch comes from the planned workspace arena.

### Engine

- `--decode-share` sets the share of time given to decode rounds while a long prefill is running.
- CUDA Graphs fit at 262,144 context.
- An FP8 GDN convolution workspace under-allocation is fixed.

### Runtime switches

These environment variables select between routes at startup. Defaults are the measured best.

| Variable | Default | Effect |
|---|---|---|
| `NINFER_VOLTA_STAGE_FP16` | on | stage activations as FP16 once per round |
| `NINFER_VOLTA_GROUP_MAJOR` | on | group-major activation layout for prepacked NVFP4 kernels |
| `NINFER_VOLTA_PAGED_FA` | on | fused paged INT8 prefill attention |
| `NINFER_GDN_PIPELINED` | on | software-pipelined GDN recurrence |
| `NINFER_NVFP4_DOWN_T1_GEMV` | off | one-token NVFP4 GEMV route for the MLP down projection |
| `NINFER_GRAPH_ALLOWANCE_MIB` | computed | overrides the memory reserved for MTP CUDA Graphs |

## Qualification

Each changed route is checked against the naive FP32 oracle in the Op tests at the route
boundaries (1, 8, 16, 17, 32, 33, 49, 64, 65, 192, 193 rows and wide prefill), and the attention
tests cover long windows on the new kernels. One test inherited from the upstream port,
`gdn_gating_proj`, fails on a single element that is 0.24% off against a 0.2% bound on a
BF16-staged route this fork does not change.
