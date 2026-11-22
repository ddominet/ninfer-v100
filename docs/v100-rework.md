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
| Model | [neroued/Qwen3.8-27B-nvfp4-NInfer](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer) at revision `11dbbbbbc33db198afe2f02c9232c771ff7031be`, sha256 `552c374c…d462c`, INT8 group-64 KV |
| Builds | upstream `b37d0dd` and this fork, built the same way, run one after the other on the same card |
| Route | `ninfer-serve` over HTTP, OpenAI Chat Completions, greedy sampling, thinking off |
| Client | [`tools/bench/run_serve_context_sweep.py`](../tools/bench/run_serve_context_sweep.py) |
| Request | a document slice of the stated size plus a task, cold (no prefix reuse), then 256 decoded tokens |
| Text | natural text from the committed corpus in `eval/corpora/perplexity-1m`: reference prose (WikiText), fiction (PG-19) and source code |
| Requests per cell | six up to 24K tokens (three domains × two tasks), three from 89K (one per domain) |
| Speculation | `--spec mtp --draft-tokens K --lm-head-draft`; "MTP off" omits `--spec` |
| Figures | what the server reports in the response `timings` object; cells are means over the requests |

Both builds were started with the same flags:

```bash
ninfer-serve qwen3_8_27b_nvfp4.ninfer --kv-dtype int8 --max-concurrency 8 \
  --max-context 200704 --kv-capacity 200704 --prefill-chunk 2048 \
  --device-state-slots 0 --host-kv-mib 6144 --host-state-slots 16 --no-cuda-graph
```

The concurrent-stream rows use `--max-context 65536 --kv-capacity 196608 --device-state-slots 8`.

Reading the results:

- "Prompt tokens" is the size the server counted. The 60-token row is six standalone prompts of
  52–66 tokens with no document.
- Tok/s with MTP depends on the text: acceptance at K=3 ranges from 36% to 74% across the
  requests, and the two builds do not always generate the same continuation, because small
  rounding differences change a greedy choice. Time per round does not depend on acceptance and is
  the like-for-like engine figure.
- CUDA Graphs were off for both builds (`--no-cuda-graph`). With graphs on and the single-stream
  flags above at K=5, the upstream build ended at startup with `std::bad_alloc` while preparing
  graphs; this fork started.
- Both servers ran with a 200,704-token context. A 262,144-token context was not benchmarked.
- The upstream port publishes 219 tok/s at K=1 with 99.2% acceptance. That figure comes from the
  `ninfer_bench` tool on a synthetic corpus and is described there as a ceiling. In this server
  test the same build decodes 32–49 tok/s at K=1 with 78–83% acceptance. The two are different
  workloads and tools; `ninfer_bench` was not run on this fork.

## Single-stream decode

Tokens per second, upstream → this fork, mean over the requests.

| Prompt tokens | Requests | MTP off | K=1 | K=2 | K=3 | K=4 | K=5 |
|---:|---:|---|---|---|---|---|---|
| 60 | 6 | 29.4 → 33.8 | 49.2 → 55.7 | 60.0 → 66.7 | 65.5 → 72.7 | 63.3 → 74.7 | 61.3 → 72.1 |
| 1.1K | 6 | 29.2 → 33.6 | 49.1 → 55.5 | 60.8 → 67.7 | 67.5 → 72.9 | 69.1 → 80.7 | 66.1 → 78.2 |
| 5.9K | 6 | 28.0 → 33.2 | 47.2 → 54.9 | 58.8 → 68.1 | 66.0 → 75.3 | 67.8 → 81.1 | 67.4 → 81.2 |
| 24K | 6 | 24.3 → 32.1 | 41.8 → 53.9 | 50.9 → 64.8 | 57.2 → 70.0 | 56.8 → 75.7 | 54.7 → 75.0 |
| 89K | 3 | 18.8 → 27.9 | 31.8 → 47.2 | 37.9 → 54.9 | 41.4 → 57.9 | 41.4 → 60.9 | 35.0 → 54.7 |
| 172K | 3 | 14.1 → 24.2 | – | – | 31.9 → 53.5 | – | – |

Gain of this fork over upstream on those means:

| Prompt tokens | MTP off | K=1 | K=2 | K=3 | K=4 | K=5 |
|---:|---|---|---|---|---|---|
| 60 | 1.15× | 1.13× | 1.11× | 1.11× | 1.18× | 1.18× |
| 1.1K | 1.15× | 1.13× | 1.11× | 1.08× | 1.17× | 1.18× |
| 5.9K | 1.19× | 1.16× | 1.16× | 1.14× | 1.20× | 1.20× |
| 24K | 1.32× | 1.29× | 1.27× | 1.22× | 1.33× | 1.37× |
| 89K | 1.48× | 1.49× | 1.45× | 1.40× | 1.47× | 1.57× |
| 172K | 1.72× | – | – | 1.68× | – | – |

With MTP off the six requests of a cell agree within 0.2 tok/s (1.1 tok/s at 172K). With MTP on
the spread across requests is wide. At K=3:

| Prompt tokens | Upstream min–max | This fork min–max |
|---:|---|---|
| 60 | 55.0–71.2 | 61.1–82.8 |
| 1.1K | 55.1–84.8 | 56.7–91.5 |
| 5.9K | 52.4–77.7 | 56.7–89.1 |
| 24K | 47.7–67.8 | 54.2–84.3 |
| 89K | 34.9–48.3 | 48.4–68.2 |
| 172K | 23.4–37.0 | 43.4–62.7 |

### By kind of text, K=3

Tok/s and draft acceptance, upstream → this fork.

| Prompt tokens | Reference prose | Fiction | Source code |
|---:|---|---|---|
| 60 | 69.6 (62%) → 75.8 (59%) | 59.1 (47%) → 65.7 (47%) | 67.8 (59%) → 76.5 (60%) |
| 1.1K | 77.0 (69%) → 84.2 (67%) | 61.0 (50%) → 65.9 (46%) | 64.4 (55%) → 68.8 (51%) |
| 5.9K | 74.6 (71%) → 85.1 (71%) | 58.0 (50%) → 66.2 (48%) | 65.4 (61%) → 74.7 (60%) |
| 24K | 64.9 (72%) → 81.0 (71%) | 53.7 (55%) → 62.5 (48%) | 53.1 (54%) → 66.6 (54%) |
| 89K | 48.3 (71%) → 68.2 (70%) | 34.9 (43%) → 48.4 (41%) | 41.0 (54%) → 57.2 (52%) |
| 172K | 35.3 (71%) → 62.7 (74%) | 23.4 (36%) → 43.4 (42%) | 37.0 (70%) → 54.4 (58%) |

From 89K each entry is a single request.

### Draft acceptance

Share of drafted tokens accepted by verification, upstream / this fork, pooled over the requests.

| Prompt tokens | K=1 | K=2 | K=3 | K=4 | K=5 |
|---:|---|---|---|---|---|
| 60 | 80% / 80% | 66% / 66% | 56% / 55% | 45% / 46% | 37% / 38% |
| 1.1K | 78% / 79% | 67% / 67% | 57% / 54% | 49% / 50% | 41% / 41% |
| 5.9K | 80% / 79% | 68% / 69% | 60% / 59% | 53% / 52% | 45% / 46% |
| 24K | 83% / 81% | 68% / 70% | 60% / 57% | 50% / 50% | 46% / 44% |
| 89K | 83% / 80% | 66% / 67% | 55% / 53% | 46% / 46% | 39% / 38% |
| 172K | – | – | 56% / 56% | – | – |

The draft layer and its weights are the same in both builds; the differences are the divergent
continuations described above.

### Time per decode round

Milliseconds per round, upstream → this fork. A round is one draft of K tokens plus one
verification pass; rounds are counted as drafted tokens divided by K.

| Prompt tokens | MTP off | K=1 | K=2 | K=3 | K=4 | K=5 |
|---:|---|---|---|---|---|---|
| 60 | 34.0 → 29.6 | 36.9 → 32.6 | 39.3 → 35.3 | 41.3 → 37.1 | 45.4 → 39.0 | 47.9 → 41.4 |
| 1.1K | 34.2 → 29.8 | 36.0 → 32.0 | 38.6 → 34.8 | 41.0 → 36.6 | 45.0 → 38.7 | 48.0 → 41.6 |
| 5.9K | 35.8 → 30.2 | 37.8 → 32.4 | 40.6 → 35.1 | 43.1 → 37.4 | 47.4 → 39.3 | 50.9 → 42.3 |
| 24K | 41.2 → 31.1 | 43.5 → 33.4 | 46.9 → 37.0 | 49.6 → 39.5 | 54.6 → 41.6 | 63.0 → 45.1 |
| 89K | 53.1 → 35.9 | 58.0 → 38.4 | 62.1 → 43.1 | 65.4 → 45.8 | 71.4 → 48.4 | 88.3 → 55.3 |
| 172K | 71.0 → 41.3 | – | – | 87.6 → 51.7 | – | – |

Upstream round time divided by this fork's:

| Prompt tokens | MTP off | K=1 | K=2 | K=3 | K=4 | K=5 |
|---:|---|---|---|---|---|---|
| 60 | 1.15× | 1.13× | 1.11× | 1.12× | 1.17× | 1.16× |
| 1.1K | 1.15× | 1.13× | 1.11× | 1.12× | 1.16× | 1.16× |
| 5.9K | 1.19× | 1.16× | 1.15× | 1.15× | 1.21× | 1.20× |
| 24K | 1.32× | 1.30× | 1.27× | 1.26× | 1.31× | 1.39× |
| 89K | 1.48× | 1.51× | 1.44× | 1.43× | 1.47× | 1.60× |
| 172K | 1.72× | – | – | 1.69× | – | – |

From 60 to 172K tokens of context the MTP-off round grows from 34.0 to 71.0 ms on the upstream
port and from 29.6 to 41.3 ms here.

### Memory throughput and arithmetic rate

Derived from the round times above, not measured directly.

- Bytes per verification pass: 19.07 GB of weights and output head (NVFP4 MLPs of layers 0–55 at
  0.5 byte per element plus one scale byte per 16, FP8 everywhere else), plus the INT8 attention
  window (16 layers × 4 KV heads × 512 bytes per token).
- Arithmetic per token: 24.35 billion matrix elements, counted as two operations each
  (48.7 GFLOP), plus attention at 16 layers × 24 heads × 256 dimensions × 4 operations per cached
  token. A round of K drafts verifies K+1 tokens.
- The draft layer's own reads and arithmetic are not counted, so both columns are lower bounds.

| Prompt tokens | GB/s, MTP off | GB/s, K=3 | TFLOPS, MTP off | TFLOPS, K=3 | TFLOPS, K=5 |
|---:|---|---|---|---|---|
| 60 | 561 → 644 | 461 → 514 | 1.43 → 1.65 | 4.72 → 5.26 | 6.11 → 7.07 |
| 1.1K | 559 → 641 | 466 → 523 | 1.44 → 1.65 | 4.80 → 5.38 | 6.14 → 7.10 |
| 5.9K | 539 → 639 | 448 → 515 | 1.43 → 1.69 | 4.75 → 5.46 | 6.02 → 7.25 |
| 24K | 482 → 638 | 401 → 503 | 1.41 → 1.87 | 4.70 → 5.90 | 5.55 → 7.74 |
| 89K | 414 → 613 | 336 → 480 | 1.57 → 2.33 | 5.11 → 7.29 | 5.68 → 9.07 |
| 172K | 348 → 599 | 282 → 478 | 1.64 → 2.82 | 5.31 → 9.00 | – |

Without MTP this fork reads 600–645 GB/s at every context length, while the upstream port falls
from 561 to 348 GB/s. Decode TFLOPS are low on both builds because a decode round reads every
weight to produce a handful of tokens.

## Prefill

Cold prompt, tokens per second as reported by the server. Prefill does not depend on K, so each
row pools the requests of all six K settings.

| Prompt tokens | Requests | Upstream tok/s (min–max) | This fork tok/s (min–max) | Gain | Upstream time | This fork time | Upstream TFLOPS | This fork TFLOPS |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 60 | 36 | 154 (122–176) | 339 (255–399) | 2.21× | 0.4 s | 0.2 s | 7.5 | 16.5 |
| 1,127 | 36 | 946 (916–983) | 978 (948–1,015) | 1.03× | 1.2 s | 1.2 s | 46.3 | 47.9 |
| 5,935 | 36 | 1,072 (1,003–1,119) | 1,163 (1,109–1,202) | 1.08× | 5.6 s | 5.1 s | 53.5 | 58.0 |
| 24,108 | 36 | 973 (950–995) | 1,078 (1,052–1,103) | 1.11× | 24.8 s | 22.4 s | 52.0 | 57.6 |
| 88,515 | 18 | 665 (647–691) | 822 (806–847) | 1.24× | 133 s | 108 s | 44.0 | 54.3 |
| 171,901 | 6 | 455 (432–488) | 625 (601–660) | 1.37× | 380 s | 276 s | 37.5 | 51.5 |

TFLOPS uses the same per-token arithmetic as above with attention taken over half the prompt (the
average window during prefill). Around 1K tokens the two builds are within 3%; the gain is at
prompts under about 200 tokens and from 24K upward.

## Concurrent streams

Aggregate output tokens per second across all streams, measured on the client's clock from
sending the requests to the last response, so it includes each request's 52–66 token prefill.
Each stream is one of the standalone prompts decoding 256 tokens. Three runs per row; mean
(min–max).

| K | Streams | Upstream | This fork | Gain |
|---:|---:|---|---|---:|
| 1 | 1 | 45.0 (44.6–45.5) | 53.1 (52.7–53.6) | 1.18× |
| 1 | 2 | 80.0 (79.5–80.9) | 95.6 (94.6–96.2) | 1.20× |
| 1 | 4 | 125.3 (124.3–125.9) | 160.0 (158.4–162.3) | 1.28× |
| 1 | 8 | 155.7 (153.8–156.7) | 243.3 (242.2–244.4) | 1.56× |
| 3 | 1 | 57.5 (56.6–58.5) | 66.4 (63.9–67.7) | 1.16× |
| 3 | 2 | 92.8 (91.5–94.1) | 113.2 (110.6–116.6) | 1.22× |
| 3 | 4 | 109.1 (106.5–112.6) | 157.3 (153.3–162.6) | 1.44× |
| 3 | 8 | 129.6 (129.0–130.0) | 211.8 (211.4–212.5) | 1.63× |
| 4 | 1 | 54.1 (one run) | 66.1 (61.3–68.7) | – |
| 4 | 2 | 77.1 (one run) | 108.8 (106.1–111.2) | – |
| 4 | 4 | 92.6 (one run) | 134.3 (132.8–136.8) | – |
| 4 | 8 | no result | 181.3 (179.3–184.3) | – |

At K=4 the upstream build completed the 1, 2 and 4 stream steps of the first run; at the first
8-stream step all eight requests returned HTTP 500 ("internal error" in its log) and the server
stopped, so the remaining runs have no result. The cause was not investigated. This fork
completed every row with no failed request.

On both builds the best aggregate at eight streams is with K=1, and a larger K costs throughput
as streams are added.

## Start-up and memory check

This fork was started with CUDA Graphs on (the default) on the same card and artifact, and each
server that came up was then sent twelve single requests (up to 5.9K tokens) and eight concurrent
streams. Common flags: `--kv-dtype int8 --prefill-chunk 2048 --host-kv-mib 6144
--host-state-slots 16 --max-context 262144 --max-concurrency 8`.

| Configuration | MTP off | K=1 | K=3 | K=5 | K=7 |
|---|---|---|---|---|---|
| `--kv-capacity 262144 --device-state-slots 0` | served | served | served | served | refused at startup |
| `--kv-capacity auto --device-state-slots 8` | served | refused | refused | refused | refused |

- "Served" means the server started, answered every request without an error and was still
  running afterwards. GPU memory left free while serving was 2.4 GB with MTP off and 0.59–0.66 GB
  with MTP at `--device-state-slots 0`.
- "Refused" means the server exited during start-up planning with a message stating the bytes
  required and available, before loading weights. No configuration ended in `std::bad_alloc` or
  failed after it had started.
- `--kv-capacity 251904` and `262144` with eight device state slots and K=3 are also refused.
- With `CUDA_MODULE_LOADING=EAGER` the driver holds about 0.12 GB more; MTP off still serves in
  both configurations, and every MTP configuration at 262,144 context is refused at startup.

## Where the remaining time goes

The figures in this section come from an earlier kernel profile of this fork on a different
artifact of the same format, not from the runs above. With MTP off at 1K context, kernels occupied
97% of the round, with the large weight-reading kernels at 700–865 GB/s (bytes read divided by
kernel time). Short-context decode is therefore mostly bound by memory throughput, which is
consistent with the gain over the upstream port being smaller at short context (8–20%) than at
long context or under concurrency (40–72%).

At K=3 the same profile attributed about 4.9 ms of a 37.5 ms round to the MTP draft layer, whose
W8 kernels measure 270–400 GB/s in the linear Op benchmark. Its weights are read in checkpoint
row order, not a prepacked fragment order.

On the natural-text requests above, K=4 has the highest single-stream mean on this fork from 1.1K
to 89K tokens, and K=3, K=4 and K=5 are within 4% of one another at 60 tokens.

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
- CUDA Graphs fit at 262,144 context with MTP up to K=5 (see the start-up check below).
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
tests cover long windows on the new kernels.

`ctest` on the V100 for this fork: 104 tests, 96 pass, 7 are skipped because they need model
artifacts that were not present, 1 fails. The failing test, `gdn_gating_proj`, is inherited from
the upstream port: a single element is 0.24% off against a 0.2% bound on a BF16-staged route this
fork does not change.
