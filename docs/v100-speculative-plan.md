# Plan: single-stream decode on par with llama-superfast-v100

This plan brings the single-stream decode speed of
[inscora/llama-superfast-v100-BWE-q8-dflash2](https://github.com/inscora/llama-superfast-v100-BWE-q8-dflash2)
(a llama.cpp fork for the V100, at `8922ee0`) into this engine, without giving up this engine's
lead with several concurrent requests and without requantising any model artifact.

## Ground rule

Everything here is added beside this engine's own work, not in place of it. MTP stays, and stays
the route with several requests, where it leads (209–246 tok/s at eight streams against 152);
prefill, the long-context decode attention and the start-up memory plan stay as they are. Every
phase is measured against the full sweep below, and a change that makes any measured row worse is
not merged, whatever it gains on single-stream decode.

## Measured starting point

One Tesla V100-PCIe-32GB, the same natural-text requests for every server
([`tools/bench/run_serve_context_sweep.py`](../tools/bench/run_serve_context_sweep.py): cold,
greedy, thinking off, 256 output tokens, six requests per row up to 24K and three from 89K).

- This engine: the published Qwen3.8-27B NVFP4 artifact (revision `11dbbbb`), INT8 KV,
  `--max-context 200704 --no-cuda-graph`.
- llama-superfast: its documented serving line (Unsloth `UD-Q4_K_XL`, the z-lab DFlash2 draft
  `Q4_K_M`, seven draft tokens, Q8_0 KV, four slots on a 262,144-token pool), except
  `--cache-ram 4096 --ctx-checkpoints 8` instead of `15360` and `16` to fit the host's memory,
  and without the web UI.

Single-stream decode, mean tok/s:

| Prompt tokens | This engine, MTP K=3 | This engine, DFlash2 ×7 | llama-superfast, DFlash2 ×7 |
|---:|---:|---:|---:|
| 60 | 71.5 | 76.8 | 81.7 |
| 1.1K | 74.8 | 83.1 | 85.3 |
| 5.9K | 75.0 | 72.8 | 85.3 |
| 24K | 70.7 | 68.3 | 78.5 |
| 89K | 58.3 | 51.3 | 61.2 |
| 172K | 51.5 | 44.6 | 51.8 (two of three requests) |

Aggregate tok/s with concurrent requests (52–66 token prompts, three runs each):

| Streams | This engine, MTP K=3 | This engine, MTP K=1 | This engine, DFlash2 ×7 | llama-superfast |
|---:|---:|---:|---:|---:|
| 1 | 67.0 | 53.0 | 65.3 | 76.3 |
| 2 | 114.1 | 95.0 | 96.2 | 117.7 |
| 4 | 159.2 | 159.8 | 95.2 | 137.9 |
| 8 | 209.0 | 246.1 | 121.3 | 151.8 |

At temperature 1.0 (top-p 0.95, top-k 20) single-stream decode was 65–72 tok/s for this engine
with MTP K=3 and 76–97 for llama-superfast.

Time per decode round and tokens per round, DFlash2 on both (rounds counted as drafted tokens
divided by seven; llama-superfast's own log reports 6.9 drafts per round on these requests):

| Prompt tokens | ms per round, this engine | ms per round, llama-superfast | Tokens per round, this engine | Tokens per round, llama-superfast |
|---:|---:|---:|---:|---:|
| 60 | 43.1 | 42.4 | 3.17 | 3.29 |
| 5.9K | 44.9 | 42.9 | 3.15 | 3.48 |
| 89K | 55.6 | 50.3 | 2.80 | 3.02 |
| 172K | 64.3 | 58.5 | 2.80 | 2.85 |

Reading these:

- With one request, seven DFlash2 drafts verified in one eight-row step give more tokens per
  round than three MTP drafts (3.2–3.5 against 2.6–2.75), for a round that costs 15% more.
- At short context the two engines' DFlash2 rounds cost the same. From 60 to 172K tokens this
  engine's eight-row round grows by 21 ms, llama-superfast's by 16 ms and this engine's four-row
  MTP round by 15 ms: the eight-row verify attention scales worse here.
- Tokens per round are 4–10% lower here on the same prompts. The two targets are quantised
  differently and generate different text, so part of this is outside the engine's control.
- With several requests every DFlash2 request verifies eight rows, and this engine's DFlash2
  falls well behind its own MTP. llama-superfast budgets 8–16 verify rows a step across its
  slots and tops out near 150 tok/s; MTP here reaches 209–246.
- Its model file is 16.4 GiB of 4-bit K-quant weights against this artifact's 19.7 GiB, about 8%
  fewer bytes per round. That difference is a property of the artifacts and is not pursued here.
- Its tensor-core K-quant products follow the QPN design of this engine's kernels (its
  `mmvq-qpn.cu` says so), so the weight kernels are not where it is ahead.

## Phase 0: kernel profile of one DFlash2 round

A per-kernel profile (`nsys`) of both engines' DFlash2 decode round at 60, 24K and 89K tokens of
context, one request, split into: target weight products, the 16 attention layers, the 48 Gated
Delta Net layers, the five draft layers and proposal head, the path selector, and GPU idle time.
It sets the order of the phases below and replaces their estimates.

The baseline for every later change is the sweep above, plus a fixed set of greedy requests whose
outputs must not change.

## Phase 1: a cheaper draft

The DFlash2 (and MTP) draft layers run on the W8 tensor-core kernels
(`src/ops/linear/w8/`, `src/ops/linear_swiglu/w8/w8_dflash2_linear_swiglu.cu`,
`src/ops/attn_input_proj/w8/w8_dflash2_attn_input*.cu`), which read their weights in checkpoint
row order at 270–400 GB/s in the linear Op benchmark. The NVFP4 kernels read weights rearranged at
load into tensor-core fragment order at 700–865 GB/s.

- **1a.** A fragment-order W8 QPN route, with the draft weights rearranged once at load (the
  stored values unchanged, only their order), following `nvfp4_volta_qpn_gemm.cuh`. Estimate:
  2–2.5 ms off a DFlash2 round at about 1.6 GB of draft weights; MTP's draft gains too.
- **1b.** If the profile puts the optimised proposal head above about 0.5 ms a round, a
  frequency-ranked draft vocabulary like llama-superfast's 98,304-id list. Verification keeps the
  output exact; only the proposals change.
- **1c.** GPU idle time between the draft's kernels, if the profile shows more than 0.5 ms a
  round.

Checks: the linear Op tests against the FP32 oracle at 1–8 and 16 rows; greedy outputs unchanged.

## Phase 2: eight-row verify attention at depth

`src/ops/softmax_attention/dense/causal_cache/direct_i8_volta.cuh` (the 2–8 row tensor-core
kernel) and its split policy `causal_direct_splits` in `small_t.cuh`, which was tuned on four-row
MTP rounds: 512 keys a split above 12K keys, at most 176 splits.

- **2a.** Re-tune keys per split and the split cap for eight rows at 24K, 89K and 172K keys.
- **2b.** Balance the warp groups: llama-superfast's verify kernel found its time in the value
  product and the widening of 8-bit values, and saved 8.4% of the kernel by having the key-side
  warps widen part of each value tile (its CHANGES section 32). The profile shows whether this
  kernel has the same imbalance.
- **2c.** Confirm a batch of several requests reads only each request's own pages.

Target: the eight-row round's growth from short context to 172K at or below 16 ms (from 21),
about 3 ms off at 89K and 5 ms at 172K. Checks: the attention Op tests at long windows for one to
eight rows; greedy outputs unchanged at 89K.

## Phase 3: DFlash2 for one request, MTP for several

Both drafts loaded (the artifact carries both), the backend chosen each round by the number of
generating requests: DFlash2 with seven drafts alone, MTP with three (or one at eight requests)
otherwise. Today the backend is fixed at startup (`src/product/speculative_options.h`,
`src/targets/qwen3_6/impl/runtime/{dflash_impl.h,mtp_impl.h,speculative_target_impl.h,schedule.h}`).

- Both drafts' state stays current on every committed token: DFlash2's target features from
  layers 5, 19, 33, 47 and 61, MTP's last hidden state.
- The fallback, if two resident drafts are too invasive, is a verify-row budget over DFlash2's
  per-request valid extent (`docs/maintainer/qwen3.8-27b-dflash2.md` section 7.4), which needs the
  physical verify columns past the live extent to be skipped (Phase 4).

Checks: greedy outputs identical to the single-backend runs; the concurrency sweep at 1, 2, 4 and
8 streams; the start-up memory check at 262,144 tokens of context and eight concurrent requests,
with CUDA Graphs on and off: a configuration that does not fit must be refused at startup with
the byte counts, never fail after it has started.

## Phase 4: drafts cut where the draft is unsure

The selector's per-position probabilities (`candidate_selector_path.cu`) cut the proposal at the
first position under a threshold, and the target verifies only the live `P+1` columns, computing
no physical columns past them. llama-superfast drafts fewer than seven tokens when its confidence
drops; on the requests above it still averaged 6.9, so the gain is expected at depth only, where
each verified row costs attention. Done only if Phase 2 leaves a gap.

## Phase 5: very short prompts

At about 60 prompt tokens llama-superfast prefills 391 tok/s against 332 here. It runs the
tensor-core matvec in 32-token passes and captures no CUDA Graph for prompt shapes (its CHANGES
section 9). Prompts under 193 tokens run here on the decode-width kernels in chunks of up to 64;
32-token chunks are measured on 52–200 token prompts.

## Not taken

- 4-bit weights for the target or the draft: a change of artifact, not of the engine.
- A Q8_0 KV cache: the INT8 group-64 cache here is the same size class.
- Lookup (n-gram) drafting: llama-superfast measured it 1–5% slower in general and a third slower
  on long copies.
- The unified KV pool, host prompt cache, CUDA Graph cache cap and two-user scheduler: they solve
  problems this engine handles with its paged cache, host KV and state slots and start-up memory
  plan.

## Expected result

If Phases 1–3 deliver their estimates, single-stream DFlash2 decode here would be about 88 tok/s
at 1.1K tokens, 76 at 5.9K, 72 at 24K and 55 at 89K, against llama-superfast's 85, 85, 79 and 61
today, with concurrent throughput unchanged from MTP. The remaining gap from 5.9K is in tokens
per round. The estimates are replaced by measurements after Phase 0.
