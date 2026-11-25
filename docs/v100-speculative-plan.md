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

## Progress

Landed (each passed the W8/Q4 linear, attention-input, LinearSwiGLU, dynamic grouped conv add,
top-K, sliding-window and causal attention Op tests, and a same-prompt server sweep in which no
MTP, concurrency or prefill row got worse):

- **Cheaper draft round.** Per-shape split-K and activation staging for the draft's W8 products,
  the draft q|k|v on the tensor-core kernel (194 -> 89 us), and the draft's sliding-window
  attention 32 keys at a time instead of one.
- **Six- to eight-token verify attention at two CTAs per SM** and at most 120 key splits:
  53.6 -> 50.4 ms a DFlash2 round at 89K tokens.
- **Draft weights in the coalesced QPN layout.** The draft block's output, gate|up and down W8
  weights are permuted at load and read with FP16-staged activations at 680-790 GB/s.

Together, 3.2-6.4 ms off a DFlash2 round at every context. Same card and prompts as the table
above (decode tok/s, mean over the requests):

| Prompt tokens | This engine, MTP K=3 | This engine, DFlash2 x7 | llama-superfast, DFlash2 x7 |
|---:|---:|---:|---:|
| 60 | 72.8 | 82.1 | 81.7 |
| 1.1K | 75.9 | 87.0 | 85.3 |
| 5.9K | 77.5 | 81.7 | 85.3 |
| 24K | 73.4 | 72.8 | 78.5 |
| 89K | 60.1 | 58.9 | 61.2 |
| 172K | 54.9 | 52.6 | 51.8 |

Aggregate tok/s with 1 / 4 / 8 streams: MTP K=3 67.3 / 162.5 / 211.0, DFlash2 66.8 / 98.9 /
122.0, llama-superfast 76.3 / 137.9 / 151.8.

This engine's DFlash2 round now costs the same as llama-superfast's or less up to 24K tokens;
the remaining gap from 5.9K to 89K is tokens per round (3.1-3.2 against 3.4-3.5 on these
prompts), which depends on how well each draft predicts its own target's output.

Measured and not adopted:

- Seven draft tokens stay best: five and nine are slower at every context, and the full proposal
  head instead of the 131,072-row optimised one adds about 5 ms a round for no net gain.
- CUDA Graphs change nothing measurable on the MTP K=3 server rounds (36.5 vs 36.6 ms at 60
  tokens, 46.0 vs 45.1 ms at 89K).
- Wave-aligned key splits and reading the rotated queries from the global plane instead of
  shared memory were slower in every case.

Constraints found:

- DFlash2's weights are 21.4 GiB resident against 19.7 GiB for MTP. On a 32 GB card it starts at
  262,144 tokens of context with one concurrent request (0.88 GiB free), but at 262,144 tokens
  with eight concurrent requests it is refused at startup: 11.34 GB of runtime needed, 10.79 GB
  available (15.37 GB with CUDA Graphs). Phase 3, which would hold both drafts, does not fit that
  configuration either.
- Resolved for graphs-off serving by keeping the 1.2 GiB token embedding in mapped host memory
  (d5baa878): DFlash2 x7 now starts at 262,144 tokens with eight requests (654 MiB free) and MTP
  K=3 gains 1.2 GiB of headroom with unchanged speed. CUDA Graphs with DFlash2 are still refused.

### Production-configuration comparison (2026-11-24)

Each engine at the production limits (262,144 tokens, eight requests, INT8/q8_0 KV), both on the
same V100 and the same prompts. Decode tok/s for one request:

| Prompt tokens | llama-superfast x7 | DFlash2 x7 | MTP K=3 | MTP K=6 | No speculation |
|---:|---:|---:|---:|---:|---:|
| 60 | 83.3 | 82.4 | 73.4 | 70.8 | 33.8 |
| 1.1K | 87.2 | 89.1 | 74.3 | 77.4 | 33.6 |
| 5.9K | 84.4 | 81.1 | 76.8 | 82.4 | 33.2 |
| 24K | 78.1 | 72.6 | 72.1 | 74.9 | 32.2 |
| 89K | 64.4 | 59.7 | 58.3 | 59.0 | 27.9 |
| 172K | 51.8* | 52.4 | 52.6 | 55.7 | 24.2 |

\* llama-superfast refused one of the 172K requests in every configuration ("new tokens do not
fit the KV pool beside the other conversations"). Prefill is level (1164 vs 1122 tok/s at 5.9K,
621 vs 621 at 172K). Aggregate tok/s at 1 / 2 / 4 / 8 streams: llama-superfast 76.8 / 120.4 /
135.0 / 151.7, MTP K=3 67.1 / 115.7 / 150.7 / 204.1, DFlash2 x7 69.2 / 107.4 / 101.4 / 123.7, no
speculation 33.1 / 61.3 / 109.3 / 179.7 (llama-superfast without speculation: 100.3 at eight).

- MTP K=4-7 helps one request at long context and costs aggregate (K=4 181.6, K=7 142.7 at
  eight streams); K=3 stays the serving choice.
- DFlash2 x6 and x8 are slower than x7 from 5.9K up (x8: 40 tok/s at 89K and 172K).
- llama-superfast at 5, 7 and 9 draft tokens is within noise; it does not draft more at 9.
- Tokens accepted per output token are 0.62-0.69 in every engine and setting, so the differences
  are round cost, not draft quality.
- Temperature 1 (top-p 0.95, top-k 20) keeps the same ordering.
- 1Cat-vLLM 1.5.3 does not load on one 32 GB V100 at any context from 262,144 to 32,768, with or
  without speculation, eager or graphs: it runs out of memory with 30.9 GiB allocated.
- This sweep found DFlash2 x8 failing every request at eight streams: 72 draft rows exceeded the
  prepacked W8 kernel's 64-row launch. Fixed in 8aaf418f (chunked launches); after the fix x8
  serves 61.1 / 82.3 / 89.6 / 106.8 tok/s at 1 / 2 / 4 / 8 streams with no errors.

### Kept-column DFlash2 verification (2026-11-24)

Under load DFlash2 verified all K+1 columns of every request through the target while accepting
about 3.3 of 8; on chat traffic it fell to half of MTP K=3's per-request speed at medium load.
Every eager DFlash round now keeps, per request, only the drafts whose expected acceptance pays
for them, and the target runs its token-local work on the kept columns only:

- Target: projections, MLPs, norms and the LM head run on kept columns; the attention core
  (expanded to the widest kept extent + 1), the Gated DeltaNet convolution and recurrence, and
  feature capture keep the padded per-request layout around `gather_columns(_group)`. Outputs
  return padded, so acceptance, rollback and the DFlash context are unchanged. The real-engine
  DFlash2 test (greedy equality with ordinary decoding, batch of eight, prefix restore, partial
  stops) passes with it.
- Draft: a block only as wide as the widest kept extent + 1, re-laid into the K-wide layout;
  the proposal head scores only kept positions; draft-side fallbacks removed (control projection
  16 columns a launch, multi-tile row-major W8 for q|k|v and the feature projection); the
  context-append block is as wide as the widest append; the Q4 proposal head is prepacked at load
  (1.4x on the fused top-K).
- Choice: a goodput rule over every request's next draft, ordered by a conditional acceptance
  estimate (decayed per-request hits and tries shrunk toward an engine-wide rate), against a round
  cost fitted online by recursive least squares (intercept, requests, columns past the weight-bound
  eight, head chunks, requests x context, draft-block columns). One request keeps its whole
  block. `NINFER_DFLASH_KEEP_ALL=1` restores full verification; `NINFER_DFLASH_TRACE=1` prints
  one line a round.

Same card, production configuration. Fixed-prompt benchmark, DFlash2 x7 (MTP K=3 for reference):

| | 1 stream | 2 | 4 | 8 |
|---|---:|---:|---:|---:|
| DFlash2 x7 before | 69.2 | 107.4 | 101.4 | 123.7 |
| DFlash2 x7 now | 71.9-73.2 | 107.3-117.3 | 137.5-143.6 | 185.1-192.2 |
| MTP K=3 | 67.2 | 117.0 | 150.3 | 202.9 |

Single-prompt decode now: 83.4 / 89.1 / 83.6 / 73.1 / 57.7 / 52.4 tok/s at 60 / 1.1K / 6K / 24K /
89K / 172K tokens (before: 81.6 / 91.3 / 81.5 / 74.0 / 59.0 / 54.5).

Seeded chat traffic (Poisson arrivals, 256-1024-token answers; median per-request decode, median
seconds per 256 tokens, throughput):

| Load | MTP K=3 | DFlash2 x7 before | DFlash2 x7 now |
|---|---|---|---|
| about 0.4 active | 72.7 / 3.65 s / 30.5 | 84.6 / 3.13 s / 30.8 | 86.6 / 3.06 s / 30.8 |
| about 2 active | 52.8 / 5.08 s / 106.6 | 26.4 / 10.50 s / 106.2 | 47.9 / 5.42 s / 108.4 |
| about 8 active | 37.4 / 9.60 s / 217.9 | 22.1 / 19.18 s / 154.2 | 30.9 / 11.90 s / 197.4 |

DFlash2 is now the better engine for one or two users and within 7% at medium load; at eight it
is about 10% behind MTP K=3 in throughput. Both accept 0.597 drafts per output token on this
traffic, so the remainder is round cost: the five-layer draft, its head and the layout copies cost
about 4 ms a round more than MTP's head at eight requests. A ragged draft block (each request
drafting only its own kept extent) would save about 1 ms; holding both drafts remains the way to
take MTP's numbers at full load.

Tried and not kept: choosing the MTP draft length by batch size (reverted); FP16-staged
activations for the Q4 head kernel (slower); hand-fitted cost constants (each suited one load).

### Wider verify rows and the kept-draft policy (2026-11-25)

The root cause under load is that every verify or draft row past eight costs real time on V100:
the 27B projections were weight-bound to eight columns and then grew about 1 ms a column, and
DFlash2 needs more rows than MTP. Changes:

- NVFP4 quadpair kernels at 17..32 columns ask for two resident CTAs instead of one (they ran
  eight warps an SM at up to 206 registers): gate/up 302 -> 263 us and down 156 -> 123 us at 32.
- FP8 projections take a new tensor-core skinny GEMM (`volta_tc_skinny.cuh`: a prepacked weight
  slice decoded once into shared memory, WMMA against all columns, ordered split-K fix-up) past
  24 columns: 1.14-1.46x at 32, up to 1.7x at 64. NVFP4 stays on quadpair (its decode-to-shared
  cost loses). `NINFER_VOLTA_TC=0` disables it.
- DFlash-only overheads: the draft control projection's scalar remainder (124 us a call) and
  one-CTA-a-column gathers (~12 us a call, ~100 a round) are gone.
- The round-cost model is fitted under the constraint it predicts with (non-negative least
  squares with a ridge toward the prior); it had overcharged wider keeps by ~10%.
- The kept-draft choice searches the draft-block width cap: one deep keep widens the five-layer
  draft for every request, and greedy alone lost 15% to a fixed keep of three under load.

Seeded chat traffic, same card (median decode tok/s / median s per 256 tokens / throughput):

| Load | MTP K=3 | DFlash2 x7 start of 11-24 | DFlash2 x7 now (two runs) |
|---|---|---|---|
| ~0.4 active | 72.5 / 3.60 s / 30.8 | 84.6 / 3.13 s / 30.8 | 85.2-85.8 / 3.11-3.16 s / 30.5 |
| ~2 active | 54.2 / 4.83 s / 104.9 | 26.4 / 10.50 s / 106.2 | 49.0-49.5 / 5.33-5.43 s / 107.4 |
| ~8 active | 40.6 / 7.99 s / 237.4 | 22.1 / 19.18 s / 154.2 | 34.7-35.5 / 9.47-9.67 s / 213-218 |

At eight requests both engines now spend the same ~53 ms of GPU a round; DFlash2 verifies the
same ~3 drafts a request with the same acceptance as MTP K=3 but pays its five-layer draft, so it
stays about 9% behind in throughput there. Ablations that did not help: early exit from the draft
(acceptance collapses: P(>=1) 0.74 -> 0.57 at four of five layers), a ragged draft block (~1 ms),
tensor-core NVFP4 (slower than quadpair).

### DFlash-only round costs and the final comparison (2026-11-25)

Profiling under the high-load chat phase showed DFlash2 rounds ~4 ms dearer than MTP K=3 rounds
for the same tokens per round. Removed since:

- the draft q|k|v and its context key/value pair, the gate/up and output projections, and the
  131 MB feature projection run prepacked, on the tensor-core skinny GEMM past 16 columns (a
  mandatory prepacked W8 decoder; prefill appends in 64-token launches);
- layout copies: the verify outputs expand with 16-byte copies and leave dropped columns
  unwritten, the GDN gate stays in the kept layout, the gated norm reads the recurrence output
  through a column map, the GDN input projection writes q/k/v straight into the padded record,
  and dropped record/g/beta columns are left unwritten (their readers stop at the valid counts);
- the draft window attention runs one CTA per K/V head for all four query heads (one read of
  each window per K/V head, 164 -> 91 us at four tokens);
- the control projection loads 16 bytes a step;
- the verify-cost model counts 8-column target tiles.

Not kept: eight warps for narrow attention blocks (slower: narrow blocks run at large batches);
the control projection on the tensor-core route (its 20 row blocks need deep split-K, whose
serial fix-up costs more than it saves); the draft down projection on the tensor-core route
(the quadpair kernel is faster to 32 columns).

Seeded chat traffic, same build (5cd25b38) and card, wall time per output token and mean decode:

| Load | MTP K=3 | DFlash2 x7 |
|---|---|---|
| ~0.4 active | 14.4 ms, 70.7 tok/s | 12.1 ms, 85.8 tok/s |
| ~2 active | 18.6 ms, 55.3 tok/s | 17.0 ms, 62.0 tok/s |
| ~8 active | 30.4 ms, 42.0 tok/s | 31.6 ms, 42.7 tok/s |

At eight active requests the decode time per request is equal (14.90 vs 14.94 s); the remaining
wall-time difference is time waiting for a free slot (4.7 vs 4.2 s mean), which varies as much
between two MTP runs (4.2 and 4.6 s). Single-prompt decode is at or above the original
baseline at every context (83.5 / 91.8 / 85.9 / 74.6 / 59.0 / 56.0 tok/s from 60 to 172K
tokens) and long-prompt prefill is ~2% faster. The full suite passes except the
gdn_gating_proj test, which failed before this work.

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
