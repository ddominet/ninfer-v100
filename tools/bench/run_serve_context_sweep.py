#!/usr/bin/env python3
"""Measure prefill and decode of a running ninfer-serve across context lengths.

The client sends natural-text requests built from the committed perplexity corpus
(eval/corpora/perplexity-1m): a document slice of the requested size followed by a task. Every
request is cold (explicit cache mode, unique header line), greedy, with thinking disabled, and
decodes a fixed number of tokens. One JSON record per request is appended to the output file.

    ninfer-serve MODEL --kv-dtype int8 ... &
    python3 tools/bench/run_serve_context_sweep.py --url http://127.0.0.1:8080 \
        --label fork-k3 --output sweep.jsonl

Server-side flags (context, speculative decoding, concurrency) are the caller's; record them with
the results. The script reports what the server measured in the response `timings` object.

`--dialect llama` sends the same requests to a llama.cpp `llama-server` (thinking off through
`chat_template_kwargs`, `cache_prompt` off), whose `timings` object has the same fields.
`--sampling` replaces the greedy default, for example
`--sampling '{"temperature": 1.0, "top_p": 0.95, "top_k": 20, "seed": 7}'`.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import pathlib
import time
import urllib.error
import urllib.request

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
CORPUS = REPO_ROOT / "eval/corpora/perplexity-1m/data"

# Documents this large and larger get one task per domain instead of two.
SINGLE_TASK_CONTEXT = 50000

# Domain -> (corpus domain, approximate characters per token, tasks appended to the document).
DOMAINS = {
    "reference": (
        "wikitext",
        4.2,
        (
            "Summarize the text above in detail, covering every topic it discusses.",
            "List the people, places and dates mentioned in the text above, with one sentence "
            "about each.",
        ),
    ),
    "fiction": (
        "pg19",
        4.0,
        (
            "Continue the story above for several more paragraphs in the same style.",
            "Describe the characters that appear in the text above and what each of them wants.",
        ),
    ),
    "code": (
        "ninfer",
        3.2,
        (
            "Explain what the code above does, file by file.",
            "Write unit tests for the most important functions in the code above.",
        ),
    ),
}

# Requests without a document: the short-prompt end of the sweep.
STANDALONE = (
    ("code", "Write a complete, well-commented C++20 implementation of a lock-free ring buffer "
             "and unit tests for it."),
    ("code", "Write a Python module that parses an Apache access log and reports the ten most "
             "frequent client addresses, with tests."),
    ("fiction", "Write the opening scene of a mystery story set in a lighthouse during a storm."),
    ("fiction", "Write a dialogue between a ship's engineer and its captain about a failing "
                "reactor."),
    ("reference", "Explain how a transformer language model processes a prompt and generates "
                  "text, for a technical reader."),
    ("reference", "Describe the causes and consequences of the Industrial Revolution."),
)


def document(domain: str, tokens: int, variant: int) -> str:
    name, chars_per_token, _ = DOMAINS[domain]
    text = "".join(path.read_text(encoding="utf-8") for path in sorted((CORPUS / name).glob("*.txt")))
    size = int(tokens * chars_per_token)
    # A different window of the stream per variant, so repeated sizes are not the same prompt.
    start = (variant * 7919) % max(1, len(text) - size)
    return text[start:start + size]


DIALECT = "ninfer"
SAMPLING = {"temperature": 0.0}


def post(url: str, content: str, max_tokens: int) -> dict:
    body = {
        "model": "qwen3.8-27b",
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        **SAMPLING,
    }
    if DIALECT == "ninfer":
        body.update(reasoning_effort="none", prompt_cache_options={"mode": "explicit"})
    else:
        body.update(chat_template_kwargs={"enable_thinking": False}, cache_prompt=False)
    request = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=7200) as response:
            result = json.loads(response.read())
    except urllib.error.HTTPError as error:
        return {"wall_s": round(time.monotonic() - started, 3), "error": error.code,
                "detail": error.read().decode(errors="replace")[:300]}
    timings = result.get("timings", {})
    return {
        "wall_s": round(time.monotonic() - started, 3),
        "prompt_tokens": timings.get("prompt_n"),
        "cached_tokens": timings.get("cache_n"),
        "prefill_tok_s": timings.get("prompt_per_second"),
        "output_tokens": timings.get("predicted_n"),
        "decode_tok_s": timings.get("predicted_per_second"),
        "draft_tokens": timings.get("draft_n"),
        "draft_accepted": timings.get("draft_n_accepted"),
        "text_head": (result.get("choices") or [{}])[0].get("message", {}).get("content", "")[:200],
    }


def requests_for(context: int, variant: int) -> list[tuple[str, str]]:
    if context == 0:
        return list(STANDALONE)
    out = []
    for domain, (_, _, tasks) in DOMAINS.items():
        text = document(domain, context, variant)
        for task in tasks[:1] if context >= SINGLE_TASK_CONTEXT else tasks:
            out.append((domain, f"{text}\n\n{task}"))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--label", required=True, help="recorded with every result")
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--contexts", default="0,1000,6000,24000,90000",
                        help="comma-separated nominal document sizes in tokens; 0 = no document")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--streams", default="",
                        help="comma-separated concurrency levels, run with the standalone requests")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--dialect", choices=("ninfer", "llama"), default="ninfer")
    parser.add_argument("--sampling", default='{"temperature": 0.0}')
    args = parser.parse_args()
    global DIALECT, SAMPLING
    DIALECT = args.dialect
    SAMPLING = json.loads(args.sampling)
    url = args.url.rstrip("/") + "/v1/chat/completions"

    with args.output.open("a", encoding="utf-8") as handle:
        def emit(record: dict) -> None:
            record["label"] = args.label
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            print(json.dumps(record), flush=True)

        nonce = time.time_ns()
        for repeat in range(args.repeats):
            for context in (int(v) for v in args.contexts.split(",") if v):
                for index, (domain, content) in enumerate(requests_for(context, repeat)):
                    row = post(url, f"[request {nonce}-{repeat}-{context}-{index}]\n{content}",
                               args.max_tokens)
                    row.update(kind="single", context=context, domain=domain, task=index,
                               repeat=repeat)
                    emit(row)
            for streams in (int(v) for v in args.streams.split(",") if v):
                pool = requests_for(0, repeat)
                jobs = [pool[i % len(pool)] for i in range(streams)]
                started = time.monotonic()
                with concurrent.futures.ThreadPoolExecutor(streams) as executor:
                    rows = list(executor.map(
                        lambda item: post(
                            url, f"[request {nonce}-{repeat}-s{streams}-{item[0]}]\n{item[1][1]}",
                            args.max_tokens),
                        enumerate(jobs)))
                wall = time.monotonic() - started
                total = sum(r.get("output_tokens") or 0 for r in rows)
                emit({
                    "kind": "concurrent", "streams": streams, "repeat": repeat,
                    "wall_s": round(wall, 3), "errors": sum("error" in r for r in rows),
                    "output_tokens": total, "aggregate_tok_s": total / wall,
                    "draft_tokens": sum(r.get("draft_tokens") or 0 for r in rows),
                    "draft_accepted": sum(r.get("draft_accepted") or 0 for r in rows),
                })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
