"""Day 11: generation latency on GPT-2 - time-to-first-token vs per-token, with and without a KV cache.

  A. Prefill (TTFT)    - time to the first token as the prompt grows, 64 -> 1024 tokens; KV cache size
  B. Decode            - ms per new token, step by step: KV cache vs re-running the whole text
  C. Same answer?      - cached and uncached greedy generation should pick the same tokens
  D. Decode speedups   - fp32 vs bf16, batch 1 / 8 / 32: what actually makes decode faster?
  E. Chart             -> latency.png, results/day11.json

Generation has two phases (Day 8 measured them on the tiny GPT; this is the real version):
  prefill - read the whole prompt in ONE forward pass, build the KV cache, pick token 1   -> TTFT
  decode  - feed only the newest token, reuse the cache, pick the next one; repeat        -> per-token latency

YOUR PART: prefill(), decode_step(), step_nocache() - marked TODO.
Reuses YOUR measure() from bench.py (Day 9).

Run:  python latency.py        (~2 min)
"""
import copy
import json
import statistics
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from bench import load_gpt2, measure

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).parent
OUT = HERE
PROMPT_LENS = (64, 128, 256, 512, 1024)   # GPT-2's context window is 1024 tokens
B_PROMPT, B_NEW = 64, 960                 # Part B: 64-token prompt + 960 new tokens = the full 1024


# ---------------------------------------------------------------- YOUR PART

@torch.no_grad()
def prefill(model, ids: torch.Tensor):
    """TODO: PREFILL - run the whole prompt once, keep the cache, pick the first new token.

        ids:     (B, T) prompt token ids
        returns: (nxt, cache)   nxt (B, 1) = the greedy next token, cache = the KV cache

      1. out = model(ids, use_cache=True)     # HF returns an object with .logits and .past_key_values
                                              #   .logits: (B, T, vocab)
                                              #   .past_key_values: every layer's K and V for all T tokens
      2. nxt = the argmax of the LAST position's logits, keeping a (B, 1) shape
             (exactly Day 8: logits[:, -1, :] then argmax over the vocab, keepdim=True)
      3. return nxt, out.past_key_values
    """
    out = model(ids, use_cache=True)
    nxt = out.logits[:,-1,:].argmax(dim=-1, keepdim=True)
    return nxt, out.past_key_values


@torch.no_grad()
def decode_step(model, nxt: torch.Tensor, cache):
    """TODO: DECODE one step WITH the cache - feed only the newest token.

        nxt:     (B, 1)  the token picked last step
        cache:   the KV cache from prefill / the previous decode_step
        returns: (nxt, cache)   the next token (B, 1) and the updated cache

      1. out = model(nxt, past_key_values=cache, use_cache=True)
             the model computes q, k, v for this ONE token, appends its k, v to the cache,
             and attends over all cached keys - so T = 1 here, not the whole text
      2. pick the next token exactly as in prefill (out.logits is (B, 1, vocab) now)
      3. return it and out.past_key_values
    """


    out = model(nxt, past_key_values=cache, use_cache=True)
    nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    return nxt, out.past_key_values


@torch.no_grad()
def step_nocache(model, ids: torch.Tensor) -> torch.Tensor:
    """TODO: DECODE one step WITHOUT a cache - Day 8's generate loop body.

        ids:     (B, T) the WHOLE text so far (prompt + everything generated)
        returns: (B, 1) the next token

      1. out = model(ids, use_cache=False)    # recompute K and V for every position, every step
      2. return the greedy pick from the last position
    """
    out = model(ids, use_cache=False)
    nxt = out.logits[:,-1,:].argmax(dim=-1, keepdim=True)
    return nxt


# ---------------------------------------------------------------- helpers (boilerplate)

def sync():
    torch.cuda.synchronize()


def timed(fn):
    """Run fn() once; return (its result, milliseconds)."""
    sync()
    t0 = time.perf_counter()
    r = fn()
    sync()
    return r, (time.perf_counter() - t0) * 1000


def shakespeare_ids(n: int, dev: str):
    """The first n GPT-2 tokens of Tiny Shakespeare, as (1, n), plus the tokenizer."""
    from transformers import GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    text = (HERE.parent / "03_tiny_gpt" / "data" / "input.txt").read_text(encoding="utf-8")
    ids = tok(text[:20_000])["input_ids"][:n]
    return torch.tensor([ids], device=dev), tok


def kv_cache_mb(cfg, B: int, T: int, bytes_per: int) -> float:
    """K and V, for every layer, for every token: 2 x layers x B x T x d_model numbers."""
    return 2 * cfg.n_layer * B * T * cfg.n_embd * bytes_per / 1e6


# ---------------------------------------------------------------- A. prefill / TTFT

def part_a(model, prompt):
    print("=" * 84, "\nA. PREFILL: time to first token (TTFT) vs prompt length  (batch 1, fp32)\n")
    print(f"{'prompt':>7} | {'TTFT ms':>8} | {'ms / prompt token':>17} | {'KV cache MB':>11}")
    rows = []
    for T in PROMPT_LENS:
        ids = prompt[:, :T]
        t = measure(lambda: prefill(model, ids), "cuda", warmup=3, iters=10)
        kv = kv_cache_mb(model.config, 1, T, 4)
        rows.append({"prompt_len": T, "ttft_ms": t["median_ms"], "kv_cache_mb": kv})
        print(f"{T:>7} | {t['median_ms']:>8.1f} | {t['median_ms'] / T:>17.3f} | {kv:>11.1f}")
    print(f"\nKV cache = 2 (K and V) x {model.config.n_layer} layers x {model.config.n_embd} numbers x 4 bytes "
          f"= {kv_cache_mb(model.config, 1, 1, 4) * 1000:.0f} KB per token (fp32)")
    print(f"  32 users x 1024 tokens = {kv_cache_mb(model.config, 32, 1024, 4):,.0f} MB of cache "
          f"vs {sum(p.numel() for p in model.parameters()) * 4 / 1e6:.0f} MB of weights")
    return rows


# ---------------------------------------------------------------- B + C. decode, cache vs no cache

def part_b(model, prompt, tok):
    print("\n" + "=" * 84, f"\nB. DECODE: ms per new token, {B_PROMPT}-token prompt + {B_NEW} new tokens (batch 1, fp32)\n")
    ids = prompt[:, :B_PROMPT]
    prefill(model, ids)                                         # warm-up
    step_nocache(model, ids)

    # with the cache: prefill once, then 1 token per step
    (nxt, cache), ttft = timed(lambda: prefill(model, ids))
    cached_tokens, cached_ms = [nxt], []
    for _ in range(B_NEW - 1):
        (nxt, cache), ms = timed(lambda: decode_step(model, nxt, cache))
        cached_tokens.append(nxt)
        cached_ms.append(ms)

    # without: re-run the whole (growing) text every step
    text, plain_tokens, plain_ms = ids, [], []
    for _ in range(B_NEW):
        nxt, ms = timed(lambda: step_nocache(model, text))
        plain_tokens.append(nxt)
        plain_ms.append(ms)
        text = torch.cat([text, nxt], dim=1)

    ctx_cached = list(range(B_PROMPT + 1, B_PROMPT + B_NEW))   # context length at each decode step
    ctx_plain = list(range(B_PROMPT, B_PROMPT + B_NEW))
    print(f"{'context length':>14} | {'with KV cache':>13} | {'no cache':>9} | {'cache is':>8}")
    for lo, hi in ((64, 128), (256, 320), (512, 576), (960, 1024)):
        c = statistics.median(m for m, n in zip(cached_ms, ctx_cached) if lo <= n < hi)
        p = statistics.median(m for m, n in zip(plain_ms, ctx_plain) if lo <= n < hi)
        print(f"{f'{lo}-{hi}':>14} | {c:>10.2f} ms | {p:>6.2f} ms | {p / c:>7.1f}x")
    tot_c, tot_p = ttft + sum(cached_ms), sum(plain_ms)
    print(f"\n{B_NEW} tokens total: {tot_c / 1000:.2f} s with the cache vs {tot_p / 1000:.2f} s without "
          f"({tot_p / tot_c:.1f}x)   |   TTFT {ttft:.1f} ms")

    # C. same tokens?
    print("\n" + "=" * 84, "\nC. SAME ANSWER?  (greedy, cached vs uncached)\n")
    a = torch.cat(cached_tokens, dim=1)[0]
    b = torch.cat(plain_tokens, dim=1)[0]
    diff = (a != b).nonzero()
    if len(diff) == 0:
        print(f"all {B_NEW} tokens identical - the cache is an optimization, not an approximation")
    else:
        i = int(diff[0])
        print(f"identical for the first {i} tokens, then they diverge (token {i}: "
              f"{tok.decode([int(a[i])])!r} vs {tok.decode([int(b[i])])!r})")
        print("  different kernels add numbers in a different order -> tiny float differences -> a near-tie flips (Day 10)")
    print("\nprompt:   ", repr(tok.decode(ids[0])[-80:]))
    print("generated:", repr(tok.decode(a[:40])))

    return {"ttft_ms": ttft, "cached_ms": cached_ms, "nocache_ms": plain_ms,
            "ctx_cached": ctx_cached, "ctx_nocache": ctx_plain,
            "total_cached_s": tot_c / 1000, "total_nocache_s": tot_p / 1000,
            "identical_tokens": int(len(diff) == 0), "first_divergence": None if len(diff) == 0 else int(diff[0])}


# ---------------------------------------------------------------- D. what makes decode faster?

def part_d(base, prompt):
    print("\n" + "=" * 84, "\nD. DECODE SPEEDUPS: precision x batch  (128-token prompt, 64 decode steps, KV cache)\n")
    print(f"{'precision':>9} | {'batch':>5} | {'ms / step':>9} | {'tokens/s':>9} | {'weights MB':>10}")
    rows = []
    for prec, dtype in (("fp32", torch.float32), ("bf16", torch.bfloat16)):
        model = copy.deepcopy(base).to(dtype)
        w = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6
        for B in (1, 8, 32):
            ids = prompt[:, :128].repeat(B, 1)
            nxt, cache = prefill(model, ids)
            for _ in range(100):                                # warm-up: a long one - batch-1 decode is so light
                nxt, cache = decode_step(model, nxt, cache)     # the GPU idles in a low-clock power state
                                                                # until it has been busy for a while (~2x slower)
            times = []
            for _ in range(64):
                (nxt, cache), ms = timed(lambda: decode_step(model, nxt, cache))
                times.append(ms)
            ms = statistics.median(times)
            rows.append({"precision": prec, "batch": B, "ms_per_step": ms, "tokens_per_s": B / ms * 1000})
            print(f"{prec:>9} | {B:>5} | {ms:>9.2f} | {B / ms * 1000:>9,.0f} | {w:>10,.0f}")
            del cache
        del model
        torch.cuda.empty_cache()
    bw_ms = sum(p.numel() for p in base.parameters()) * 4 / 1e9 / 224 * 1000
    print(f"\nlower bound for one fp32 step: read 498 MB of weights at ~224 GB/s = {bw_ms:.1f} ms "
          f"(RTX 3050 memory bandwidth)")
    return rows


# ---------------------------------------------------------------- E. chart

def part_e(a_rows, b, d_rows):
    print("\n" + "=" * 84, "\nE. CHART\n")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))

    ax = axes[0]
    x = [r["prompt_len"] for r in a_rows]
    y = [r["ttft_ms"] for r in a_rows]
    ax.plot(x, y, color="#2a78d6", linewidth=2, marker="o", markersize=5)
    for xi, yi in zip(x, y):
        ax.annotate(f"{yi:.0f} ms", (xi, yi), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8)
    ax.set_xlabel("prompt length (tokens)")
    ax.set_ylabel("ms")
    ax.set_title("Prefill: time to first token")
    ax.set_ylim(bottom=0)

    ax = axes[1]
    ax.plot(b["ctx_nocache"], b["nocache_ms"], color="#eb6834", linewidth=1, alpha=0.8)
    ax.plot(b["ctx_cached"], b["cached_ms"], color="#2a78d6", linewidth=1, alpha=0.8)
    ax.annotate("no cache", (b["ctx_nocache"][-1], b["nocache_ms"][-1]), xytext=(6, 0),
                textcoords="offset points", va="center", fontsize=9, color="#333")
    ax.annotate("KV cache", (b["ctx_cached"][-1], b["cached_ms"][-1]), xytext=(6, 0),
                textcoords="offset points", va="center", fontsize=9, color="#333")
    ax.set_xlabel("context length (tokens so far)")
    ax.set_ylabel("ms per new token")
    ax.set_title("Decode: cost of each new token")
    ax.set_ylim(bottom=0)
    ax.margins(x=0.12)

    for ax in axes:
        ax.grid(alpha=0.25)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.suptitle("GPT-2 (124M) generation latency, batch 1, fp32")
    fig.tight_layout()
    fig.savefig(OUT / "latency.png", dpi=110)
    plt.close(fig)
    (OUT / "results").mkdir(exist_ok=True)
    (OUT / "results" / "day11.json").write_text(json.dumps({"prefill": a_rows, "decode": b, "speedups": d_rows}, indent=1))
    print("saved latency.png and results/day11.json")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        sys.exit("this benchmark needs a CUDA GPU")
    model = load_gpt2("cuda")
    prompt, tok = shakespeare_ids(1024, "cuda")
    a_rows = part_a(model, prompt)
    b = part_b(model, prompt, tok)
    d_rows = part_d(model, prompt)
    part_e(a_rows, b, d_rows)
