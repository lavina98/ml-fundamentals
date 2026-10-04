"""Day 5: self-attention from scratch, checked against PyTorch.

  A. Scaled dot-product attention (+ causal mask) vs F.scaled_dot_product_attention
  B. What the attention weights look like                         -> attention.png
  C. Multi-head attention: split -> attend -> merge, vs nn.MultiheadAttention
  D. Infra: memory and time as the sequence gets longer (why FlashAttention exists)

YOUR PART: attention(), split_heads(), merge_heads() - marked TODO.

Run:  python attention.py
"""
import math
import sys
import time

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn

sys.stdout.reconfigure(encoding="utf-8")


# ---------------------------------------------------------------- YOUR PART

def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
              causal: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """TODO: scaled dot-product attention. Return (out, weights).

    Shapes (the leading "..." can be (B,) or (B, H) - write it so both work):
        q, k, v:  (..., T, d)
        weights:  (..., T, T)   weights[i, j] = how much token i looks at token j; each row sums to 1
        out:      (..., T, d)   each token's output = weighted average of all value vectors

    Steps:
      1. scores = q @ k.transpose(-2, -1)        # (..., T, d) @ (..., d, T) -> (..., T, T)
                                                 # transpose(-2, -1) swaps the LAST TWO axes only,
                                                 # so it works with or without a heads axis
      2. scores = scores / math.sqrt(d)          # d = q.shape[-1]   ("scaled")
      3. if causal: block the future - token i may only look at tokens j <= i
             future = torch.triu(torch.ones(T, T, dtype=torch.bool, device=q.device), diagonal=1)
             scores = scores.masked_fill(future, float("-inf"))
         (exp(-inf) = 0, so softmax gives those positions exactly 0 weight)
      4. weights = softmax over the LAST axis (each row = one token's distribution over tokens)
      5. out = weights @ v                       # (..., T, T) @ (..., T, d) -> (..., T, d)
    """
    scores = q @ k.transpose(-2,-1)
    d = q.shape[-1]
    t = q.shape[-2]
    scores = scores / math.sqrt(d)
    if causal:
        future = torch.triu(torch.ones(t, t, dtype=torch.bool, device=q.device), diagonal=1)
        scores = scores.masked_fill(future, float("-inf"))

    weights = torch.softmax(scores, dim=-1)
    out = weights @ v

    return out, weights


def split_heads(x: torch.Tensor, n_heads: int) -> torch.Tensor:
    """TODO: (B, T, D) -> (B, H, T, D // H)

    Cut each token's D numbers into H chunks, then move the head axis next to the batch:
      1. x.view(B, T, H, D // H)      # same numbers, regrouped: last axis split into (H, d_head)
      2. .transpose(1, 2)             # swap the T and H axes -> (B, H, T, d_head)

    Example: B=1, T=2, D=4, H=2
        token 0: [a, b, c, d]   ->  head 0 sees [a, b],  head 1 sees [c, d]
    """

    b = x.shape[0]
    t =x.shape[1]
    d =x.shape[2]

    x = x.view(b, t, n_heads, d//n_heads)
    x = x.transpose(1,2)

    return x

def merge_heads(x: torch.Tensor) -> torch.Tensor:
    """TODO: (B, H, T, d_head) -> (B, T, H * d_head)   - the exact inverse of split_heads

      1. .transpose(1, 2)             # -> (B, T, H, d_head)
      2. .contiguous()                # transpose only changes how memory is READ; view() needs
                                      # the data physically in order, so make a reordered copy
      3. .view(B, T, H * d_head)      # glue the heads back together per token
    """

    b, h, t, d_head = x.shape
    x = x.transpose(1,2)
    x = x.contiguous()
    x = x.view(b, t, h*d_head)
    return x    


# ---------------------------------------------------------------- A. correctness

def part_a():
    print("=" * 70, "\nA. SCALED DOT-PRODUCT ATTENTION\n")

    # A tiny example you can check by hand: 3 tokens, d = 2
    q = torch.tensor([[1., 0], [0, 1], [1, 1]])
    k = torch.tensor([[1., 0], [0, 1], [1, 1]])
    v = torch.tensor([[10., 0], [0, 10], [5, 5]])
    out, w = attention(q, k, v)
    print("3 tokens, d=2.  weights (row i = where token i looks):")
    print(w.round(decimals=2))
    print("output (each row = weighted average of v's rows):")
    print(out.round(decimals=2))
    print(f"rows of weights sum to 1: {torch.allclose(w.sum(-1), torch.ones(3))}")

    # Against PyTorch's built-in on random data, with and without the causal mask
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 5, 8) for _ in range(3))
    for causal in (False, True):
        mine, w = attention(q, k, v, causal=causal)
        ref = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        print(f"\ncausal={causal}: matches F.scaled_dot_product_attention: "
              f"{torch.allclose(mine, ref, atol=1e-6)}   (max diff {(mine - ref).abs().max():.1e})")
    print("causal weights for one sequence (zeros above the diagonal = no peeking ahead):")
    print(w[0].round(decimals=2))


# ---------------------------------------------------------------- B. picture

def part_b(path="attention.png"):
    print("\n" + "=" * 70, "\nB. ATTENTION WEIGHTS PICTURE\n")
    tokens = ["The", "cat", "sat", "on", "the", "mat"]
    torch.manual_seed(1)
    x = torch.randn(len(tokens), 16)
    x[4] = x[0]                       # "the" and "The": same vector -> watch them attend to each other
    _, full = attention(x, x, x)
    _, causal = attention(x, x, x, causal=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, (title, w) in zip(axes, [("full (encoder-style)", full), ("causal (GPT-style)", causal)]):
        im = ax.imshow(w, cmap="Blues", vmin=0, vmax=1)   # sequential: 0 = white, 1 = dark blue
        ax.set_title(title)
        ax.set_xticks(range(len(tokens)), tokens)
        ax.set_yticks(range(len(tokens)), tokens)
        ax.set_xlabel("attends to (key)")
        for i in range(len(tokens)):
            for j in range(len(tokens)):
                ax.text(j, i, f"{w[i, j]:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if w[i, j] > 0.6 else "#222")
    axes[0].set_ylabel("token (query)")
    fig.colorbar(im, ax=axes, shrink=0.8, label="attention weight")
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print("toy vectors (random, not trained) - the point is the SHAPE of the patterns:")
    print("  full:   every token can look at every token")
    print("  causal: token i only sees tokens 0..i  ->  'The' can only look at itself")
    print(f"saved {path}")


# ---------------------------------------------------------------- C. multi-head

class MultiHeadSelfAttention(nn.Module):
    """The real layer: project x into q, k, v -> split into heads -> attend -> merge -> project out."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, causal: bool = False) -> torch.Tensor:
        q = split_heads(self.q_proj(x), self.n_heads)     # (B, T, D) -> (B, H, T, d_head)
        k = split_heads(self.k_proj(x), self.n_heads)
        v = split_heads(self.v_proj(x), self.n_heads)
        out, _ = attention(q, k, v, causal=causal)        # (B, H, T, d_head): each head attends independently
        return self.out_proj(merge_heads(out))            # (B, T, D)

 
def part_c():
    print("\n" + "=" * 70, "\nC. MULTI-HEAD ATTENTION\n")
    B, T, D, H = 2, 5, 16, 4
    x = torch.randn(B, T, D)

    s = split_heads(x, H)
    print(f"split_heads: {tuple(x.shape)} -> {tuple(s.shape)}")
    print(f"  head 1 of token 0 = numbers 4..7 of token 0: {torch.equal(s[0, 1, 0], x[0, 0, 4:8])}")
    print(f"merge_heads(split_heads(x)) == x: {torch.equal(merge_heads(s), x)}")

    torch.manual_seed(0)
    mine = MultiHeadSelfAttention(D, H)
    ref = nn.MultiheadAttention(D, H, batch_first=True)
    with torch.no_grad():   # copy my weights into PyTorch's layer so both compute the same function
        ref.in_proj_weight.copy_(torch.cat([mine.q_proj.weight, mine.k_proj.weight, mine.v_proj.weight]))
        ref.in_proj_bias.copy_(torch.cat([mine.q_proj.bias, mine.k_proj.bias, mine.v_proj.bias]))
        ref.out_proj.weight.copy_(mine.out_proj.weight)
        ref.out_proj.bias.copy_(mine.out_proj.bias)
        future = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
        for causal in (False, True):
            a = mine(x, causal=causal)
            b, _ = ref(x, x, x, attn_mask=future if causal else None, need_weights=False)
            print(f"causal={causal}: matches nn.MultiheadAttention: {torch.allclose(a, b, atol=1e-5)}")
    print(f"params: 4 x Linear({D},{D}) = {sum(p.numel() for p in mine.parameters()):,}")


# ---------------------------------------------------------------- D. infra

def part_d():
    print("\n" + "=" * 70, "\nD. COST VS SEQUENCE LENGTH (GPU, fp16, B=1, 8 heads, d_head=64)\n")
    if not torch.cuda.is_available():
        print("no GPU - skipping")
        return
    dev, H, d = "cuda", 8, 64

    def measure(fn, q, k, v):
        fn(q, k, v)                                   # warm-up
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        t0 = time.perf_counter()
        for _ in range(5):
            fn(q, k, v)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / 5 * 1000
        mb = (torch.cuda.max_memory_allocated() - base) / 1e6
        return ms, mb

    print(f"{'T':>6} | {'scores (H,T,T)':>14} | {'mine: ms':>9} {'extra MB':>9} | {'F.sdpa: ms':>10} {'extra MB':>9}")
    for T in (512, 1024, 2048, 4096, 8192):
        q, k, v = (torch.randn(1, H, T, d, device=dev, dtype=torch.float16) for _ in range(3))
        score_mb = H * T * T * 2 / 1e6
        row = f"{T:>6} | {score_mb:>11,.0f} MB | "
        try:
            ms, mb = measure(lambda a, b, c: attention(a, b, c, causal=True), q, k, v)
            row += f"{ms:>9.2f} {mb:>9,.0f} | "
        except torch.cuda.OutOfMemoryError:
            row += f"{'OOM':>9} {'-':>9} | "
        torch.cuda.empty_cache()
        ms, mb = measure(lambda a, b, c: F.scaled_dot_product_attention(a, b, c, is_causal=True), q, k, v)
        row += f"{ms:>10.2f} {mb:>9,.0f}"
        print(row)
        del q, k, v
        torch.cuda.empty_cache()


if __name__ == "__main__":
    part_a()
    part_b()
    part_c()
    part_d()
