"""Day 7: train the tiny GPT on Tiny Shakespeare for real.

  1. Benchmark: your Day 5 attention vs F.scaled_dot_product_attention inside the full model
  2. Train:     random batches, train/val loss every N steps, keep the best checkpoint
  3. Results:   train_curve.png + tokens/sec + peak GPU memory

YOUR PART: train_step(), estimate_loss() - marked TODO.

Run:
    python train.py --bench                  # ~1 min: mine vs F.sdpa, tokens/sec + memory at T=128 and T=512
    python train.py --attn sdpa              # ~5-10 min: the real training run (use the faster attention)
Out: checkpoints/gpt.pt (best val loss), train_curve.png
"""
import argparse
import math
import sys
import time
from dataclasses import asdict

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from gpt import HERE, GPT, Config, get_batch, load_data
import attention as attn_mod   # the Day 5 module, already on sys.path thanks to gpt.py

sys.stdout.reconfigure(encoding="utf-8")


# ---------------------------------------------------------------- YOUR PART

def train_step(model: GPT, opt: torch.optim.Optimizer, x: torch.Tensor, y: torch.Tensor) -> float:
    """TODO: one optimization step on one batch. Return the loss as a plain Python float.

    Same 4 lines as Day 2's train_one_epoch, plus one new line:
      1. forward:  logits, loss = model(x, y)
      2. opt.zero_grad()
      3. loss.backward()
      4. NEW - gradient clipping:  torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
         If one unlucky batch produces a huge gradient, this scales the whole gradient down so
         its total length is at most 1.0 - one bad step can't wreck the weights.
         (It must go AFTER backward - the gradients have to exist - and BEFORE step.)
      5. opt.step()
      6. return loss.item()
    """
    _, loss = model(x, y)
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    return loss.item()


@torch.no_grad()
def estimate_loss(model: GPT, train: torch.Tensor, val: torch.Tensor, cfg: Config,
                  batch_size: int, dev: str, n_batches: int = 20) -> dict[str, float]:
    """TODO: average loss over `n_batches` random batches from train AND from val.

    Returns {"train": float, "val": float}.

    Why average? One batch is 64 random windows - its loss is noisy. 20 batches gives a stable number.
    Why measure train loss here too, instead of using train_step's numbers? Those were computed WITH
    dropout on, while the weights were changing; here both splits are measured the same way.

    Steps:
      1. model.eval()                          # dropout OFF (Day 3)
      2. for each split ("train" -> train, "val" -> val):
             losses = []
             repeat n_batches times:
                 x, y = get_batch(data, batch_size, cfg.block_size, dev)
                 _, loss = model(x, y)
                 losses.append(loss.item())
             out[split] = average of losses
      3. model.train()                         # dropout back ON - easy to forget!
      4. return out

    (@torch.no_grad() above the def = the whole function runs without building a graph - Day 2.)
    """

    out ={}
    model.eval()
    for (split, data) in  (("train", train) , ("val", val)):
        losses =[]
        for i in range(n_batches):
            x,y = get_batch(data, batch_size, cfg.block_size, dev)
            _, loss = model(x,y)
            losses.append(loss.item())

        out[split] = sum(losses)/len(losses)

    model.train()
    return out


# ---------------------------------------------------------------- attention switch

def use_attention(kind: str):
    """Swap the attention function used inside every MultiHeadSelfAttention layer.

    'mine' = your Day 5 attention() (builds the full (B, H, T, T) score matrix)
    'sdpa' = F.scaled_dot_product_attention (fused kernel, never stores the T x T matrix)
    Same math, same weights - a checkpoint trained with one loads fine with the other.
    """
    if kind == "mine":
        attn_mod.attention = _MINE
    else:
        attn_mod.attention = lambda q, k, v, causal=False: (
            F.scaled_dot_product_attention(q, k, v, is_causal=causal), None)


_MINE = attn_mod.attention


# ---------------------------------------------------------------- 1. benchmark

def bench(dev: str, steps: int = 30):
    print("=" * 70, "\nBENCHMARK: training speed, your attention vs F.sdpa (same model, same batch)\n")
    if dev != "cuda":
        print("no GPU - skipping")
        return
    tok, train, _ = load_data()
    print(f"{'T':>5} {'B':>4} | {'attention':>9} | {'tokens/sec':>11} | {'peak MB':>8}")
    for T in (128, 512):
        B = 8192 // T                                     # same tokens per batch at both lengths
        for kind in ("mine", "sdpa"):
            use_attention(kind)
            torch.manual_seed(0)
            cfg = Config(vocab_size=tok.vocab_size, block_size=T)
            model = GPT(cfg).to(dev)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
            try:
                x, y = get_batch(train, B, T, dev)
                for _ in range(3):                        # warm-up
                    train_step(model, opt, x, y)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                t0 = time.perf_counter()
                for _ in range(steps):
                    train_step(model, opt, x, y)
                torch.cuda.synchronize()
                tps = steps * B * T / (time.perf_counter() - t0)
                mb = torch.cuda.max_memory_allocated() / 1e6
                print(f"{T:>5} {B:>4} | {kind:>9} | {tps:>11,.0f} | {mb:>8,.0f}")
            except torch.cuda.OutOfMemoryError:
                print(f"{T:>5} {B:>4} | {kind:>9} | {'OOM':>11} | {'-':>8}")
            del model, opt
            torch.cuda.empty_cache()


# ---------------------------------------------------------------- 2. training

def plot_curve(log: list[dict], best: dict, path):
    steps = [r["step"] for r in log]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for key, color, label in (("train", "#2a78d6", "train"), ("val", "#eb6834", "val")):
        vals = [r[key] for r in log]
        ax.plot(steps, vals, color=color, linewidth=2, marker="o", markersize=3, label=label)
        ax.annotate(f"{label} {vals[-1]:.2f}", (steps[-1], vals[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=9, color="#333")
    ax.scatter([best["step"]], [best["val"]], s=70, facecolor="none", edgecolor="#333", linewidth=1.5, zorder=3)
    ax.annotate(f"best val {best['val']:.3f}\n(checkpoint saved)", (best["step"], best["val"]),
                xytext=(0, 14), textcoords="offset points", ha="center", va="bottom", fontsize=9, color="#333")
    ax.axhline(math.log(65), color="#999", linewidth=1, linestyle="--")
    ax.text(steps[0], math.log(65), "  random guessing (ln 65)", va="bottom", fontsize=8, color="#666")
    ax.margins(x=0.1)
    ax.set_xlabel("step")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title("Tiny GPT on Tiny Shakespeare")
    ax.grid(alpha=0.25)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true", help="only run the attention benchmark")
    ap.add_argument("--attn", choices=["mine", "sdpa"], default="sdpa")
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval-every", type=int, default=250)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    if args.bench:
        bench(dev)
        return

    print("=" * 70, f"\nTRAINING  (attention = {args.attn}, device = {dev})\n")
    use_attention(args.attn)
    tok, train, val = load_data()
    cfg = Config(vocab_size=tok.vocab_size)
    torch.manual_seed(0)
    model = GPT(cfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    n_params = sum(p.numel() for p in model.parameters())
    tokens_per_step = args.batch * cfg.block_size
    print(f"{n_params:,} params | batch {args.batch} x {cfg.block_size} = {tokens_per_step:,} tokens/step "
          f"| {args.steps} steps = {args.steps * tokens_per_step / len(train):.0f} passes over the train set\n")

    ckpt_path = HERE / "checkpoints" / "gpt.pt"
    ckpt_path.parent.mkdir(exist_ok=True)
    log, best = [], {"step": -1, "val": float("inf")}
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t_start = t_last = time.perf_counter()
    train_time = 0.0

    for step in range(args.steps + 1):
        if step % args.eval_every == 0 or step == args.steps:
            if dev == "cuda":
                torch.cuda.synchronize()
            train_time += time.perf_counter() - t_last           # don't count eval time as training
            losses = estimate_loss(model, train, val, cfg, args.batch, dev)
            log.append({"step": step, **losses})
            mark = ""
            if losses["val"] < best["val"]:
                best = {"step": step, "val": losses["val"]}
                torch.save({"cfg": asdict(cfg), "chars": tok.chars, "model": model.state_dict(),
                            "step": step, "val_loss": losses["val"]}, ckpt_path)
                mark = "  <- best, saved"
            tps = step * tokens_per_step / train_time if step else 0
            print(f"step {step:>5} | train {losses['train']:.3f} | val {losses['val']:.3f} "
                  f"| {tps:>9,.0f} tok/s | {time.perf_counter() - t_start:5.0f}s{mark}")
            t_last = time.perf_counter()
            if step == args.steps:
                break

        x, y = get_batch(train, args.batch, cfg.block_size, dev)
        train_step(model, opt, x, y)

    peak = torch.cuda.max_memory_allocated() / 1e6 if dev == "cuda" else 0
    print(f"\ndone in {time.perf_counter() - t_start:.0f}s | training throughput "
          f"{args.steps * tokens_per_step / train_time:,.0f} tokens/sec | peak GPU memory {peak:,.0f} MB")
    print(f"best val loss {best['val']:.3f} at step {best['step']}  ->  {ckpt_path.relative_to(HERE)}")
    plot_curve(log, best, HERE / "train_curve.png")
    print("saved train_curve.png")


if __name__ == "__main__":
    main()
