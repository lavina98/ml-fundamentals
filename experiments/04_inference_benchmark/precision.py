"""Day 10: precision - fp32 vs TF32 vs bf16 vs fp16 on GPT-2. Faster? Smaller? Same answers?

  A. Number formats   - what each format can represent (range vs precision), with 3 tiny demos
  B. Speed + memory   - the same forward pass in 5 modes; weights MB, latency, throughput, max batch
  C. Accuracy         - on real Shakespeare: does the loss change? do the predicted next tokens change?
  D. Chart            -> precision.png, results/day10.json

YOUR PART: to_precision(), compare_outputs() - marked TODO.
Reuses YOUR measure() and peak_memory_mb() from bench.py (Day 9).

Run:  python precision.py
"""
import contextlib
import copy
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from bench import SEQ_LEN, load_gpt2, measure, peak_memory_mb

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).parent
MODES = ["fp32", "tf32", "autocast_bf16", "bf16", "fp16"]
BATCH = 16


# ---------------------------------------------------------------- YOUR PART

def to_precision(model, mode: str):
    """TODO: put an fp32 model into `mode`. Return (model, ctx).

    ctx is a context manager the forward pass runs inside:  `with ctx: model(ids)`.
    Use contextlib.nullcontext() when nothing special is needed (it does nothing).

      "fp32"           weights fp32, math fp32 (the baseline)
                         -> return model, contextlib.nullcontext()
      "tf32"           weights fp32, but matmuls run on tensor cores with a 10-bit mantissa
                         -> torch.backends.cuda.matmul.allow_tf32 = True     (a global switch - the
                            harness turns it back off after this mode)
                         -> return model, contextlib.nullcontext()
      "autocast_bf16"  weights STAY fp32; PyTorch casts to bf16 on the fly, op by op
                         (matmuls in bf16, sensitive ops like softmax/layernorm kept in fp32)
                         -> return model, torch.autocast("cuda", dtype=torch.bfloat16)
      "bf16"           weights CONVERTED to bf16 (half the memory), everything runs in bf16
                         -> return model.to(torch.bfloat16), contextlib.nullcontext()
      "fp16"           weights converted to fp16
                         -> return model.half(), contextlib.nullcontext()

    Note: .half() / .to(dtype) change the model IN PLACE - that's why the harness gives you a
    fresh fp32 copy for every mode (converting back to fp32 would NOT restore the lost digits).
    """

    if mode == "fp32":
        return model, contextlib.nullcontext()
    if mode == "tf32":
        torch.backends.cuda.matmul.allow_tf32 = True
        return model, contextlib.nullcontext()
    if mode == "autocast_bf16":
        return model, torch.autocast("cuda", dtype=torch.bfloat16)
    if mode == "bf16":
        return model.to(torch.bfloat16), contextlib.nullcontext()
    if mode == "fp16":
        return model.half(), contextlib.nullcontext()

    raise ValueError(mode) 

def compare_outputs(ref: torch.Tensor, out: torch.Tensor) -> dict[str, float]:
    """TODO: how different are `out` logits from the fp32 `ref` logits?

        ref, out: (B, T, vocab) logits - convert BOTH to fp32 first with .float()
                  (you can't subtract a bf16 tensor from an fp32 one cleanly, and you want the
                   comparison itself done at full precision)

    Return:
      "max_abs_diff": the biggest |ref - out| anywhere                  ->  (ref - out).abs().max().item()
      "top1_agree":   fraction of positions where both models pick the SAME most-likely next token
                      1. argmax over the vocab axis (which dim?) for each  -> (B, T) token ids
                      2. compare them with ==                               -> (B, T) of True/False
                      3. .float().mean().item()                             -> e.g. 0.998
    """
    ref = ref.float()
    out = out.float()

    max_abs_diff = (ref-out).abs().max().item()
    ref_top = ref.argmax(dim=-1)
    out_top = out.argmax(dim=-1)

    same = ref_top == out_top

    top1_agree = same.float().mean().item()

    return {"max_abs_diff":max_abs_diff, "top1_agree": top1_agree}



# ---------------------------------------------------------------- A. formats

def part_a():
    print("=" * 84, "\nA. NUMBER FORMATS\n")
    print(f"{'format':>9} | {'bits':>4} | {'exponent':>8} | {'mantissa':>8} | {'largest number':>14} | "
          f"{'step size near 1.0':>18}")
    for name, dt, e, m in (("fp32", torch.float32, 8, 23), ("tf32*", torch.float32, 8, 10),
                           ("bf16", torch.bfloat16, 8, 7), ("fp16", torch.float16, 5, 10)):
        fi = torch.finfo(dt)
        eps = 2.0 ** -m
        bits = {"fp32": 32, "tf32*": 19, "bf16": 16, "fp16": 16}[name]
        print(f"{name:>9} | {bits:>4} | {e:>8} | {m:>8} | {fi.max:>14.3g} | {eps:>18.2g}")
    print("  * TF32 is not a storage type: numbers are STORED as fp32, only matmul inputs are rounded to 10 bits")
    print("  exponent bits = RANGE (how big/small a number can be), mantissa bits = PRECISION (how many digits)\n")

    x = torch.tensor(70000.0)
    print(f"  70000 in fp16 = {x.half().item()}   in bf16 = {x.bfloat16().item()}       "
          "<- fp16 overflows past 65,504; bf16 has fp32's range")
    y = torch.tensor(1.001)
    print(f"  1.001 in fp16 = {y.half().item():.6f}   in bf16 = {y.bfloat16().item():.6f}  "
          "<- bf16 has only ~3 significant digits")
    z = torch.tensor(1e-8)
    print(f"  1e-8  in fp16 = {z.half().item()}        in bf16 = {z.bfloat16().item():.2e}   "
          "<- tiny numbers vanish in fp16 (gradients!)")


# ---------------------------------------------------------------- data (boilerplate)

def shakespeare_batch(B: int, dev: str) -> torch.Tensor:
    """B windows of SEQ_LEN + 1 real GPT-2 tokens from Tiny Shakespeare (input + next-token labels)."""
    from transformers import GPT2TokenizerFast
    text = (HERE.parent / "03_tiny_gpt" / "data" / "input.txt").read_text(encoding="utf-8")
    ids = GPT2TokenizerFast.from_pretrained("gpt2")(text[:200_000])["input_ids"]
    ids = torch.tensor(ids)
    g = torch.Generator().manual_seed(0)
    starts = torch.randint(0, len(ids) - SEQ_LEN - 1, (B,), generator=g)
    return torch.stack([ids[s:s + SEQ_LEN + 1] for s in starts]).to(dev)


def weights_mb(model) -> float:
    return sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6


# ---------------------------------------------------------------- B + C. run every mode

def run_modes(base, dev: str) -> list[dict]:
    data = shakespeare_batch(BATCH, dev)
    x, y = data[:, :-1], data[:, 1:]                       # (B, 128) inputs, (B, 128) next-token labels
    vram = torch.cuda.get_device_properties(0).total_memory / 1e6
    n_params = sum(p.numel() for p in base.parameters())
    rows, ref = [], None

    print("\n" + "=" * 84, f"\nB. SPEED + MEMORY  (GPT-2, batch {BATCH} x {SEQ_LEN} tokens, {torch.cuda.get_device_name(0)})\n")
    print(f"{'mode':>14} | {'weights MB':>10} | {'median ms':>9} | {'seq/s':>6} | {'TFLOPS':>6} | "
          f"{'MB / seq':>8} | {'est. max batch':>14}")
    for mode in MODES:
        model = copy.deepcopy(base).to(dev)                 # a fresh fp32 copy every time
        try:
            model, ctx = to_precision(model, mode)

            @torch.no_grad()
            def fwd():
                with ctx:
                    return model(x).logits

            t = measure(fwd, dev)
            mem = peak_memory_mb(fwd, dev)
            logits = fwd()
            loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), y.reshape(-1)).item()
        finally:
            torch.backends.cuda.matmul.allow_tf32 = False   # undo TF32's global switch
        w = weights_mb(model)
        per_seq = mem / BATCH
        row = {"mode": mode, "weights_mb": w, **t, "seq_per_s": BATCH / (t["median_ms"] / 1000),
               "tflops": 2 * n_params * BATCH * SEQ_LEN / (t["median_ms"] / 1000) / 1e12,
               "mb_per_seq": per_seq, "max_batch_est": int((vram - w - 300) / per_seq), "loss": loss}
        if ref is None:
            ref = logits.float().cpu()
            row.update({"max_abs_diff": 0.0, "top1_agree": 1.0})
        else:
            row.update(compare_outputs(ref, logits.cpu()))
        rows.append(row)
        print(f"{mode:>14} | {w:>10,.0f} | {t['median_ms']:>9.1f} | {row['seq_per_s']:>6.0f} | "
              f"{row['tflops']:>6.2f} | {per_seq:>8.1f} | {row['max_batch_est']:>14,}")
        del model, logits
        torch.cuda.empty_cache()
    print("  (est. max batch = (VRAM - weights - ~300 MB for CUDA) / MB per sequence)")

    print("\n" + "=" * 84, f"\nC. ACCURACY  (the same {BATCH} real Shakespeare windows, compared with fp32)\n")
    print(f"{'mode':>14} | {'loss':>6} | {'loss vs fp32':>12} | {'max |logit diff|':>16} | {'same top-1 token':>16}")
    for r in rows:
        print(f"{r['mode']:>14} | {r['loss']:>6.4f} | {r['loss'] - rows[0]['loss']:>+12.4f} | "
              f"{r['max_abs_diff']:>16.4f} | {r['top1_agree']:>15.2%}")
    return rows


# ---------------------------------------------------------------- D. chart

def part_d(rows):
    print("\n" + "=" * 84, "\nD. CHART\n")
    modes = [r["mode"] for r in rows][::-1]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    for ax, key, title, fmt in ((axes[0], "seq_per_s", "Throughput (sequences/sec, higher is better)", "{:.0f}"),
                                (axes[1], "weights_mb", "Weight memory (MB, lower is better)", "{:,.0f}")):
        vals = [r[key] for r in rows][::-1]
        bars = ax.barh(modes, vals, color="#2a78d6", height=0.6)
        for b, v in zip(bars, vals):
            ax.text(b.get_width(), b.get_y() + b.get_height() / 2, "  " + fmt.format(v),
                    va="center", fontsize=9, color="#333")
        ax.set_title(title, fontsize=11)
        ax.margins(x=0.15)
        ax.grid(axis="x", alpha=0.25)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.suptitle(f"GPT-2 (124M), batch {BATCH} x {SEQ_LEN} tokens")
    fig.tight_layout()
    fig.savefig(HERE / "precision.png", dpi=110)
    plt.close(fig)
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "day10.json").write_text(json.dumps(rows, indent=1))
    print("saved precision.png and results/day10.json")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        sys.exit("this benchmark needs a CUDA GPU")
    part_a()
    base = load_gpt2("cpu")
    rows = run_modes(base, "cuda")
    part_d(rows)
