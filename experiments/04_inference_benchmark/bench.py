"""Day 9: an inference benchmark harness + batch-size sweep on a real model (GPT-2, 124M params).

  A. The model     - params, weight memory, FLOPs per token
  B. GPU sweep     - batch 1 -> OOM: latency (median + p95), throughput, peak memory, achieved TFLOPS
  C. CPU sweep     - the same, batch 1 -> 16
  D. Charts        - latency vs batch, throughput vs batch      -> batch_sweep.png, results/day9.json

Every run is one forward pass over B sequences of 128 tokens - the "prefill" step an
inference server runs when requests arrive (Day 11 does token-by-token generation).

YOUR PART: measure(), peak_memory_mb() - marked TODO.

Run:  python bench.py        (first run downloads GPT-2, ~550 MB)
"""
import json
import statistics
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import torch

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).parent
SEQ_LEN = 128


# ---------------------------------------------------------------- YOUR PART

def measure(fn, dev: str, warmup: int = 3, iters: int = 10) -> dict[str, float]:
    """TODO: time fn() properly. Return {"median_ms": ..., "p95_ms": ...}.

    The 3 rules of GPU timing (you've seen all of them - Day 3 and Day 5 Part D):
      1. WARM UP: call fn() `warmup` times first and throw the timings away
         (the first calls pay one-time costs: CUDA init, kernel selection, memory allocation)
      2. SYNC: the GPU runs asynchronously - Python queues work and moves on. Call
         torch.cuda.synchronize() right BEFORE starting the clock and right BEFORE stopping it,
         or you time how long it took to QUEUE the work, not to DO it.   (only if dev == "cuda")
      3. REPEAT + MEDIAN: time each of `iters` calls separately, keep a list of ms values,
         report the median (robust to a random slow run) and p95 (the tail - what SLOs are about)

    Steps:
      - warm-up loop
      - times = []
      - for each iter: sync -> t0 = time.perf_counter() -> fn() -> sync -> times.append((now - t0) * 1000)
      - median: statistics.median(times)
      - p95:    sorted(times)[int(0.95 * (len(times) - 1))]
    """

    sync = ""
    if dev == "cuda":
        sync = torch.cuda.synchronize

    times = []
    for i in range(warmup):
        fn()
    
    for i in range(iters):
        if sync!="" : sync()
        t0 = time.perf_counter()
        fn()
        if sync!="" : sync()
        times.append((time.perf_counter()-t0) * 1000)

    return {"median_ms": statistics.median(times), "p95_ms": sorted(times)[int(0.95 * len(times)-1)]}
    

def peak_memory_mb(fn, dev: str) -> float:
    """TODO: the EXTRA GPU memory fn() needs at its peak, in MB (0.0 on CPU).

    Steps (Day 5 Part D did exactly this):
      1. if dev != "cuda": return 0.0
      2. torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()   # reset the high-water mark
      3. base = torch.cuda.memory_allocated()       # what's already there (the model weights)
      4. fn(); torch.cuda.synchronize()
      5. return (torch.cuda.max_memory_allocated() - base) / 1e6
    """

    if dev !="cuda":
        return 0.0

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    fn()
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated()-base) /1e6


# ---------------------------------------------------------------- model (boilerplate)

def load_gpt2(dev: str):
    from transformers import GPT2LMHeadModel
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(dev).eval()
    return model


def make_forward(model, B: int, dev: str):
    ids = torch.randint(0, model.config.vocab_size, (B, SEQ_LEN), device=dev)   # content doesn't matter for speed

    @torch.no_grad()
    def fwd():
        return model(ids).logits                    # (B, 128, 50257)
    return fwd


# ---------------------------------------------------------------- A. the model

def part_a(model):
    print("=" * 78, "\nA. THE MODEL: GPT-2 small\n")
    n = sum(p.numel() for p in model.parameters())
    c = model.config
    print(f"params:        {n / 1e6:.0f}M   ({c.n_layer} layers, d_model {c.n_embd}, {c.n_head} heads, vocab {c.vocab_size:,})")
    print(f"weights:       {n * 4 / 1e6:,.0f} MB in fp32 (4 bytes/param)  |  {n * 2 / 1e6:,.0f} MB in fp16/bf16 (Day 10)")
    print(f"compute:       ~2 x params = {2 * n / 1e9:.2f} GFLOPs per token (one multiply + one add per weight)")
    print(f"logits/batch:  B x {SEQ_LEN} x {c.vocab_size:,} x 4 bytes = {SEQ_LEN * c.vocab_size * 4 / 1e6:.0f} MB per sequence "
          f"(watch this in the memory column)")
    print(f"vs your tiny GPT: 3.2M params -> GPT-2 is {n / 3.2e6:.0f}x bigger")
    return n


# ---------------------------------------------------------------- B/C. sweeps

def sweep(model, n_params: int, dev: str, batches, iters: int) -> list[dict]:
    print(f"\n{'batch':>6} | {'median ms':>9} | {'p95 ms':>8} | {'samples/s':>9} | {'tokens/s':>9} | "
          f"{'TFLOPS':>6} | {'peak MB':>8}")
    rows = []
    vram_mb = torch.cuda.get_device_properties(0).total_memory / 1e6 if dev == "cuda" else 0
    for B in batches:
        fwd = make_forward(model, B, dev)
        try:
            t = measure(fwd, dev, warmup=2 if dev == "cpu" else 3, iters=iters)
            mem = peak_memory_mb(fwd, dev)
        except torch.cuda.OutOfMemoryError:
            print(f"{B:>6} | {'OOM - the largest batch that fits is the previous row':>60}")
            break
        finally:
            del fwd
            if dev == "cuda":
                torch.cuda.empty_cache()
        sps = B / (t["median_ms"] / 1000)
        tflops = 2 * n_params * B * SEQ_LEN / (t["median_ms"] / 1000) / 1e12
        rows.append({"device": dev, "batch": B, **t, "samples_per_s": sps,
                     "tokens_per_s": sps * SEQ_LEN, "tflops": tflops, "peak_mb": mem})
        spill = ""
        if dev == "cuda" and mem + torch.cuda.memory_allocated() / 1e6 > vram_mb:
            spill = "  <- more than the GPU has! Windows spilled into system RAM (slow)"
        print(f"{B:>6} | {t['median_ms']:>9.1f} | {t['p95_ms']:>8.1f} | {sps:>9.1f} | {sps * SEQ_LEN:>9,.0f} | "
              f"{tflops:>6.2f} | {mem:>8,.0f}{spill}")
    return rows


def part_b(model, n_params):
    print("\n" + "=" * 78, f"\nB. GPU SWEEP ({torch.cuda.get_device_name(0)}, fp32, {SEQ_LEN} tokens/sequence)")
    return sweep(model, n_params, "cuda", [2 ** i for i in range(10)], iters=10)    # 1, 2, 4 ... 512


def part_c(model, n_params):
    print("\n" + "=" * 78, f"\nC. CPU SWEEP (fp32, {torch.get_num_threads()} threads, {SEQ_LEN} tokens/sequence)")
    return sweep(model, n_params, "cpu", [1, 2, 4, 8, 16], iters=5)


# ---------------------------------------------------------------- D. charts

def part_d(rows):
    print("\n" + "=" * 78, "\nD. CHARTS\n")
    colors = {"cuda": "#2a78d6", "cpu": "#eb6834"}
    names = {"cuda": "GPU", "cpu": "CPU"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for dev in ("cuda", "cpu"):
        r = [x for x in rows if x["device"] == dev]
        if not r:
            continue
        b = [x["batch"] for x in r]
        for ax, key in zip(axes, ("median_ms", "samples_per_s")):
            ax.plot(b, [x[key] for x in r], color=colors[dev], linewidth=2, marker="o", markersize=5, label=names[dev])
            ax.annotate(names[dev], (b[-1], r[-1][key]), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=9, color="#333")
    for ax, title, ylabel in ((axes[0], "Latency per batch (lower is better)", "median ms per forward pass"),
                              (axes[1], "Throughput (higher is better)", "sequences per second")):
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.set_yscale("log")
        ax.set_xlabel("batch size")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.25, which="both")
        ax.margins(x=0.12)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(frameon=False)
    fig.suptitle(f"GPT-2 (124M) forward pass, {SEQ_LEN} tokens per sequence, fp32")
    fig.tight_layout()
    fig.savefig(HERE / "batch_sweep.png", dpi=110)
    plt.close(fig)
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "day9.json").write_text(json.dumps(rows, indent=1))
    print("saved batch_sweep.png and results/day9.json")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        sys.exit("this benchmark needs a CUDA GPU")
    model = load_gpt2("cuda")
    n = part_a(model)
    rows = part_b(model, n)
    model = model.to("cpu")
    torch.cuda.empty_cache()
    rows += part_c(model, n)
    part_d(rows)
