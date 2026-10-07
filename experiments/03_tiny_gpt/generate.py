"""Day 8: make the trained GPT write - and measure why LLM generation is slow.

  A. Untrained vs trained       - the same prompt through a random model and your checkpoint
  B. Temperature                - how dividing the logits changes the next-char probabilities and the text
  C. Infra: cost per token      - every new token re-runs the WHOLE context (why the KV cache exists)
  D. Infra: batched generation  - 1 sequence vs many at once (why servers batch users together)

YOUR PART: pick_next(), generate() - marked TODO.

Run:  python generate.py        (needs checkpoints/gpt.pt from Day 7)
"""
import sys
import time

import torch
import torch.nn.functional as F

from gpt import HERE, GPT, CharTokenizer, Config
from train import use_attention

sys.stdout.reconfigure(encoding="utf-8")
PROMPT = "ROMEO:\n"


# ---------------------------------------------------------------- YOUR PART

def pick_next(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """TODO: choose the next token for every sequence in the batch.

        logits:  (B, vocab)   the scores for the NEXT token (already just the last position)
        returns: (B, 1)       one token id per sequence (the 1 makes it easy to append later)

    temperature == 0  ->  GREEDY: always take the highest score
        torch.argmax(logits, dim=-1, keepdim=True)
    temperature > 0   ->  SAMPLE: roll weighted dice
        1. probs = torch.softmax(logits / temperature, dim=-1)
               t < 1 -> scores spread apart -> the top choice gets MORE likely (safer)
               t > 1 -> scores squashed together -> unlikely chars get a real chance (wilder)
        2. torch.multinomial(probs, num_samples=1)     # picks index i with probability probs[i]
    """


    if temperature==0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    
    probs = torch.softmax(logits/temperature, dim=-1)
    return torch.multinomial(probs, num_samples=1)



@torch.no_grad()
def generate(model: GPT, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0) -> torch.Tensor:
    """TODO: autoregressive generation - predict one token, append it, repeat.

        idx:     (B, T0)                    the prompt token ids
        returns: (B, T0 + max_new_tokens)   prompt + generated tokens

    Repeat max_new_tokens times:
      1. idx_cond = idx[:, -model.cfg.block_size:]   # keep only the last 128 tokens: pos_emb has 128 rows.
                                                    # This IS the "context window" - older text is forgotten
      2. logits, _ = model(idx_cond)                 # (B, T, vocab) - no targets, so the loss is None
      3. logits = logits[:, -1, :]                   # (B, vocab) - only the LAST position predicts the new token
      4. nxt = pick_next(logits, temperature)        # (B, 1)
      5. idx = torch.cat([idx, nxt], dim=1)          # append along the time axis -> (B, T + 1)
    Return idx.
    """

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.cfg.block_size:]   # (B, <=128)  the context window
        logits, _ = model(idx_cond)                 # (B, T, vocab)  a prediction at every position
        logits = logits[:, -1, :]                   # (B, vocab)     keep only the last one
        nxt = pick_next(logits, temperature)        # (B, 1)         one new token per sequence
        idx = torch.cat([idx, nxt], dim=1)          # (B, T + 1)     append along the time axis

    return idx


# ---------------------------------------------------------------- loading (boilerplate)

def load_model(dev: str, trained: bool = True) -> tuple[GPT, CharTokenizer]:
    ckpt = torch.load(HERE / "checkpoints" / "gpt.pt", map_location=dev)
    tok = CharTokenizer("".join(ckpt["chars"]))        # same 65 chars -> same ids as in training
    cfg = Config(**ckpt["cfg"])
    torch.manual_seed(0)
    model = GPT(cfg).to(dev)
    if trained:
        model.load_state_dict(ckpt["model"])
    model.eval()                                       # dropout OFF for generation
    return model, tok


def write(model, tok, dev, n=300, temperature=1.0, prompt=PROMPT, seed=0) -> str:
    torch.manual_seed(seed)
    idx = torch.tensor([tok.encode(prompt)], device=dev)   # (1, T0)
    return tok.decode(generate(model, idx, n, temperature)[0])


def show(title: str, text: str):
    print(f"--- {title} " + "-" * max(0, 60 - len(title)))
    print(text.rstrip())
    print()


def sync(dev):
    if dev == "cuda":
        torch.cuda.synchronize()


# ---------------------------------------------------------------- A. untrained vs trained

def part_a(dev):
    print("=" * 70, "\nA. UNTRAINED vs TRAINED  (same prompt, same seed, temperature 1.0)\n")
    for trained, title in ((False, "untrained (random weights, loss ~4.2)"),
                           (True, "trained (your Day 7 checkpoint, val loss 1.49)")):
        model, tok = load_model(dev, trained)
        show(title, write(model, tok, dev))


# ---------------------------------------------------------------- B. temperature

def part_b(dev):
    print("=" * 70, "\nB. TEMPERATURE\n")
    model, tok = load_model(dev)
    context = "ROMEO:\nWhat light through yonder w"
    idx = torch.tensor([tok.encode(context)], device=dev)
    with torch.no_grad():
        logits = model(idx)[0][0, -1]                 # (vocab,) scores for the char after "...yonder w"
    print(f"next-char probabilities after {context[-20:]!r}:")
    print(f"  {'temp':>5} | top 5")
    for t in (0.5, 1.0, 1.5):
        probs = torch.softmax(logits / t, dim=-1)
        top = torch.topk(probs, 5)
        cells = "  ".join(f"{tok.decode([int(i)])!r} {p:.2f}" for p, i in zip(top.values, top.indices))
        print(f"  {t:>5} | {cells}")
    print("  (low temp sharpens toward the top choice; high temp flattens toward uniform)\n")

    for t, note in ((0, "greedy: always the top choice"), (0.5, "safe"), (1.0, "the model's own distribution"),
                    (1.5, "wild")):
        show(f"temperature {t}  ({note})", write(model, tok, dev, n=250, temperature=t))


# ---------------------------------------------------------------- C. cost per token

def part_c(dev, n_new=50):
    print("=" * 70, "\nC. INFRA: COST PER GENERATED TOKEN vs CONTEXT LENGTH\n")
    print("Every step re-runs the model over the whole context, but only the LAST position is used.\n")
    print(f"{'device':>6} | {'context':>9} | {'ms / token':>10} | {'positions computed':>18} | {'wasted':>6}")
    for d in ([dev, "cpu"] if dev == "cuda" else ["cpu"]):
        model, tok = load_model(d)
        for start in (1, 64, 128):
            idx = torch.zeros(1, start, dtype=torch.long, device=d)
            generate(model, idx, 3)                         # warm-up
            sync(d)
            t0 = time.perf_counter()
            generate(model, idx, n_new)
            sync(d)
            ms = (time.perf_counter() - t0) / n_new * 1000
            ctx = [min(start + i, model.cfg.block_size) for i in range(n_new)]
            avg = sum(ctx) / n_new                          # positions the model ran per new token
            label = f"{ctx[0]}-{ctx[-1]}" if ctx[0] != ctx[-1] else f"{ctx[0]}"
            print(f"{d:>6} | {label:>9} | {ms:>10.2f} | {avg:>18.0f} | {1 - 1 / avg:>6.0%}")
    print("\n'wasted' = positions recomputed only to throw away. The keys/values of old tokens never change,")
    print("so a KV cache stores them and each step processes just 1 new position.")


# ---------------------------------------------------------------- D. batching

def part_d(dev, n_new=100):
    print("\n" + "=" * 70, f"\nD. INFRA: BATCHED GENERATION ({n_new} new tokens per sequence, on {dev})\n")
    model, tok = load_model(dev)
    prompt = torch.tensor(tok.encode(PROMPT), device=dev)
    print(f"{'batch':>6} | {'total time':>10} | {'tokens/sec':>11} | {'ms per step':>11}")
    base = None
    for B in (1, 8, 32, 64, 128, 256):
        idx = prompt.repeat(B, 1)                           # (B, T0): B users, same prompt
        generate(model, idx, 3)                             # warm-up
        sync(dev)
        t0 = time.perf_counter()
        generate(model, idx, n_new)
        sync(dev)
        secs = time.perf_counter() - t0
        tps = B * n_new / secs
        base = base or tps
        print(f"{B:>6} | {secs:>9.2f}s | {tps:>11,.0f} | {secs / n_new * 1000:>11.2f}   ({tps / base:.0f}x batch 1)")
    print("\nSmall batches: a step costs about the same for 1 user or 8 - the GPU is mostly idle.")
    print("Once it's saturated, ms per step grows with the batch and tokens/sec stops improving.")


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    use_attention("sdpa")
    part_a(dev)
    part_b(dev)
    part_c(dev)
    part_d(dev)
