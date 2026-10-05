"""Day 6: a tiny character-level GPT - data, model, and a sanity check.

  A. Data   - Tiny Shakespeare -> characters -> ids; random (B, T) training batches
  B. Model  - embeddings -> N transformer blocks -> logits; check the untrained loss
  C. Sanity - overfit ONE batch: if the loss can't go to ~0, something is broken

Day 7 (train.py) trains this model for real; Day 8 (generate.py) samples from it.

YOUR PART: get_batch(), Block.forward(), GPT.forward() - marked TODO.
Your MultiHeadSelfAttention from Day 5 is imported and used as-is.

Run:  python gpt.py      (first run downloads ~1 MB of Shakespeare)
"""
import math
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "02_transformer_building_blocks"))
from attention import MultiHeadSelfAttention  # noqa: E402  (your Day 5 layer)

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


@dataclass
class Config:
    vocab_size: int = 65
    block_size: int = 128   # T: the longest context the model sees (in characters)
    d_model: int = 256      # D: numbers per token
    n_heads: int = 4        # H: d_head = 256 / 4 = 64
    n_layers: int = 4       # number of transformer blocks stacked
    dropout: float = 0.1


# ---------------------------------------------------------------- data (boilerplate)

def load_text() -> str:
    path = HERE / "data" / "input.txt"
    if not path.exists():
        path.parent.mkdir(exist_ok=True)
        print(f"downloading {DATA_URL}")
        urllib.request.urlretrieve(DATA_URL, path)
    return path.read_text(encoding="utf-8")


class CharTokenizer:
    """The simplest tokenizer: every distinct character is one token (vocab 65, vs cl100k's 100k)."""

    def __init__(self, text: str):
        self.chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(self.chars)}

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids) -> str:
        return "".join(self.chars[int(i)] for i in ids)


def load_data() -> tuple[CharTokenizer, torch.Tensor, torch.Tensor]:
    """Returns (tokenizer, train_ids, val_ids). The ids are one long 1-D LongTensor each."""
    text = load_text()
    tok = CharTokenizer(text)
    data = torch.tensor(tok.encode(text), dtype=torch.long)
    n = int(0.9 * len(data))            # first 90% train, last 10% val
    return tok, data[:n], data[n:]


# ---------------------------------------------------------------- YOUR PART

def get_batch(data: torch.Tensor, batch_size: int, block_size: int,
              device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
    """TODO: pick `batch_size` random windows from the 1-D `data`. Return (x, y), both (B, T) on `device`.

    y is x shifted ONE character to the left, so at every position the target is the NEXT character:
        data:  ... F  i  r  s  t  _  C ...
        x   :      F  i  r  s  t           (T = 5)
        y   :         i  r  s  t  _        position 0 learns "after F comes i", position 4 "after t comes _"
    One window of T characters = T training examples at once.

    Steps:
      1. starts = torch.randint(0, len(data) - block_size, (batch_size,))    # B random start offsets
         (why "- block_size"? y needs one character past the end of x)
      2. x = torch.stack([data[s : s + block_size] for s in starts])          # (B, T)
      3. y = the same, but each window starts one later                        # (B, T)
      4. return both moved .to(device)
    """
    starts = torch.randint(0, len(data) - block_size, (batch_size,))
    x = torch.stack([data[s:s+block_size] for s in starts])
    y = torch.stack([data[s+1:s+block_size+1] for s in starts])

    return x.to(device=device), y.to(device=device)

class Block(nn.Module):
    """One transformer block = communicate (attention) then think (MLP), each wrapped in a residual."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = MultiHeadSelfAttention(cfg.d_model, cfg.n_heads)        # yours, from Day 5
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(                                            # per-token feed-forward
            nn.Linear(cfg.d_model, 4 * cfg.d_model),
            nn.GELU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model),
        )
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """TODO: (B, T, D) -> (B, T, D)

          1. x = x + self.drop(self.attn(self.ln1(x), causal=True))   # tokens gather info from earlier tokens
          2. x = x + self.drop(self.mlp(self.ln2(x)))                 # each token processes what it gathered
          3. return x

        - ln1/ln2 (LayerNorm): rescale each token's D numbers to mean 0, std 1, so values stay
          in a sane range however many blocks you stack
        - "x + ...": the RESIDUAL connection. A block only ADDS a correction to x; gradients flow
          straight back through the "+" to the early layers, which is what makes deep stacks trainable
        - causal=True: GPT predicts the next character, so no token may see the future
        """

        x = x +self.drop(self.attn(self.ln1(x), causal = True))
        x = x +self.drop(self.mlp(self.ln2(x)))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)   # WHICH character (Day 4's lookup table)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.d_model)   # WHERE it is: position 0..T-1
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size)       # D numbers -> one score per character

    def forward(self, idx: torch.Tensor,
                targets: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        """TODO: idx (B, T) token ids -> logits (B, T, vocab). Also the loss if targets (B, T) are given.

          1. B, T = idx.shape
             pos = torch.arange(T, device=idx.device)              # [0, 1, ..., T-1]
          2. x = self.tok_emb(idx) + self.pos_emb(pos)             # (B, T, D) + (T, D) -> broadcasting!
             x = self.drop(x)
          3. run x through every block in self.blocks, in order
          4. logits = self.lm_head(self.ln_f(x))                   # (B, T, vocab)
          5. loss = None. If targets is given:
                 cross_entropy wants (N, classes) and (N,) - so fold B and T into one axis:
                 loss = F.cross_entropy(logits.view(B * T, -1), targets.view(B * T))
          6. return logits, loss

        Why pos_emb? Attention is a weighted average - it has no idea of ORDER. Without it,
        "dog bites man" and "man bites dog" look identical. pos_emb adds "I'm at position t".
        """

        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) +  self.pos_emb(pos)
        x = self.drop(x)
        for b in self.blocks:
            x = b(x)

        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(B*T, -1), targets.view(B*T))

        return logits, loss

# ---------------------------------------------------------------- A. data

def part_a():
    print("=" * 70, "\nA. DATA\n")
    tok, train, val = load_data()
    print(f"{len(train) + len(val):,} characters, vocab = {tok.vocab_size} distinct characters:")
    print(f"  {''.join(tok.chars)!r}")
    print(f"  train {len(train):,} / val {len(val):,}")
    print(f"  encode('Hello') = {tok.encode('Hello')}  ->  decode = {tok.decode(tok.encode('Hello'))!r}")

    torch.manual_seed(0)
    x, y = get_batch(train, batch_size=4, block_size=8)
    print(f"\nget_batch(B=4, T=8): x {tuple(x.shape)}, y {tuple(y.shape)}")
    print(f"  y is x shifted by one: {torch.equal(x[:, 1:], y[:, :-1])}")
    print(f"  example window:  x = {tok.decode(x[0])!r}")
    print(f"                   y = {tok.decode(y[0])!r}")
    print("  the 8 training examples inside that one window:")
    for t in range(8):
        print(f"    context {tok.decode(x[0, :t + 1])!r:<12} -> next {tok.decode(y[0, t:t + 1])!r}")


# ---------------------------------------------------------------- B. model

def part_b(dev: str):
    print("\n" + "=" * 70, "\nB. MODEL\n")
    tok, train, _ = load_data()
    cfg = Config(vocab_size=tok.vocab_size)
    torch.manual_seed(0)
    model = GPT(cfg).to(dev)

    count = lambda m: sum(p.numel() for p in m.parameters())
    print(f"{cfg}")
    print(f"params: {count(model):,} total")
    print(f"  embeddings (tok + pos): {count(model.tok_emb) + count(model.pos_emb):>9,}")
    print(f"  per block:              {count(model.blocks[0]):>9,}   "
          f"(attn {count(model.blocks[0].attn):,} + mlp {count(model.blocks[0].mlp):,} + layernorms)")
    print(f"  x {cfg.n_layers} blocks:             {count(model.blocks):>9,}")
    print(f"  lm_head:                {count(model.lm_head):>9,}")

    x, y = get_batch(train, 8, cfg.block_size, dev)
    logits, loss = model(x, y)
    print(f"\nforward: idx {tuple(x.shape)} -> logits {tuple(logits.shape)}")
    print(f"untrained loss: {loss.item():.3f}   vs  -ln(1/{cfg.vocab_size}) = {math.log(cfg.vocab_size):.3f}"
          f"  (random guessing among {cfg.vocab_size} characters - same idea as Day 2's 2.30 for 10 digits)")
    _, no_loss = model(x)
    print(f"no targets -> loss is {no_loss}")


# ---------------------------------------------------------------- C. sanity check

def part_c(dev: str, steps: int = 300):
    print("\n" + "=" * 70, "\nC. SANITY CHECK: OVERFIT ONE BATCH\n")
    print("Same batch every step. A working model + loop memorizes it (loss -> ~0).")
    print("If it doesn't, there's a bug - find it now, before a 20-minute training run.\n")
    tok, train, _ = load_data()
    cfg = Config(vocab_size=tok.vocab_size, dropout=0.0)
    torch.manual_seed(0)
    model = GPT(cfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x, y = get_batch(train, 16, cfg.block_size, dev)

    t0 = time.perf_counter()
    for step in range(steps + 1):
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 50 == 0:
            print(f"  step {step:>3}: loss {loss.item():.3f}")
    if dev == "cuda":
        torch.cuda.synchronize()
    secs = time.perf_counter() - t0
    print(f"{steps} steps in {secs:.1f}s  ->  {steps * x.numel() / secs:,.0f} tokens/sec")
    print("memorized the batch" if loss.item() < 0.2 else "loss did NOT go to ~0 - check your code")


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    part_a()
    part_b(dev)
    part_c(dev)
