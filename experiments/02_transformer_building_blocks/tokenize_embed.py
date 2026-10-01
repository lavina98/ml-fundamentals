"""Day 4: text -> token ids -> padded batch -> embeddings -> sentence similarity.

  A. Tokenization    - how different text types turn into tokens (and what it costs)
  B. Padding + masks - batching sentences of different lengths
  C. Embeddings      - token ids -> vectors; random vs trained; sentence similarity

YOUR PART: pad_batch(), mean_pool(), cosine_similarity_matrix() - marked TODO.

Run:  python tokenize_embed.py      (first run downloads a ~90 MB embedding model)
Out:  printed tables + similarity.png
"""
import sys

import matplotlib.pyplot as plt
import tiktoken
import torch
from torch import nn

sys.stdout.reconfigure(encoding="utf-8")  # Windows defaults to cp1252, which can't print Hindi

SENTENCES = [
    "The cat sat on the mat.",
    "A kitten was sitting on the rug.",
    "Stock markets fell sharply today.",
    "Shares dropped across global exchanges this afternoon.",
    "How do I reset my password?",
    "I forgot my login credentials and can't get into my account.",
]


# ---------------------------------------------------------------- YOUR PART

def pad_batch(seqs, pad_id=0):
    """TODO: turn a list of token-id lists (different lengths) into two tensors:

        ids:  LongTensor (B, T_max) - each row is a sequence, right-padded with pad_id
        mask: LongTensor (B, T_max) - 1 where there's a real token, 0 where it's padding

    Example:
        pad_batch([[5, 8, 2], [7]])  ->  ids  = [[5, 8, 2], [7, 0, 0]]
                                         mask = [[1, 1, 1], [1, 0, 0]]

    Hints:
      - T_max = the longest sequence's length
      - plain Python is fine: build padded lists, then torch.tensor(...)
    """
    max_len=0
    for s in seqs:
        max_len = max(len(s), max_len)

    result=[]
    mask=[]

    for s in seqs:
        r = list.copy(s)
        m =[1 for v in range(0, len(r))]
        if len(r) <max_len:
            for v in range(len(r), max_len):
                r.append(pad_id)
                m.append(0)

        result.append(r)
        mask.append(m)

    return torch.tensor(result), torch.tensor(mask)

def mean_pool(token_embs, mask):
    """TODO: average each sentence's token vectors into ONE sentence vector, ignoring padding.

        token_embs: (B, T, D)   one D-dim vector per token
        mask:       (B, T)      1 = real token, 0 = padding
        returns:    (B, D)

    Steps:
      1. m = mask.unsqueeze(-1).float()       # (B, T) -> (B, T, 1) so it lines up with (B, T, D)
      2. zero out padding vectors:  token_embs * m
      3. sum over the token dimension (which dim is T?)
      4. divide by the number of REAL tokens per sentence: m.sum(dim=1)   -> shape (B, 1)

    Why not just token_embs.mean(dim=1)? Part C prints both so you can see the difference.
    """
    m = mask.unsqueeze(-1).float()
    summed = (token_embs *m).sum(dim=1)
    counts = m.sum(dim=1)
    return summed/counts


def cosine_similarity_matrix(vecs):
    """TODO: all-pairs cosine similarity.

        vecs:    (N, D)
        returns: (N, N)  where out[i, j] = cos(angle between vecs[i] and vecs[j])

    cos(a, b) = (a . b) / (|a| |b|)   ->  1 = same direction, 0 = unrelated, -1 = opposite

    Steps:
      1. normalize every row to length 1:  vecs / vecs.norm(dim=1, keepdim=True)
      2. once rows have length 1, the dot product IS the cosine
      3. all pairs of dot products at once = one matrix multiply: normed @ normed.T
    """

    normed = vecs/vecs.norm(dim=1, keepdim=True)
    return normed @ normed.T


# ---------------------------------------------------------------- A. tokenization

def part_a():
    print("=" * 70, "\nA. TOKENIZATION\n")
    samples = {
        "English": "The quick brown fox jumps over the lazy dog.",
        "Code": "def add(a, b):\n    return a + b",
        "Hindi": "मुझे मशीन लर्निंग सीखना पसंद है।",
        "Rare word": "Pneumonoultramicroscopicsilicovolcanoconiosis",
        "Numbers": "Order 1234567 shipped on 2026-09-30 for $1,299.99",
    }
    encoders = {"gpt2 (50k vocab)": tiktoken.get_encoding("gpt2"),
                "cl100k (100k vocab)": tiktoken.get_encoding("cl100k_base")}

    enc = encoders["cl100k (100k vocab)"]
    ids = enc.encode(samples["English"])
    print(f"'{samples['English']}'")
    print(f"  ids:    {ids}")
    print(f"  pieces: {[enc.decode([i]) for i in ids]}")
    print(f"  decode(ids) round-trips: {enc.decode(ids) == samples['English']}\n")

    print(f"{'text':<11}{'chars':>6} | " + " | ".join(f"{n:>20}" for n in encoders))
    for name, text in samples.items():
        cells = []
        for enc in encoders.values():
            n = len(enc.encode(text))
            cells.append(f"{n:>3} tok ({len(text) / n:4.1f} ch/tok)")
        print(f"{name:<11}{len(text):>6} | " + " | ".join(f"{c:>20}" for c in cells))

    enc = encoders["cl100k (100k vocab)"]
    print("\nHow the rare word and the Hindi get split (cl100k):")
    for name in ("Rare word", "Hindi"):
        pieces = [enc.decode_single_token_bytes(i).decode("utf-8", errors="replace")
                  for i in enc.encode(samples[name])]
        print(f"  {name}: {len(pieces)} tokens, first 8 = {pieces[:8]}")
    print("  (� = a token holding only PART of a character's UTF-8 bytes)")


# ---------------------------------------------------------------- B. padding

def part_b():
    print("\n" + "=" * 70, "\nB. PADDING + ATTENTION MASKS\n")
    enc = tiktoken.get_encoding("cl100k_base")
    seqs = [enc.encode(s) for s in SENTENCES[:3]]

    ids, mask = pad_batch(seqs)
    print(f"lengths: {[len(s) for s in seqs]}  ->  ids {tuple(ids.shape)}, mask {tuple(mask.shape)}")
    print(f"ids:\n{ids}\nmask:\n{mask}")
    assert ids.shape == mask.shape == (3, max(len(s) for s in seqs)), "shape is off"
    assert mask.sum().item() == sum(len(s) for s in seqs), "mask should count real tokens exactly"
    print(f"note: pad id 0 is a REAL token in cl100k ({enc.decode([0])!r}) - only the mask says it's padding")

    # Padding waste on a realistic mix of lengths
    torch.manual_seed(0)
    lengths = torch.randint(5, 200, (512,)).tolist()          # 512 requests, 5..200 tokens
    fake = [[1] * n for n in lengths]

    def waste(batches):
        total = real = 0
        for b in batches:
            _, m = pad_batch(b)
            total += m.numel()
            real += m.sum().item()
        return 1 - real / total

    bs = 32
    arrival_order = [fake[i:i + bs] for i in range(0, len(fake), bs)]
    by_length = sorted(fake, key=len)
    bucketed = [by_length[i:i + bs] for i in range(0, len(by_length), bs)]
    print(f"\n512 sequences of 5-200 tokens, batch size {bs}:")
    print(f"  padding waste, batches in arrival order: {waste(arrival_order):.0%}")
    print(f"  padding waste, batches sorted by length: {waste(bucketed):.0%}")
    print("  (every padded position still costs compute and memory on the GPU)")


# ---------------------------------------------------------------- C. embeddings

def plot_similarity(mats, path="similarity.png"):
    labels = [s if len(s) < 34 else s[:31] + "..." for s in SENTENCES]
    fig, axes = plt.subplots(1, len(mats), figsize=(7.5 * len(mats), 6.2))
    for k, (ax, (title, m)) in enumerate(zip(axes, mats.items())):
        im = ax.imshow(m, cmap="RdBu_r", vmin=-1, vmax=1)   # diverging: -1 blue, 0 white, +1 red
        ax.set_title(title, fontsize=11)
        ax.set_xticks(range(len(labels)), [str(i) for i in range(len(labels))])
        ytick_labels = [f"{i}  {l}" for i, l in enumerate(labels)] if k == 0 else [str(i) for i in range(len(labels))]
        ax.set_yticks(range(len(labels)), ytick_labels, fontsize=8)
        for i in range(len(m)):
            for j in range(len(m)):
                ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if abs(m[i, j]) > 0.6 else "#222")
    fig.colorbar(im, ax=axes, shrink=0.8, label="cosine similarity")
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def part_c():
    print("\n" + "=" * 70, "\nC. EMBEDDINGS\n")
    enc = tiktoken.get_encoding("cl100k_base")
    ids, mask = pad_batch([enc.encode(s) for s in SENTENCES])

    # C1. An embedding layer is just a lookup table: row i = the vector for token i.
    torch.manual_seed(0)
    emb = nn.Embedding(enc.n_vocab, 64)
    out = emb(ids)
    print(f"nn.Embedding({enc.n_vocab:,}, 64): ids {tuple(ids.shape)} -> vectors {tuple(out.shape)}")
    print(f"  is it just row lookup? emb(ids) == emb.weight[ids]: {torch.equal(out, emb.weight[ids])}")
    print(f"  table size: {emb.weight.numel():,} params")
    for vocab, dim, name in [(100_277, 4096, "cl100k x 4096 (a 7B-class model)"),
                             (128_256, 8192, "128k x 8192 (a 70B-class model)")]:
        print(f"  {name}: {vocab * dim / 1e6:,.0f}M params = {vocab * dim * 2 / 1e9:.1f} GB in bf16")

    # C2. Masked vs naive mean - does padding leak into the sentence vector?
    naive = out.mean(dim=1)
    pooled = mean_pool(out, mask)
    shortest = int(mask.sum(dim=1).argmin())
    diff = (naive[shortest] - pooled[shortest]).norm() / pooled[shortest].norm()
    print(f"\nshortest sentence ({int(mask[shortest].sum())} real tokens, "
          f"{int((1 - mask[shortest]).sum())} pads): naive mean differs from masked mean by {diff:.0%}")

    # C3. Random embeddings vs a trained sentence-embedding model
    random_sim = cosine_similarity_matrix(pooled.detach())

    from transformers import AutoModel, AutoTokenizer
    name = "sentence-transformers/all-MiniLM-L6-v2"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModel.from_pretrained(name).eval()
    batch = tok(SENTENCES, padding=True, return_tensors="pt")   # the library does pad_batch for you
    with torch.no_grad():
        hidden = model(**batch).last_hidden_state                # (B, T, 384): one vector per token
    trained_sim = cosine_similarity_matrix(mean_pool(hidden, batch["attention_mask"]))
    print(f"\nMiniLM ({sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params): "
          f"token vectors {tuple(hidden.shape)} -> sentence vectors ({len(SENTENCES)}, {hidden.shape[-1]})")

    print("\ncosine similarity with sentence 0 ('The cat sat on the mat.'):")
    print(f"  {'':<52}{'random':>8}{'trained':>9}")
    for i, s in enumerate(SENTENCES):
        print(f"  {i} {s[:50]:<50}{random_sim[0, i]:>8.2f}{trained_sim[0, i]:>9.2f}")

    plot_similarity({"random nn.Embedding (untrained)": random_sim.numpy(),
                     "all-MiniLM-L6-v2 (trained)": trained_sim.numpy()})
    print("\nsaved similarity.png")


if __name__ == "__main__":
    part_a()
    part_b()
    part_c()
