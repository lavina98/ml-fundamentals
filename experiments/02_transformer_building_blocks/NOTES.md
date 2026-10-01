# 02 · Transformer building blocks

**Days 4–5.** Goal: tokenization → embeddings → attention, each implemented by hand and checked against PyTorch.

## Tasks
- [x] Tokenize text with `tiktoken`; compare tokens per word on English, code, and another language
- [x] Batch variable-length sequences: padding + attention mask; measure the padding waste (%)
- [x] `nn.Embedding` lookup; cosine similarity between a few sentence embeddings
- [ ] Scaled dot-product attention by hand, with a causal mask; `torch.allclose` against `F.scaled_dot_product_attention`
- [ ] Multi-head: reshape to (B, heads, T, d_head) and back

## Day 4: Text → tokens → padded batch → embeddings (`tokenize_embed.py`)

```
"The cat sat"  →  [791, 8415, 7731]  →  ids + mask (B, T)  →  vectors (B, T, D)  →  sentence vector (B, D)
   text           tokenizer              pad_batch             nn.Embedding / model    mean_pool
```

### A. Tokenization
| Text | Chars | gpt2 (50k vocab) | cl100k (100k vocab) |
|---|---|---|---|
| English | 44 | 10 tok (4.4 ch/tok) | 10 tok (4.4 ch/tok) |
| Code | 31 | 15 tok (2.1) | 11 tok (2.8) |
| Hindi | 32 | 53 tok (**0.6**) | 35 tok (**0.9**) |
| Rare word | 45 | 15 tok (3.0) | 17 tok (2.6) |
| Numbers | 49 | 19 tok (2.6) | 21 tok (2.3) |

- **Tokens aren't words.** Common English words are one token each, with the leading space included (`' quick'`).
- **Hindi costs 5–7× more tokens per character than English.** The tokenizers were trained mostly on English. More tokens means higher API cost, slower generation, and more of the context window used up for the same meaning.
- **Tokenizers work on bytes:** a Devanagari character is 3 bytes in UTF-8, and some tokens hold only part of one (`�`). So nothing is ever out of vocabulary: the rare word becomes 17 subword pieces (`P`, `ne`, `um`, …).
- **A bigger vocabulary helps code and non-English** (code 15 → 11, Hindi 53 → 35), at the cost of a bigger embedding table.
- **Numbers split into chunks** (`1234567` isn't one token), which is part of why LLMs struggle with arithmetic.

### B. Padding + attention masks
- `pad_batch` right-pads to the longest sequence and returns `ids` and `mask` (1 = real, 0 = pad), both `(B, T_max)`.
- **The pad id 0 is a real token in cl100k (`'!'`).** Only the mask tells the model which positions are padding, so the mask must always travel with the ids.
- **Padding waste**, 512 sequences of 5–200 tokens, batch size 32:
  - batches in arrival order: **50%** of positions are padding
  - batches sorted by length: **6%**
- Every padded position still costs GPU compute and memory, so random batching wastes half the work. That's why inference servers bucket requests by length or use continuous batching (e.g. vLLM).

### C. Embeddings
- **`nn.Embedding` is a lookup table:** `emb(ids) == emb.weight[ids]` → True. `(B, T)` ids → `(B, T, D)` vectors. Row *i* is the vector for token *i*.
- **The table is big at scale:** 100k × 64 = 6.4M params here; 100k × 4096 = 411M (0.8 GB bf16) for a 7B-class model; 128k × 8192 ≈ 1B (2.1 GB) for a 70B-class model.
- **Masked vs naive mean:** for the shortest sentence (6 real tokens, 7 pads), `mean(dim=1)` differed from the masked mean by **128%**. The padding dominated the average.
- **Random vs trained embeddings**, cosine similarity with "The cat sat on the mat.":

  | Sentence | Random `nn.Embedding` | Trained MiniLM |
  |---|---|---|
  | A kitten was sitting on the rug. | 0.57 | 0.65 |
  | Stock markets fell sharply today. | 0.33 | 0.04 |
  | Shares dropped across global exchanges… | 0.04 | 0.03 |
  | How do I reset my password? | −0.14 | 0.08 |
  | I forgot my login credentials… | −0.10 | 0.01 |

  ![similarity](similarity.png)

  - **Random embeddings measure word overlap, not meaning.** Cat ↔ kitten scored 0.57 only because they share ` on`, ` the`, `.`, and cat ↔ stocks scored 0.33 from sharing `.`.
  - **The trained model groups by meaning:** stocks ↔ shares 0.57 and password ↔ login 0.69, even though those pairs share **no words**. Unrelated topics sit near 0.
  - **Embeddings only mean something after training.** Embed → mean-pool → cosine similarity is the core of semantic search and RAG.

### PyTorch concepts that clicked today
- **`dim=N` collapses axis N**: `(B, T, D).sum(dim=1)` → `(B, D)`. Cross the named axis out of the shape. dim 0 is almost always the batch, so most reductions use `dim=1` or `dim=-1`.
- **Broadcasting:** shapes line up from the right, and size-1 axes stretch to match. `(B, T, D) * (B, T, 1)` works; `(B, T, D) * (B, T)` doesn't. Hence `unsqueeze(-1)` in `mean_pool` and `keepdim=True` in the cosine norm.
- **Masks do math too:** multiplying by a 0/1 mask zeros out padding before a sum (Day 3 used masks to *select*; today they *weight*).
- **All-pairs similarity = one matrix multiply:** normalize the rows, then `normed @ normed.T` → `(N, N)`. Attention's `Q @ K.T` (Day 5) is the same shape of computation.
