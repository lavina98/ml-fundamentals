# 03 · Tiny GPT

**Days 6–8.** Goal: a character-level GPT (~1–5M params) trained on Tiny Shakespeare, trained and sampled end to end.

## Tasks
- [x] Data: character vocab, train/val split, random (B, T) batches
- [x] Model: embeddings + N transformer blocks (reuse your attention from 02) + LM head
- [ ] Train for 10–20 minutes on the RTX 3050; log train/val loss; save a checkpoint
- [ ] `generate.py`: load the checkpoint and sample autoregressively (greedy + temperature)
- [ ] Record: param count, tokens/sec during training, peak GPU memory


## Day 6: Data + model + sanity check (`gpt.py`)

```
text ─► char ids (1-D, 1.1M) ─► get_batch: x, y (B, T) ─► tok_emb + pos_emb (B,T,D) ─► 4 × Block ─► ln_f ─► lm_head ─► logits (B,T,65) ─► cross_entropy
                                                                                      Block: x = x + attn(ln1(x), causal)   ← my Day 5 attention
                                                                                             x = x + mlp(ln2(x))
```

### A. Data
- Tiny Shakespeare: 1,115,394 characters, **vocab = 65** (one token per character), 90/10 train/val split.
- `get_batch`: `randint(0, len - T, (B,))` random starts (`high` is exclusive, so the last start still leaves room for y), then `torch.stack` the windows → `(B, T)`.
- **y = x shifted by one.** One window of T characters = T training examples (`'h'→'e'`, `'he'→' '`, …): the model predicts the next character at every position at once.
- `stack` adds a NEW axis (B windows of `(T,)` → `(B, T)`); `cat` glues along an existing axis (Day 5's q/k/v weights → `(3D, D)`).

### B. Model (`Config`: T=128, D=256, 4 heads, 4 layers)
| Part | Params |
|---|---|
| Embeddings (tok + pos) | 49,408 |
| One block: attention / MLP | 263,168 / 525,568 |
| 4 blocks | 3,159,040 (98%) |
| lm_head | 16,705 |
| **Total** | **3,225,665** |

- **The MLP is 2× attention in every block:** 8·D² (D→4D→D) vs 4·D² (q, k, v, out). Most of a transformer's parameters are in the MLPs.
- The embeddings are tiny because the vocab is 65 (vs 411M for a 100k × 4096 table, Day 4).
- **Untrained loss 4.29 ≈ ln(65) = 4.17**: random guessing among 65 characters (Day 2: 2.30 = ln 10).
- `pos_emb(arange(T))` is `(T, D)` with no B axis, because position t means the same thing in every sequence; broadcasting adds it to all B rows. Without it, attention has no notion of order.
- Loss: `cross_entropy(logits.view(B*T, -1), targets.view(B*T))`. Every position is one prediction; the targets are class **indices** `(N,)`, not scores.
- Residual `x = x + f(x)` works because the shape stays `(B, T, D)` through every block; gradients flow straight back through the `+`.

### C. Sanity check: overfit one batch
- The same batch (16 × 128) for 300 steps: loss **4.31 → 0.006** in 5.4 s, **~114k tokens/sec** (RTX 3050, my naive attention).
- Proves that the model, gradients, and optimizer are wired correctly *before* a long run. Memorizing one batch ≠ learning; the val loss on Day 7 is the real test.

### Bugs I hit
- `if targets:` → "Boolean value of Tensor … ambiguous". Use `if targets is not None:`.
- Forgot `ln_f` before `lm_head`: no error, it just trains worse. Silent bugs only get caught by review/checks.
