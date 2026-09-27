# 03 · Tiny GPT

**Days 6–8.** Goal: a character-level GPT (~1–5M params) trained on Tiny Shakespeare, trained and sampled end to end.

## Tasks
- [ ] Data: character vocab, train/val split, random (B, T) batches
- [ ] Model: embeddings + N transformer blocks (reuse your attention from 02) + LM head
- [ ] Train for 10–20 minutes on the RTX 3050; log train/val loss; save a checkpoint
- [ ] `generate.py`: load the checkpoint and sample autoregressively (greedy + temperature)
- [ ] Record: param count, tokens/sec during training, peak GPU memory

## Observations
-
