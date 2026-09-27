# 02 · Transformer building blocks

**Days 4–5.** Goal: tokenization → embeddings → attention, each implemented by hand and checked against PyTorch.

## Tasks
- [ ] Tokenize text with `tiktoken`; compare tokens per word on English, code, and another language
- [ ] Batch variable-length sequences: padding + attention mask; measure the padding waste (%)
- [ ] `nn.Embedding` lookup; cosine similarity between a few sentence embeddings
- [ ] Scaled dot-product attention by hand, with a causal mask; `torch.allclose` against `F.scaled_dot_product_attention`
- [ ] Multi-head: reshape to (B, heads, T, d_head) and back

## Observations
-
