# ML Fundamentals for Software Engineers

A 2-week sprint of hands-on PyTorch experiments covering the ML fundamentals ML infrastructure roles expect: training vs inference, loss and gradient descent, overfitting, tokenization, embeddings, attention, transformers, and batching. The experiments lean toward measurements: latency, throughput, and memory.

## Experiments

| # | Experiment | Days | Status |
|---|---|---|---|
| 01 | [Training loop](experiments/01_training_loop/): gradient descent, FashionMNIST train/infer, overfitting | 1–3 | ✅ |
| 02 | [Transformer building blocks](experiments/02_transformer_building_blocks/): tokenization, embeddings, attention from scratch | 4–5 | ✅ |
| 03 | [Tiny GPT](experiments/03_tiny_gpt/): character-level transformer trained on Tiny Shakespeare | 6–8 | ✅ |
| 04 | [Inference benchmark](experiments/04_inference_benchmark/): batch size, CPU vs GPU, fp32 vs bf16, memory, KV cache, TTFT vs per-token latency | 9–11 | ✅ |

Each folder has a `NOTES.md` with the task checklist and observations, including measured numbers.

## Key findings

All numbers were measured on one RTX 3050 (8 GB). GPT-2 small (124M) for experiment 04.

| | Finding | Number | Where |
|---|---|---|---|
| 1 | **The latency/throughput knee.** Throughput saturates at small batches; past the knee you pay latency for almost nothing | Batch 4 reaches **95% of peak throughput at 33 ms**; batch 128 takes **1,004 ms** for the last 5% | [04 · Day 9](experiments/04_inference_benchmark/NOTES.md) |
| 2 | **bf16 is faster and the same quality, but not bit-identical** | **2.5× faster** prefill, **half the memory** (2× the batch), loss within **0.01%**, yet **11.5%** of greedy next-token picks flip (all near-ties) | [04 · Day 10](experiments/04_inference_benchmark/NOTES.md) |
| 3 | **The KV cache turns decode from quadratic to ~flat, and its memory limits users per GPU** | **11.6×** faster per token at 1k context, identical output; 32 users × 1k tokens = **2.4 GB of cache, 5× the weights** | [04 · Day 11](experiments/04_inference_benchmark/NOTES.md) |
| 4 | **Batch-1 decode is overhead-bound, not memory-bound.** The GPU waits on kernel launches, so making the math cheaper doesn't help | bf16 decode **1.04×** at batch 1 vs **2.14×** at batch 32; batch 8 cost the same as batch 1 | [04 · Day 11](experiments/04_inference_benchmark/NOTES.md) |
| 5 | **Amdahl's law applies to attention.** A fused attention kernel is huge on its own but modest inside a full model | `F.scaled_dot_product_attention`: **278× less memory, 14× faster** at 8k tokens in isolation → **1.16–1.42×** end-to-end training | [02 · Day 5](experiments/02_transformer_building_blocks/NOTES.md), [03 · Day 7](experiments/03_tiny_gpt/NOTES.md) |

Also: a from-scratch 3.2M-param GPT trains at **157k tokens/s** here; at that rate GPT-3's training run would take **~3,000 years**, which is the case for clusters and distributed training ([03 · Day 7](experiments/03_tiny_gpt/NOTES.md)). Removing `zero_grad()` or `loss.backward()` raises **no error**: the model silently trains wrong or not at all ([01 · Day 1](experiments/01_training_loop/NOTES.md)).

## What's next
The skills that come up most in ML infrastructure job descriptions and aren't covered here yet: **model serving** (vLLM, a batching server, load-tested on Kubernetes), **profiling** (`torch.profiler`, Nsight Systems, CUDA graphs), and **quantization** (int8/int4).

## Setup

Hardware used: NVIDIA RTX 3050, Windows 11, Python 3.10, PyTorch 2.6 (CUDA 12.6).

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

## Plan

| Days | Work |
|---|---|
| 1 | Setup + gradient descent warm-up |
| 2–3 | FashionMNIST train/infer + overfitting |
| 4–5 | Tokenization, embeddings, attention |
| 6–8 | Tiny GPT |
| 9–11 | Inference benchmark |
| 12 | Requirements map from JDs |
| 13 | This README: key findings + push to GitHub |
| 14 | Buffer; practice a 5-minute explanation of the training and inference flow |
