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
<!-- Fill in on day 13: the 3–5 most interesting numbers across all experiments. -->

## Job market research
[`job-research/`](job-research/) tracks ML infrastructure job descriptions (`jds.csv`), with a tally script and a [requirements map](job-research/requirements_map.md) of recurring skills.

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
| 13 | This README: key findings |
| 14 | Buffer; practice a 5-minute explanation of the training and inference flow |
