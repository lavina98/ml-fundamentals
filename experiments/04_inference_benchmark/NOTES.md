# 04 · Inference benchmark

**Days 9–11.** Goal: the infra experiment. Measure how batching, device, and precision change latency, throughput, and memory.

## Tasks
- [ ] Benchmark harness: warm-up runs, `torch.cuda.synchronize()` before timing, median over N runs
- [ ] Batch size sweep (1 → OOM): latency/batch, samples/sec, peak memory (`torch.cuda.max_memory_allocated`)
- [ ] CPU vs GPU at each batch size
- [ ] fp32 vs fp16/bf16 (`torch.autocast`)
- [ ] Generation: time-to-first-token vs per-token latency for the tiny GPT
- [ ] Charts: throughput vs batch size, latency vs batch size

## Observations
<!-- The one number you'd quote in an interview goes here. -->
-
