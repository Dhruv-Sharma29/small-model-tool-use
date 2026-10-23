# Latency and memory, one model at a time (M5)

`eval/bench_latency.py` (2026-10-23, 17:16–17:35 UTC): each model served alone with the flags of
every accuracy run (llama.cpp b11439, Q4_K_M, 4 slots × 8,192 tokens), the first 40 val tasks run
one request at a time, temperature 0. No other process was on the GPU for any condition
(`gpu_others_at_start` is empty in every `results/summary/latency-*.json`). Prefill and decode speed
are llama.cpp's own per-request timings; memory is the server process's GPU memory (nvidia-smi,
sampled each second).

| Condition | Model call p50 / p95 | Episode p50 / p95 | Prompt tokens per call | Prefill tok/s | Decode tok/s | Server GPU memory |
| --- | --- | --- | --- | --- | --- | --- |
| base 7B, base prompt | 0.55 / 1.09 s | 2.63 / 12.03 s | 1,695 | 1,546 | 42.4 | 6.3 GB |
| **tuned 7B (M5), base prompt** | **0.55 / 0.79 s** | **3.27 / 3.99 s** | 1,715 | 1,924 | 42.7 | **6.3 GB** |
| 7B, improved prompt | 0.53 / 3.04 s | 4.34 / 10.55 s | 2,588 | 2,165 | 42.2 | 6.3 GB |
| 14B, improved prompt | 1.75 / 6.76 s | 16.22 / 36.55 s | 2,645 | 1,050 | 21.3 | 14.6 GB |

**The cost half of the research question.** Against its strongest competitor, the 14B with the
improved prompt (73% on test_unseen vs the tuned 7B's 84%), the tuned 7B is about 5x faster per
episode at the median (3.3 vs 16.2 s), 9x at p95 (4.0 vs 36.6 s), and needs 2.3x less GPU memory
(6.3 vs 14.6 GB): half the decode cost per token, 930 fewer prompt tokens per call (no long
instructions), and shorter replies (21 vs 50 tokens). The tuned model's tail is also the tightest of
the four (episode p95 4.0 s): it rarely writes long text.

Caveat: the 40 tasks are only the timing workload. Success on them (base 7/40, tuned 26/40, 7B +
improved 21/40, 14B + improved 33/40) is not the accuracy result; test_unseen is (`m5.md`).
