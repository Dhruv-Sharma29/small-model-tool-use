# 05 — Evaluation

## What question the evaluation answers

Not "is the tuned model good." Specifically:

> On held-out tasks it has never seen in any form, does LoRA tuning get Qwen2.5-7B closer to
> Qwen2.5-14B than a better prompt does, measured by resulting state, and at what latency and memory
> cost?

That framing fixes the comparisons. The improved prompt is the honest baseline: it is free, and it
already moves the 7B a long way (44% → 58% on test_unseen). A tuned model that only matches it has
not earned its training cost.

## Conditions

| Condition | Model (all Q4_K_M, llama.cpp b11439) | System prompt | Sets run |
| --- | --- | --- | --- |
| `base-7b` | Qwen2.5-7B-Instruct a09a354 | `base` | val + all five test sets |
| `improved-7b` | same | `improved` (v1) | val only (superseded) |
| `improved-v2-7b` | same | `improved-v2` (frozen) | val + all five test sets |
| `base-14b` | Qwen2.5-14B-Instruct cf98f3b | `base` | val + test_unseen |
| `improved-v2-14b` | same | `improved-v2` | test_unseen |
| `lora-pilot-7b`, `-f16`, `-reload` | 7B + pilot 1 adapter (gold data) | `base` | val |
| `lora-pilot-runbook-7b` | 7B + pilot 2 adapter (runbook data) | `base` | val |

The tuned models get the **base** prompt: the comparison is weights vs prompt, not both at once.

**The improved prompt** (`baselines/prompts.py`) is a runbook-style checklist written from the base
7B's val failures (`results/summary/m3-val-analysis.md`: 75/155 episodes start with an action, 135/155
finish `resolved` against 95 expected, `config_history` called 6 times in 155 episodes). v1 fixed
incidents but broke explicit requests (51/100 vs 63/100); v2 added four clarifications from v1's
traces. Val: base 44%, v1 48%, v2 62%. v2 was frozen before any test run and no prompt work happened
after (PLAN.md).

**The 14B** runs on val and test_unseen only. At about one episode per minute on the shared GPU, five
test sets would take ~13 h, and memory pressure from other users stopped it twice. Untuned models
score alike on the per-factor sets (base 7B: 42–48%), so test_unseen carries the 14B comparison; the
per-factor sets exist to diagnose the tuned model (PLAN.md).

## Fixed for every condition

| | Value | Where |
| --- | --- | --- |
| Tools | 10 + `finish`, strict schemas, offered natively (`tool_choice: auto`, `parallel_tool_calls: false`) | `tasks/tools.py`, `baselines/llm.py` |
| Budget | 12 steps including `finish` and invalid calls; 3 retries, the 4th invalid call ends the episode | `executor/runner.py` |
| Sampling | temperature 0, seed 0, max 512 output tokens | `baselines/llm.py` |
| Serving | `-c 32768 --parallel 4` (8192 tokens per slot), `--jinja` | `env/models.sh` |
| Tool-call format | Strict: a call written as text is an invalid reply | [09](09-implementation-notes.md) § 12 |
| Counting | No episode is filtered: budget or transport failures count as failures | `eval/summarize.py` |

## Metrics

Per (condition, set), from `eval/summarize.py`:

| Metric | Meaning |
| --- | --- |
| Success, 95% CI | State-verified task completion, Wilson interval |
| Incidents / Requests | Success per family (55 / 100 tasks per set) |
| By expected outcome, failed checks | Where failures land: wrong outcome, unallowed effects, unhealthy end, ... |
| Invalid calls | Share of steps that were schema violations, unknown tools or text replies |
| Tool selection ok | Share of episodes whose set of action tools equals the reference's |
| Unallowed effects | Episodes with at least one effect the task didn't allow (unnecessary or unauthorised calls) |
| Mean steps, latency p50/p95, tokens | Effort; latency here is **not** the research question's latency (§ Latency and memory) |
| Paired comparison | Exact McNemar on the same tasks: how many only A solved, only B solved |

Counts are reported with percentages. With n = 155 a Wilson interval is about ±8 points; one task is
0.6 points.

## Results so far

### test_unseen, the primary set (`results/summary/m3.md`)

| Condition | Success | 95% CI | Incidents | Requests | Invalid calls | Unallowed effects |
| --- | --- | --- | --- | --- | --- | --- |
| base-7b | 68/155 (44%) | 0.36–0.52 | 9/55 | 59/100 | 0.0% | 77 |
| improved-v2-7b | 90/155 (58%) | 0.50–0.66 | 21/55 | 69/100 | 1.8% | 58 |
| base-14b | 98/155 (63%) | 0.55–0.70 | 30/55 | 68/100 | 4.6% | 45 |
| improved-v2-14b | 113/155 (73%) | 0.65–0.79 | 38/55 | 75/100 | 11.1% | 24 |
| LoRA 7B | not run | | | | | |

| Pair (test_unseen) | Only A | Only B | p (exact McNemar) |
| --- | --- | --- | --- |
| base-7b vs improved-v2-7b | 0 | 22 | < 0.001 |
| base-7b vs base-14b | 5 | 35 | < 0.001 |
| improved-v2-7b vs improved-v2-14b | 12 | 35 | 0.0011 |
| base-14b vs improved-v2-14b | 14 | 29 | 0.0315 |
| base-14b vs improved-v2-7b | 23 | 15 | 0.2559 |

What this settles: the improved prompt helps the 7B on every set (paired p ≤ 0.0072 on all six) and
helps the 14B too. **A prompted 7B is not distinguishable from the unprompted 14B** on test_unseen,
and a prompted 14B is clearly ahead of both. For the tuned 7B to matter, it has to be measured against
`improved-v2-7b` and `improved-v2-14b`, not against `base-7b`.

### Base and improved 7B on every set (`results/summary/m3.md`)

| Set | base-7b | improved-v2-7b |
| --- | --- | --- |
| val | 68/155 | 96/155 |
| test_seen | 66/155 | 85/155 |
| test_template | 65/155 | 81/155 |
| test_entity | 74/155 | 91/155 |
| test_scenario | 69/155 | 82/155 |
| test_unseen | 68/155 | 90/155 |

Untuned models don't benefit from familiar phrasing (42–48% for base across sets), as expected: they
never saw train. The per-factor sets become informative only for a tuned model, where a gap between
test_seen and test_unseen would measure memorisation.

### LoRA pilots, val only (`results/summary/m4-pilot.md`)

lora-pilot-7b 96/155, lora-pilot-7b-f16 99/155, lora-pilot-runbook-7b 89/155, against improved-v2-7b
96/155 and base-14b 100/155 on the same set. Analysis in [04](04-fine-tuning.md). Two caveats make
these weaker than they look: val chose the improved prompt, and val diagnosed the pilots.

### M5: LoRA on 1,240 runbook trajectories, one seed (`results/summary/m5-analysis.md`, `m5.md`)

| Condition | test_unseen | 95% CI | vs tuned 7B (only other / only tuned, exact McNemar) |
| --- | --- | --- | --- |
| base-7b | 68/155 (44%) | 0.36–0.52 | 5 / 67, p < 0.001 |
| improved-v2-7b | 90/155 (58%) | 0.50–0.66 | 10 / 50, p < 0.001 |
| base-14b | 98/155 (63%) | 0.55–0.70 | 13 / 45, p < 0.001 |
| improved-v2-14b | 113/155 (73%) | 0.65–0.79 | 10 / 27, p = 0.0076 |
| **lora-runbook-1240-7b** | **130/155 (84%)** | 0.77–0.89 | — |

Per factor: test_template 88%, test_entity 86%, test_scenario 84%, test_seen 90%; the 6-point
seen-to-unseen gap is far above the 37–40% lookup ceiling, so this is not prompt memorisation.
General capability on project 01's 60 held-out prompts: 59/60 for base and tuned, 0 tool-call
replies to plain prompts. The checkpoint's recipe was fixed before any of these numbers were seen
(4x the second pilot's data, nothing else changed).

### A failure pattern worth knowing: text-written calls

The 14B sometimes writes `finish({...})` as text instead of a tool call. Under the strict format that
is an invalid reply: improved-v2-14b has 76 text replies and an 11.1% invalid-call rate on
test_unseen, and 12 of its failures are "did not finish" (`results/summary/m3.json`). Lenient parsing
would probably score it higher; by how much is not measured. The rule is the same for every
condition and was fixed before the 14B ran; [09](09-implementation-notes.md) § 12.

## Noise

Two sources, both measured:

- **Sampling.** 155 tasks per set; Wilson intervals are about ±8 points.
- **Serving.** With 4 parallel slots, llama.cpp batches concurrent requests and temperature 0 is not
  deterministic: the same GGUF rerun on val gave 94 vs 96/155, 4 tasks flipped. One request at a
  time, two 30-task runs were identical (`results/summary/m4-pilot.md`).

All baselines ran at 4 slots, so it stays (changing it now would break comparability). Differences
of a few tasks on 155 are treated as noise; comparisons rest on paired tests, and M5 adds repeated
seeds. The measured spread (4 flips, net 2) is one sample, not a bound.

## Latency and memory

The research question asks for them, and they are **not measured yet**. Accuracy runs overlap on a
shared GPU with other users' jobs, which is fine for accuracy (temperature 0, fixed tasks) and
useless for latency: per-call p50 latency across the M3 runs ranges from 3.2 s to 23.7 s
(`results/summary/m3.json`), mostly from load, not from the model. Latency and memory will come from
a controlled benchmark with one model loaded at a time (PLAN.md, Decisions). The only resource
numbers so far are training ones: peak 19.7 GiB allocated for the pilot.

## Not built yet

| Item | Why it matters | Plan |
| --- | --- | --- |
| Repeated seeds | Spread across training runs, not just serving | M5 (running) |
| Transfer check on 02's 30 live incidents | The only check against real processes; 02 reports 15/30 for its simple agent (`02-reliable-operations-agent/results/summary/m5-comparison.md`) | M5 |
| Controlled latency/memory benchmark | The cost half of the question | M5, one model at a time |
| Ablation (one factor) | SPEC.md definition of done | M6 |
