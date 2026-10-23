# 08 — Learning checklist

Ordered by milestone, because each milestone forced a specific skill. "Done when" points at the
artifact that shows it, not at a tutorial.

## M1 — task definition

| Skill | Why | Done when |
| --- | --- | --- |
| JSON Schema (Draft 2020-12) for tool arguments | Invalid calls must fail before anything runs | `tasks/tools.py` validators; `tests/test_validation.py` rejects out-of-scope targets, injected fields, unknown tools, malformed JSON |
| OpenAI function-calling message shape | One transcript format for the runner, the LLM policy and the training data | `executor/runner.py` records exactly what `baselines/llm.py` sends |
| State-based success specs | A pass must not depend on the model's report | `executor/scorer.py`; every failure names its check |
| Adversarial tests for a scorer | A scorer that passes the reference proves nothing until wrong trajectories fail | `tests/test_scorer.py`: wrong outcome, claim without action, extra action, wrong worker, wrong action |
| Writing an upper-bound policy | Shows tasks are solvable from what a model can see | `baselines/runbook.py`, 930/930 |

## M2 — leak-resistant splits

| Skill | Why | Done when |
| --- | --- | --- |
| Factor-level splitting (template, entity, scenario) | Random task splits leak phrasing and names | `tasks/splits/splits.json`, frozen before generation |
| Leakage through parallel structure | Holding out a template per family leaks it through sister families | Templates held out by frame position in every family |
| Near-duplicate audit (n-gram Jaccard) and a lookup ceiling | Quantifies how much a set rewards memorisation | `results/summary/m2_audit.md` |
| Reproducible data by hash | "Same data" must be checkable | `results/summary/m2_manifest.json`; byte-for-byte rebuild |

## M3 — baselines

| Skill | Why | Done when |
| --- | --- | --- |
| Serving GGUF models with llama.cpp (`--jinja`, slots, context per slot) | All conditions served identically | `env/models.sh` |
| Prompt iteration on val only, then freezing | A prompt tuned on test is a leak | v1 → v2 on val, frozen before test (`baselines/prompts.py`) |
| Wilson intervals and exact McNemar | n = 155 needs intervals and paired tests | `eval/summarize.py` |
| Failure analysis by outcome and failed check | Where the improved prompt came from | `results/summary/m3-val-analysis.md` |
| Running long jobs on a shared machine | Others' jobs can OOM yours, and yours theirs | `env/mem_guard.sh`, `env/run_guarded.sh` |

## M4 — pilot fine-tune

| Skill | Why | Done when |
| --- | --- | --- |
| Chat templates with tool calls | Training text must equal inference text, byte for byte | `train/build_sft.py` (`--check-url` against `/apply-template`) |
| Loss masking by character span | Train on actions, not on tool output | `train/sft.py` `encode()` |
| LoRA with PEFT in plain PyTorch | Small enough to read; no trainer hiding the loop | `train/sft.py`, per-step loss, time and peak memory |
| A CUDA training env on arm64 without breaking torch | pip will replace the vendor build if allowed | `train/ENVIRONMENT.md` |
| Merge → GGUF → quantise, with hashes | Tuned and base models served the same way; reload checkable | `train/export.py`; byte-identical re-export |
| Diagnosing what a tuned model learned | A tie can hide a shortcut | The gold-leak analysis and `tests/test_sft_policy.py` |
| Nondeterminism in batched inference | Temperature 0 isn't enough with parallel slots | The 1-slot vs 4-slot reruns in `results/summary/m4-pilot.md` |

## M5–M6 — still to learn by doing

| Skill | Why | Done when |
| --- | --- | --- |
| Seed variance for fine-tuning runs | Training spread, separate from serving noise | M5 seed runs reported with spread |
| Controlled latency/memory benchmarking | The cost half of the question | One model at a time, quiet GPU |
| One-factor ablation design | SPEC.md requires one | M6 result with paired tests |
| Writing an adapter/model card | A deliverable | Card next to the published adapter |

## Interview talking points, mapped to what was built

| Question | Where the answer lives |
| --- | --- |
| "How do you know the model actually fixed it?" | [02](02-tasks-and-scoring.md) § Scoring: state and effects, never the message; strict per-instance checks |
| "How do you know it isn't memorising the test set?" | [03](03-dataset-card.md): factor splits fixed before generation, lookup ceiling, test_unseen primary |
| "Did fine-tuning help?" | [05](05-evaluation.md) and [04](04-fine-tuning.md): not shown yet; the pilot's val tie came through a leak |
| "What went wrong?" | [07](07-risks.md) § R2–R3: the gold-trajectory leak and the majority-branch failure, and the test that now guards the first |
| "Why not just prompt better?" | [07](07-risks.md) § The question: the prompted 7B already ties the base 14B on test_unseen |
| "Is your evaluation reproducible?" | Deterministic sandbox and data hashes; [05](05-evaluation.md) § Noise for the 4-slot caveat |
| "Why a simulator?" | [09](09-implementation-notes.md) § 1: speed and repeatability, at the cost of realism, with a planned live transfer check |
| "What would you do differently?" | Check training trajectories for answer leaks before the first training run; measure 1-slot vs 4-slot determinism before the baselines |
