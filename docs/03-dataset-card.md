# 03 — Dataset card: sandbox tool-use tasks

## What it is

Synthetic operations tasks for a small jobs service (API, 1–4 workers, report exporter, database,
spool, external payments provider). Each task is an alert or an operator request, a starting state
for a deterministic sandbox model, a success specification and a reference tool-call trajectory. A
model solves a task by calling 10 tools plus `finish`. Success is judged from the sandbox's
resulting state and the effects it performed, never from the model's own report.

Built for one question: does LoRA fine-tuning let a 3–8B model approach a larger model at this
narrow task? It is not a general benchmark for operations or tool use.

| | |
| --- | --- |
| Size | 3,100 train + 6 evaluation sets × 155 = 4,030 tasks (`results/summary/m2_manifest.json`) |
| Unit | One task: prompt, starting world, spec, reference trajectory, factor labels (kind, variant, distractor, template, seed) |
| Format | JSON Lines, one file per set under `data/` (git-ignored, rebuildable) |
| Intended use | Training and evaluating tool-calling policies for this sandbox; comparing conditions under identical tools and budgets |
| Not for | Claims about live operations, other domains, or general tool-use ability |
| Tasks, scoring | [02](02-tasks-and-scoring.md) |

## How it is made

- **Generator:** `tasks/generator/`, deterministic. The same (kind, variant, distractor, template,
  seed, entity pools) always produces the same task. No language model wrote any task, prompt or
  reference answer; there is no teacher model and no external data.
- **Content:** 22 task kinds (10 incidents, 12 requests), 31 with state-dependent variants, 6
  distractors, 8 phrasing templates per family, and entity pools (people, notes, release series,
  error texts, bad DB values, kinds of impossible request). See `tasks/generator/vocab.py`.
- **Reference trajectories** are written per kind from the task's setup. They include investigation
  reads and a verification read, so they double as training targets. Every one is checked to
  succeed (tests). **Caveat found in M4:** they are the shortest path for the fault the generator
  planted, so in 3 of 18 prompt-indistinguishable groups their first call already encodes the
  outcome. Training now uses runbook trajectories instead; see § Training data derived from it.
- **Licence / provenance:** all content was written for this project. Names are invented.

## Splits

Defined once in `tasks/splits/splits.json` (`make_splits.py`, seed 20261009) **before any task was
generated**; the file refuses to be regenerated. Three factors are held out:

| Factor | Held out | How chosen |
| --- | --- | --- |
| Template | 2 of 8 phrasing frames (positions 3 and 6) in **every** family | Random. Frames are parallel across families, so holding out a position everywhere avoids leaking the phrasing through a sister family |
| Entity | ~30% of every pool (e.g. 7/24 authors, 4/12 release series, 2/6 kinds of impossible request) | Random; DB values held out per key, so every key stays in training |
| Scenario | 1 of 6 distractors for each of the 31 kind-variants | Random |

| Set | Held out | Tasks | Use |
| --- | --- | --- | --- |
| train | none | 3,100 (100 per kind-variant) | Training pool; M4/M5 choose subsets |
| val | none | 155 | Prompt and checkpoint selection |
| test_seen | none | 155 | Fit on familiar phrasing (new seeds) |
| test_template | template | 155 | Unfamiliar phrasing |
| test_entity | entity | 155 | Unfamiliar names, versions, errors, values |
| test_scenario | scenario | 155 | Unfamiliar noise combinations |
| **test_unseen** | all three | 155 | **Primary held-out result** |

How close each set is to train (`results/summary/m2_audit.md`):

| Set | Same prompt + reference actions as a train task | Prompt-lookup ceiling | Closest-train-prompt similarity, median / max |
| --- | --- | --- | --- |
| val | 127/155 | 37% | 1.0 / 1.0 |
| test_seen | 134/155 | 38% | 1.0 / 1.0 |
| test_template | 0/155 | 0% | 0.0 / 0.6 |
| test_entity | 100/155 | 29% | 1.0 / 1.0 |
| test_scenario | 131/155 | 40% | 1.0 / 1.0 |
| test_unseen | 0/155 | 0% | 0.0 / 0.333 |

The **prompt-lookup ceiling** is what a policy would score by answering each task with the most
common reference answer (actions with arguments, plus outcome) among train tasks with the identical
prompt, without calling any tool (`eval/m2_audit.py`). "Same example" compares normalised prompts
(lowercase, digits to 0) plus reference actions and outcome. Similarity is word 3-gram
Jaccard to the closest train prompt.

Every set has 5 (train: 100) tasks per kind-variant, so the outcome mix is identical across sets:
61% resolved, 16% escalated, 13% no_action_needed, 10% need_information.

`data/` is not committed. `python -m tasks.splits.build` regenerates it byte for byte; the SHA-256
of every file is in `results/summary/m2_manifest.json`.

## Leakage audit

`python -m eval.m2_audit` → `results/summary/m2_audit.md`. Tests (`tests/test_splits.py`) enforce
the hard rules and were mutation-checked: each kind of factor leak makes them fail.

- **Held-out factors in train: 0** templates, 0 scenario combinations, 0 entities.
- **Held-out sets use only held-out values** of their factors (checked per task, both directions).
- **No task appears in two sets.**
- **test_unseen / test_template:** no task has the same prompt and reference actions as any train
  task; the closest train prompt has word 3-gram Jaccard ≤ 0.33 / 0.6; held-out templates (slots
  masked) are ≤ 0.2 from any train template.

## Known weaknesses

- **Seen-template sets reward memorisation.** On val, test_seen and test_scenario, 82–86% of tasks
  share their exact prompt and reference actions with a train task. Answering from an exact-prompt
  lookup, without calling any tool, would score up to 37–40% there (0% on test_template and
  test_unseen). Gains on those sets are not evidence of generalisation; **report test_unseen first.**
- **The scenario split is weak.** Distractors change what the tools return, never the prompt or the
  correct answer. test_scenario tests robustness to unfamiliar noise combinations, nothing more.
- **Synthetic regularity.** 8 frames per family and fixed reference trajectories are far more
  regular than real incidents. A fine-tuned model may learn the generator. The unfamiliar-template
  split and the M5 transfer check on 02's live incidents are the guards.
- **Strict success.** Reasonable but broader actions (restarting all workers for one crashed worker)
  fail. The scorer reports which check failed, so this is visible in results.
- **Small held-out sets.** 155 tasks per set: a difference of a few points between conditions is
  within noise; comparisons must be paired and report uncertainty. Serving adds its own noise:
  rerunning the same GGUF on val with 4 parallel slots flipped 4 tasks
  (`results/summary/m4-pilot.md`).
- **val is no longer clean.** The improved prompt was chosen on val after two iterations, and both
  LoRA pilots were diagnosed on val. Val scores of those conditions are optimistic; test sets have
  not been used for any design decision, and no test set has seen a tuned model.

## Training data derived from it

`train/build_sft.py` turns a subset of `data/train.jsonl` into chat transcripts; nothing from val or
test is read. Tasks are chosen balanced over the 31 kind-variants (`--n 310` gives 10 each). Each
trajectory is executed through the real runner, so observations are the sandbox's, and kept only if
it scores as a success. Provenance per example: task id, kind, variant; per file: a manifest with the
source, base-model revision, policy, seed and SHA-256 next to the data (`.run/sft/`, git-ignored).

| Source (`--policy`) | What decides the next call | Leaks the outcome before an observation | Used for |
| --- | --- | --- | --- |
| `gold` (default) | The generator's knowledge of the planted fault | Yes, in 3 of 18 prompt groups (`req_scale`, `req_restart_instance`, `alert.jobs_lagging`) | First pilot (sha 0337885451c3) |
| `runbook` | Only what the tools returned (plus a request's slots) | No, 0 of 18 (`tests/test_sft_policy.py`) | Second pilot (sha b7ba3fb3d20a) and later runs |

Same 310 tasks: gold gives 900 assistant turns and 519,186 tokens, runbook 1,020 and 531,644
(`results/summary/m4-pilot.md`). Gold stays the default so the first pilot's data rebuilds byte for
byte. Details and the effect on training: [04](04-fine-tuning.md).

## Reproducing it

```bash
uv run python -m tasks.splits.build   # data/<set>.jsonl + results/summary/m2_manifest.json
uv run python -m eval.m2_audit        # results/summary/m2_audit.{md,json}
uv run pytest tests/test_splits.py
```

`build.py` rewrites the manifest with a new `generated` timestamp; the per-set `sha256` values are
what must match. `make_splits.py` refuses to overwrite `splits.json`; the splits hash
(`c5b228f3fc10c969`) is recorded in every evaluation run's metadata line.
