# 04 — Small model for accurate tool use: documentation

An experimental study: does a LoRA-tuned Qwen2.5-7B match a better prompt and a 14B model at tool
use for ops diagnostics, scored by executing its calls in a sandbox and checking the resulting state?

These docs are the plan and the design record. Read them in order the first time and use them as
reference afterwards. The brief is [SPEC.md](../SPEC.md); the dated milestone log and the list of
decisions as they were made is [PLAN.md](../PLAN.md). To run things, start at [start.md](../start.md);
for the steps that need the GPU, see [RUNBOOK.md](../RUNBOOK.md).

## Reading order

| # | Doc | What it settles |
|---|---|---|
| 00 | [Charter](00-charter.md) | The question, scope, non-goals, what "done" means |
| 01 | [Architecture](01-architecture.md) | The pipeline, the three rules it rests on, components, repo layout |
| 02 | [Tasks and scoring](02-tasks-and-scoring.md) | Tools, the simulated sandbox, task kinds, the state-based scorer and why it can't be talked round |
| 03 | [Dataset card](03-dataset-card.md) | How tasks are generated, the leak-resistant splits, the overlap audit, known weaknesses |
| 04 | [Fine-tuning](04-fine-tuning.md) | Training data (gold vs runbook trajectories), LoRA setup, export to GGUF, the reload check |
| 05 | [Evaluation](05-evaluation.md) | Conditions, budgets, metrics, statistics, every result so far and what it does and doesn't show |
| 06 | [Roadmap](06-roadmap.md) | Milestones, acceptance criteria, what is done and what is next |
| 07 | [Risks](07-risks.md) | Thesis and delivery risks, the two that already happened, claim hygiene |
| 08 | [Learning checklist](08-learning-checklist.md) | Skills per milestone, and interview talking points mapped to the code |
| 09 | [Implementation notes](09-implementation-notes.md) | Where the build departs from the plan, and why |
| 10 | [Adapter card](10-adapter-card.md) | The M5 checkpoint: base, licence, data, training, results, limits |

## The one-paragraph version

Tool-use demos usually score a model by reading its final message ("I restarted the worker"). This
study never does: every call is validated against a strict schema before it runs, executed against a
deterministic model of a small jobs service, and the task passes only if the resulting state is right
and nothing else changed. That makes "better" measurable, and it is what exposed the project's main
finding so far: a fine-tuned 7B can match a tuned prompt on validation for the wrong reason. The first
LoRA pilot learned that the generator's reference trajectories gave the answer away in their first
call; retrained on observation-driven runbook trajectories it lost that shortcut but, at 310 examples,
followed the runbook's call order without reading the tool results. The non-trivial part is
therefore not the training. It is building splits, training targets and comparisons where a gain
cannot come from memorised prompts, leaked answers or serving noise.

## Status

**As of 2026-10-23: M1–M4 done; M5 has its main result, with seed spread and the transfer check still running; M6 has its required ablation.**

- **M1 task definition, M2 leak-resistant splits:** done and tested (2,476 tests pass with
  `pytest -q` on CPU).
- **M3 baselines:** done. On test_unseen: base 7B 68/155 (44%), 7B + improved prompt 90/155
  (58%), base 14B 98/155 (63%), 14B + improved prompt 113/155 (73%). 7B + improved prompt and base
  14B are not distinguishable (23 vs 15 discordant tasks, p = 0.26)
  (`results/summary/m3.md`).
- **M4 pilot fine-tune:** two 310-example pilots, **val only**. Gold trajectories: 96/155, equal
  to the improved prompt, but through a leak. Runbook trajectories: 89/155, leak gone, observation
  reading not learned. The adapter reload check passed: re-export gives a byte-identical GGUF, and
  4-slot serving is not deterministic at temperature 0 (`results/summary/m4-pilot.md`).
- **M5, one seed:** LoRA on 1,240 runbook trajectories scores **130/155 (84%) on test_unseen**,
  ahead of every baseline including the 14B with the improved prompt (73%; 10 vs 27 discordant
  tasks, p = 0.0076). Per factor 84–90%. No general-capability regression on project 01's 60
  held-out prompts (59/60 base and tuned) (`results/summary/m5-analysis.md`, `m5.md`).
- **M6, data quantity:** 310 runbook examples give 49% on test_unseen, not distinguishable from
  the untuned 44% (p = 0.36) though with 9 unallowed effects instead of 77; 1,240 give 84% (5 vs 59
  discordant against 310) (`results/summary/m6-data-quantity.md`).
- **Not done yet:** seed spread, the controlled latency and memory benchmark (needs the GPU to
  itself), the transfer check on project 02's live incidents (running), further ablations and the write-up.

The research question has a first answer on accuracy: in this simulated setting a tuned 7B beats
both better prompting and a 2x larger model. Seed spread and the latency/memory cost are still owed. See [06](06-roadmap.md) for what comes next and
[09](09-implementation-notes.md) for the places where building it changed the plan.

## Conventions used in these docs

- **MUST / SHOULD / MAY** are load-bearing. MUST items are acceptance criteria.
- Every number cites the file it comes from, usually under `results/summary/`. If a number isn't in
  a committed file, it isn't in these docs.
- **[verify]** marks a claim not confirmed against a primary source or a committed file (for example
  a fact about another project, or a record that lives only in the git-ignored `.run/`).
- Set names (`val`, `test_unseen`, ...) and condition names (`base-7b`, `improved-v2-7b`,
  `lora-pilot-7b`, ...) are the ones in the result files.
