# 06 — Roadmap

[PLAN.md](../PLAN.md) holds the milestone checklists and the dated decision log; this doc is the
acceptance view: what each milestone must show, what closed it, and what is next.

## Where things stand (2026-10-23)

| Milestone | State | Closed by / next |
| --- | --- | --- |
| M1 Task definition | **Done** | Runbook 930/930 from observations only; scorer tests ([02](02-tasks-and-scoring.md)) |
| M2 Leak-resistant splits | **Done** | 0 factor leakage, test_unseen shares no example with train ([03](03-dataset-card.md)) |
| M3 Baselines | **Done** | `results/summary/m3.md` ([05](05-evaluation.md)) |
| M4 Pilot fine-tune | **Pilots done, val only** | Two 310-example pilots, reload check passed ([04](04-fine-tuning.md)) |
| M5 Full comparison | **Done**: test_unseen 84% / 88% / 94% over three seeds (`results/summary/m5-seeds.md`); transfer check (12/30 vs base 16/30 on 02's live harness); latency (`results/summary/latency.md`) | LoRA on 1,240 runbook trajectories (40 per kind-variant, `.run/adapters/runbook-1240`), then export and evaluation |
| M6 Ablations | **Data quantity done** (310 vs 1,240 on test_unseen, paired) | `results/summary/m6-data-quantity.md` |

## What changed from the original plan, and why

| Change | Reason |
| --- | --- |
| Simulated sandbox instead of 02's live processes | Live episodes take 10–200 s and disagree run to run; a transfer check on 02's live incidents moves into M5 ([09](09-implementation-notes.md) § 1) |
| test_unseen became the primary set | The audit showed seen-template sets can be partly solved by prompt lookup (37–40%) |
| 14B on val + test_unseen only | ~13 h for five test sets on a shared GPU, and two memory stops |
| llama.cpp (01's b11439) instead of a vLLM gateway | Every condition served by the same build at the same quantisation ([09](09-implementation-notes.md) § 6) |
| A second M4 pilot | The first pilot's score came through a training-data leak; the second isolates the data source with all else equal |
| Accuracy and latency measured separately | Accuracy runs share the GPU; latency needs one model loaded at a time |

## M1 — Task definition (done)

**Done when:** 5–10 tools with strict schemas; success decided from resulting state; a hand-written
policy solves the tasks; no-action, missing-information and unavailable-tool cases exist.
- [x] `tasks/tools.py`: 10 + `finish`.
- [x] `executor/scorer.py`: outcome, allowed effects, unchanged state, health, required state.
- [x] `baselines/runbook.py` 930/930; trivial floor at most 16.1% (`results/summary/m1.md`).
- [x] Wrong trajectories fail wherever they apply (`tests/test_scorer.py`).

## M2 — Leak-resistant splits (done)

**Done when:** template, entity and scenario held out before generation; exact and near-duplicate
audit written; no teacher, or teacher provenance recorded.
- [x] `tasks/splits/splits.json` frozen, `make_splits.py` refuses to overwrite.
- [x] `results/summary/m2_audit.md`: 0 factor leakage; 0 exact duplicates dropped at build.
- [x] No teacher model ([03](03-dataset-card.md)).

## M3 — Baselines (done)

**Done when:** base model, improved prompt and larger model run under the same tools, budget and
steps.
- [x] base-7b and improved-v2-7b on all six sets; base-14b on val + test_unseen; improved-v2-14b on
      test_unseen.
- [x] Prompt frozen on val before any test run.

## M4 — Pilot fine-tune (pilots done)

**Done when:** a small LoRA run trains, its peak memory and step time are recorded, and the reloaded
adapter reproduces its evaluated scores.
- [x] Pilot 1 (gold): 310 examples, r16, peak 19.7 GiB, ~105 s/step, val 96/155.
- [x] Pilot 2 (runbook): same tasks, val 89/155, leak removed.
- [x] Reload: byte-identical GGUF; identical trajectories with one request in flight.
- [ ] PLAN.md's "stop the 01 server and other heavy processes first" is still unticked there; the
      pilots ran on a GPU shared with other users ([07](07-risks.md) § R9).

## M5 — Full comparison (main result in; see the table above)

Done on 2026-10-23: the 1,240-example checkpoint on all six held-out sets with paired tests against
`improved-v2-7b` and `improved-v2-14b`, a failure analysis by kind and failed check, and the
general-capability check (`results/summary/m5-analysis.md`). Also done: the latency/memory benchmark: against the 14B with the improved prompt, the tuned 7B is about 5x faster per episode at the
median (3.3 vs 16.2 s), 9x at p95 (4.0 vs 36.6 s) and needs 2.3x less GPU memory (6.3 vs 14.6 GB),
measured one model at a time on an otherwise idle GPU (`results/summary/latency.md`). Still owed: training-seed
spread. The transfer check is done: in 02's own live harness, with 02's tools and prompt and only the weights changed, the tuned 7B
scored 12/30 against the base model's 16/30 (p = 0.42): the gain does not carry over to another tool
interface (`results/summary/m5-transfer.md`).

**MUST:**
- Scale runbook data from the pilot's 310 examples (first step: 1,240). Scale
  further only while val keeps improving and the failure analysis still points at data quantity.
- Evaluate the chosen checkpoint, chosen on val, on test_unseen first, then the four per-factor sets.
  The gap between test_seen and test_unseen is the memorisation measurement.
- Compare against `improved-v2-7b` and `improved-v2-14b` with paired tests, not only against `base-7b`.
- Repeat the key condition with different training seeds and report the spread.

**SHOULD:**
- A small general-capability set, base vs tuned, to catch regressions from specialisation.
- ~~The transfer check on 02's 30 live incidents.~~ Done 2026-10-23 (see above).
- ~~The controlled latency and peak-memory benchmark.~~ Done 2026-10-23.

**Done when:** a test_unseen result for a tuned model with paired comparisons and seed spread
exists in `results/summary/`, plus a failure analysis by kind and failed check.

## M6 — Ablations

One factor at a time; at least one is required (SPEC.md).

| Ablation | Factor | Already have |
| --- | --- | --- |
| Data quantity | 310 → 1,240 → more runbook examples | **Done 2026-10-23** on test_unseen (below) |
| Data quality | Gold vs runbook trajectories at equal size; with/without hard negatives | Gold vs runbook at 310 (pilots 1 and 2, val) |
| Adapter | LoRA rank / target modules | Nothing yet |

**Done when:** at least one ablation is run on the same held-out set as M5 with paired tests.
Met on 2026-10-23 (`results/summary/m6-data-quantity.md`):

| Runbook training examples (seed 0, all else equal) | test_unseen | Unallowed effects | Paired |
| --- | --- | --- | --- |
| 0 (base 7B) | 68/155 (44%) | 77 | — |
| 310 | 76/155 (49%) | 9 | vs base: 25 / 33 discordant, p = 0.36 |
| 1,240 | 130/155 (84%) | 22 | vs 310: 5 / 59 discordant, p < 0.001 |

At 310 examples the tuned model completes no more tasks than the untuned one, but it causes far
fewer unallowed effects (9 vs 77): it has learned the procedure's caution before its decisions. Four
times the data turns that into task success. A third point (3,100, the whole training split) would
show whether the curve is flattening.

## Cut line

If GPU time runs short, drop in this order, and say so in the write-up: further data-scaling points
beyond the first one that answers the question; the adapter ablation; the per-factor test sets for
the 14B (already dropped). Do not drop: test_unseen for the tuned model, paired comparison against
the prompted 7B and 14B, at least one ablation, and the failure analysis. Those are the definition of
done.

## Deliverables (SPEC.md)

| Deliverable | State |
| --- | --- |
| Dataset card, split definitions | Done: [03](03-dataset-card.md), `tasks/splits/splits.json` |
| Training and evaluation scripts, configurations | Done for M1–M4: `train/`, `eval/`, `env/` |
| Adapter / model card | Done 2026-10-23: [10](10-adapter-card.md) |
| Results with variance | Done: CIs and paired tests throughout; three training seeds (88.4% ± 4.9 on test_unseen) |
| Experimental write-up with failure analysis | Done: [11](11-write-up.md) |
