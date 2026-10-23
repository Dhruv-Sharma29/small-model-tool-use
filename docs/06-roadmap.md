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
| M5 Full comparison | **In progress: scaling started 2026-10-23, results pending** | LoRA on 1,240 runbook trajectories (40 per kind-variant, `.run/adapters/runbook-1240`), then export and evaluation |
| M6 Ablations | Not started | Needs M5 |

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

## M5 — Full comparison (in progress; see the table above)

**MUST:**
- Scale runbook data from the pilot's 310 examples (first step: 1,240). Scale
  further only while val keeps improving and the failure analysis still points at data quantity.
- Evaluate the chosen checkpoint, chosen on val, on test_unseen first, then the four per-factor sets.
  The gap between test_seen and test_unseen is the memorisation measurement.
- Compare against `improved-v2-7b` and `improved-v2-14b` with paired tests, not only against `base-7b`.
- Repeat the key condition with different training seeds and report the spread.

**SHOULD:**
- A small general-capability set, base vs tuned, to catch regressions from specialisation.
- The transfer check on 02's 30 live incidents, comparable with 02's published result **[verify]**.
- The controlled latency and peak-memory benchmark, one model loaded at a time.

**Done when:** a test_unseen result for a tuned model with paired comparisons and seed spread
exists in `results/summary/`, plus a failure analysis by kind and failed check.

## M6 — Ablations

One factor at a time; at least one is required (SPEC.md).

| Ablation | Factor | Already have |
| --- | --- | --- |
| Data quantity | 310 → 1,240 → more runbook examples | The 310-example pilot 2 is the smallest point |
| Data quality | Gold vs runbook trajectories at equal size; with/without hard negatives | Gold vs runbook at 310 (pilots 1 and 2, val) |
| Adapter | LoRA rank / target modules | Nothing yet |

**Done when:** at least one ablation is run on the same held-out set as M5 with paired tests.

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
| Adapter / model card | Not written; needs the M5 checkpoint and the 7B licence **[verify]** |
| Results with variance | Partial: M3 with CIs and paired tests; no seed spread yet |
| Experimental write-up with failure analysis | Not written; pilot analyses in `results/summary/m3-val-analysis.md` and `m4-pilot.md` |
