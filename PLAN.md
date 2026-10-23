# 04 — Small model for accurate tool use: execution plan

Brief: [SPEC.md](SPEC.md). This is an **experimental study**. The research question:

> On the 02 sandbox-diagnostics task, how close does a LoRA-tuned 3–8B model get to (a) the same model with better prompting and (b) a larger general model, in task completion measured by resulting state — and at what latency/memory cost?

**Needs:** 02 sandbox tools + verifier, 01 serving/benchmark harness.

## Progress (2026-10-23)

| Milestone | State |
| --- | --- |
| M1 task definition | **Done.** 11 tools (5 read, 5 action, `finish`), 22 task kinds / 31 kind-variants x 6 distractors = 186 scenarios. Reference 100%, observation-only runbook 100%, best trivial strategy 16.1% (`results/summary/m1.md`). 2,465 tests; each scorer check shown load-bearing by mutation |
| M2 leak-resistant splits | **Done.** Splits frozen before generation (template frames, ~30% of entities, 1 distractor per kind-variant); 3,100 train + 6 x 155 eval tasks, rebuilt byte for byte. Audit: 0 factor leakage; test_unseen shares no example with train. Seen-template sets reward memorisation (lookup ceiling 37–40%), so test_unseen is primary (`results/summary/m2_audit.md`, `docs/03-dataset-card.md`) |
| M3 baselines | **Done.** test_unseen: base 7B 44%, improved prompt (v2) 58%, base 14B 63%, 14B + v2 prompt 73%. 7B + v2 vs base 14B not distinguishable (23 vs 15 tasks, p = 0.26). 7B on all six sets, 14B on val + test_unseen (`results/summary/m3.md`) |
| M4 pilot fine-tune | **Done.** Pilot (310 gold trajectories, 39 steps, peak 19.7 GiB): val 62% (Q4) / 64% (f16), equal to the v2 prompt overall; incidents up, requests down. Cause: gold trajectories leak the answer through the first call. Second pilot on runbook trajectories (no leak): val 57%, incidents 18/55, unallowed effects 2 (was 32); it follows the runbook's call order but takes the majority branch instead of reading the tool result. Next: more runbook data (`results/summary/m4-pilot.md`) |
| M5 full comparison | **Main result in (one seed).** 1,240 runbook trajectories: test_unseen 130/155 (84%) vs 14B + v2 prompt 73% (p = 0.0076); per factor 84–90%; general capability unchanged (`results/summary/m5-analysis.md`). Seeds, latency benchmark and transfer check pending |
| M6 ablations | **One done (data quantity).** test_unseen: 310 runbook examples 49% (vs base 44%, p = 0.36), 1,240 examples 84% (vs 310: 5 / 59 discordant, p < 0.001) (`results/summary/m6-data-quantity.md`) |

### Decisions and deviations

- **2026-10-09 — simulated sandbox, not 02's live processes (deviation).** Tasks run against a
  deterministic in-process model of 02's sandbox (`executor/sandbox.py`): same components, fault
  kinds and action semantics. Why: live episodes take 10–200 s (fault settling, health checks) and
  their timing noise makes repeat runs disagree; the study needs thousands of training trajectories
  and hundreds of held-out episodes x conditions x seeds. Cost: the results describe the model, not
  live operations. Mitigation: **M5 adds a transfer check on 02's 30 live incidents**, comparable
  with 02's published 15/30.
- **Tool set cut to 10 + finish** (brief: 5–10). 02's six metric tools are one `get_metrics`;
  `escalate` became a `finish` outcome. `rotate_credential` was left out on purpose: the expired
  credential is the *unavailable tool* case. `config_history`/`deploy_history` kept separate,
  because never calling them was 02's main failure.
- **Strict success.** Restarting healthy workers alongside the broken one fails the task (the tool
  supports per-instance restarts). Failures report which check failed, so strictness is visible.
- **2026-10-09 — template split by frame position.** Templates are parallel across families
  (position 3 is always a ticket, 6 is "Go ahead and ..."), so frames 3 and 6 are held out in every
  family. Holding out per family would leak the phrasing frame through sister families.
- **2026-10-09 — primary result is test_unseen.** The audit showed seen-template sets can be
  partly solved by memorising prompts (up to 37–40% without any tool call).
- **2026-10-09 — model pair: Qwen2.5-7B (tuned) vs Qwen2.5-14B-Instruct** (user's choice; 14B rev
  cf98f3b, Apache-2.0). Both served by 01's llama.cpp b11439 at Q4_K_M, same context (8192 per
  slot) and 4 slots (`env/models.sh`). The LoRA model will be merged and quantised the same way
  (option a); quantisation loss gets measured separately in M4.
- **2026-10-09 — accuracy and latency are measured separately.** Accuracy runs overlap on the shared
  GPU (temperature 0, fixed tasks: success doesn't depend on load). Latency and memory for the
  research question come from a controlled benchmark with one model loaded at a time.
- **2026-10-09 — improved prompt: v2, frozen.** v1 (from the base val failure analysis) fixed
  incidents but broke explicit requests (51/100 vs 63/100); v2 adds four clarifications from v1's
  traces. Val: base 44%, v1 48%, v2 62% (v2 vs base: 30 vs 2 tasks, p < 0.001). Chosen on val after
  two iterations, so its val score is optimistic; no prompt work after the test runs started.
- **2026-10-09 — strict tool-call format.** A call written as text (the 14B sometimes writes
  `finish({...})` instead of a tool call) is an invalid reply for every condition, reported as its
  own metric. Not parsed leniently: the harness was already frozen for the running baselines.
- **2026-10-09 — 14B test runs paused** to free memory when another user's training jobs left 3 GB
  on the shared machine. base-14b val (complete) is kept; its partial test run is in
  `results/raw/partial/`; redone in full the same day (base-14b test_unseen 98/155).
- **2026-10-09 — 14B evaluated on val + test_unseen only.** At ~1 episode/min on the shared GPU,
  five test sets would take ~13 h, and memory pressure from other users stopped it twice (the second
  time by our own guard, `env/mem_guard.sh`, at < 6 GB available). The per-factor sets diagnose
  the *fine-tuned* model's generalisation; untuned models score alike on all of them (base 7B:
  42–48%), so test_unseen carries the 14B comparison.
- **2026-10-23 — training data from the runbook, not gold (decision pending a second pilot).** The
  pilot (gold trajectories) failed request kinds every baseline solves (req_scale 6/10): gold calls
  are the shortest path for the planted fault, so in 3 of 18 prompt-indistinguishable groups the
  first call already encodes the outcome, and the model learned that instead of reading the tool
  result ("Scale workers to 1." → service_status → "There are already 4 workers"). The runbook is
  one observation-driven tree per family (0 groups leak, `tests/test_sft_policy.py`). Diagnosed on
  val; no test set has seen a tuned model. Gold stays the default so the pilot's data rebuilds byte for byte.
  Second pilot (2026-10-23): no shortcut left, but at 310 examples the model takes the majority
  branch (service_status → more reads in 90/110 incident examples) and never restarts a broken worker
  on val (0/10). Next factor to change: data quantity, keeping the runbook source.
- **2026-10-23 — 4-slot serving is not deterministic at temperature 0.** Same GGUF, val rerun: 94 vs
  96/155, 4 tasks flipped; one request at a time, two 30-task runs were identical. Kept 4 slots (all
  baselines ran that way, and changing it now would break comparability); differences of a few tasks
  on 155 are treated as noise, and comparisons rest on paired tests and M5 seeds.
- **Runbook policy gets the request's meaning, not its answer.** For requests it receives kind +
  slots (what the text says); conditions, existence and "is this the latest change" come from tools.
  For incidents it gets nothing but tool output.

## Target layout

```text
04-small-model-tool-use/
  tasks/
    tools/             5–10 tool schemas (import or copy from 02)
    generator/         deterministic task generator (templates × entities × scenarios)
    splits/            split definitions written BEFORE generating examples
  executor/            runs tool calls in sandbox, scores by resulting state
  baselines/           base model, improved prompting, larger model (sequential)
  train/               LoRA configs + scripts, pinned versions
  eval/                held-out sets incl. unfamiliar templates, no-action, ambiguous, general-capability set
  results/raw/  results/summary/
  docs/                dataset card, model/adapter card, write-up
```

## Milestones

### M1 — Task definition (1 week)
- [x] Freeze 5–10 tools with strict schemas and observable effects (`tasks/tools.py`: 10 + `finish`).
- [x] Success = resulting sandbox state matches the target, checked by the executor (`executor/scorer.py`: outcome, allowed effects, unchanged state, health, required state).
- [x] Hand-written policy solves the tasks → proves the scorer works (`baselines/runbook.py`, 930/930 from observations only; plus wrong-trajectory tests that must fail).
- [x] Task types: single-tool → multi-step; plus no-action, missing-information, unavailable-tool cases (`tasks/generator/scenarios.py`).

### M2 — Leak-resistant splits (½–1 week)
- [x] Split by **template**, **entity**, and **scenario** before generating any example (`tasks/splits/splits.json`, written by `make_splits.py`, refuses to overwrite).
- [x] Dedup (exact + near-duplicate, e.g. normalized text / n-gram overlap); write an overlap audit report (`eval/m2_audit.py` → `results/summary/m2_audit.md`; exact duplicates dropped at build: 0).
- [x] If any teacher model is used: record licence/terms + provenance; teacher never sees held-out tasks. Prefer the deterministic generator. (No teacher: deterministic generator only; `docs/03-dataset-card.md`.)

### M3 — Baselines (1 week)
- [x] Base model, zero-shot (`base` prompt).
- [x] Improved prompting (`improved-v2`, frozen on val; `baselines/prompts.py`).
- [x] Larger model, same tasks (Qwen2.5-14B, val + test_unseen).
- [x] Same tool set, retry budget, and max steps for every condition (12 steps, 3 retries).

### M4 — Pilot fine-tune (½–1 week)
- [ ] Stop the 01 server and other heavy processes first. (Not possible: the GPU is shared with other users' jobs; runs use env/mem_guard.sh instead.)
- [x] Small subset, LoRA, modest seq length. Record peak memory and step time (310 examples, r16, 19.7 GiB peak, ~105 s/step).
- [x] Save adapter → reload into serving stack → reproduce the evaluated scores. The GGUF is byte-identical, and single-request runs are deterministic. With 4 parallel slots, val gave 94 vs 96/155 (4 flips) from batching, not the adapter.
- [x] Second pilot on runbook trajectories (`.run/sft/pilot-runbook.jsonl`), val only: 89/155; leak gone, observation reading not learned at 310 examples.

### M5 — Full comparison (1–2 weeks)
- [x] Scale data (hundreds → thousands only once generator/scorer are trusted). (310 → 1,240 runbook trajectories.)
- [ ] Evaluate on untouched held-out splits. Multiple seeds where compute permits; report spread. (All six sets done for seed 0; more seeds running.)
- [x] Check a small general-capability set for regression. (01's 60 held-out prompts: 59/60 base and tuned.)

### M6 — Ablations (1–2 weeks)
One factor at a time: data quantity · data quality (e.g. with/without hard negatives) · LoRA rank/targets. At least one is required.

## Metrics
Task completion (state-verified) · tool-selection accuracy · argument validity · unnecessary/unauthorized calls · retries · latency · peak memory · general-capability delta.

## Tests
- [x] Invalid args rejected before execution (`tests/test_validation.py`) · [x] scorer rejects plausible but wrong-state calls (`tests/test_scorer.py`: wrong outcome, claim without action, extra action, wrong worker, wrong action; mutation-checked) · [x] no train/eval overlap (automated check: `tests/test_splits.py`, mutation-checked) · [x] adapter reload reproduces scores (byte-identical GGUF; deterministic at 1 request in flight) · [x] no-action + missing-info scored explicitly

## Risks
- Synthetic tasks can be too regular → model learns templates, not the task. The unfamiliar-template split is the guard.
- Training-stack support on GB10/arm64 — follow NVIDIA's Spark fine-tuning recipe; quantized training only if verified.

## Definition of done
- [x] Baselines · [x] ≥1 controlled ablation · [ ] held-out results with variance · [x] failure analysis · [ ] dataset card (done: `docs/03-dataset-card.md`) + adapter card · [ ] write-up (negative result acceptable)
