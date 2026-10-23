# 00 — Charter

## The problem

Small open-weight models are cheap to serve and easy to specialise, and tool use is where
specialisation looks most promising: a narrow domain, a fixed tool set, a known procedure. The
question every team that tries it has to answer is whether the tuned small model is actually as good
as the alternatives, which are cheaper to try: write a better prompt, or run a bigger model.

That question is easy to answer wrongly. If success is read from the model's own final message, a
model that says "restarted worker-3, all healthy" scores the same whether or not it did. If the
evaluation tasks share phrasing, names or noise patterns with the training tasks, a tuned model can
score by recognition. And if the training targets carry the answer before the evidence does, the
model learns the shortcut. All three happened, or nearly happened, in this project
([07](07-risks.md)).

The research question ([PLAN.md](../PLAN.md)):

> On the 02 sandbox-diagnostics task, how close does a LoRA-tuned 3–8B model get to (a) the same
> model with better prompting and (b) a larger general model, in task completion measured by
> resulting state, and at what latency/memory cost?

## What it does

**Input:** a task: an alert ("jobs seem stuck or slow, 782 in the queue") or an operator request
("Scale workers to 3."), plus a hidden starting state for the sandbox.

**Output, per episode and condition:**
1. The model's tool calls, each validated against a strict JSON Schema before it runs
   (`tasks/tools.py`), executed against a deterministic sandbox model (`executor/sandbox.py`).
2. A score computed from the sandbox's resulting state and recorded effects, with the check that
   failed (`executor/scorer.py`).
3. One JSONL line per episode with every step, its latency and tokens, and the full transcript
   (`results/raw/`), summarised with confidence intervals and paired tests (`eval/summarize.py`).

## Scope

### In
- One domain: diagnostics and fixes for a small jobs service (API, 1–4 workers, report exporter,
  database, spool, external payments provider), modelled on project 02's sandbox.
- 10 tools plus `finish`, with strict argument schemas and observable effects.
- 22 task kinds (31 with state-dependent variants) including no-action, missing-information and
  unavailable-tool cases.
- One tuned model (Qwen2.5-7B-Instruct, LoRA) against the same model with a base and an improved
  prompt, and against Qwen2.5-14B-Instruct, all served the same way (llama.cpp, Q4_K_M).

### Out (deliberately, not from running out of time)
- **Live processes.** Episodes run against an in-process model of 02's sandbox. Live episodes take
  10–200 s and their timing noise makes repeat runs disagree; the study needs thousands of training
  trajectories and hundreds of held-out episodes per condition (PLAN.md, Decisions). A transfer
  check on 02's live incidents is planned instead ([06](06-roadmap.md)).
- **A teacher model.** Every task, prompt and training trajectory comes from a deterministic
  generator or a hand-written runbook. No licence or provenance questions, and no way for a teacher
  to see held-out tasks ([03](03-dataset-card.md)).
- **Lenient parsing.** A tool call written as text is an invalid reply, for every condition
  ([09](09-implementation-notes.md) § 12).
- **Full fine-tuning, quantised training, training from scratch.** LoRA on a bf16 base only, as the
  brief's hardware constraints suggest (SPEC.md, GX10 constraints).
- **A general tool-use benchmark.** The task set answers one question about one domain.

### Why this scoping holds up in an interview
Every narrowing serves the same constraint: **success must be decidable from state, with no
judgement call.** A simulated sandbox gives the same observations and score for the same calls every
time, so differences between conditions are differences between models. A deterministic generator
makes split leakage checkable by a test rather than argued about. One domain with ~10 tools is small
enough for a hand-written runbook to solve every task from tool output alone (930/930), which proves
the tasks are solvable from what a model can see. Widen any of these and "better" turns back into
an opinion.

## Non-goals

- A production operations agent. The results describe the models on this task, not live operations.
- Beating the 14B model at everything. A negative result is acceptable if the measurements explain it
  (SPEC.md, Definition of done).
- A large benchmark. Each evaluation set has 155 tasks; the docs report counts, intervals and paired
  tests rather than headline percentages ([05](05-evaluation.md)).

## Success criteria

The study is done when all of these hold. They follow SPEC.md's "Important tests" and "Definition of
done"; [06](06-roadmap.md) tracks them per milestone.

| # | Criterion | How it's checked | State |
|---|---|---|---|
| S1 | Invalid tool arguments fail before execution | `tests/test_validation.py` | Done |
| S2 | The scorer rejects plausible-looking calls that produce the wrong state | `tests/test_scorer.py` (wrong outcome, claim without action, extra action, wrong worker, wrong action, over the whole catalogue) | Done |
| S3 | Training data can't include evaluation answers or near-duplicate templates | `tests/test_splits.py`; `results/summary/m2_audit.md`; `tests/test_sft_policy.py` for answer leaks inside training trajectories | Done |
| S4 | No-action and missing-information tasks are scored explicitly | Outcome check per task; per-outcome breakdown in `eval/summarize.py` | Done |
| S5 | Adapter loading reproduces the evaluated checkpoint's results | Byte-identical GGUF on re-export; identical trajectories with one request in flight (`results/summary/m4-pilot.md`) | Done, with the 4-slot caveat |
| S6 | Base, improved-prompt and larger-model baselines under identical tools and budgets | `results/summary/m3.md` | Done |
| S7 | A LoRA model evaluated on untouched held-out sets, test_unseen first, with run-to-run spread | M5 | Not done |
| S8 | At least one controlled ablation (one factor at a time) | M6 | Not done |
| S9 | Latency and peak memory from a controlled benchmark, one model loaded at a time | Not built yet | Not done |
| S10 | General-capability regression check for the tuned model | Not built yet | Not done |
| S11 | Dataset card, adapter card, failure analysis and write-up published | [03](03-dataset-card.md) exists; the rest are not written | Partly done |

## The honest risk to the thesis

A fine-tuned model on synthetic tasks can win by learning the generator instead of the task: its
phrasing frames, its entity names, or (as it turned out) the order of the reference trajectory's
calls. The guards are built in rather than added later: held-out templates, entities and scenario
combinations, with test_unseen holding out all three; a prompt-lookup ceiling that shows how much a
set rewards memorisation; and a test that training trajectories don't reveal the outcome before an
observation does. [07](07-risks.md) § R1–R3 covers what each guard catches and what it can't.
