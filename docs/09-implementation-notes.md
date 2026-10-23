# 09 — Implementation notes: where the build departs from the plan

The docs are the design record, so the places where building the study changed the plan belong here,
not only in commit messages. Each entry is a change against [SPEC.md](../SPEC.md) or the original
plan, with the reason and what it costs. Most come from PLAN.md § Decisions and deviations, where they
are dated as they were made.

None of them changes the research question. Entries 1–6 change the setup, 7–11 are bugs found while
building, 12–16 are honest downgrades, 17 is a rule that came out of building the runbook.

## Deviations from the plan

### 1. A simulated sandbox, not 02's live processes (2026-10-09)

SPEC.md says to reuse project 02's sandbox tools. Tasks run against a deterministic in-process model
of it instead (`executor/sandbox.py`): same components, fault kinds and action semantics. Live
episodes take 10–200 s (fault settling, health checks), and their timing noise makes repeat runs
disagree; the study needs thousands of training trajectories and hundreds of held-out episodes ×
conditions × seeds. **Cost:** the results describe the model, not live operations. **Mitigation:** M5
adds a transfer check on 02's 30 live incidents, comparable with 02's published 15/30 (simple agent, `02-reliable-operations-agent/results/summary/m5-comparison.md`).

### 2. Tool set cut to 10 + `finish`

The brief allows 5–10 tools. 02's six metric tools became one `get_metrics(component)`; `escalate`
became a `finish` outcome. `rotate_credential` is left out on purpose: the expired credential is the
unavailable-tool case. `config_history` and `deploy_history` stay separate because never calling them
was 02's main failure ([02](02-tasks-and-scoring.md)).

### 3. Template split by frame position (2026-10-09)

Templates are parallel across families (position 3 is always a ticket, 6 is "Go ahead and ..."), so
positions 3 and 6 are held out in every family. Holding out a random frame per family would have leaked
the phrasing frame through sister families.

### 4. test_unseen is the primary result (2026-10-09)

The plan treated all held-out sets alike. The audit showed seen-template sets can be partly solved by
memorising prompts (up to 37–40% without any tool call), so test_unseen, which holds out all three
factors and has a 0% lookup ceiling, is reported first.

### 5. The 14B runs on val and test_unseen only (2026-10-09)

At ~1 episode/min on the shared GPU, five test sets would take ~13 h, and memory pressure stopped the
run twice (the second time by `env/mem_guard.sh` at < 6 GB available). base-14b's partial test runs are
in `results/raw/partial/`; the full rerun gave 98/155. Untuned models score alike on all sets (base
7B: 42–48%), so test_unseen carries the 14B comparison.

### 6. llama.cpp for every condition, not a vLLM gateway

The README's reuse table used to describe project 01 as "an OpenAI-compatible gateway in front of
vLLM" (corrected 2026-10-23). What runs is 01's pinned llama.cpp (b11439) at Q4_K_M: the 7B served by 01 on port 21000, the
14B and every LoRA model by `env/models.sh` with identical flags (`-c 32768 --parallel 4 --jinja`).
The LoRA model is merged and quantised the same way (PLAN.md "option a"), so tuned and untuned models
differ only in weights. Quantisation loss is measured separately (f16 vs Q4_K_M, [04](04-fine-tuning.md)).

## Bugs found while building

### 7. Gold trajectories leak the outcome through the first call (2026-10-23)

The first pilot failed request kinds every baseline solves (req_scale 6/10). Gold calls are the
shortest path for the planted fault, so in 3 of 18 prompt-indistinguishable groups the first call
already encodes the outcome, and the model learned that instead of reading the tool result ("Scale
workers to 1." → service_status → "There are already 4 workers"). Fix: `build_sft.py --policy runbook`,
guarded by `tests/test_sft_policy.py`. Diagnosed on val; no test set has seen a tuned model. Gold stays
the default so the pilot's data rebuilds byte for byte.

### 8. The memory guard watched the wrong process (2026-10-09 pilot run)

`setsid cmd &` can fork, which once left `mem_guard.sh` watching a parent that had already exited,
so the real job was unguarded. `env/run_guarded.sh` now captures the job's PID inside the new session
(`echo $$; exec ...`), and the guard kills the whole process group.

### 9. Tool-call arguments double-encoded in training text

llama.cpp parses tool-call arguments into JSON objects before applying the chat template. Passing them
to the template as strings (as the runner stores them) would double-encode them, and the model would
train on text it never sees at inference. `build_sft.py` converts them to objects first and can check
the rendering against the server's `/apply-template`.

### 10. Installing PEFT nearly replaced the CUDA torch

A plain `uv pip install peft` tried to resolve torch from PyPI, a multi-GB download that would have
shadowed the host's NVIDIA build; it was stopped after 10 minutes. `train/.venv` now uses
`--system-site-packages` and installs `peft==0.21.2` with `--no-deps` (`train/ENVIRONMENT.md`).

### 11. A smoke run on stale data

`results/raw/smoke/20261009T061636Z-smoke-7b-val.jsonl` ran on data built before a fix to the queue
age (`stalled_oldest_s`), so its tasks no longer match `data/`. Smoke and partial runs live in their
own folders with a README each and are never summarised as results.

## Honest downgrades

### 12. Strict tool-call format (2026-10-09)

A call written as text (the 14B sometimes writes `finish({...})` instead of a tool call) is an invalid
reply for every condition, reported as its own metric. It is not parsed leniently because the harness
was already frozen for the running baselines. Cost: the 14B's scores are lower than lenient parsing
would give (improved-v2-14b: 76 text replies on test_unseen, `results/summary/m3.json`).

### 13. 4-slot serving is not deterministic at temperature 0 (2026-10-23)

The same GGUF rerun on val: 94 vs 96/155, 4 tasks flipped; one request at a time, two 30-task runs
were identical. 4 slots were kept, because every baseline ran that way and changing it now would break
comparability. Differences of a few tasks on 155 are treated as noise; comparisons rest on paired
tests and M5 seeds.

### 14. Accuracy and latency are measured separately (2026-10-09)

Accuracy runs overlap on the shared GPU (temperature 0, fixed tasks: success doesn't depend on load).
Latency and memory for the research question are to come from a controlled benchmark with one model
loaded at a time. That benchmark has not been run; the latencies in the result files are not usable
for the research question ([05](05-evaluation.md) § Latency and memory).

### 15. Strict success

Restarting healthy workers alongside the broken one fails the task, because the tool supports
per-instance restarts. A reasonable operator might accept it. Failures report which check failed, so
the strictness is visible in every summary.

### 16. The improved prompt was chosen on val (2026-10-09)

v1 (from the base val failure analysis) fixed incidents but broke explicit requests (51/100 vs
63/100); v2 adds four clarifications from v1's traces. Val: base 44%, v1 48%, v2 62% (v2 vs base: 30
vs 2 tasks, p < 0.001). Chosen after two iterations on val, so its val score is optimistic. No prompt
work happened after the test runs started.

## A rule that came out of building the runbook

### 17. The runbook gets the request's meaning, not its answer

For requests, the runbook receives the kind and slots (what the text says: which instance, how many
replicas). Conditions, existence and "is this the latest change" come from tools. For incidents it
gets nothing but tool output. This is what makes it a fair solvability proof (930/930) and, since M4,
a training source with no answer leak.

## Deferred, not dropped

| Item | Where it is planned |
| --- | --- |
| Transfer check on 02's 30 live incidents | M5 ([06](06-roadmap.md)) |
| Controlled latency and peak-memory benchmark | M5 |
| General-capability regression set | M5 |
| Repeated training seeds | M5 |
| At least one controlled ablation | M6 |
| Adapter card and experimental write-up | After M5; SPEC.md deliverables |
| PLAN.md M4 "stop the 01 server and other heavy processes first" | Unticked in PLAN.md; the GPU is shared with other users regardless |
