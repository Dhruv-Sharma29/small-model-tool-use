# Start here

A guide to running the study's code and reading what it produces. The [README](README.md) states the
question; [docs/](docs/README.md) is the design record.

Everything in steps 1–6 runs **on CPU, with no model server and no GPU**. That is deliberate: the
task generator, sandbox, validator and scorer are the part every result depends on, and they can be
checked without a model. Steps that need the GPU are in [RUNBOOK.md](RUNBOOK.md).

---

## 1. Install

Python 3.12 and [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

This installs `jsonschema`, `httpx` and `pytest` (`pyproject.toml`). Training uses a separate
environment, `train/.venv`, described in `train/ENVIRONMENT.md`; you don't need it for anything here.

## 2. Build the data

`data/` is git-ignored. Rebuild it from the frozen split definitions:

```bash
uv run python -m tasks.splits.build
```

```
train           3100 tasks, 0 duplicates dropped
val              155 tasks, 0 duplicates dropped
...
```

It writes `data/<set>.jsonl` and rewrites `results/summary/m2_manifest.json`. Compare the per-set
`sha256` values with the committed ones (`git diff results/summary/m2_manifest.json`): only the
`generated` timestamp should change. If a hash differs, your data is not the data the results were
computed on.

## 3. Run the tests

```bash
uv run pytest
```

About 2,476 tests in ~15 s. Most are in `tests/test_scorer.py`, parametrised over every scenario:
the reference trajectory must pass and five kinds of plausible wrong trajectory must fail.
`tests/test_sft_policy.py` reads `data/train.jsonl`, so it needs step 2 first.

## 4. See one task scored, before any model is involved

Generate one task per scenario (186) and look at the first:

```bash
uv run python -m tasks.generator.generate --catalog /tmp/catalog.jsonl
head -1 /tmp/catalog.jsonl | python -m json.tool | less
```

Each task has a `prompt`, a hidden `world`, a `spec` (outcome, allowed effects, required state) and a
reference trajectory (`gold`). Now score two policies on it, the observation-only runbook and a
plausible-looking wrong one:

```bash
uv run python - <<'EOF'
from tasks.generator.generate import generate
from executor.runner import run_episode, ScriptedPolicy
from executor.scorer import score
from baselines.runbook import RunbookPolicy
t = generate("inc_worker_crashed", "default", "none", "alert.jobs_lagging.5", seed=0)
print(t["prompt"])
r = score(t, run_episode(t, RunbookPolicy(t)))
print("runbook:", r["success"], r["effects"])
bad = [["restart_service", {"service": "worker"}], ["finish", {"outcome": "resolved", "message": "restarted the workers"}]]
r = score(t, run_episode(t, ScriptedPolicy(bad)))
print("restart all:", r["success"], r["failed_checks"], r["unallowed_effects"])
EOF
```

```
Chen: jobs seem stuck or slow, 782 in the queue. Can you check what's going on and sort it out?
runbook: True ['restart_service:worker-3']
restart all: False ['effects_allowed'] ['restart_service:worker-1', 'restart_service:worker-2', 'restart_service:worker-4']
```

Restarting every worker did fix the crashed one, and the message says "resolved", but the task fails:
three restarts were not needed. That is the scorer's whole point ([02](docs/02-tasks-and-scoring.md)).

## 5. Reproduce the M1 and M2 checks

```bash
uv run python -m eval.m1         # reference, runbook and trivial policies on 930 tasks
uv run python -m eval.m2_audit   # overlap audit over data/
```

Both overwrite their files in `results/summary/` (`m1.*`, `m2_audit.*`); `git diff` should show only
timestamps. Expected: reference and runbook 100%, best trivial strategy 16.1%; 0 factor leakage,
test_unseen 0/155 same-example and 0% lookup ceiling.

## 6. Read the model results already in the repo

Every evaluation episode is in `results/raw/` (one JSONL per condition and set; the first line is the
run's metadata). `eval/summarize.py` turns them into tables without a server:

```bash
uv run python -m eval.summarize \
  results/raw/20261009T062734Z-base-7b-test_unseen.jsonl \
  results/raw/20261009T092318Z-improved-v2-7b-test_unseen.jsonl \
  results/raw/20261009T125623Z-base-14b-test_unseen.jsonl \
  results/raw/20261009T131322Z-improved-v2-14b-test_unseen.jsonl \
  --pair base-14b improved-v2-7b
```

The rows should match `results/summary/m3.md`: 68, 90, 98 and 113 of 155, and 23 vs 15 discordant
tasks (p = 0.2559) for the pair. Without `--out` nothing is written. Each line of a raw file also holds
the full transcript (`messages`) and per-step detail (`steps_detail`), which is where failure analysis
starts. Files under `results/raw/partial/` and `results/raw/smoke/` are not results (see their
READMEs).

## 7. Where to read next

| If you want | Read |
| --- | --- |
| The question, scope and success criteria | [docs/00-charter.md](docs/00-charter.md) |
| How tasks become a score | [docs/02-tasks-and-scoring.md](docs/02-tasks-and-scoring.md) |
| Why test_unseen is the primary set | [docs/03-dataset-card.md](docs/03-dataset-card.md) |
| What the fine-tuning pilots learned, and the leak | [docs/04-fine-tuning.md](docs/04-fine-tuning.md) |
| Every result and what it doesn't show | [docs/05-evaluation.md](docs/05-evaluation.md) |
| Current status | [docs/README.md](docs/README.md#status) |
| GPU steps: serving, evaluation, training, export | [RUNBOOK.md](RUNBOOK.md) |

## Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| `FileNotFoundError: .../data/train.jsonl` in `tests/test_sft_policy.py` or `eval.m2_audit` | `data/` not built | Step 2 |
| A per-set `sha256` differs from the committed manifest | Changed generator, vocab or splits code | Check `git status` under `tasks/`; results were computed on the committed hashes |
| `make_splits.py` exits with "splits are frozen" | Intended: `splits.json` must not be regenerated | Don't; the existing file defines every set |
| `ModuleNotFoundError: transformers` or `peft` | `train/` scripts run in `train/.venv`, not the uv environment | [RUNBOOK.md](RUNBOOK.md) § 0 |
| `eval.run_model` fails to connect | It needs a model server | [RUNBOOK.md](RUNBOOK.md) § 1 |
