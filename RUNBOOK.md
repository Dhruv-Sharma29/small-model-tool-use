# Runbook: the steps that need the GPU

Everything in [start.md](start.md) runs on CPU. The steps below need a GPU, a running model server,
or both: serving models, evaluating a condition, building training data, training, exporting, and the
reload check. They are the commands that produced the files in `results/raw/` and
`results/summary/m3.md` / `m4-pilot.md`, in order.

The machine this was built on is a shared GB10/arm64 host with other users' jobs on the same GPU.
That shapes two rules: **every long job runs under the memory guard**, and **latency numbers from
these runs are not benchmarks** ([docs/05](docs/05-evaluation.md) § Latency and memory).

---

## 0. Prerequisites

| Need | Where it comes from |
| --- | --- |
| Project 01 checked out next to this one (`../01-llm-inference-lab`) with its llama.cpp build (b11439) and converter venv under `.tools/` | `env/models.sh`, `train/export.py` use these paths |
| Hugging Face snapshots in the local cache: `Qwen/Qwen2.5-7B-Instruct` rev `a09a35458c702b33eeacc393d103063234e8bc28`, `Qwen/Qwen2.5-14B-Instruct` rev `cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8` | Loaded with `local_files_only=True`; download them first |
| `data/` built | [start.md](start.md) § 2 |
| Training environment | Below |

```bash
uv venv --system-site-packages --python /usr/bin/python3 train/.venv
uv pip install --python train/.venv/bin/python --no-deps peft==0.21.2
```

This reuses the host's NVIDIA PyTorch build and adds only PEFT without dependencies, so pip can't
replace the CUDA torch (`train/ENVIRONMENT.md` for versions and why).

**Run long jobs guarded.** `env/run_guarded.sh FLOOR_GB LOGFILE -- command ...` starts the job in its
own session and stops its whole process group if available memory falls below the floor (the 14B
evaluation used 6 GB):

```bash
env/run_guarded.sh 6 .run/logs/<job>.log -- <command> <args>
```

## 1. Serve the models

All conditions use the same llama.cpp build, Q4_K_M, `-c 32768 --parallel 4 --jinja` (8192 tokens per
slot).

| Model | Command | Port |
| --- | --- | --- |
| Base 7B (Qwen2.5-7B-Instruct a09a354, Q4_K_M) | Served by project 01; see that project | 21000 |
| 14B: build the GGUF once (HF snapshot → f16 → Q4_K_M, SHA-256 appended to `models/SHA256SUMS`) | `env/models.sh build-14b` | |
| 14B: serve / stop | `env/models.sh serve-14b` / `env/models.sh stop-14b` | 21001 |
| Any GGUF, e.g. a LoRA export | `env/models.sh serve models/<file>.gguf <port> <alias>` / `env/models.sh stop <port>` | yours |

Run one large model at a time if memory is tight; the 14B test runs were stopped twice for memory
(`results/raw/partial/README.md`).

## 2. Evaluate a condition

```bash
uv run python -m eval.run_model --url http://127.0.0.1:21000 --condition base-7b \
    --prompt base --sets val test_unseen
```

| Flag | Meaning |
| --- | --- |
| `--condition` | Name used in the output file and the summaries (`base-7b`, `improved-v2-14b`, `lora-<name>-7b`) |
| `--prompt` | `base`, `improved` (v1, superseded) or `improved-v2` (frozen). Tuned models use `base` |
| `--sets` | Any of `val test_seen test_template test_entity test_scenario test_unseen` |
| `--workers` | Episodes in flight (default 4, one per server slot). `--workers 1` makes runs repeat exactly ([docs/04](docs/04-fine-tuning.md) § Export and the reload check) |
| `--limit` | First N tasks only; for smoke runs, which belong in `results/raw/smoke/` |

Output: `results/raw/<UTC stamp>-<condition>-<set>.jsonl`, first line the run metadata (model file,
prompt hash, splits hash, budget, sampling). At ~1 episode/min a 155-task 14B set takes hours on the
shared GPU (PLAN.md).

**Rules that keep results comparable:** don't change `executor/`, `tasks/`, `baselines/llm.py` or the
server flags between conditions; don't tune prompts on any test set; a run that stops early goes to
`results/raw/partial/` with a note, and is rerun in full.

## 3. Summarise

```bash
uv run python -m eval.summarize results/raw/<run A>.jsonl results/raw/<run B>.jsonl \
    --pair <condition A> <condition B> --out results/summary/<name>
```

Writes `<name>.md` and `<name>.json`: success with Wilson 95% CI, per family and per expected outcome,
failed checks, invalid calls, tool selection, unallowed effects, steps, latency, tokens, and exact
McNemar for each `--pair`.

## 4. Build training data

```bash
train/.venv/bin/python -m train.build_sft --n 310 --policy runbook \
    --out .run/sft/pilot-runbook.jsonl --check-url http://127.0.0.1:21000
```

Picks `--n` train tasks balanced over the 31 kind-variants (seeded by `--seed`, default 0), runs the
policy through the real runner, keeps only successes, and renders with the base model's chat template.
`--check-url` compares `--check-n` samples (default 20) with the server's `/apply-template` byte for
byte, so it needs the base 7B served (§ 1). `--policy gold` rebuilds the first pilot's leaky data
([docs/04](docs/04-fine-tuning.md)). Writes the data and `<out>.manifest.json` (hash, token counts,
assistant turns).

## 5. Train

```bash
env/run_guarded.sh 6 .run/logs/sft-pilot-runbook.log -- \
    train/.venv/bin/python -m train.sft --data .run/sft/pilot-runbook.jsonl \
    --out .run/adapters/pilot-runbook --epochs 1
```

Defaults are the pilots' settings: r 16, alpha 32, lr 2e-4, 8-step accumulation, 5 warmup steps, seed
0. Writes `<out>/adapter/`, `train_log.jsonl` (loss, step time, peak memory per optimizer step) and
`run.json`. The pilot peaked at 19.7 GiB allocated at ~105 s/step for 310 examples
(`results/summary/m4-pilot.md`). Stopping other heavy GPU processes first is in PLAN.md's M4 checklist.

## 6. Export to GGUF

```bash
train/.venv/bin/python -m train.export .run/adapters/pilot-runbook [--keep-f16]
```

Merges on CPU, converts with 01's converter, quantises to Q4_K_M, writes
`models/qwen2.5-7b-lora-<adapter dir name>-Q4_K_M.gguf` and `<adapter dir>/export.json` with the
adapter, merged and GGUF SHA-256. `--keep-f16` keeps the f16 GGUF for the quantisation-loss
comparison. Then serve it (§ 1) and evaluate with `--prompt base` (§ 2).

## 7. Reload check

1. Run § 6 again on the same adapter and compare `gguf_q4_k_m_sha256` in `export.json`: it must be
   identical (pilot 1: c1c55a74a49d both times).
2. Rerun val with the same GGUF and summarise both runs together. With 4 slots expect a few flipped
   tasks (pilot 1: 94 vs 96/155).
3. To separate the adapter from serving noise, run the same `--limit 30` subset twice with
   `--workers 1`: trajectories must be identical (pilot 1: 30/30).

## Not scripted yet

The controlled latency and peak-memory benchmark (one model loaded at a time, quiet GPU), the
general-capability set and the transfer check on project 02's live incidents have no command in this
repo yet ([docs/06](docs/06-roadmap.md) § M5).
