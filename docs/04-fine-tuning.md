# 04 — Fine-tuning

How training data is made, how the adapter is trained and served, and what the two pilots showed.
The pilots ran on val only; no test set has seen a tuned model. Current run: [Status](README.md#status).

## The pipeline

```text
data/train.jsonl ─► train/build_sft.py ─► .run/sft/<name>.jsonl (+ .manifest.json)
                     │  pick N tasks balanced over 31 kind-variants (seeded)
                     │  run gold or runbook policy through executor/runner.py, keep only successes
                     │  render with Qwen2.5's own chat template + the frozen tool schemas
                     │  store character spans of every assistant reply
                     ▼
                   train/sft.py ─► .run/adapters/<name>/adapter  (+ train_log.jsonl, run.json)
                     │  bf16 base, LoRA on all attention + MLP projections, loss on assistant spans only
                     ▼
                   train/export.py ─► models/qwen2.5-7b-lora-<name>-Q4_K_M.gguf  (+ export.json)
                     │  merge on CPU → f16 GGUF (01's converter) → Q4_K_M (01's llama-quantize)
                     ▼
                   env/models.sh serve <gguf> <port> <alias> ─► eval/run_model.py --prompt base
```

All of `.run/`, `models/` and `data/` are git-ignored. The committed record of each run is its
summary under `results/summary/` and its raw episodes under `results/raw/`.

## Training data: what the model imitates

Two details matter more than the hyperparameters.

**The transcript must be byte-identical to what the model sees at inference.** Training text is
rendered with the base model's tokenizer chat template and `openai_tools()`, exactly as llama.cpp
renders it with `--jinja`. llama.cpp parses tool-call arguments into JSON objects before templating,
so `build_sft.py` passes them as objects too; passed as strings, the template double-encodes them
([09](09-implementation-notes.md) § 9). `--check-url` compares a sample against the server's
`/apply-template` byte for byte. Whether the pilot builds used that check is recorded in their
manifests under `.run/sft/`, which are not committed **[verify]**.

**The trajectory must not give the answer away before an observation does.** This is the lesson of the
first pilot, and it is now a test. Tasks a model can't tell apart from the prompt (same request kind,
or for incidents the same alert family) must start with the same tool call. Gold trajectories break
this in 3 of 18 groups:

| Group | Gold first call → outcome |
| --- | --- |
| req_scale | service_status → no_action_needed; scale_service → resolved |
| req_restart_instance | service_status → need_information; restart_service → resolved |
| alert.jobs_lagging | service_status → worker restart; get_metrics → backlog / slow dependency |

The runbook is one observation-driven tree per family, so what it does next depends only on what the
tools returned: 0 groups leak. `tests/test_sft_policy.py` asserts both facts, so the check would also
catch a future runbook that leaks.

## Training setup

`train/sft.py`, plain PyTorch + PEFT, no trainer framework (`train/ENVIRONMENT.md` for versions:
torch 2.7.0a0 NVIDIA build, transformers 5.14.1, accelerate 1.15.0, peft 0.21.2).

| Setting | Value | Why |
| --- | --- | --- |
| Base | Qwen/Qwen2.5-7B-Instruct, rev a09a35458c70, bf16 | Same weights the baselines serve (as Q4_K_M) |
| LoRA | r 16, alpha 32, dropout 0.05, on q, k, v, o, gate, up, down projections | Modest default; rank is an M6 ablation candidate |
| Optimiser | AdamW, lr 2e-4, weight decay 0, 5 warmup steps, cosine decay, grad clip 1.0 | |
| Batch | 1 example per forward, 8-step gradient accumulation | Trajectories vary in length; no padding or truncation |
| Memory | Gradient checkpointing, KV cache off | Pilot peak 19.7 GiB allocated |
| Loss | Assistant tokens only (tool call + `<|im_end|>`), via the stored spans | The model learns to act, not to predict tool output |
| Record | `train_log.jsonl` per optimizer step (loss, step time, peak memory); `run.json` with config, data hash, versions | |

## Export and the reload check

The LoRA model is served like every other condition: merged into the bf16 base, converted with project
01's pinned llama.cpp converter, quantised to Q4_K_M with 01's `llama-quantize` (b11439), and
served by `env/models.sh serve` with the baselines' flags (PLAN.md, option a). `export.py` records
the adapter, merged and GGUF SHA-256. `--keep-f16` keeps the f16 GGUF to measure quantisation loss.

**Reload check (SPEC: "Adapter loading reproduces the evaluated checkpoint's results").** Re-exporting
the first pilot's adapter reproduced the same Q4_K_M byte for byte (c1c55a74a49d). Serving that file
again and rerunning val gave 94/155 against 96/155: 109/155 trajectories identical, 4 tasks changed
success (3 lost, 1 gained). The cause was tested: with 4 requests in flight llama.cpp batches them,
and the batch a request shares changes its floating-point results. Two 30-task runs with one request
at a time were identical (30/30 trajectories) (`results/summary/m4-pilot.md`). So the checkpoint
reloads exactly; the evaluation harness at 4 slots does not repeat exactly. [05](05-evaluation.md)
§ Noise explains how comparisons handle that.

**Quantisation loss** for the first pilot: f16 99/155 vs Q4_K_M 96/155, 10 vs 7 discordant tasks,
p = 0.63. Not distinguishable at n = 155.

## Pilot 1: 310 gold trajectories

10 train tasks per kind-variant, 1 epoch = 39 steps of 8 examples, 73 min, ~105 s/step, peak 19.7 GiB
(`results/summary/m4-pilot.md`). Val, base system prompt:

| Condition | Success | Incidents | Requests | Unallowed effects |
| --- | --- | --- | --- | --- |
| base-7b | 68/155 (44%) | 5/55 | 63/100 | 79 |
| improved-v2-7b | 96/155 (62%) | 23/55 | 73/100 | 51 |
| base-14b | 100/155 (64%) | 24/55 | 76/100 | 41 |
| **lora-pilot-7b** | **96/155 (62%)** | 30/55 | 66/100 | 32 |

Equal to the improved prompt overall (23 vs 23 discordant, p = 1.0), but not for a good reason.
Incidents improved, requests regressed, including kinds every baseline solves (req_scale 6/10, all
baselines 10/10). All four req_scale failures call service_status and then finish `no_action_needed`
with "There are already N workers" while N differs from the request: the model learned the gold
shortcut (first call → outcome), not to read the observation.

## Pilot 2: the same 310 tasks, runbook trajectories

Identical hyperparameters; only the data changed (1,020 assistant turns vs 900). 39 steps, final loss
0.029, peak 19.6 GiB, 1 h 45 min on a shared GPU.

| Condition | Success | Incidents | Requests | Invalid calls | Unallowed effects |
| --- | --- | --- | --- | --- | --- |
| lora-pilot-7b (gold) | 96/155 (62%) | 30/55 | 66/100 | 0.0% | 32 |
| **lora-pilot-runbook-7b** | **89/155 (57%)** | 18/55 | 71/100 | 0.0% | **2** |

Against pilot 1: 30 vs 23 discordant tasks, p = 0.41; not a difference on 155 tasks. What did change
is the failure pattern. The shortcut is gone: unallowed effects fall from 32 to 2, false alarms 5/5
(pilot 1: 1/5), req_restart_instance 7/10 (5/10). But on the 10 crashed or hung-worker incidents it
calls service_status first, as the runbook does, and never restarts (0/10): it continues down the
"nothing found" branch, which is the majority continuation in its data (after service_status,
get_metrics 90 vs restart_service 20 of 110 incident examples). 38 of its 66 failures finish
`no_action_needed` where an action was needed.

**Reading.** At 10 examples per kind-variant and 39 steps, the model learns the trajectory shape and
the most common branch, not the conditional on the tool result. Part of pilot 1's incident gain is
consistent with the shortcut it could take. The next factor to change is data quantity, keeping the
runbook source; the 310-task run becomes the smallest point of the data-quantity ablation
([06](06-roadmap.md)).

## M5: 1,240 runbook trajectories

Same settings, 4x the second pilot's data (`train/build_sft.py --n 1240 --policy runbook`; 155
steps, 6 h 41 min on the shared GPU, peak 19.8 GiB). Val 135/155 (87%) against 89/155 for the second
pilot (3 vs 49 discordant tasks): crashed and hung workers went from 0/10 to 10/10, so the
majority-branch behaviour was data quantity. The 20 val failures sit at a few value comparisons
(bad config "buried", bad deploy vs bad config, queue growth, latest vs not-latest version,
heartbeat threshold). On test_unseen it scores 130/155 (84%) ([05](05-evaluation.md)).

## What is not claimed

- One training seed. The spread across training runs isn't measured yet.
- The two pilots have val results only, and val was used to diagnose them.
- The improved prompt's val score is optimistic too (chosen on val), so "pilot 1 equals the improved
  prompt on val" is not evidence that the two would tie on test_unseen.
- Training time and step time come from a shared GPU (pilot 2's early steps ran up to 2× slower while
  another user's jobs ran); they are not a benchmark of training cost.
