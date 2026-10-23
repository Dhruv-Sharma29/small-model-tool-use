# M4 pilot: LoRA on 310 gold trajectories, val only

Adapter `.run/adapters/pilot` (r16/alpha 32, all attention + MLP projections, lr 2e-4, 1 epoch = 39
steps of 8 examples; 73 min, ~105 s/step, peak 19.7 GiB allocated). Data `.run/sft/pilot.jsonl`
(sha 0337885451c3, 10 train tasks per kind-variant, gold trajectories). Served at Q4_K_M like every
other condition, plus the f16 GGUF for quantisation loss; base system prompt. Re-exporting the
adapter reproduced the same Q4_K_M hash (c1c55a74a49d).

Val was used to diagnose this pilot; no test set has seen it.

| Condition | Set | Success | 95% CI | Incidents | Requests | Invalid calls | Tool selection ok | Unallowed effects | Mean steps |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base-7b | val | 68/155 (44%) | 0.36–0.52 | 5/55 | 63/100 | 0.0% | 45% | 79 | 4.01 |
| improved-v2-7b | val | 96/155 (62%) | 0.54–0.69 | 23/55 | 73/100 | 4.8% | 57% | 51 | 4.19 |
| base-14b | val | 100/155 (64%) | 0.57–0.72 | 24/55 | 76/100 | 7.4% | 66% | 41 | 4.02 |
| lora-pilot-7b | val | 96/155 (62%) | 0.54–0.69 | 30/55 | 66/100 | 0.0% | 64% | 32 | 3.13 |
| lora-pilot-7b-f16 | val | 99/155 (64%) | 0.56–0.71 | 33/55 | 66/100 | 0.0% | 69% | 26 | 3.17 |


**Paired: base-7b vs lora-pilot-7b**

| Set | Tasks | Only A | Only B | p (exact McNemar) |
| --- | --- | --- | --- | --- |
| val | 155 | 10 | 38 | 0.0001 |

**Paired: improved-v2-7b vs lora-pilot-7b**

| Set | Tasks | Only A | Only B | p (exact McNemar) |
| --- | --- | --- | --- | --- |
| val | 155 | 23 | 23 | 1.0 |

**Paired: base-14b vs lora-pilot-7b**

| Set | Tasks | Only A | Only B | p (exact McNemar) |
| --- | --- | --- | --- | --- |
| val | 155 | 34 | 30 | 0.708 |

**Paired: lora-pilot-7b vs lora-pilot-7b-f16**

| Set | Tasks | Only A | Only B | p (exact McNemar) |
| --- | --- | --- | --- | --- |
| val | 155 | 7 | 10 | 0.6291 |

## Where it changed

Incidents improve (30/55 vs 23/55 for the improved prompt), mostly where no prompt helped:
expired credential 5/5 (all baselines 0/5), slow dependency 4/5, unavailable-tool requests 5/5.
Requests regress against the improved prompt (66 vs 73/100), including kinds every baseline solves:

| Kind | base-7b | improved-v2-7b | base-14b | lora-pilot-7b |
| --- | --- | --- | --- | --- |
| req_scale | 10/10 | 10/10 | 10/10 | 6/10 |
| req_restart_instance | 5/10 | 8/10 | 9/10 | 5/10 |
| req_if_hung | 5/10 | 7/10 | 9/10 | 4/10 |
| inc_bad_deploy | 0/5 | 3/5 | 3/5 | 0/5 |

## Cause: the gold trajectories give the answer away

The generator's gold calls are the shortest path for the fault it planted, so the first call already
encodes the answer. In 3 of 18 groups that the prompt can't tell apart, the first call differs by outcome:

| Group | Gold first call → outcome |
| --- | --- |
| req_scale | service_status → no_action_needed; scale_service → resolved |
| req_restart_instance | service_status → need_information; restart_service → resolved |
| alert.jobs_lagging | service_status → worker restart; get_metrics → backlog / slow dependency |

The pilot learned the shortcut, not the observation. All four req_scale failures call
service_status and then finish `no_action_needed` with "There are already N workers" while N differs
from the request (e.g. "Scale workers to 1." → "There are already 4 workers."). Two restart_instance
failures restart a worker that doesn't exist, then finish with the "There is no worker-N" message.

Fix: `train/build_sft.py --policy runbook` builds the same tasks from the observation-only runbook
(one decision tree per family; 0 of 18 groups leak; `tests/test_sft_policy.py`). Same 310 tasks:
1,020 assistant turns (gold 900), 531,644 tokens (gold 519,186), sha b7ba3fb3d20a.

## Second pilot: same 310 tasks, runbook trajectories

Adapter `.run/adapters/pilot-runbook`, identical hyperparameters; only the data changed
(`.run/sft/pilot-runbook.jsonl`). 39 steps, final loss 0.029, peak 19.6 GiB; 1 h 45 min, early steps ran up to 2x
slower than the first pilot while another user's jobs shared the GPU. Q4_K_M sha a30f9975446d.
Raw: `results/raw/20261023T064426Z-lora-pilot-runbook-7b-val.jsonl`.

| Condition | Set | Success | 95% CI | Incidents | Requests | Invalid calls | Tool selection ok | Unallowed effects | Mean steps |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| improved-v2-7b | val | 96/155 (62%) | 0.54–0.69 | 23/55 | 73/100 | 4.8% | 57% | 51 | 4.19 |
| lora-pilot-7b | val | 96/155 (62%) | 0.54–0.69 | 30/55 | 66/100 | 0.0% | 64% | 32 | 3.13 |
| lora-pilot-runbook-7b | val | 89/155 (57%) | 0.49–0.65 | 18/55 | 71/100 | 0.0% | 65% | 2 | 3.1 |

Paired against the first pilot: 30 vs 23 tasks, p = 0.41 (exact McNemar); not a difference on 155 tasks.

The leak is gone and so is the shortcut: unallowed effects fall from 32 to 2, false alarms 5/5
(first pilot 1/5), req_restart_instance 7/10 (5/10). But the model now follows the runbook's call
order without reading the decisive observation. On the 10 crashed/hung-worker incidents it calls
service_status first, as the runbook does, and never restarts (0/10): it continues down the
"nothing found" branch and finishes `no_action_needed`. That is the majority continuation in its data
(after service_status: get_metrics 90, restart_service 20 of 110 incident examples). 38 of its 66
failures finish `no_action_needed` where an action was needed.

Reading: at 10 examples per kind-variant and 39 steps, the model learns the trajectory shape and the
most common branch, not the conditional on the tool result. Part of the first pilot's incident gain
is consistent with the shortcut it could take (it rose on the kinds the leak covers). Next: one
factor at a time, more runbook data first (planned M5 scaling; the 310-task run becomes the
smallest point of the data-quantity ablation).

## Reload check: same GGUF, rerun on val

Re-exporting the adapter reproduced the Q4_K_M byte for byte (c1c55a74a49d). Serving that file again
and rerunning val (`results/raw/20261023T070231Z-lora-pilot-7b-reload-val.jsonl`) gave 94/155 against
96/155: 109/155 trajectories identical, 4 tasks changed success (3 lost, 1 gained), most other
differences only in the finish message wording.

Cause, tested: with 4 requests in flight, llama.cpp batches them together, and the batch a request
shares changes its floating-point results, so temperature 0 is not deterministic. Two runs of 30 val tasks
one request at a time (`results/raw/smoke/*-det1-*`) were identical (30/30 trajectories, 14/14 success).

Consequence for every comparison in this study (all runs use 4 slots): a val difference of a few
tasks is within serving noise. The run-to-run spread measured here (4 flips, net 2) is one sample, not
a bound; the M5 seeds and paired tests carry the comparisons.
