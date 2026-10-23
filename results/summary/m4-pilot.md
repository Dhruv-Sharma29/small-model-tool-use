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
