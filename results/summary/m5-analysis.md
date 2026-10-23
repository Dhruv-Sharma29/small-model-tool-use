# M5: LoRA on 1,240 runbook trajectories (one seed)

Adapter `.run/adapters/runbook-1240` (same settings as both pilots: r16/alpha 32, all attention + MLP
projections, lr 2e-4, 1 epoch = 155 steps of 8; 6 h 41 min on the shared GPU, peak 19.8 GiB). Data:
`train/build_sft.py --n 1240 --policy runbook` (40 train tasks per kind-variant, 0 of 18 prompt
groups leak, sha bbc7172cbe1c). Served at Q4_K_M with the same flags as every condition; base prompt.
The only change from the second pilot is 4x the data, decided before either result was seen.

## Held-out results

`results/summary/m5.md` has every row and the paired tests. test_unseen is primary (M2 audit).

| Condition | test_unseen | 95% CI | Incidents | Requests | Unallowed effects |
| --- | --- | --- | --- | --- | --- |
| base-7b | 68/155 (44%) | 0.36–0.52 | 9/55 | 59/100 | 77 |
| improved-v2-7b (prompt) | 90/155 (58%) | 0.50–0.66 | 21/55 | 69/100 | 58 |
| base-14b | 98/155 (63%) | 0.55–0.70 | 30/55 | 68/100 | 45 |
| improved-v2-14b | 113/155 (73%) | 0.65–0.79 | 38/55 | 75/100 | 24 |
| **lora-runbook-1240-7b** | **130/155 (84%)** | 0.77–0.89 | 39/55 | 91/100 | 22 |

Paired on test_unseen (exact McNemar, tasks only the other condition solved vs only the tuned 7B):
base-7b 5 vs 67, improved-v2-7b 10 vs 50, base-14b 13 vs 45 (all p < 0.001), and improved-v2-14b
10 vs 27 (p = 0.0076).

By held-out factor: test_template 137/155 (88%), test_entity 134/155 (86%), test_scenario 130/155
(84%), test_seen 140/155 (90%). The gap between seen and fully unseen is 6 points, so the gain is not
prompt memorisation (the M2 audit puts the lookup ceiling at 37–40%).

## Val failures (20 of 155)

Not random: they sit at a few comparisons inside otherwise correct trajectories.

| Pattern | Count | What goes wrong |
| --- | --- | --- |
| inc_bad_config, "buried" variant | 5 | rolls back although the bad change isn't the latest version (should escalate) |
| inc_bad_deploy | 4 | takes the config branch (rollback_config) instead of deploy_history → rollback_release |
| inc_queue_backlog | 3 | skips the "queue growing" check and escalates instead of scaling |
| req_undo_config_by_author | 4 | latest vs not-latest confused |
| req_if_hung | 2 | heartbeat age misread against the threshold |

## General capability

Project 01's held-out benchmark prompts, no tools attached (`eval/general_capability.py`):
base 7B 59/60, tuned 7B 59/60 (by category, tuned: {'extraction': '20/20', 'reasoning': '19/20', 'tool_call': '20/20'}); tool-call
replies to plain prompts: base 0, tuned 0. No regression on this set; 60
prompts can only show a large one.

## Not settled yet

- One training seed. Seed spread needs more runs (PLAN M5).
- Latency and memory: `eval/bench_latency.py` is ready but needs the GPU to itself.
- Transfer to project 02's live incidents: running.
