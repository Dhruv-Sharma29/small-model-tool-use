# M3: base 7B on val, failure analysis (input to the improved prompt)

Run: `results/raw/20261009T061841Z-base-7b-val.jsonl`. Val only; the improved prompt was designed from this and no test set.

Success 68/155. Incidents 5/55, requests 63/100.

| Expected outcome | Success | Model's finish outcome (count) |
| --- | --- | --- |
| resolved | 59/95 | {'resolved': 92, 'need_information': 1, 'escalated': 1, 'no_action_needed': 1} |
| no_action_needed | 9/20 | {'resolved': 11, 'no_action_needed': 9} |
| escalated | 0/25 | {'resolved': 22, 'escalated': 2, None: 1} |
| need_information | 0/15 | {'resolved': 10, 'escalated': 4, 'no_action_needed': 1} |

- **Acts before looking:** 75/155 episodes start with an action.
- **Default to 'resolved':** 135/155 episodes finish resolved (expected: 95).
- **Change history ignored:** config_history called 6 times, deploy_history 1, in 155 episodes (02's main failure, reproduced).
- **Unneeded effects** in 79/155 episodes; most common: {'restart_service:worker-1': 28, 'restart_service:api': 28, 'rollback_config': 17, 'restart_service:worker-2': 16, 'scale_service:worker': 16}.

| Kind | Success |
| --- | --- |
| inc_bad_config | 0/10 |
| inc_bad_deploy | 0/5 |
| inc_disk_fill | 3/5 |
| inc_expired_credential | 0/5 |
| inc_exporter_leak | 0/5 |
| inc_false_alarm | 0/5 |
| inc_queue_backlog | 1/5 |
| inc_slow_dependency | 0/5 |
| inc_worker_crashed | 1/5 |
| inc_worker_hung | 0/5 |
| req_cleanup | 5/5 |
| req_if_hung | 5/10 |
| req_if_spool_full | 8/10 |
| req_restart_ambiguous | 4/10 |
| req_restart_failed | 6/10 |
| req_restart_instance | 5/10 |
| req_restart_service | 10/10 |
| req_rollback_release | 5/5 |
| req_scale | 10/10 |
| req_scale_unspecified | 0/5 |
| req_unavailable | 0/5 |
| req_undo_config_by_author | 5/10 |
