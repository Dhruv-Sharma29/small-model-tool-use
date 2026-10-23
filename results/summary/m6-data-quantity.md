| Condition | Set | Success | 95% CI | Incidents | Requests | Invalid calls | Tool selection ok | Unallowed effects | Mean steps |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lora-pilot-runbook-7b | test_unseen | 76/155 (49%) | 0.41–0.57 | 19/55 | 57/100 | 0.0% | 59% | 9 | 3.01 |
| lora-runbook-1240-7b | test_unseen | 130/155 (84%) | 0.77–0.89 | 39/55 | 91/100 | 0.0% | 84% | 22 | 3.25 |
| base-7b | test_unseen | 68/155 (44%) | 0.36–0.52 | 9/55 | 59/100 | 0.0% | 46% | 77 | 4.14 |

**Paired: lora-pilot-runbook-7b vs lora-runbook-1240-7b**

| Set | Tasks | Only A | Only B | p (exact McNemar) |
| --- | --- | --- | --- | --- |
| test_unseen | 155 | 5 | 59 | 0.0 |

**Paired: base-7b vs lora-pilot-runbook-7b**

| Set | Tasks | Only A | Only B | p (exact McNemar) |
| --- | --- | --- | --- | --- |
| test_unseen | 155 | 25 | 33 | 0.3581 |
