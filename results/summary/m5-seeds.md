| Condition | Set | Success | 95% CI | Incidents | Requests | Invalid calls | Tool selection ok | Unallowed effects | Mean steps |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lora-runbook-1240-7b | val | 135/155 (87%) | 0.81–0.92 | 41/55 | 94/100 | 0.0% | 87% | 12 | 3.25 |
| lora-runbook-1240-7b | test_unseen | 130/155 (84%) | 0.77–0.89 | 39/55 | 91/100 | 0.0% | 84% | 22 | 3.25 |
| lora-runbook-1240-s1-7b | val | 138/155 (89%) | 0.83–0.93 | 46/55 | 92/100 | 0.0% | 89% | 10 | 3.3 |
| lora-runbook-1240-s1-7b | test_unseen | 136/155 (88%) | 0.82–0.92 | 46/55 | 90/100 | 0.0% | 88% | 12 | 3.35 |
| lora-runbook-1240-s2-7b | val | 152/155 (98%) | 0.94–0.99 | 52/55 | 100/100 | 0.0% | 98% | 2 | 3.28 |
| lora-runbook-1240-s2-7b | test_unseen | 145/155 (94%) | 0.89–0.96 | 51/55 | 94/100 | 0.0% | 94% | 8 | 3.34 |
| improved-v2-14b | test_unseen | 113/155 (73%) | 0.65–0.79 | 38/55 | 75/100 | 11.1% | 74% | 24 | 4.42 |

**Paired: improved-v2-14b vs lora-runbook-1240-s1-7b**

| Set | Tasks | Only A | Only B | p (exact McNemar) |
| --- | --- | --- | --- | --- |
| test_unseen | 155 | 5 | 28 | 0.0001 |

**Paired: improved-v2-14b vs lora-runbook-1240-s2-7b**

| Set | Tasks | Only A | Only B | p (exact McNemar) |
| --- | --- | --- | --- | --- |
| test_unseen | 155 | 3 | 35 | 0.0 |

**Paired: lora-runbook-1240-7b vs lora-runbook-1240-s2-7b**

| Set | Tasks | Only A | Only B | p (exact McNemar) |
| --- | --- | --- | --- | --- |
| test_unseen | 155 | 3 | 18 | 0.0015 |
| val | 155 | 1 | 18 | 0.0001 |

**Paired: lora-runbook-1240-7b vs lora-runbook-1240-s1-7b**

| Set | Tasks | Only A | Only B | p (exact McNemar) |
| --- | --- | --- | --- | --- |
| test_unseen | 155 | 7 | 13 | 0.2632 |
| val | 155 | 7 | 10 | 0.6291 |
