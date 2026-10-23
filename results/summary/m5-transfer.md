# M5 transfer check: project 02's 30 live incidents

Project 02's own agent loop (`eval.run --responder llm-act`, prompt v2s), its own tools
(10 read tools, 8 actions, `submit_diagnosis`) and its live sandbox processes, with **only the model
weights changed**: base Qwen2.5-7B-Instruct vs the M5 checkpoint (`lora-runbook-1240-7b`), both
Q4_K_M on the same llama.cpp build, run back to back on 2026-10-23. Raw:
`02-reliable-operations-agent/results/raw/20261023T*-llm-act-*-transfer-*.jsonl`.

| Model | Success | Diagnosis correct | Diagnosis "unknown" | Escalated | Unnecessary actions |
| --- | --- | --- | --- | --- | --- |
| base 7B | 16/30 | 18 | 3 | 4 | 6 |
| tuned 7B (M5) | 12/30 | 13 | 10 | 2 | 8 |

Paired by case: only base 9, only tuned 5 (exact McNemar p = 0.42). Not distinguishable, and the
tuned model is not better. The base rerun matches 02's published 15/30.

**Why.** The tuned model carried its learned procedure over, not the skill behind it: it opened
with `service_status` in all 30 cases (base: once) and called `search_logs` once (base: 37). 04's
procedure rarely needs logs, but 02's diagnosis has to cite evidence, much of it in logs (bad
deploys, expired credentials), so its diagnosis came back "unknown" 10 times. It did better where
04's procedure matches 02's evidence: crashed and hung workers (3 of the 5 cases only it solved).

**What it means for the M5 result.** The 84% on test_unseen is a result for the interface it was
trained on (04's tools and prompt). Moving to another tool set and prompt for the same services
loses the gain. Training on the deployment's own tool interface, or on several, would be the fix
to test.
