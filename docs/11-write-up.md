# 11 — Write-up: can a tuned 7B match better prompting and a 2x larger model?

## Question

> On the 02 sandbox-diagnostics task, how close does a LoRA-tuned 3–8B model get to (a) the same
> model with better prompting and (b) a larger general model, in task completion measured by
> resulting state — and at what latency/memory cost? (PLAN.md)

## Setup, in one paragraph

186 scenarios over 22 task kinds (incidents and operator requests, including no-action,
missing-information and unavailable-tool cases) on a deterministic model of project 02's services
([02](02-tasks-and-scoring.md)). Success is decided from the resulting sandbox state, strictly: a
correct diagnosis with an extra restart fails. Splits by template, entity and scenario were frozen
before any example was generated; **test_unseen**, which holds out all three, is the primary set,
because seen-template sets can be partly solved by prompt lookup (37–40%) ([03](03-dataset-card.md)).
Every condition gets the same 10 tools plus `finish`, the same 12-step and 3-retry budget, temperature
0, and the same llama.cpp build at Q4_K_M ([05](05-evaluation.md)).

## Results

**Accuracy** (test_unseen, 155 tasks; `results/summary/m5.md`):

| Condition | Success | vs tuned 7B (only other / only tuned) |
| --- | --- | --- |
| Qwen2.5-7B, base prompt | 44% | 5 / 67, p < 0.001 |
| Qwen2.5-7B, improved prompt | 58% | 10 / 50, p < 0.001 |
| Qwen2.5-14B, base prompt | 63% | 13 / 45, p < 0.001 |
| Qwen2.5-14B, improved prompt | 73% | 10 / 27, p = 0.0076 |
| **Qwen2.5-7B + LoRA on 1,240 runbook trajectories** (seed 0; seeds 1, 2: 88%, 94%) | **84%** | — |

The tuned 7B holds up on every held-out factor (templates 88%, entities 86%, scenarios 84%, seen
90%), and it has no general-capability regression on project 01's 60 held-out prompts (59/60 for base
and tuned).

**Cost** (one model at a time, idle GPU; `results/summary/latency.md`): the tuned 7B's episodes take
3.3 s at the median and 4.0 s at p95, against 16.2 s and 36.6 s for the 14B with the improved prompt,
in 6.3 GB of GPU memory against 14.6 GB.

**So, answer (a) and (b):** in this setting the tuned 7B doesn't just get close; it beats both, at
about a fifth of the larger model's latency and less than half its memory.

## What made it work, and what nearly didn't

1. **The first pilot learned a shortcut.** Gold trajectories took the shortest path for the planted
   fault, so their first call gave the answer away (3 of 18 prompt groups). The pilot matched the
   improved prompt on val *through the leak* and failed easy requests every baseline solves
   ([04](04-fine-tuning.md)). Fix: train on the observation-only runbook, which takes the same first
   steps for every task a prompt can't tell apart (`tests/test_sft_policy.py`).
2. **Then it learned the majority branch.** With the leak gone and 310 examples, the model followed
   the runbook's call order but took the most common continuation: 0 of 10 broken workers restarted.
3. **Data quantity fixed it** (M6, `results/summary/m6-data-quantity.md`): 310 examples give 49% on
   test_unseen, no better than the untuned model's 44% (p = 0.36) though with 9 unallowed effects
   instead of 77; 1,240 examples give 84% (5 vs 59 discordant).

## Where it fails

The M5 checkpoint's 20 val failures cluster at a few comparisons inside otherwise correct
trajectories: telling a bad config from a bad deploy (it rolls back the config), escalating when the
bad change isn't the latest version, reading queue growth, and threshold comparisons on heartbeat
age (`results/summary/m5-analysis.md`). It still causes unallowed effects (22 in 155 test_unseen
episodes, the 14B with the improved prompt 24), so anything real would need 02's approval layer.

## The result that limits the claim

**It does not transfer to another tool interface.** In project 02's own live harness, with 02's
tools (separate metric tools, a runbook tool, `submit_diagnosis`) and prompt, and only the weights
changed, the tuned 7B scored 12/30 against the base model's 16/30 (p = 0.42;
`results/summary/m5-transfer.md`). It kept its procedure (it opened with `service_status` in all 30
cases) but barely searched logs (1 call vs 37), which 02's evidence-citing diagnoses need. The 84%
is a result for the interface it was trained on.

## Threats to validity

- **Simulated environment and synthetic data.** Tasks run against a deterministic model of 02's
  sandbox; the training data is generated, not collected. The live transfer check is the one
  contact with real processes, and it went against the tuned model.
- **The runbook is the teacher.** The tuned model imitates a hand-written policy that solves the
  simulated tasks perfectly. The held-out templates, entities and scenarios test generalisation
  within that policy's world, not beyond it.
- **Seeds.** Three training seeds give 84%, 88% and 94% on test_unseen (next section): the
  conclusion holds for every seed, but the exact number moves by up to 10 points with the seed.
- **Serving noise.** 4-slot serving at temperature 0 is not bit-deterministic: a rerun of the same
  model on val changed 4 of 155 tasks ([04](04-fine-tuning.md)). The differences above are far larger.
- **The improved prompt was tuned on val** (two iterations), so its numbers are if anything
  optimistic; the tuned model's recipe was fixed before its test numbers were seen.

## Seed spread

Three training seeds of the same recipe (same 1,240 examples; only the training seed changes the
LoRA initialisation and example order), each evaluated on val and test_unseen
(`results/summary/m5-seeds.md`):

| Training seed | val | test_unseen | Unallowed effects (test_unseen) | vs 14B + improved prompt (only 14B / only tuned) |
| --- | --- | --- | --- | --- |
| 0 | 135/155 (87%) | 130/155 (84%) | 22 | 10 / 27, p = 0.0076 |
| 1 | 138/155 (89%) | 136/155 (88%) | 12 | 5 / 28, p = 0.0001 |
| 2 | 152/155 (98%) | 145/155 (94%) | 8 | 3 / 35, p < 0.001 |
| **mean ± sd** | | **88.4% ± 4.9** | | every seed ahead |

Every seed beats the 14B with the improved prompt (73%) by a significant margin, so the conclusion
does not hang on one run. But the training seed alone moves test_unseen by 10 points (seeds 0 and 2
differ: 3 vs 18 discordant tasks, p = 0.0015), more than serving noise (about 4 tasks) and more than
the gap between some baselines. Seed 0, the checkpoint in the other sections and the adapter card,
is the weakest of the three, so the single-seed numbers above are, if anything, conservative.
Any comparison of tuned models in this setup needs several seeds per condition.

## Conclusion

A 7B model fine-tuned with LoRA on 1,240 procedure traces completes 84–94% of held-out tool-use
tasks by resulting state (three training seeds, mean 88%), ahead of the same model with a better prompt (58%) and of a 2x larger model with
that prompt (73%), at about a fifth of the larger model's latency. Two conditions make the claim
honest: the traces must not leak the answer through their structure, and the model must be used
with the tool interface it was trained on; on a different interface for the same services the gain
disappeared.
