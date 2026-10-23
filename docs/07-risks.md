# 07 — Risks and failure modes

Two kinds of risk: **thesis risks** (the study runs but doesn't show what it claims) and **delivery
risks** (it doesn't get finished). The first kind is worse, because a results table can look complete
while measuring the wrong thing. Two of the thesis risks below have already happened; they are kept
here with what detected them.

## Thesis risks

### R1 — The model learns the generator, not the task *(high likelihood, fatal to the thesis)*

Synthetic tasks are regular: 8 phrasing frames per family, fixed entity pools, fixed trajectories. A
tuned model can score by recognising a prompt rather than reading the system.

**Detection:** the prompt-lookup ceiling per set (37–40% on seen-template sets, 0% on test_template
and test_unseen, `results/summary/m2_audit.md`); the gap between test_seen and test_unseen for the
tuned model (M5); the transfer check on 02's live incidents (M5, not built).

**Mitigation:** splits fixed before generation, test_unseen holding out template, entity and scenario
at once, reported first; `tests/test_splits.py` fails on any factor leak.

### R2 — Training targets leak the answer *(happened in pilot 1)*

Gold trajectories are the generator's shortest path for the fault it planted, so in 3 of 18 prompt
groups the first call encodes the outcome. Pilot 1 learned exactly that: req_scale 6/10 where every
baseline scores 10/10, all four failures finishing `no_action_needed` with the wrong worker count.

**Detection:** per-kind failure analysis against the baselines, then reading the failing trajectories
(`results/summary/m4-pilot.md`). Now automated: `tests/test_sft_policy.py` asserts that
prompt-indistinguishable tasks start with the same call, and that the check catches the gold leak.

**Mitigation:** train on runbook trajectories (0 leaking groups). Gold stays available so the leaky
data can be rebuilt for the data-quality ablation.

### R3 — Imitating the majority branch instead of reading the observation *(happened in pilot 2)*

With the leak removed, the 310-example model follows the runbook's call order and takes the most
common continuation: it never restarts a broken worker on val (0/10), and 38 of 66 failures finish
`no_action_needed` where an action was needed.

**Detection:** success per kind, plus outcome counts among failures; the branch statistics of the
training data (after service_status: get_metrics 90, restart_service 20 of 110 incident examples).

**Mitigation:** more data first (M5); if the imbalance persists at scale, rebalance branches or add
hard negatives (an M6 data-quality ablation). If neither helps, that is the negative result, and it is
reportable.

### R4 — Selecting on val, then reporting val *(medium likelihood)*

The improved prompt was chosen on val after two iterations; both pilots were diagnosed on val. Val
scores of those conditions are optimistic, and val comparisons between them are weak.

**Mitigation:** no design decision has used a test set; no test set has seen a tuned model. The tuned
checkpoint is chosen on val and then run on test_unseen once.

### R5 — Noise read as an effect *(high likelihood)*

n = 155 per set (±8 points), and 4-slot serving flips about 4 tasks between identical runs.

**Detection / mitigation:** counts with Wilson intervals; paired exact McNemar on the same tasks; M5
seeds for training spread. Pilot 2 vs pilot 1 (89 vs 96, p = 0.41) is reported as no difference.

### R6 — The simulator is not operations *(certain, by design)*

Results describe models on a deterministic model of 02's sandbox, not live incident response.

**Mitigation:** say so in every claim; the transfer check on 02's 30 live incidents gives one point of
contact with real processes.

### R7 — Strictness penalises models unevenly *(medium)*

Strict success and strict tool-call format hit some models harder: the 14B writes calls as text
(76 text replies for improved-v2-14b on test_unseen).

**Mitigation:** the same rules for every condition, fixed before the runs; invalid calls and failed
checks reported per condition so the cost is visible ([05](05-evaluation.md)).

### R8 — Quantisation hides or creates the effect *(low)*

**Mitigation:** every condition runs at Q4_K_M from the same llama.cpp build; pilot 1's f16 vs Q4_K_M
was 99 vs 96/155, p = 0.63.

## Delivery risks

### R9 — A shared GPU *(high likelihood, already biting)*

Other users' jobs share the machine. The 14B test run was stopped twice for memory (once by hand at
3 GB free, once by `env/mem_guard.sh` at < 6 GB available, `results/raw/partial/README.md`); pilot 2's
early steps ran up to 2× slower.

**Mitigation:** long jobs go through `env/run_guarded.sh` with the memory guard; partial runs are kept
apart and never summarised as results.

### R10 — Training stack on arm64 + CUDA *(medium)*

A plain `uv pip install peft` tried to pull torch from PyPI over the host's CUDA build.

**Mitigation:** `train/.venv` with `--system-site-packages` and PEFT installed `--no-deps`; versions
pinned in `train/ENVIRONMENT.md`.

### R11 — No quiet GPU time for the latency/memory benchmark *(medium)*

**Mitigation:** the benchmark is separate from accuracy runs and needs only one model at a time. If it
cannot be run cleanly, report accuracy only and say the cost half of the question is open.

### R12 — Ablation sprawl *(medium)*

**Mitigation:** one factor at a time, one required; data quantity first, because M5 produces its
points anyway ([06](06-roadmap.md) § Cut line).

## Claim hygiene

| Don't claim | Do claim |
| --- | --- |
| "Fine-tuning matched the improved prompt" | "On val, a 310-example pilot tied the improved prompt (96 vs 96/155), through a leak in its training trajectories; no tuned model has a test-set result yet" |
| "The 7B matches the 14B" | "With the improved prompt, the 7B is not distinguishable from the unprompted 14B on test_unseen (90 vs 98/155, p = 0.26); the prompted 14B is ahead (113/155)" |
| "62% accuracy" | "96/155 on val, 95% CI 0.54–0.69" |
| "Deterministic evaluation" | "Deterministic sandbox and scoring; temperature-0 serving with 4 slots flips about 4 of 155 tasks between runs" |
| "Tested on real incidents" | "Tested on a deterministic model of project 02's sandbox; the live transfer check is planned" |
| "Faster and smaller" | Nothing yet: latency and memory are not measured |
| "No leakage" | "No held-out template, entity or scenario combination occurs in train (tested); seen-template sets still allow 37–40% by prompt lookup, so test_unseen is primary" |

## The question to have a real answer for

> *"A better prompt takes the 7B from 44% to 58% and level with the 14B. Why fine-tune at all?"*

That is the comparison the study is built around, and so far the honest answer is "not shown yet."
The pilots show what fine-tuning on this task picks up first: whatever in the data predicts the
answer most cheaply (a leaked first call, then the majority branch). The case for tuning would be a
tuned 7B beating `improved-v2-7b` on test_unseen with paired significance, at the 7B's serving cost.
If M5 doesn't show that, the result is that prompting is the better tool here at this data scale, and
the leak and branch analysis is the useful finding.
