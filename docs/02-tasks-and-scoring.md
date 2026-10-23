# 02 — Tasks and scoring

What a model is asked to do, what it can call, and how a pass is decided. This is the part every
result in [05](05-evaluation.md) rests on, so it was built and tested first (M1) and has not changed
since the baselines started.

## The tools: 10 plus `finish`

All schemas are strict (`additionalProperties: false`, enums and patterns on every target) and live in
`tasks/tools.py`. They are derived from project 02's tool schemas: same names, scopes and enums where
the tool exists there.

| Kind | Tool | Arguments | Effect |
| --- | --- | --- | --- |
| Read | `service_status` | none | Every instance: running or exited, health, heartbeat age (workers), restarts, uptime |
| Read | `get_metrics` | `component`: api, workers, queue, database, disk, payments | 60 s of metrics for that component |
| Read | `search_logs` | `instance`, optional `contains`, `level`, `last` (≤ 50) | Recent JSON log records |
| Read | `config_history` | optional `last` (≤ 20) | Config versions with changed keys, author, note |
| Read | `deploy_history` | optional `last` (≤ 20) | API releases with author and note |
| Action | `restart_service` | `service`: api, worker, report-exporter; optional `instance`: worker-1..4 | Restart every instance of the service, or one worker |
| Action | `scale_service` | `service`: worker, `replicas`: 1–4 | Set worker count |
| Action | `rollback_config` | none | Restore the previous config version and restart api and workers |
| Action | `rollback_release` | none | Redeploy the previous API release |
| Action | `cleanup_spool` | `pattern`: only `*.tmp` | Delete temporary spool files |
| End | `finish` | `outcome`: resolved, no_action_needed, escalated, need_information; `message` (1–300 chars) | Ends the episode; changes nothing |

Three choices in this set are deliberate (PLAN.md, Decisions):

- **02's six metric tools became one `get_metrics(component)`,** and `escalate` became a `finish`
  outcome, to stay within the brief's 5–10 tools.
- **There is no `rotate_credential`.** An expired payments credential is the *unavailable tool* case:
  the right answer is to escalate with nothing changed. A call to a tool that doesn't exist is
  rejected by the validator like any other invalid call (`tests/test_validation.py`).
- **`config_history` and `deploy_history` stay separate,** although they could have been one tool,
  because never calling them was project 02's main failure. The base 7B reproduces it: 6
  `config_history` calls and 1 `deploy_history` call in 155 val episodes
  (`results/summary/m3-val-analysis.md`).

## The sandbox: a deterministic model of 02's jobs service

`executor/sandbox.py` holds a `World`: instances (state, hung flag, restarts, uptime), config
versions, releases, spool, queue, database connections and the payments provider. Read tools return
views of it; action tools change it and append an **effect** record (`restart_service:worker-3`,
`scale_service:worker`, ...). It mirrors 02's semantics: `rollback_config` restores the previous
version and restarts api and workers; a hung worker still answers `/health` but its heartbeat is
900 s old; restarting the exporter stops its connection leak.

It also has an independent health check, `World.problems()`, computed from state alone: api down,
database config broken, bad release live, exited or hung worker, exporter leak, queue inflow above
worker capacity, spool over quota, payments slow or credential expired. This plays the role of 02's
verifier.

Why a model and not 02's live processes: same task plus same calls gives the same observations and
the same score every time, and episodes are cheap: the 2,476 tests, most of them full scripted
episodes, run in about 13 s on CPU, against 10–200 s per live episode
([09](09-implementation-notes.md) § 1). The price is realism; [07](07-risks.md) § R6.

## The tasks

22 kinds, 31 kind-variants (a variant is a state-dependent branch: the condition in "if hung,
restart" is true or false; "restart the worker" has one worker or several), each combinable with 6
distractors: **186 scenarios** (`tasks/generator/scenarios.py`, `results/summary/m1.md`).

| | Kinds |
| --- | --- |
| Incidents (an alert; diagnose and fix) | crashed or hung worker, bad DB config (latest change, or buried under a later one), connection leak, queue backlog, full spool, bad deploy, slow payments provider, expired credential, client-error false alarm |
| Requests (an operator asks) | restart an instance / service, "restart the worker" (one vs several), restart the failed one, scale to N / already N / no N given, "if hung, restart", "if over quota, clean", clean up, roll back release, undo X's config change (latest or not), asks no tool can do |

Distractors add unrelated noise before the fault: none, a benign config change, a recent good deploy,
old errors in the logs, client-error (4xx) noise, spool near quota. They change what the tools return,
never the prompt or the right answer ([03](03-dataset-card.md) § Known weaknesses).

Each task carries a **spec** and a **reference trajectory** (`gold`):

```text
prompt   "Chen: jobs seem stuck or slow, 782 in the queue. Can you check what's going on and sort it out?"
spec     outcome resolved · allowed_effects [restart_service:worker-3] · require_healthy · require.restarted [worker-3]
gold     service_status → restart_service(worker, worker-3) → service_status → finish(resolved, ...)
```

(`python -m tasks.generator.generate --catalog out.jsonl`, first line.) Across the 930-task M1
catalogue, 570 tasks need `resolved`, 150 `escalated`, 120 `no_action_needed`, 90 `need_information`:
39% must change nothing, so a model that always acts can't score well.

## Scoring: from state, never from the message

`executor/scorer.py`. A task succeeds only if every applicable check passes:

| Check | Passes when | Applies |
| --- | --- | --- |
| `finished` | The episode ended with `finish`, not on the step or retry budget | Always |
| `outcome` | `finish` outcome equals the spec's | Always |
| `effects_allowed` | Every effect the sandbox recorded matches an allowed pattern | Always |
| `unchanged` | Final state snapshot equals the initial one | When no effect is allowed |
| `healthy` | `World.problems()` is empty at the end | When a fix is expected |
| `replicas`, `release`, `restarted`, `spool_tmp_files`, `config_values` | The required state holds | Per task |

The message is free text and never read. The scorer also records, per episode, the failed checks,
read and action call counts, invalid calls, unallowed effects, and whether the set of action tools
used equals the reference's (`tool_selection_ok`). The summaries aggregate these
([05](05-evaluation.md) § Metrics).

**Success is strict.** Restarting all workers when one crashed fails `effects_allowed`, because the
tool supports per-instance restarts:

```text
restart_service(worker) → finish(resolved)   failed_checks [effects_allowed]
                                              unallowed     [restart_service:worker-1, worker-2, worker-4]
```

Strictness is visible rather than hidden: every failure names its check, and the "restart all
workers, claim resolved" floor below shows what it costs a lazy policy (7.2%).

`unchanged` compares state, not effects, on purpose: it is the backstop for a sandbox bug where an
action changes state without recording an effect (`test_state_change_without_a_recorded_effect_is_still_caught`).

## Proof the tasks are solvable, and the floor

`python -m eval.m1`, 930 tasks = 186 scenarios × 5 seeds (`results/summary/m1.md`):

| Policy | Success |
| --- | --- |
| Reference trajectories (written from each task's setup) | 100% |
| Hand-written runbook (sees only tool output, plus what a request asks) | 100% |
| Always `finish: escalated` (best trivial strategy) | 16.1% |
| Always `finish: no_action_needed` | 12.9% |
| Always `finish: need_information` | 9.7% |
| Restart all workers, claim resolved | 7.2% |
| Always `finish: resolved` | 0% |

**The runbook** (`baselines/runbook.py`) is one decision tree per family. It never reads the task's
world, variant, spec or gold. For incidents it gets nothing but tool output (the alert text isn't
parsed); for requests it gets the request's meaning (kind plus slots such as which instance or how many
replicas), while whether a condition holds, whether an instance exists, or whether a config change is
the latest always comes from a tool. Solving 930/930 that way shows every task is decidable from
what a model can see. It was written from the catalogue, so it is an upper bound, not a realistic
baseline. Since M4 it is also the training-trajectory source ([04](04-fine-tuning.md)).

## The tests that hold this up

2,476 tests pass on CPU (`pytest -q`, 2026-10-23); 2,453 of them are in `tests/test_scorer.py`,
parametrised over every scenario for seeds 0–2.

| Claim | Test |
| --- | --- |
| Invalid arguments, unknown tools, injected fields and malformed JSON never execute and cost a retry; the 4th invalid call ends the episode | `tests/test_validation.py` |
| The reference passes every scenario | `test_reference_solution_succeeds` |
| Wrong outcome, success claimed without acting, one extra unneeded action, the wrong worker, a plausible wrong action: each fails wherever it applies | `tests/test_scorer.py` |
| Fix-incidents start broken; no-change tasks pass with `finish` alone | same file |
| Effects and state changes agree for every reference trajectory | `test_effects_and_state_changes_agree` |
| Generation is deterministic | `test_generation_is_deterministic` |

PLAN.md records that disabling any scorer check makes at least one test fail (a mutation check). The
mutation runs themselves are not committed as a script, so that claim can be re-checked only by
repeating it by hand **[verify]**.
