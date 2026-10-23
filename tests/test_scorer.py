"""The scorer's central claims (PLAN.md M1 + Tests): the reference solution passes every scenario,
and plausible-looking wrong trajectories fail. Run over the whole catalog (every kind x variant x
distractor) for several seeds."""

from __future__ import annotations

import pytest

from executor.runner import Budget, ScriptedPolicy, run_episode
from executor.sandbox import World
from executor.scorer import score
from tasks.generator.generate import catalog
from tasks.tools import ACTION_TOOLS, OUTCOMES

SEEDS = [0, 1, 2]
TASKS = [t for s in SEEDS for t in catalog(s)]


def run(task, calls):
    return score(task, run_episode(task, ScriptedPolicy(calls)))


def ids(tasks):
    return [f"{t['scenario']}#{t['seed']}" for t in tasks]


def _actions(gold):
    return [c for c in gold if c[0] in ACTION_TOOLS]


@pytest.mark.parametrize("task", TASKS, ids=ids(TASKS))
def test_reference_solution_succeeds(task):
    r = run(task, task["gold"])
    assert r["success"], r["failed_checks"]


@pytest.mark.parametrize("task", TASKS, ids=ids(TASKS))
def test_wrong_outcome_fails(task):
    gold = [list(c) for c in task["gold"]]
    wrong = next(o for o in OUTCOMES if o != task["spec"]["outcome"])
    gold[-1] = ["finish", {**gold[-1][1], "outcome": wrong}]
    assert not run(task, gold)["success"]


FIX_TASKS = [t for t in TASKS if t["spec"]["allowed_effects"]]


@pytest.mark.parametrize("task", FIX_TASKS, ids=ids(FIX_TASKS))
def test_claiming_success_without_acting_fails(task):
    """Same reads, same 'resolved' finish, but the action is dropped: the claim alone scores nothing."""
    calls = [c for c in task["gold"] if c[0] not in ACTION_TOOLS]
    r = run(task, calls)
    assert not r["success"] and r["outcome"] == "resolved"


def _extra_action(task):
    allowed = task["spec"]["allowed_effects"]
    if "restart_service:api" not in allowed and "rollback_config" not in allowed:
        return ["restart_service", {"service": "api"}]
    return ["restart_service", {"service": "report-exporter"}]


@pytest.mark.parametrize("task", TASKS, ids=ids(TASKS))
def test_extra_unneeded_action_fails(task):
    gold = [list(c) for c in task["gold"]]
    r = run(task, gold[:-1] + [_extra_action(task)] + gold[-1:])
    assert not r["success"] and r["unallowed_effects"]


WORKER_RESTARTS = [t for t in TASKS if any(c[0] == "restart_service" and "instance" in c[1] for c in t["gold"])
                   and len(World.from_dict(t["world"]).workers()) > 1]


@pytest.mark.parametrize("task", WORKER_RESTARTS, ids=ids(WORKER_RESTARTS))
def test_restarting_the_wrong_worker_fails(task):
    workers = World.from_dict(task["world"]).workers()
    calls = []
    for name, args in task["gold"]:
        if name == "restart_service" and "instance" in args:
            args = {**args, "instance": next(w for w in workers if w != args["instance"])}
        calls.append([name, args])
    assert not run(task, calls)["success"]


SWAP = {"rollback_config": ["rollback_release", {}], "rollback_release": ["rollback_config", {}],
        "cleanup_spool": ["restart_service", {"service": "report-exporter"}],
        "scale_service": ["restart_service", {"service": "worker"}],
        "restart_service": ["rollback_config", {}]}
ONE_ACTION = [t for t in TASKS if len(_actions(t["gold"])) == 1]


@pytest.mark.parametrize("task", ONE_ACTION, ids=ids(ONE_ACTION))
def test_plausible_wrong_action_fails(task):
    calls = [SWAP[c[0]] if c[0] in ACTION_TOOLS else list(c) for c in task["gold"]]
    assert not run(task, calls)["success"]


def test_fix_incidents_start_broken_and_no_change_tasks_need_nothing():
    for t in TASKS:
        w = World.from_dict(t["world"])
        if t["family"] == "incident" and t["spec"]["outcome"] == "resolved":
            assert w.problems(), t["scenario"]
        if t["kind"].startswith("req_") and t["kind"] not in ("req_restart_failed",) and t["variant"] not in ("true",):
            assert not w.problems(), (t["scenario"], w.problems())


def test_no_change_tasks_pass_with_finish_alone():
    for t in TASKS:
        if not t["spec"]["allowed_effects"]:
            assert run(t, [["finish", {"outcome": t["spec"]["outcome"], "message": "x"}]])["success"], t["scenario"]


def test_effects_and_state_changes_agree():
    for t in TASKS:
        ep = run_episode(t, ScriptedPolicy(t["gold"]))
        changed = ep.world.snapshot() != World.from_dict(t["world"]).snapshot()
        assert changed == bool(ep.world.effects), t["scenario"]


def test_never_finishing_fails_on_step_budget():
    t = TASKS[0]
    ep = run_episode(t, ScriptedPolicy([["service_status", {}]] * 20), Budget(max_steps=5))
    r = score(t, ep)
    assert ep.stop_reason == "max_steps" and not r["success"] and r["steps"] == 5


def test_generation_is_deterministic():
    assert catalog(0) == catalog(0)
    assert [t["task_id"] for t in catalog(0)] != [t["task_id"] for t in catalog(1)]


def test_state_change_without_a_recorded_effect_is_still_caught(monkeypatch):
    """Backstop for a sandbox bug: an action that changes state but forgets its effect record must
    still fail a task where nothing may change (the `unchanged` check reads state, not effects)."""
    t = next(t for t in TASKS if t["kind"] == "req_unavailable")

    def silent_restart(self, service, instance=None):
        self.instances["api"].restarts += 1
        return {"ok": True}

    monkeypatch.setattr(World, "restart_service", silent_restart)
    r = run(t, [["restart_service", {"service": "api"}], t["gold"][-1]])
    assert r["effects"] == [] and r["failed_checks"] == ["unchanged"]
