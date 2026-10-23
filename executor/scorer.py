"""Scoring from resulting state (PLAN.md M1). The model's `finish` message is never trusted. The
only thing taken from it is the outcome category, which must match the task's. Everything else is
read from the sandbox: the effects it performed and its final state.

A task succeeds only if every applicable check passes:
    finished          the episode ended with finish (not the step or retry budget)
    outcome           finish outcome == spec outcome
    effects_allowed   every effect the sandbox performed matches spec.allowed_effects
    unchanged         if nothing may change: final snapshot == initial snapshot
    healthy           if required: the sandbox's own health check finds no problem
    require.*         replicas, release, restarted instances, spool temp files, config values
"""

from __future__ import annotations

from fnmatch import fnmatch

from executor.runner import Episode
from executor.sandbox import World
from tasks.tools import ACTION_TOOLS, READ_TOOLS


def score(task: dict, ep: Episode) -> dict:
    spec, final = task["spec"], ep.world
    initial = World.from_dict(task["world"])
    effects = [e["effect"] for e in final.effects]
    allowed = spec["allowed_effects"]
    checks = {"finished": ep.stop_reason == "finished",
              "outcome": bool(ep.finish) and ep.finish["outcome"] == spec["outcome"],
              "effects_allowed": all(any(fnmatch(e, p) for p in allowed) for e in effects)}
    if not allowed:
        checks["unchanged"] = final.snapshot() == initial.snapshot()
    if spec["require_healthy"]:
        checks["healthy"] = not final.problems()
    req = spec["require"]
    if "replicas" in req:
        checks["replicas"] = len(final.workers()) == req["replicas"]
    if "release" in req:
        checks["release"] = final.release() == req["release"]
    if "restarted" in req:
        checks["restarted"] = all(f"restart_service:{i}" in effects for i in req["restarted"])
    if "spool_tmp_files" in req:
        checks["spool_tmp_files"] = final.spool["tmp_files"] == req["spool_tmp_files"]
    if "config_values_of" in req:
        want = next(v["values"] for v in initial.config_versions if v["version"] == req["config_values_of"])
        checks["config_values"] = final.config() == want

    valid = [s for s in ep.steps if s["valid"]]
    gold_actions = sorted({c[0] for c in task["gold"] if c[0] in ACTION_TOOLS})
    used_actions = sorted({s["name"] for s in valid if s["name"] in ACTION_TOOLS})
    return {
        "task_id": task["task_id"], "success": all(checks.values()), "checks": checks,
        "failed_checks": [k for k, v in checks.items() if not v],
        "outcome": ep.finish["outcome"] if ep.finish else None, "expected_outcome": spec["outcome"],
        "stop_reason": ep.stop_reason, "steps": len(ep.steps), "invalid_calls": ep.invalid_calls,
        "read_calls": sum(s["name"] in READ_TOOLS for s in valid),
        "action_calls": sum(s["name"] in ACTION_TOOLS for s in valid),
        "effects": effects, "unallowed_effects": [e for e in effects if not any(fnmatch(e, p) for p in allowed)],
        "tool_selection_ok": used_actions == gold_actions,
        "problems_after": final.problems(),
    }
