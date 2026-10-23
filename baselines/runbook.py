"""Hand-written runbook policy (PLAN.md M1 exit: "a hand-written policy can be evaluated").

It shows every task can be solved from what the tools return. It never reads the task's world,
variant, spec or gold:
- incidents: only tool observations (the alert text isn't even parsed);
- requests: what the request says (task kind + `request_slots`, e.g. which instance or how many
  replicas) plus tool observations. Whether a condition holds, whether an instance exists, or
  whether a config change is the latest is always read from the tools.

It is also a non-LLM baseline, like 02's scripted runbook: rules written from the scenario
catalogue, so it measures what's achievable with this evidence, not what's realistic for unseen
situations.
"""

from __future__ import annotations

import json
import math
from collections.abc import Generator

from executor.runner import run_episode
from executor.scorer import score

Obs = dict
Plan = Generator[tuple[str, dict], Obs, None]
HUNG_AFTER_S = 60


def _finish(outcome: str, message: str) -> tuple[str, dict]:
    return ("finish", {"outcome": outcome, "message": message})


def _broken_worker(status: Obs) -> str | None:
    for row in status["instances"]:
        if row["service"] == "worker" and (row["state"] != "running" or (row.get("heartbeat_age_s") or 0) > HUNG_AFTER_S):
            return row["instance"]
    return None


def incident() -> Plan:
    status = yield ("service_status", {})
    victim = _broken_worker(status)
    if victim:
        yield ("restart_service", {"service": "worker", "instance": victim})
        yield _finish("resolved", f"restarted {victim}")
        return
    api = yield ("get_metrics", {"component": "api"})
    reqs = api.get("requests_60s", {})
    if reqs.get("5xx", 0) > 0:
        db = yield ("get_metrics", {"component": "database"})
        if db["rejected_60s"] > 0 and db["connections_by_app"]["report-exporter"] > 10:
            yield ("restart_service", {"service": "report-exporter"})
            yield _finish("resolved", "report-exporter was exhausting database connections; restarted it")
            return
        hist = yield ("config_history", {"last": 5})
        db_changes = [v for v in hist["versions"] if any(k.startswith("api.db_") for k in v["changed"])]
        if db_changes and api["by_route"]["GET /health"]["5xx"] > 0:   # every route failing: the database is unreachable
            if db_changes[-1]["version"] == hist["current"]:
                yield ("rollback_config", {})
                yield _finish("resolved", f"rolled back config version {hist['current']}")
            else:
                yield _finish("escalated", f"bad db change in version {db_changes[-1]['version']} is not the latest")
            return
        yield ("deploy_history", {"last": 3})
        yield ("rollback_release", {})
        yield _finish("resolved", "latest release fails GET /jobs/{id}; rolled back")
        return
    if reqs.get("4xx", 0) > 100:
        yield _finish("no_action_needed", "client errors only; service healthy")
        return
    disk = yield ("get_metrics", {"component": "disk"})
    if disk["used_pct"] >= 100:
        yield ("cleanup_spool", {"pattern": "*.tmp"})
        yield _finish("resolved", "cleaned the spool")
        return
    queue = yield ("get_metrics", {"component": "queue"})
    if queue["trend"] == "growing":
        workers = yield ("get_metrics", {"component": "workers"})
        if any(w.get("last_error") for w in workers["workers"].values()):
            yield _finish("escalated", "payments provider problem")
            return
        need = math.ceil(queue["inflow_per_s"] / queue["per_worker_capacity_per_s"])
        if need <= 4:
            yield ("scale_service", {"service": "worker", "replicas": need})
            yield _finish("resolved", f"scaled to {need}")
            return
    pay = yield ("get_metrics", {"component": "payments"})
    if "401" in pay["status_60s"] or pay["p95_ms"] > 1000:
        yield _finish("escalated", "payments provider credential or latency problem; no tool for it")
        return
    yield _finish("no_action_needed", "found nothing wrong")


def request(kind: str, slots: dict) -> Plan:
    if kind in ("req_restart_instance", "req_if_hung"):
        status = yield ("service_status", {})
        row = next((r for r in status["instances"] if r["instance"] == slots["instance"]), None)
        if row is None:
            yield _finish("need_information", f"no {slots['instance']}")
        elif kind == "req_if_hung" and (row.get("heartbeat_age_s") or 0) <= HUNG_AFTER_S:
            yield _finish("no_action_needed", f"{slots['instance']} is not hung")
        else:
            yield ("restart_service", {"service": "worker", "instance": slots["instance"]})
            yield _finish("resolved", f"restarted {slots['instance']}")
    elif kind == "req_restart_service":
        service = {"API": "api", "report exporter": "report-exporter"}[slots["service_name"]]
        yield ("restart_service", {"service": service})
        yield _finish("resolved", f"restarted {service}")
    elif kind == "req_restart_ambiguous":
        status = yield ("service_status", {})
        workers = [r["instance"] for r in status["instances"] if r["service"] == "worker"]
        if len(workers) == 1:
            yield ("restart_service", {"service": "worker", "instance": workers[0]})
            yield _finish("resolved", f"restarted {workers[0]}")
        else:
            yield _finish("need_information", "which worker?")
    elif kind == "req_restart_failed":
        status = yield ("service_status", {})
        victim = _broken_worker(status)
        if victim:
            yield ("restart_service", {"service": "worker", "instance": victim})
            yield _finish("resolved", f"restarted {victim}")
        else:
            yield _finish("no_action_needed", "no failed worker")
    elif kind == "req_scale":
        status = yield ("service_status", {})
        n = sum(r["service"] == "worker" for r in status["instances"])
        if n == slots["n"]:
            yield _finish("no_action_needed", f"already {n}")
        else:
            yield ("scale_service", {"service": "worker", "replicas": slots["n"]})
            yield _finish("resolved", f"scaled to {slots['n']}")
    elif kind == "req_scale_unspecified":
        yield _finish("need_information", "how many replicas?")
    elif kind in ("req_if_spool_full", "req_cleanup"):
        if kind == "req_if_spool_full":
            disk = yield ("get_metrics", {"component": "disk"})
            if disk["used_pct"] < 100:
                yield _finish("no_action_needed", "spool under quota")
                return
        yield ("cleanup_spool", {"pattern": "*.tmp"})
        yield _finish("resolved", "cleaned the spool")
    elif kind == "req_rollback_release":
        yield ("rollback_release", {})
        yield _finish("resolved", "rolled back")
    elif kind == "req_undo_config_by_author":
        hist = yield ("config_history", {"last": 20})
        mine = [v for v in hist["versions"] if v["author"] == slots["author"]]
        if mine and mine[-1]["version"] == hist["current"]:
            yield ("rollback_config", {})
            yield _finish("resolved", f"reverted version {hist['current']}")
        else:
            yield _finish("escalated", "the change is not the latest version")
    elif kind == "req_unavailable":
        yield _finish("escalated", "no tool for that")
    else:
        raise ValueError(kind)


class RunbookPolicy:
    """Drives a plan generator: each tool result in the transcript is sent back into the plan."""

    def __init__(self, task: dict):
        self.plan = incident() if task["family"] == "incident" else request(task["kind"], task["request_slots"])
        self.started = False

    def act(self, messages: list[dict]) -> dict:
        try:
            if not self.started:
                self.started = True
                name, args = next(self.plan)
            else:
                name, args = self.plan.send(json.loads(messages[-1]["content"]))
        except StopIteration:
            return {"text": "(plan finished)"}
        return {"name": name, "arguments": args}


def evaluate(tasks: list[dict]) -> list[dict]:
    return [score(t, run_episode(t, RunbookPolicy(t))) for t in tasks]
