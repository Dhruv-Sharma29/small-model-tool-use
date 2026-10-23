"""The frozen tool set (PLAN.md M1): 5 read tools, 5 action tools and `finish`, with strict JSON
Schemas in the OpenAI function-calling shape.

Derived from 02-reliable-operations-agent/agent/tools/schemas.py (same names, scopes and enums where
the tool exists there). Changes, so the set stays within the brief's 5-10 tools:
- 02's six metric tools (api_metrics, worker_metrics, queue_stats, worker_heartbeats,
  db_connections, plus disk) are one `get_metrics(component)`.
- No `stop_service`, `runbook` or `escalate` tool. Escalating is a `finish` outcome.
- No `rotate_credential`: an expired credential is the "unavailable tool" case. The right response
  is to escalate without changing anything.
- `finish` replaces 02's submit_diagnosis. It reports what the agent did, never whether it worked:
  the scorer checks state itself.

Arguments are validated against these schemas before anything executes. The sandbox refuses
out-of-scope targets again on its own.
"""

from __future__ import annotations

from jsonschema import Draft202012Validator

WORKER_INSTANCE = {"type": "string", "pattern": "^worker-[1-4]$"}
INSTANCE = {"type": "string", "pattern": "^(api|report-exporter|worker-[1-4])$"}
OUTCOMES = ["resolved", "no_action_needed", "escalated", "need_information"]


def _obj(properties: dict | None = None, required: tuple[str, ...] = ()) -> dict:
    return {"type": "object", "properties": properties or {}, "required": list(required), "additionalProperties": False}


READ_TOOLS = {
    "service_status": ("State of every component instance: running or exited, health answer, seconds since the "
                       "last heartbeat (workers), restarts and uptime.", _obj()),
    "get_metrics": ("Metrics for one component over the last 60 s. api: requests by status, p95 latency. "
                    "workers: jobs done and failed, payments-call latency and status codes. queue: pending jobs, "
                    "inflow and processing rates. database: connections by application and the limit. "
                    "disk: spool usage and quota. payments: provider latency and error codes.", _obj({
                        "component": {"enum": ["api", "workers", "queue", "database", "disk", "payments"]},
                    }, required=("component",))),
    "search_logs": ("Most recent JSON log records of one instance, newest last, optionally filtered by a "
                    "substring and/or level.", _obj({
                        "instance": INSTANCE,
                        "contains": {"type": "string", "maxLength": 100},
                        "level": {"enum": ["info", "warning", "error"]},
                        "last": {"type": "integer", "minimum": 1, "maximum": 50},
                    }, required=("instance",))),
    "config_history": ("Recent config versions, newest last, with the keys each one changed, author and note.", _obj({
        "last": {"type": "integer", "minimum": 1, "maximum": 20}})),
    "deploy_history": ("Recent API releases, newest last, with author and note.", _obj({
        "last": {"type": "integer", "minimum": 1, "maximum": 20}})),
}

ACTION_TOOLS = {
    "restart_service": ("Restart a service: every instance of it, or only `instance` (workers).", _obj({
        "service": {"enum": ["api", "worker", "report-exporter"]},
        "instance": WORKER_INSTANCE,
    }, required=("service",))),
    "scale_service": ("Set the number of worker replicas.", _obj({
        "service": {"enum": ["worker"]}, "replicas": {"type": "integer", "minimum": 1, "maximum": 4},
    }, required=("service", "replicas"))),
    "rollback_config": ("Restore the previous config version and restart api and workers.", _obj()),
    "rollback_release": ("Redeploy the previous API release.", _obj()),
    "cleanup_spool": ("Delete temporary files from the spool directory.", _obj(
        {"pattern": {"enum": ["*.tmp"]}}, required=("pattern",))),
}

FINISH_TOOL = {
    "finish": ("End the task. outcome: resolved (you made the change that fixes or fulfils it), no_action_needed "
               "(nothing needed changing), escalated (it needs a person or a tool you don't have), "
               "need_information (the request is ambiguous; say what you need). Changes nothing.", _obj({
                   "outcome": {"enum": OUTCOMES},
                   "message": {"type": "string", "minLength": 1, "maxLength": 300},
               }, required=("outcome", "message"))),
}

ALL_TOOLS = {**READ_TOOLS, **ACTION_TOOLS, **FINISH_TOOL}
_VALIDATORS = {name: Draft202012Validator(schema) for name, (_, schema) in ALL_TOOLS.items()}


class ToolArgumentError(ValueError):
    pass


def validate_call(name: str, args: object) -> None:
    """Raise ToolArgumentError unless `name` is a known tool and `args` match its schema exactly."""
    if name not in _VALIDATORS:
        raise ToolArgumentError(f"unknown tool {name!r}")
    if not isinstance(args, dict):
        raise ToolArgumentError(f"{name}: arguments must be a JSON object")
    errors = sorted(_VALIDATORS[name].iter_errors(args), key=lambda e: list(e.path))
    if errors:
        raise ToolArgumentError(f"{name}: " + "; ".join(
            f"{'/'.join(map(str, e.path)) or '(args)'}: {e.message}" for e in errors[:3]))


def openai_tools() -> list[dict]:
    """Tool definitions as sent to an OpenAI-compatible server."""
    return [{"type": "function", "function": {"name": n, "description": d, "parameters": s}}
            for n, (d, s) in ALL_TOOLS.items()]
