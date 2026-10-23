"""Invalid tool arguments fail before execution (PLAN.md Tests), and cost a retry."""

from __future__ import annotations

import pytest

from executor.runner import Budget, ScriptedPolicy, run_episode
from tasks.generator.generate import generate
from tasks.tools import ToolArgumentError, openai_tools, validate_call

TASK = generate("req_restart_instance", "exists", "none", "request.restart_instance.0", seed=3)


@pytest.mark.parametrize("name,args", [
    ("restart_service", {"service": "worker", "instance": "worker-9"}),        # out of scope
    ("restart_service", {"service": "database"}),                              # not a service we may touch
    ("restart_service", {"service": "worker", "approved": True}),              # injected field
    ("scale_service", {"service": "worker", "replicas": 12}),
    ("scale_service", {"service": "worker"}),                                  # missing required
    ("cleanup_spool", {"pattern": "*"}),                                       # only *.tmp
    ("finish", {"outcome": "fixed", "message": "done"}),
    ("rotate_credential", {"name": "payments"}),                               # unavailable tool
    ("get_metrics", {"component": "everything"}),
])
def test_invalid_calls_are_rejected_and_never_execute(name, args):
    with pytest.raises(ToolArgumentError):
        validate_call(name, args)
    ep = run_episode(TASK, ScriptedPolicy([[name, args]]))
    assert ep.steps[0]["valid"] is False and ep.world.effects == [] and ep.invalid_calls >= 1


def test_malformed_json_arguments_are_rejected():
    ep = run_episode(TASK, ScriptedPolicy([["restart_service", '{"service": "worker",']]))
    assert not ep.steps[0]["valid"] and "not valid JSON" in ep.steps[0]["error"]


def test_retry_budget_ends_the_episode():
    bad = [["scale_service", {"service": "worker", "replicas": 99}]] * 10
    ep = run_episode(TASK, ScriptedPolicy(bad), Budget(max_retries=3))
    assert ep.stop_reason == "max_retries" and ep.invalid_calls == 4 and ep.world.effects == []


def test_tool_set_size_and_shape():
    tools = openai_tools()
    assert len(tools) == 11   # 5 read + 5 action + finish
    assert all(t["function"]["parameters"]["additionalProperties"] is False for t in tools)
