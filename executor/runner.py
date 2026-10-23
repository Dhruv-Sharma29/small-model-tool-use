"""Episode runner: the same loop for every policy and model (PLAN.md M3: identical tool set, retry
budget and max steps for every condition).

Each step the policy returns one tool call. The call is validated against the frozen schema
*before* it can execute. An invalid call, an unknown tool or a reply with no tool call costs one
retry and the policy sees the error. The episode ends at `finish`, at the step limit, or when the
retry budget is spent. Only `finish` ends it as finished; the other two are failures.

Messages use the OpenAI chat shape (assistant tool_calls + tool results), so an LLM policy (M3)
reads the same transcript that is recorded here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Protocol

from executor.sandbox import World
from tasks.tools import ACTION_TOOLS, ToolArgumentError, validate_call

SYSTEM_PROMPT = """You are the on-call operator for a small jobs service: an API, 1-4 workers (worker-1 .. worker-4), a report exporter, a PostgreSQL database, a spool directory and an external payments provider.

You get an alert or a request. Use the tools to look at the system, make only the changes the situation needs, then call `finish` exactly once.

- Change nothing that isn't needed. Restart only the instances that need it.
- If the request is ambiguous or refers to something that doesn't exist, change nothing and finish with need_information.
- If the fix needs a person or a tool you don't have, change nothing and finish with escalated.
- If nothing is wrong or nothing needs changing, finish with no_action_needed.
- Call one tool at a time."""


@dataclass(frozen=True)
class Budget:
    max_steps: int = 12       # tool calls, including finish and invalid ones
    max_retries: int = 3      # invalid calls tolerated; the next one ends the episode


@dataclass
class Episode:
    task_id: str
    steps: list[dict] = field(default_factory=list)   # {name, args, valid, error?, observation?}
    finish: dict | None = None
    stop_reason: str = ""                              # finished | max_steps | max_retries
    invalid_calls: int = 0
    world: World | None = None
    messages: list[dict] = field(default_factory=list)


class Policy(Protocol):
    def act(self, messages: list[dict]) -> dict:
        """Return {"name": str, "arguments": dict | str} for one tool call, or {"text": str} if
        the reply contained no tool call. Extra keys (usage, latency, raw) are recorded."""


def initial_messages(task: dict, system_prompt: str = SYSTEM_PROMPT) -> list[dict]:
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": task["prompt"]}]


def run_episode(task: dict, policy: Policy, budget: Budget = Budget(), system_prompt: str = SYSTEM_PROMPT) -> Episode:
    world = World.from_dict(task["world"])
    ep = Episode(task_id=task["task_id"], world=world, messages=initial_messages(task, system_prompt))
    for step in range(budget.max_steps):
        reply = policy.act(ep.messages)
        call_id = f"call_{step}"
        if "name" not in reply:
            ep.invalid_calls += 1
            ep.steps.append({"name": None, "args": None, "valid": False, "error": "no tool call", **_extra(reply)})
            ep.messages += [{"role": "assistant", "content": reply.get("text", "")},
                            {"role": "user", "content": "Error: reply with exactly one tool call."}]
        else:
            name, args = reply["name"], reply["arguments"]
            ep.messages.append({"role": "assistant", "content": None, "tool_calls": [{
                "id": call_id, "type": "function",
                "function": {"name": name, "arguments": args if isinstance(args, str) else json.dumps(args)}}]})
            try:
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except ValueError as e:
                        raise ToolArgumentError(f"{name}: arguments are not valid JSON ({e})") from e
                validate_call(name, args)
            except ToolArgumentError as e:
                ep.invalid_calls += 1
                ep.steps.append({"name": name, "args": args, "valid": False, "error": str(e), **_extra(reply)})
                ep.messages.append({"role": "tool", "tool_call_id": call_id, "content": json.dumps({"error": str(e)})})
            else:
                if name == "finish":
                    ep.steps.append({"name": name, "args": args, "valid": True, **_extra(reply)})
                    ep.finish, ep.stop_reason = args, "finished"
                    return ep
                obs = world.call(name, args)
                ep.steps.append({"name": name, "args": args, "valid": True, "observation": obs,
                                 "action": name in ACTION_TOOLS, **_extra(reply)})
                ep.messages.append({"role": "tool", "tool_call_id": call_id, "content": json.dumps(obs)})
        if ep.invalid_calls > budget.max_retries:
            ep.stop_reason = "max_retries"
            return ep
    ep.stop_reason = "max_steps"
    return ep


def _extra(reply: dict) -> dict:
    return {k: v for k, v in reply.items() if k not in ("name", "arguments", "text")}


class ScriptedPolicy:
    """Replays a fixed list of (tool, args) calls; returns a text reply once they run out."""

    def __init__(self, calls: list):
        self.calls = [tuple(c) for c in calls]
        self.i = 0

    def act(self, messages: list[dict]) -> dict:
        if self.i >= len(self.calls):
            return {"text": "(no more calls)"}
        name, args = self.calls[self.i]
        self.i += 1
        return {"name": name, "arguments": args}
