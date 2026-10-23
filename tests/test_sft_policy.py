"""Training trajectories must not give the answer away before an observation does (train/build_sft.py).

Tasks a model can't tell apart from the prompt (same request kind; for incidents, same alert) must
start with the same tool call. The generator's gold calls break this (a scale request that needs no
change starts with service_status, one that does starts with scale_service), so a model trained on
them can pick the outcome from its own first call instead of from the tool result."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pytest

from baselines.runbook import RunbookPolicy
from executor.runner import ScriptedPolicy, run_episode

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def train():
    return [json.loads(line) for line in (ROOT / "data/train.jsonl").read_text().splitlines()]


def first_calls(tasks: list[dict], policy) -> dict[str, set[str]]:
    by_group = defaultdict(set)
    for t in tasks:
        group = t["kind"] if t["family"] == "request" else t["template_id"].rsplit(".", 1)[0]
        call = next(m for m in run_episode(t, policy(t)).messages if m.get("tool_calls"))
        by_group[group].add(call["tool_calls"][0]["function"]["name"])
    return by_group


def test_runbook_trajectories_start_alike_within_each_prompt_group(train):
    assert {g: c for g, c in first_calls(train, RunbookPolicy).items() if len(c) > 1} == {}


def test_check_detects_the_gold_leak(train):
    leaky = {g for g, c in first_calls(train, lambda t: ScriptedPolicy(t["gold"])).items() if len(c) > 1}
    assert leaky == {"req_scale", "req_restart_instance", "alert.jobs_lagging"}
