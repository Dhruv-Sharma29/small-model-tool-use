"""No train/eval overlap (PLAN.md Tests: automated check). Sets are rebuilt in memory from the frozen
splits, exactly as tasks.splits.build makes them."""

from __future__ import annotations

import pytest

from eval.m2_audit import entity_hits
from tasks.splits import factors, load
from tasks.splits.build import ORDER, build_set, content_key

SPLITS = load()
HELD = {p: v["held_out"] for p, v in SPLITS["entities"].items()}
TRAIN_SIDE = {p: v["train"] for p, v in SPLITS["entities"].items()}
HELD_TEMPLATES = set(SPLITS["templates"]["held_out"])
HELD_SCENARIOS = {f"{k}:{d}" for k, d in SPLITS["scenarios"]["held_out_distractor"].items()}


@pytest.fixture(scope="module")
def sets():
    seen: set[str] = set()
    return {name: build_set(name, seen)[0] for name in ORDER}


def test_train_contains_no_held_out_factor(sets):
    train = sets["train"]
    assert not {t["template_id"] for t in train} & HELD_TEMPLATES
    assert not {t["scenario"] for t in train} & HELD_SCENARIOS
    assert not [h for t in train for h in entity_hits(HELD, t)]


@pytest.mark.parametrize("name", ["test_template", "test_entity", "test_scenario", "test_unseen"])
def test_held_out_sets_use_only_held_out_values_of_their_factors(sets, name):
    held = set(SPLITS["sets"][name]["held_out"])
    for t in sets[name]:
        assert (t["template_id"] in HELD_TEMPLATES) == ("template" in held)
        assert (t["scenario"] in HELD_SCENARIOS) == ("scenario" in held)
        if "entity" in held:
            assert not entity_hits(TRAIN_SIDE, t), t["task_id"]


def test_no_task_appears_in_two_sets(sets):
    keys = [content_key(t) for name in ORDER for t in sets[name]]
    assert len(keys) == len(set(keys))


def test_sets_are_balanced_and_sized_as_defined(sets):
    for name, tasks in sets.items():
        per = SPLITS["sets"][name]["per_kind_variant"]
        counts = {}
        for t in tasks:
            counts[(t["kind"], t["variant"])] = counts.get((t["kind"], t["variant"]), 0) + 1
        assert set(counts.values()) == {per} and len(counts) == 31


def test_every_set_task_passes_its_reference(sets):
    from executor.runner import ScriptedPolicy, run_episode
    from executor.scorer import score
    for name in ORDER[1:]:
        for t in sets[name]:
            assert score(t, run_episode(t, ScriptedPolicy(t["gold"])))["success"], (name, t["scenario"])


def test_factors_cover_every_kind_variant():
    for name in ORDER:
        f = factors(name)
        assert all(f["distractors"][kv] for kv in f["distractors"]) and len(f["distractors"]) == 31
