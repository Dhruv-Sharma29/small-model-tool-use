"""Reading the frozen split definitions (splits.json) and turning a set name into generator inputs."""

from __future__ import annotations

import hashlib
import json
from functools import cache
from pathlib import Path

from tasks.generator.generate import DISTRACTORS, KINDS

SPLITS = Path(__file__).with_name("splits.json")


@cache
def load() -> dict:
    return json.loads(SPLITS.read_text())


def splits_sha() -> str:
    return hashlib.sha256(SPLITS.read_bytes()).hexdigest()[:16]


def factors(set_name: str) -> dict:
    """Allowed templates, entity pools and distractors for one set."""
    s = load()
    held = set(s["sets"][set_name]["held_out"])
    side = lambda factor: "held_out" if factor in held else "train"   # noqa: E731
    templates = set(s["templates"][side("template")])
    pools = {pool: v[side("entity")] for pool, v in s["entities"].items()}
    distractors = {}
    for kind in KINDS.values():
        for variant in kind.variants:
            h = s["scenarios"]["held_out_distractor"][f"{kind.name}:{variant}"]
            distractors[(kind.name, variant)] = [h] if "scenario" in held else [d for d in DISTRACTORS if d != h]
    return {"templates": templates, "pools": pools, "distractors": distractors}
