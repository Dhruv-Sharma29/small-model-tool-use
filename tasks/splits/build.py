"""Generate every set from the frozen split definitions (PLAN.md M2).

    python -m tasks.splits.build              # -> data/<set>.jsonl + results/summary/m2_manifest.json

Each set has the same number of tasks per kind-variant (31 of them), so the outcome mix is the same
in every set. Per task: distractor, template family and template are drawn from what the set
allows; entities come from the set's pools; the seed comes from the set's own range. Exact
duplicates (same prompt and same world) are dropped within a set and against every earlier set;
the audit (eval/m2_audit.py) checks near-duplicates.
"""

from __future__ import annotations

import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path

from tasks.generator.generate import KINDS, generate
from tasks.splits import factors, load, splits_sha

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
ORDER = ["train", "val", "test_seen", "test_template", "test_entity", "test_scenario", "test_unseen"]


def content_key(task: dict) -> str:
    return hashlib.sha256(json.dumps([task["prompt"], task["world"]], sort_keys=True).encode()).hexdigest()


def build_set(name: str, seen: set[str]) -> tuple[list[dict], int]:
    cfg, f = load()["sets"][name], factors(name)
    rng = random.Random(f"build|{name}|{load()['seed']}")
    tasks, dropped, seed = [], 0, cfg["seed_offset"]
    for kind in KINDS.values():
        for variant in kind.variants:
            made = 0
            while made < cfg["per_kind_variant"]:
                seed += 1
                family = rng.choice(kind.template_families)
                template = rng.choice(sorted(t for t in f["templates"] if t.rsplit(".", 1)[0] == family))
                distractor = rng.choice(f["distractors"][(kind.name, variant)])
                task = generate(kind.name, variant, distractor, template, seed, pools=f["pools"])
                key = content_key(task)
                if key in seen:
                    dropped += 1
                    continue
                seen.add(key)
                task["set"] = name
                tasks.append(task)
                made += 1
    return tasks, dropped


def main() -> None:
    DATA.mkdir(exist_ok=True)
    seen: set[str] = set()
    manifest = {"generated": datetime.now(timezone.utc).isoformat(timespec="seconds"), "splits_sha": splits_sha(), "sets": {}}
    for name in ORDER:
        tasks, dropped = build_set(name, seen)
        path = DATA / f"{name}.jsonl"
        body = "".join(json.dumps(t, sort_keys=True) + "\n" for t in tasks)
        path.write_text(body)
        outcomes = {}
        for t in tasks:
            outcomes[t["spec"]["outcome"]] = outcomes.get(t["spec"]["outcome"], 0) + 1
        manifest["sets"][name] = {"tasks": len(tasks), "exact_duplicates_dropped": dropped,
                                  "sha256": hashlib.sha256(body.encode()).hexdigest(), "outcomes": outcomes}
        print(f"{name:14} {len(tasks):5} tasks, {dropped} duplicates dropped")
    out = ROOT / "results/summary/m2_manifest.json"
    out.write_text(json.dumps(manifest, indent=1) + "\n")


if __name__ == "__main__":
    main()
