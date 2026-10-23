"""Write the split definitions ONCE, before any example is generated (PLAN.md M2).

    python -m tasks.splits.make_splits          # refuses to overwrite tasks/splits/splits.json

Three factors are held out, each chosen at random with a recorded seed (no hand-picking):

- templates: 2 of the 8 phrasing *frames*, the same positions in every family. Templates are
  parallel across families (position 0 is "Please ...", 3 is a ticket, ...), so holding out a
  frame in one family while training on it in another would leak the phrasing.
- entities: about 30% of every entity pool (authors, requesters, notes, release series, error
  texts, bad DB values, kinds of impossible request).
- scenarios: for each kind-variant, one of the six distractors. The held-out combinations are
  unseen, while each component appears in training.

Sets (`build.py` generates them): train and val use only training factors; `test_seen` too (new
seeds), which measures fit; each `test_<factor>` holds out one factor; `test_unseen` holds out all
three at once and is the primary held-out result.
"""

from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from pathlib import Path

from tasks.generator import vocab
from tasks.generator.generate import DISTRACTORS, KINDS

OUT = Path(__file__).with_name("splits.json")
SEED = 20261009
FRAMES = 8
HELD_OUT_FRAMES = 2
ENTITY_SHARE = 0.3

SETS = {   # name: (held-out factors, tasks per kind-variant, seed offset)
    "train": ((), 100, 0),
    "val": ((), 5, 1_000_000),
    "test_seen": ((), 5, 2_000_000),
    "test_template": (("template",), 5, 3_000_000),
    "test_entity": (("entity",), 5, 4_000_000),
    "test_scenario": (("scenario",), 5, 5_000_000),
    "test_unseen": (("template", "entity", "scenario"), 5, 6_000_000),
}


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"{OUT} exists; splits are frozen. Delete it deliberately to redefine them.")
    rng = random.Random(SEED)
    assert all(len(t) == FRAMES for t in vocab.TEMPLATES.values())
    held_frames = sorted(rng.sample(range(FRAMES), HELD_OUT_FRAMES))
    templates = {"held_out_frames": held_frames,
                 "held_out": sorted(f"{fam}.{i}" for fam in vocab.TEMPLATES for i in held_frames),
                 "train": sorted(f"{fam}.{i}" for fam in vocab.TEMPLATES for i in range(FRAMES) if i not in held_frames)}
    entities = {}
    for pool, items in sorted(vocab.POOLS.items()):
        if pool == "bad_db_changes":   # hold out values per key, so every key stays in training
            by_key: dict[str, list[str]] = {}
            for item in items:
                by_key.setdefault(item.split("=")[0], []).append(item)
            held = sorted(x for vs in by_key.values() for x in rng.sample(vs, max(1, round(ENTITY_SHARE * len(vs)))))
        else:
            held = sorted(rng.sample(items, max(1, round(ENTITY_SHARE * len(items)))))
        entities[pool] = {"held_out": held, "train": [x for x in items if x not in held]}
    scenarios = {}
    names = sorted(DISTRACTORS)
    for kind in sorted(KINDS):
        for variant in KINDS[kind].variants:
            scenarios[f"{kind}:{variant}"] = rng.choice(names)
    splits = {"created": datetime.now(timezone.utc).isoformat(timespec="seconds"), "seed": SEED,
              "note": "Frozen before any example was generated. Do not edit; see make_splits.py.",
              "templates": templates, "entities": entities,
              "scenarios": {"held_out_distractor": scenarios},
              "sets": {n: {"held_out": list(f), "per_kind_variant": k, "seed_offset": o} for n, (f, k, o) in SETS.items()}}
    OUT.write_text(json.dumps(splits, indent=1) + "\n")
    print(f"held-out frames {held_frames}; wrote {OUT}")


if __name__ == "__main__":
    main()
