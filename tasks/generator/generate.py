"""Deterministic task generation: (kind, variant, distractor, template, seed, entity pools) -> task.

The same inputs always give the same task, byte for byte. Splits (M2) restrict the *inputs*:
which templates, distractor/scenario combinations and entity pool items each split may use. They
are fixed before any task exists.

    python -m tasks.generator.generate --catalog out.jsonl   # one task per scenario, for checking
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from tasks.generator import vocab
from tasks.generator.scenarios import DISTRACTORS, KINDS, Builder, alert_slots


def scenario_id(kind: str, variant: str, distractor: str) -> str:
    return f"{kind}:{variant}:{distractor}"


def all_scenarios() -> list[tuple[str, str, str]]:
    return [(k.name, v, d) for k in KINDS.values() for v in k.variants for d in DISTRACTORS]


def generate(kind: str, variant: str, distractor: str, template_id: str, seed: int, pools: dict | None = None) -> dict:
    pools = pools or vocab.POOLS
    k = KINDS[kind]
    if variant not in k.variants:
        raise ValueError(f"{kind} has no variant {variant!r}")
    if template_id.rsplit(".", 1)[0] not in k.template_families:
        raise ValueError(f"template {template_id} is not used by {kind}")
    rng = random.Random(f"{kind}|{variant}|{distractor}|{template_id}|{seed}")
    b = Builder(rng, pools, k.replicas(variant, rng))
    DISTRACTORS[distractor](b)
    spec, gold, request_slots = k.build(b, variant)
    world = b.finalize()
    slots = {**alert_slots(world, rng), "requester": rng.choice(pools["requesters"]), "ticket": rng.randint(1000, 9999),
             **request_slots}
    prompt = vocab.template_text(template_id).format(**slots)
    task = {"kind": kind, "family": k.family, "variant": variant, "distractor": distractor,
            "scenario": scenario_id(kind, variant, distractor), "template_id": template_id, "seed": seed,
            "prompt": prompt, "request_slots": request_slots,   # what the request text says (instance, n, author, ...)
            "world": world.to_dict(), "spec": spec, "gold": [list(c) for c in gold]}
    task["task_id"] = hashlib.sha256(json.dumps(task, sort_keys=True).encode()).hexdigest()[:16]
    return task


def catalog(seed: int = 0) -> list[dict]:
    """One task per scenario (kind x variant x distractor), template chosen by seed."""
    out = []
    for kind, variant, distractor in all_scenarios():
        rng = random.Random(f"catalog|{kind}|{variant}|{distractor}|{seed}")
        family = rng.choice(KINDS[kind].template_families)
        out.append(generate(kind, variant, distractor, rng.choice(vocab.template_ids(family)), seed))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    tasks = catalog(args.seed)
    args.catalog.parent.mkdir(parents=True, exist_ok=True)
    args.catalog.write_text("".join(json.dumps(t) + "\n" for t in tasks))
    print(f"{len(tasks)} tasks -> {args.catalog}")


if __name__ == "__main__":
    main()
