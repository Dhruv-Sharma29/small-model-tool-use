"""Overlap audit between train and every evaluation set (PLAN.md M2).

    python -m eval.m2_audit        # needs data/ from tasks.splits.build

1. Factor leakage (must be zero): held-out templates, entities or scenario combinations found in
   train.
2. Same example: normalised prompt (lowercase, digits -> 0) AND the same reference actions and
   outcome as some train task. Expected for test_seen (same templates and entities, by design).
3. Near-duplicate prompts: word 3-gram Jaccard of each test prompt with its closest train prompt.
   Prompt-lookup ceiling: answer each test task with the most common reference answer (actions with
   arguments + outcome) among train tasks with the identical prompt. What memorising prompt -> answer,
   without using any tool, could score.
4. Template frames: each held-out template, slots masked, against its closest train template.
Writes results/summary/m2_audit.json and m2_audit.md.
"""

from __future__ import annotations

import json
import re
import statistics
from pathlib import Path

from tasks.generator import vocab
from tasks.splits import load
from tasks.splits.build import DATA, ORDER
from tasks.tools import ACTION_TOOLS

ROOT = Path(__file__).resolve().parents[1]
NEAR = 0.8


def read(name: str) -> list[dict]:
    return [json.loads(line) for line in (DATA / f"{name}.jsonl").read_text().splitlines()]


def norm(text: str) -> str:
    return re.sub(r"\d+", "0", text.lower())


def grams(text: str, n: int = 3) -> frozenset:
    w = re.findall(r"[a-z0-9_.{}<>-]+", norm(text))
    return frozenset(tuple(w[i:i + n]) for i in range(max(1, len(w) - n + 1)))


def jaccard(a: frozenset, b: frozenset) -> float:
    return len(a & b) / len(a | b) if a or b else 1.0


def signature(task: dict) -> str:
    acts = [c for c in task["gold"] if c[0] in ACTION_TOOLS]
    return json.dumps([norm(task["prompt"]), acts, task["spec"]["outcome"]], sort_keys=True)


def answer(task: dict) -> str:
    return json.dumps([[c for c in task["gold"] if c[0] in ACTION_TOOLS], task["spec"]["outcome"]], sort_keys=True)


def entity_hits(held: dict, task: dict) -> list[str]:
    """Entity values from `held` present in a task. Impossible requests are matched in the prompt
    only, where they are used (a reference message saying "set api.db_port to 6432" is not one)."""
    text, hits = json.dumps(task, sort_keys=True), []
    for pool, items in held.items():
        for item in items:
            if pool == "release_series":
                found = f'"version": "{item}.' in text
            elif pool == "unavailable_kinds":
                found = any(ask.lower() in task["prompt"].lower() for ask in vocab.UNAVAILABLE_ASKS[item])
            elif pool == "bad_db_changes":
                key, value = item.split("=", 1)
                found = f'"{key}": {json.dumps(int(value) if key == "api.db_port" else value)}' in text
            else:
                found = re.search(rf"(?<![\w-]){re.escape(item)}(?![\w-])", text) is not None
            if found:
                hits.append(f"{pool}:{item}")
    return hits


def main() -> None:
    splits = load()
    sets = {n: read(n) for n in ORDER}
    train = sets["train"]
    held_entities = {p: v["held_out"] for p, v in splits["entities"].items()}
    held_templates = set(splits["templates"]["held_out"])
    held_scen = {f"{k}:{d}" for k, d in splits["scenarios"]["held_out_distractor"].items()}

    leakage = {"templates": sorted({t["template_id"] for t in train} & held_templates),
               "scenarios": sorted({t["scenario"] for t in train} & held_scen),
               "entities": sorted({h for t in train for h in entity_hits(held_entities, t)})}

    train_sigs = {signature(t) for t in train}
    by_prompt: dict[str, dict[str, int]] = {}
    for t in train:
        counts = by_prompt.setdefault(t["prompt"], {})
        counts[answer(t)] = counts.get(answer(t), 0) + 1
    lookup = {p: max(c, key=c.get) for p, c in by_prompt.items()}
    train_grams = list({grams(t["prompt"]) for t in train})
    report = {"splits_sha": None, "leakage": leakage, "sets": {}}
    for name in ORDER[1:]:
        ts = sets[name]
        sims = [max(jaccard(grams(t["prompt"]), g) for g in train_grams) for t in ts]
        held_used = {
            "templates": sum(t["template_id"] in held_templates for t in ts),
            "scenarios": sum(t["scenario"] in held_scen for t in ts),
            "tasks_with_held_out_entities": sum(bool(entity_hits(held_entities, t)) for t in ts)}
        report["sets"][name] = {
            "tasks": len(ts), "held_out_factors": splits["sets"][name]["held_out"], "uses_held_out": held_used,
            "same_example_as_train": sum(signature(t) in train_sigs for t in ts),
            "prompt_lookup_ceiling": round(sum(lookup.get(t["prompt"]) == answer(t) for t in ts) / len(ts), 3),
            "prompt_similarity": {"median": round(statistics.median(sims), 3), "p95": round(sorted(sims)[int(0.95 * (len(sims) - 1))], 3),
                                  "max": round(max(sims), 3), f"share_ge_{NEAR}": round(sum(s >= NEAR for s in sims) / len(sims), 3)}}

    masked = {tid: re.sub(r"\{[^}]+\}", "<slot>", vocab.template_text(tid)) for tid in vocab.TEMPLATES
              for tid in vocab.template_ids(tid)}
    train_t = [t for t in masked if t not in held_templates]
    frames = []
    for tid in sorted(held_templates):
        best = max(train_t, key=lambda o: jaccard(grams(masked[tid]), grams(masked[o])))
        frames.append({"held_out": tid, "closest_train": best, "jaccard": round(jaccard(grams(masked[tid]), grams(masked[best])), 3),
                       "text": masked[tid], "closest_text": masked[best]})
    frames.sort(key=lambda f: -f["jaccard"])
    report["template_frames"] = {"max": frames[0]["jaccard"], "median": statistics.median(f["jaccard"] for f in frames),
                                 "top": frames[:5]}
    from tasks.splits import splits_sha
    report["splits_sha"] = splits_sha()

    out = ROOT / "results/summary"
    (out / "m2_audit.json").write_text(json.dumps(report, indent=1) + "\n")
    lines = ["# M2: overlap audit", "", "Generated by `python -m eval.m2_audit` from `data/` (`python -m tasks.splits.build`).", "",
             f"**Factor leakage into train:** templates {len(leakage['templates'])}, scenarios {len(leakage['scenarios'])}, "
             f"entities {len(leakage['entities'])}.", "",
             "| Set | Held out | Same example as a train task | Prompt-lookup ceiling | Prompt similarity to closest train prompt: median / p95 / max | Share ≥ 0.8 |",
             "| --- | --- | --- | --- | --- | --- |"]
    for name, r in report["sets"].items():
        p = r["prompt_similarity"]
        lines.append(f"| {name} | {', '.join(r['held_out_factors']) or 'none'} | {r['same_example_as_train']}/{r['tasks']} | {r['prompt_lookup_ceiling']:.0%} | "
                     f"{p['median']} / {p['p95']} / {p['max']} | {p[f'share_ge_{NEAR}']:.0%} |")
    lines += ["", f"**Held-out templates vs closest train template** (slots masked, word 3-grams): max {report['template_frames']['max']}, "
              f"median {report['template_frames']['median']}.", "", "| Held out | Closest train | Jaccard |", "| --- | --- | --- |"]
    lines += [f"| `{f['text']}` | `{f['closest_text']}` | {f['jaccard']} |" for f in frames[:5]]
    (out / "m2_audit.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
