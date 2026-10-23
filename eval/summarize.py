"""Summarise evaluation runs (M3 onwards).

    python -m eval.summarize RUN.jsonl [RUN.jsonl ...] [--pair A B] [--out results/summary/m3]

For each (condition, set): success with a Wilson 95% interval, success by family and by expected
outcome, which scorer check failed, invalid calls, tool selection, steps, latency and tokens.
`--pair A B` adds a paired comparison of conditions A and B on identical tasks (exact McNemar).
Never filters episodes: a run that stopped on budget or transport errors counts as a failure.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(c - h, 3), round(c + h, 3))


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    return round(min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n), 4)


def load(path: Path) -> tuple[dict, list[dict]]:
    lines = path.read_text().splitlines()
    return json.loads(lines[0])["meta"], [json.loads(x) for x in lines[1:]]


def summarise(meta: dict, rs: list[dict]) -> dict:
    n, k = len(rs), sum(r["success"] for r in rs)
    group = lambda key: {g: f"{sum(r['success'] for r in rs if r[key] == g)}/{sum(r[key] == g for r in rs)}"  # noqa: E731
                         for g in sorted({r[key] for r in rs})}
    steps = [s for r in rs for s in r["steps_detail"]]
    calls = [s for s in steps if s.get("name")]
    lat = sorted(s["latency_s"] for s in steps if "latency_s" in s)
    return {
        "condition": meta["condition"], "set": meta["set"], "model_file": meta.get("model_file"), "prompt": meta["prompt"],
        "tasks": n, "success": k, "rate": round(k / n, 3) if n else None, "ci95": wilson(k, n),
        "by_family": group("family"), "by_expected_outcome": group("expected_outcome"),
        "failed_checks": dict(collections.Counter(c for r in rs for c in r["failed_checks"]).most_common()),
        "stop_reasons": dict(collections.Counter(r["stop_reason"] for r in rs)),
        "invalid_call_rate": round(sum(not s["valid"] for s in steps) / max(1, len(steps)), 3),
        "text_replies": sum(s.get("name") is None for s in steps),
        "transport_errors": sum(bool(s.get("transport_error")) for s in steps),
        "tool_selection_ok": round(sum(r["tool_selection_ok"] for r in rs) / n, 3) if n else None,
        "episodes_with_unallowed_effects": sum(bool(r["unallowed_effects"]) for r in rs),
        "mean_steps": round(sum(r["steps"] for r in rs) / n, 2) if n else None,
        "calls_never_reading": sum(r["read_calls"] == 0 for r in rs),
        "latency_s_p50": lat[len(lat) // 2] if lat else None, "latency_s_p95": lat[int(0.95 * (len(lat) - 1))] if lat else None,
        "completion_tokens": sum(s.get("usage", {}).get("completion_tokens", 0) for s in steps),
        "max_prompt_tokens": max((s.get("usage", {}).get("prompt_tokens", 0) for s in steps), default=0),
        "dropped_parallel_calls": sum(s.get("dropped_calls", 0) for s in calls),
    }


def pair(runs: dict, a: str, b: str) -> list[dict]:
    out = []
    for set_name in sorted({s for _, s in runs}):
        if (a, set_name) not in runs or (b, set_name) not in runs:
            continue
        ra = {r["task_id"]: r["success"] for r in runs[(a, set_name)]}
        rb = {r["task_id"]: r["success"] for r in runs[(b, set_name)]}
        common = sorted(set(ra) & set(rb))
        only_a = sum(ra[t] and not rb[t] for t in common)
        only_b = sum(rb[t] and not ra[t] for t in common)
        out.append({"set": set_name, "tasks": len(common), f"{a}": sum(ra[t] for t in common), f"{b}": sum(rb[t] for t in common),
                    f"only_{a}": only_a, f"only_{b}": only_b, "p_mcnemar": mcnemar_exact(only_a, only_b)})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--pair", nargs=2, action="append", default=[])
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    runs, summaries = {}, []
    for p in args.runs:
        meta, rs = load(p)
        runs[(meta["condition"], meta["set"])] = rs
        summaries.append({**summarise(meta, rs), "raw": str(p)})
    report = {"runs": summaries, "pairs": {f"{a} vs {b}": pair(runs, a, b) for a, b in args.pair}}
    lines = ["| Condition | Set | Success | 95% CI | Incidents | Requests | Invalid calls | Tool selection ok | Unallowed effects | Mean steps |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for s in summaries:
        lines.append(f"| {s['condition']} | {s['set']} | {s['success']}/{s['tasks']} ({s['rate']:.0%}) | {s['ci95'][0]:.2f}–{s['ci95'][1]:.2f} | "
                     f"{s['by_family'].get('incident', '-')} | {s['by_family'].get('request', '-')} | {s['invalid_call_rate']:.1%} | "
                     f"{s['tool_selection_ok']:.0%} | {s['episodes_with_unallowed_effects']} | {s['mean_steps']} |")
    for name, rows in report["pairs"].items():
        lines += ["", f"**Paired: {name}**", "", "| Set | Tasks | Only A | Only B | p (exact McNemar) |", "| --- | --- | --- | --- | --- |"]
        a, b = name.split(" vs ")
        lines += [f"| {r['set']} | {r['tasks']} | {r[f'only_{a}']} | {r[f'only_{b}']} | {r['p_mcnemar']} |" for r in rows]
    print("\n".join(lines))
    if args.out:
        args.out.with_suffix(".json").write_text(json.dumps(report, indent=1) + "\n")
        args.out.with_suffix(".md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
