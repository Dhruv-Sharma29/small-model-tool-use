"""General-capability regression check (PLAN.md M5): does tuning for tool use cost ordinary skills?

    python -m eval.general_capability --url http://127.0.0.1:21004 --condition lora-runbook-1240-7b

The set is project 01's held-out benchmark prompts (60: reasoning, JSON extraction and plain-text
tool-call formatting, `../01-llm-inference-lab/benchmarks/prompts/heldout.jsonl`), scored with 01's
own deterministic validators, so nothing here is written for this check. Each prompt is sent as an
ordinary chat with no tools attached, temperature 0. Also counted: replies that are a tool call
although no tool was offered, a likely way for a tool-use fine-tune to regress.
Writes results/summary/general-<condition>.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
L01 = ROOT.parent / "01-llm-inference-lab"
sys.path.insert(0, str(L01))
from benchmarks.validators.validators import validate  # noqa: E402

PROMPTS = L01 / "benchmarks/prompts/heldout.jsonl"


def ask(client: httpx.Client, prompt: dict, max_tokens: int) -> dict:
    r = client.post("/v1/chat/completions", json={"messages": prompt["messages"], "temperature": 0, "seed": 0,
                                                  "max_tokens": max_tokens}).json()
    msg = r["choices"][0]["message"]
    text = msg.get("content") or ""
    passed, reason = validate(prompt["validator"], text)
    return {"id": prompt["id"], "category": prompt["category"], "passed": passed, "reason": reason,
            "tool_call_reply": bool(msg.get("tool_calls")) or "<tool_call>" in text, "finish_reason": r["choices"][0].get("finish_reason")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", required=True)
    ap.add_argument("--condition", required=True)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    prompts = [json.loads(line) for line in PROMPTS.read_text().splitlines()]
    with httpx.Client(base_url=args.url, timeout=600) as client, ThreadPoolExecutor(args.workers) as pool:
        rows = list(pool.map(lambda p: ask(client, p, args.max_tokens), prompts))
    by_cat = defaultdict(lambda: [0, 0])
    for r in rows:
        by_cat[r["category"]][0] += r["passed"]
        by_cat[r["category"]][1] += 1
    out = {"condition": args.condition, "set": str(PROMPTS.relative_to(ROOT.parent)), "prompts": len(rows),
           "passed": sum(r["passed"] for r in rows), "by_category": {k: f"{a}/{b}" for k, (a, b) in sorted(by_cat.items())},
           "tool_call_replies": sum(r["tool_call_reply"] for r in rows),
           "failure_reasons": dict(Counter(r["reason"] for r in rows if not r["passed"]).most_common(8)), "rows": rows}
    (ROOT / f"results/summary/general-{args.condition}.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))


if __name__ == "__main__":
    main()
