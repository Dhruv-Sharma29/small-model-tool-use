"""Run one condition (model x system prompt) on evaluation sets (PLAN.md M3).

    python -m eval.run_model --url http://127.0.0.1:21000 --condition base-7b --prompt base \\
        --sets val test_unseen [--limit 20] [--workers 4]

Same loop, tools, budget and scorer for every condition. Episodes run in parallel (one per server
slot); each episode's calls are sequential. Every episode is written to
results/raw/<stamp>-<condition>-<set>.jsonl: score, every step (arguments, validity, observation
size, latency, tokens) and the transcript. The first line holds the run's metadata: model file, prompt
hash, splits hash, budget, sampling settings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import httpx

from baselines.llm import MAX_TOKENS, LLMPolicy
from baselines.prompts import PROMPTS
from executor.runner import Budget, run_episode
from executor.scorer import score
from tasks.splits import splits_sha
from tasks.splits.build import DATA

ROOT = Path(__file__).resolve().parents[1]


def server_meta(url: str) -> dict:
    try:
        props = httpx.get(f"{url}/props", timeout=10).json()
        return {"model_file": props.get("model_path"), "n_ctx_per_slot": props.get("default_generation_settings", {}).get("n_ctx"),
                "slots": props.get("total_slots")}
    except (httpx.HTTPError, ValueError) as e:
        return {"model_file": None, "error": str(e)[:200]}


def run_set(name: str, args, meta: dict) -> dict:
    tasks = [json.loads(line) for line in (DATA / f"{name}.jsonl").read_text().splitlines()][: args.limit or None]
    system = PROMPTS[args.prompt]
    budget = Budget()
    client = httpx.Client(base_url=args.url, timeout=600, limits=httpx.Limits(max_connections=args.workers * 2))

    def one(task):
        t0 = time.perf_counter()
        ep = run_episode(task, LLMPolicy(args.url, args.model, client), budget, system_prompt=system)
        r = score(task, ep)
        steps = [{k: v for k, v in s.items() if k != "observation"} | {"obs_chars": len(json.dumps(s.get("observation", "")))}
                 for s in ep.steps]
        return {**r, "set": name, "kind": task["kind"], "variant": task["variant"], "family": task["family"],
                "scenario": task["scenario"], "template_id": task["template_id"], "wall_s": round(time.perf_counter() - t0, 2),
                "steps_detail": steps, "messages": ep.messages}

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = ROOT / f"results/raw/{stamp}-{args.condition}-{name}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    t0, done, ok = time.monotonic(), 0, 0
    with path.open("w") as f, ThreadPoolExecutor(args.workers) as pool:
        f.write(json.dumps({"meta": {**meta, "set": name, "tasks": len(tasks)}}) + "\n")
        for rec in pool.map(one, tasks):
            f.write(json.dumps(rec, default=str) + "\n")
            f.flush()
            done, ok = done + 1, ok + rec["success"]
            if done % 20 == 0 or done == len(tasks):
                print(f"  {name}: {done}/{len(tasks)}  success {ok}/{done}  {time.monotonic() - t0:.0f} s", flush=True)
    return {"set": name, "raw": str(path.relative_to(ROOT)), "tasks": len(tasks), "success": ok,
            "wall_s": round(time.monotonic() - t0, 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", default="local")
    ap.add_argument("--condition", required=True, help="name used in result files, e.g. base-7b")
    ap.add_argument("--prompt", choices=sorted(PROMPTS), default="base")
    ap.add_argument("--sets", nargs="+", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    meta = {"condition": args.condition, "url": args.url, **server_meta(args.url), "prompt": args.prompt,
            "prompt_sha": hashlib.sha256(PROMPTS[args.prompt].encode()).hexdigest()[:12], "splits_sha": splits_sha(),
            "budget": {"max_steps": Budget().max_steps, "max_retries": Budget().max_retries},
            "sampling": {"temperature": 0, "seed": 0, "max_tokens": MAX_TOKENS, "tool_choice": "auto"},
            "started": datetime.now(timezone.utc).isoformat(timespec="seconds"), "limit": args.limit}
    print(json.dumps(meta))
    for name in args.sets:
        print(json.dumps(run_set(name, args, meta)))


if __name__ == "__main__":
    main()
