"""Training examples from reference trajectories (PLAN.md M4/M5).

    train/.venv/bin/python -m train.build_sft --n 310 --out .run/sft/pilot.jsonl [--policy runbook] [--check-url http://127.0.0.1:21000]

Each train task's trajectory is executed through the real runner (so observations are
the sandbox's), with the base system prompt and the frozen tool set. The transcript is rendered with
the base model's own chat template, exactly as llama.cpp renders it at inference: llama.cpp parses
tool-call arguments into JSON objects before templating, so they are passed as objects here (as
strings, the template would double-encode them). `--check-url` compares a sample against the
server's /apply-template byte for byte.

Trajectory source (`--policy`): `gold` replays the generator's reference calls (the pilot's data).
They are the shortest path for the known fault, so the first call gives the answer away: a scale
request that needs no change starts with service_status and one that does starts with scale_service,
and a jobs_lagging alert starts with service_status only when a worker is down. `runbook` uses the
observation-only runbook (baselines/runbook.py), one decision tree for every task in a family, so
what it does next depends only on what the tools returned.

Only assistant turns are trained on: each example stores the character spans of the assistant
replies (tool call + <|im_end|>). Tasks are chosen balanced over kind-variants from data/train.jsonl
only; nothing from val or test is read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from executor.runner import ScriptedPolicy, run_episode  # noqa: E402
from executor.scorer import score  # noqa: E402
from baselines.runbook import RunbookPolicy  # noqa: E402
from tasks.tools import openai_tools  # noqa: E402

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
BASE_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"


def args_as_objects(messages: list[dict]) -> list[dict]:
    out = []
    for m in messages:
        if m.get("tool_calls"):
            m = {**m, "tool_calls": [{**c, "function": {**c["function"], "arguments": json.loads(c["function"]["arguments"])}}
                                     for c in m["tool_calls"]]}
        out.append(m)
    return out


def render(tok, messages: list[dict], generation_prompt: bool = False) -> str:
    return tok.apply_chat_template(messages, tools=openai_tools(), tokenize=False, add_generation_prompt=generation_prompt)


def example(tok, task: dict, policy: str = "gold") -> dict:
    ep = run_episode(task, ScriptedPolicy(task["gold"]) if policy == "gold" else RunbookPolicy(task))
    if not score(task, ep)["success"]:
        raise RuntimeError(f"{policy} trajectory fails for {task['task_id']}")
    msgs = args_as_objects(ep.messages)
    text = render(tok, msgs)
    spans = []
    for i, m in enumerate(msgs):
        if m["role"] == "assistant":
            start = len(render(tok, msgs[:i], generation_prompt=True))
            end = len(render(tok, msgs[:i + 1]))
            if not text.startswith(render(tok, msgs[:i + 1])):
                raise RuntimeError("template is not prefix-consistent")
            spans.append([start, end])
    return {"task_id": task["task_id"], "kind": task["kind"], "variant": task["variant"], "text": text,
            "assistant_spans": spans, "n_tokens": len(tok(text)["input_ids"]), "_messages": msgs}


def select(n: int, seed: int) -> list[dict]:
    tasks = [json.loads(line) for line in (ROOT / "data/train.jsonl").read_text().splitlines()]
    groups: dict[tuple, list] = {}
    for t in tasks:
        groups.setdefault((t["kind"], t["variant"]), []).append(t)
    rng = random.Random(seed)
    per = -(-n // len(groups))
    chosen = [t for key in sorted(groups) for t in rng.sample(groups[key], min(per, len(groups[key])))]
    rng.shuffle(chosen)
    return chosen[:n]


def main() -> None:
    from transformers import AutoTokenizer
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--policy", choices=["gold", "runbook"], default="gold")
    ap.add_argument("--check-url", default="")
    ap.add_argument("--check-n", type=int, default=20)
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, revision=BASE_REVISION, local_files_only=True)
    rows = [example(tok, t, args.policy) for t in select(args.n, args.seed)]
    checked = 0
    if args.check_url:
        import httpx
        for r in rows[: args.check_n]:
            history = r["_messages"][:-1]   # inference always renders history ending on a tool result
            want = httpx.post(f"{args.check_url}/apply-template", json={"messages": history, "tools": openai_tools()},
                              timeout=60).json()["prompt"]
            if render(tok, history, generation_prompt=True) != want:
                raise SystemExit(f"rendering differs from llama.cpp for {r['task_id']}")
            checked += 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(json.dumps({k: v for k, v in r.items() if not k.startswith("_")}) + "\n" for r in rows)
    args.out.write_text(body)
    toks = sorted(r["n_tokens"] for r in rows)
    manifest = {"out": str(args.out), "examples": len(rows), "seed": args.seed, "policy": args.policy, "base_model": BASE_MODEL,
                "base_revision": BASE_REVISION, "source": "data/train.jsonl", "sha256": hashlib.sha256(body.encode()).hexdigest(),
                "tokens": {"min": toks[0], "median": toks[len(toks) // 2], "max": toks[-1], "total": sum(toks)},
                "assistant_turns": sum(len(r["assistant_spans"]) for r in rows), "checked_against_llamacpp": checked}
    args.out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
