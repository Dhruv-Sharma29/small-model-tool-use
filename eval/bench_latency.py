"""Controlled latency and memory benchmark (PLAN.md M5): one model loaded at a time, one request at a time.

    python -m eval.bench_latency --gguf models/qwen2.5-14b-instruct-cf98f3b-Q4_K_M.gguf --alias base-14b --port 21001
    python -m eval.bench_latency --gguf models/qwen2.5-7b-lora-runbook-1240-Q4_K_M.gguf --alias lora-1240-7b --port 21004

Accuracy runs share the GPU and run 4 requests in flight; their timings say little. This serves one
model with the same flags as every accuracy run (env/models.sh), runs the first --tasks val tasks
with a single worker, and measures:
  - per model call and per episode: wall latency (from the episode records, eval.run_model);
  - prefill and decode speed: llama.cpp's own per-request timings from the server log;
  - peak GPU memory of the server process (nvidia-smi, sampled every second).
It refuses to start while another process is on the GPU unless --allow-shared, and records what it
found either way. Writes results/summary/latency-<alias>.json; the episode records go to results/raw/.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TIMING = re.compile(r"(prompt eval|\beval) time =\s*([\d.]+) ms /\s*(\d+) tokens")


def gpu_apps() -> list[dict]:
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout
    apps = []
    for line in out.strip().splitlines():
        pid, mem = (x.strip() for x in line.split(","))
        cmd = Path(f"/proc/{pid}/cmdline")
        name = cmd.read_bytes().replace(b"\0", b" ").decode()[:120] if cmd.exists() else "?"
        apps.append({"pid": int(pid), "used_mib": int(mem) if mem.isdigit() else None, "cmd": name})
    return apps


def server_pid(port: int) -> int:
    return int((ROOT / f".run/llama-{port}.pid").read_text())


def sample_memory(pid: int, stop: threading.Event, peak: list) -> None:
    while not stop.is_set():
        for a in gpu_apps():
            if a["pid"] == pid and a["used_mib"]:
                peak[0] = max(peak[0], a["used_mib"])
        stop.wait(1)


def pct(values: list[float], q: float) -> float:
    values = sorted(values)
    return round(values[min(len(values) - 1, int(q * len(values)))], 3) if values else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--alias", required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--tasks", type=int, default=40)
    ap.add_argument("--prompt", default="base")
    ap.add_argument("--allow-shared", action="store_true", help="run even if other processes are on the GPU")
    args = ap.parse_args()

    others = gpu_apps()
    for _ in range(60):   # a server stopped just before may take a few seconds to leave the GPU
        if not others or args.allow_shared:
            break
        time.sleep(1)
        others = gpu_apps()
    if others and not args.allow_shared:
        raise SystemExit(f"GPU not quiet, refusing (use --allow-shared to record anyway): {others}")
    log = ROOT / f".run/logs/llama-{args.port}.log"
    subprocess.run(["env/models.sh", "serve", args.gguf, str(args.port), args.alias], cwd=ROOT, check=True)
    log_start = log.stat().st_size
    pid, peak, stop = server_pid(args.port), [0], threading.Event()
    sampler = threading.Thread(target=sample_memory, args=(pid, stop, peak), daemon=True)
    sampler.start()
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    try:
        run = subprocess.run([str(ROOT / ".venv/bin/python"), "-m", "eval.run_model", "--url", f"http://127.0.0.1:{args.port}",
                              "--condition", f"latency-{args.alias}", "--prompt", args.prompt, "--sets", "val",
                              "--limit", str(args.tasks), "--workers", "1"], cwd=ROOT, capture_output=True, text=True, check=True)
    finally:
        stop.set()
        sampler.join()
        subprocess.run(["env/models.sh", "stop", str(args.port)], cwd=ROOT, check=False)
        for _ in range(60):   # leave the GPU as we found it before the next condition starts
            if not any(a["pid"] == pid for a in gpu_apps()):
                break
            time.sleep(1)
    summary_line = json.loads(run.stdout.strip().splitlines()[-1])
    rows = [json.loads(line) for line in (ROOT / summary_line["raw"]).read_text().splitlines()][1:]
    steps = [s for r in rows for s in r["steps_detail"]]
    with log.open() as f:
        f.seek(log_start)
        timings = TIMING.findall(f.read())
    prefill = [int(n) / (float(ms) / 1000) for kind, ms, n in timings if kind == "prompt eval" and float(ms) > 0]
    decode = [int(n) / (float(ms) / 1000) for kind, ms, n in timings if kind == "eval" and float(ms) > 0]
    result = {
        "alias": args.alias, "gguf": args.gguf, "started": started, "tasks": len(rows),
        "gpu_others_at_start": others, "shared_gpu": bool(others),
        "success": sum(r["success"] for r in rows), "raw": summary_line["raw"],
        "call_latency_s": {"p50": pct([s["latency_s"] for s in steps], 0.5), "p95": pct([s["latency_s"] for s in steps], 0.95),
                           "calls": len(steps)},
        "episode_wall_s": {"p50": pct([r["wall_s"] for r in rows], 0.5), "p95": pct([r["wall_s"] for r in rows], 0.95)},
        "prompt_tokens_mean": round(statistics.mean(s["usage"]["prompt_tokens"] for s in steps if s.get("usage")), 1),
        "completion_tokens_mean": round(statistics.mean(s["usage"]["completion_tokens"] for s in steps if s.get("usage")), 1),
        "prefill_tokens_per_s_median": round(statistics.median(prefill), 1) if prefill else None,
        "decode_tokens_per_s_median": round(statistics.median(decode), 1) if decode else None,
        "server_gpu_peak_mib": peak[0] or None,
    }
    out = ROOT / f"results/summary/latency-{args.alias}.json"
    out.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "gpu_others_at_start"}, indent=1))


if __name__ == "__main__":
    main()
