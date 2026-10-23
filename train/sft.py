"""LoRA supervised fine-tuning on reference trajectories (PLAN.md M4 pilot, M5 full runs).

    train/.venv/bin/python -m train.sft --data .run/sft/pilot.jsonl --out .run/adapters/pilot --epochs 1

Plain PyTorch + PEFT, no trainer framework: bf16 base weights, LoRA on every attention and MLP
projection, gradient checkpointing, batch 1 with gradient accumulation (examples vary in length).
Loss only on assistant tokens (the spans build_sft.py stored). Every optimizer step logs loss, step
time and peak GPU memory to <out>/train_log.jsonl; <out>/run.json records config, data hash, versions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from train.build_sft import BASE_MODEL, BASE_REVISION

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def encode(tok, row: dict) -> tuple[list[int], list[int]]:
    enc = tok(row["text"], return_offsets_mapping=True, add_special_tokens=False)
    labels = []
    for tid, (a, b) in zip(enc["input_ids"], enc["offset_mapping"]):
        inside = any(s <= a and b <= e for s, e in row["assistant_spans"])
        labels.append(tid if inside else -100)
    return enc["input_ids"], labels


def main() -> None:
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=float, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=0, help="stop after this many optimizer steps (pilot)")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in args.data.read_text().splitlines()]
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, revision=BASE_REVISION, local_files_only=True)
    data = [encode(tok, r) for r in rows]
    n_target = sum(sum(t != -100 for t in lab) for _, lab in data)

    torch.cuda.reset_peak_memory_stats()
    t_load = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, revision=BASE_REVISION, local_files_only=True,
                                                 dtype=torch.bfloat16, device_map={"": 0})
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False
    lora = LoraConfig(r=args.rank, lora_alpha=args.alpha, lora_dropout=0.05, target_modules=TARGETS, task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    load_s = time.perf_counter() - t_load
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    mem_after_load = torch.cuda.max_memory_allocated() / 2**30

    total_steps = math.ceil(len(data) * args.epochs / args.accum)
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / args.warmup) *
                                              0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, total_steps)))))
    order = []
    while len(order) < total_steps * args.accum:
        epoch = list(range(len(data)))
        random.shuffle(epoch)
        order += epoch
    order = order[: total_steps * args.accum]

    run = {"started": datetime.now(timezone.utc).isoformat(timespec="seconds"), "args": {k: str(v) for k, v in vars(args).items()},
           "base_model": BASE_MODEL, "base_revision": BASE_REVISION, "lora_targets": TARGETS,
           "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(), "examples": len(data),
           "target_tokens": n_target, "trainable_params": trainable, "total_steps": total_steps,
           "load_s": round(load_s, 1), "peak_gib_after_load": round(mem_after_load, 2),
           "versions": {"torch": torch.__version__, "transformers": __import__("transformers").__version__,
                        "peft": __import__("peft").__version__}}
    print(json.dumps(run), flush=True)

    model.train()
    log = (args.out / "train_log.jsonl").open("w")
    t_step, loss_acc = time.perf_counter(), 0.0
    for i, idx in enumerate(order):
        ids, labels = data[idx]
        x = torch.tensor([ids], device="cuda")
        y = torch.tensor([labels], device="cuda")
        loss = model(input_ids=x, labels=y).loss / args.accum
        loss.backward()
        loss_acc += loss.item()
        if (i + 1) % args.accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step = (i + 1) // args.accum
            rec = {"step": step, "loss": round(loss_acc, 4), "lr": sched.get_last_lr()[0],
                   "step_s": round(time.perf_counter() - t_step, 2),
                   "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                   "reserved_gib": round(torch.cuda.memory_reserved() / 2**30, 2)}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(json.dumps(rec), flush=True)
            t_step, loss_acc = time.perf_counter(), 0.0
    model.save_pretrained(args.out / "adapter")
    run["finished"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    run["peak_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    (args.out / "run.json").write_text(json.dumps(run, indent=1) + "\n")
    print(json.dumps({"saved": str(args.out / "adapter"), "peak_gib": run["peak_gib"]}))


if __name__ == "__main__":
    main()
