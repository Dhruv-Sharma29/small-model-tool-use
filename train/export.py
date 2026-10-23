"""Adapter -> merged model -> GGUF Q4_K_M, the same path the base 7B took in project 01 (option a:
every condition is served by the same llama.cpp build at the same quantisation).

    train/.venv/bin/python -m train.export .run/adapters/pilot [--keep-f16]

1. Merge the LoRA adapter into the bf16 base (on CPU) and save it as safetensors.
2. Convert with 01's pinned llama.cpp converter (f16 GGUF), quantise to Q4_K_M with 01's
   llama-quantize, delete the intermediate files unless --keep-f16 (the f16 GGUF is needed to measure
   quantisation loss).
3. Write <adapter dir>/export.json: adapter, merged and GGUF SHA-256s, tool versions.
Re-running on the same adapter must reproduce the same GGUF hash (the reload check, M4).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
L01 = ROOT.parent / "01-llm-inference-lab"
sys.path.insert(0, str(ROOT))

from train.build_sft import BASE_MODEL, BASE_REVISION  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--keep-f16", action="store_true")
    args = ap.parse_args()
    name = args.run_dir.name
    merged = ROOT / f".run/merged/{name}"
    f16 = ROOT / f"models/qwen2.5-7b-lora-{name}-f16.gguf"
    q4 = ROOT / f"models/qwen2.5-7b-lora-{name}-Q4_K_M.gguf"
    (ROOT / "models").mkdir(exist_ok=True)

    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, revision=BASE_REVISION, local_files_only=True,
                                                dtype=torch.bfloat16, device_map={"": "cpu"})
    model = PeftModel.from_pretrained(base, args.run_dir / "adapter").merge_and_unload()
    model.save_pretrained(merged, safe_serialization=True)
    AutoTokenizer.from_pretrained(BASE_MODEL, revision=BASE_REVISION, local_files_only=True).save_pretrained(merged)
    del model, base

    env = {"PYTHONPATH": str(L01 / ".tools/llama.cpp/gguf-py"), "PATH": "/usr/bin:/bin"}
    subprocess.run([str(L01 / ".tools/convert-venv/bin/python"), str(L01 / ".tools/llama.cpp/convert_hf_to_gguf.py"),
                    str(merged), "--outtype", "f16", "--outfile", str(f16)], check=True, env=env,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run([str(L01 / ".tools/llama.cpp/build/bin/llama-quantize"), str(f16), str(q4), "Q4_K_M"], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    record = {"adapter_sha256": sha256(args.run_dir / "adapter/adapter_model.safetensors"),
              "merged_sha256": {p.name: sha256(p) for p in sorted(merged.glob("*.safetensors"))},
              "gguf_q4_k_m": str(q4.relative_to(ROOT)), "gguf_q4_k_m_sha256": sha256(q4),
              "base_model": BASE_MODEL, "base_revision": BASE_REVISION, "llama_cpp": "b11439 (project 01)"}
    if args.keep_f16:
        record["gguf_f16"], record["gguf_f16_sha256"] = str(f16.relative_to(ROOT)), sha256(f16)
    else:
        f16.unlink()
    shutil.rmtree(merged)
    (args.run_dir / "export.json").write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps(record))


if __name__ == "__main__":
    main()
