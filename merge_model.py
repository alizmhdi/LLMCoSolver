"""Merge a LoRA adapter checkpoint into a full HuggingFace model directory."""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer


def _resolve_local_base_snapshot(base_model: str) -> str | None:
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
    hub_dir = hf_home / "hub" / f"models--{base_model.replace('/', '--')}" / "snapshots"
    if not hub_dir.is_dir():
        return None
    snaps = sorted(hub_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    return str(snaps[0]) if snaps else None


def main() -> None:
    model_dir = os.environ.get("MODEL_DIR")
    output_dir = os.environ.get("OUTPUT_DIR")
    if not model_dir or not output_dir:
        raise SystemExit("Set MODEL_DIR and OUTPUT_DIR environment variables.")

    adapter_path = Path(model_dir)
    out_path = Path(output_dir)
    if not adapter_path.is_dir():
        raise SystemExit(f"Adapter checkpoint not found: {adapter_path}")

    adapter_config = json.loads((adapter_path / "adapter_config.json").read_text())
    base_model = adapter_config["base_model_name_or_path"]
    local_base = _resolve_local_base_snapshot(base_model)

    print(f"HF_HOME={os.environ.get('HF_HOME', '')}")
    print(f"Adapter base: {base_model}")
    if local_base:
        print(f"Using local base snapshot: {local_base}")

    print(f"Merging: {adapter_path}")
    print(f"Output:  {out_path}")

    model = AutoPeftModelForCausalLM.from_pretrained(
        str(adapter_path),
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    merged = model.merge_and_unload()

    out_path.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(out_path, safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_path))
    tokenizer.save_pretrained(out_path)

    print(f"Done: {out_path}")


if __name__ == "__main__":
    main()
