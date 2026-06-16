#!/bin/bash
set -euo pipefail

# Merge the latest RL LoRA checkpoint into a full model on scratch.
#
# Usage:
#   bash cmd.sh
#   MODEL_DIR=/anvil/scratch/.../checkpoint-650 bash cmd.sh
#
# Optional:
#   OUTPUT_DIR=/anvil/scratch/x-malizadeh/MetaRL/models/merged/my_run bash cmd.sh

METARL_ROOT="/anvil/scratch/x-malizadeh/MetaRL"

if [ -z "${MODEL_DIR:-}" ]; then
  MODEL_DIR="$(ls -td "$METARL_ROOT"/runs/rl/*/checkpoint-* 2>/dev/null | head -1 || true)"
fi

OUTPUT_DIR="${OUTPUT_DIR:-$METARL_ROOT/models/merged/$(basename "$MODEL_DIR")}"

if [ -z "$MODEL_DIR" ] || [ ! -d "$MODEL_DIR" ]; then
  echo "ERROR: RL checkpoint not found (MODEL_DIR=${MODEL_DIR:-unset})" >&2
  exit 1
fi

echo "Latest RL checkpoint: $MODEL_DIR"
echo "Merged output:        $OUTPUT_DIR"

export MODEL_DIR OUTPUT_DIR
bash "$METARL_ROOT/repo/LLMCO_cmd.sh"
