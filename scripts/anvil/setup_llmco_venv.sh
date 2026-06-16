#!/bin/bash -l
# Create LLMCO Python env on scratch (avoids home disk quota).
#
# Run once on a login or compute node:
#   cd src/problems/cluster_scheduling/solvers/LLMCO
#   bash scripts/anvil/setup_llmco_venv.sh

set -euo pipefail

LLMCO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV="${LLMCO_VENV:-/anvil/scratch/x-malizadeh/MetaRL/envs/llmco-py312}"
PYTHON="${PYTHON:-/anvil/projects/x-cis260760/local/python-3.12/bin/python3}"

export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/anvil/scratch/x-malizadeh/MetaRL/cache/pip}"
export TMPDIR="${TMPDIR:-/anvil/scratch/x-malizadeh/MetaRL/tmp}"
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR" "$(dirname "$VENV")"

if [ ! -x "$PYTHON" ]; then
  echo "ERROR: Python not found: $PYTHON" >&2
  exit 1
fi

if [ ! -x "$VENV/bin/python" ]; then
  echo "Creating venv at $VENV"
  "$PYTHON" -m venv "$VENV"
fi

source "$VENV/bin/activate"
pip install --upgrade pip
pip install -r "$LLMCO/requirements.txt"

python - <<'PY'
from pathlib import Path
import huggingface_hub
import torch
root = Path(huggingface_hub.__file__).resolve().parent
tpl = root / "templates" / "modelcard_template.md"
print("venv:", __import__("sys").prefix)
print("huggingface_hub template:", tpl.is_file())
print("torch:", torch.__version__)
PY

echo "LLMCO venv ready: $VENV"
