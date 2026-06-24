#!/bin/bash -l
# GRPO RL training for CS (optimality + feasibility rewards), 1 epoch.
#
# Submit:
#   cd src/problems/cluster_scheduling/solvers/LLMCO
#   sbatch scripts/anvil/cs_rl.slurm
#
# Override SFT checkpoint, output dir, or disable num_nodes filter for j15-25 data:
#   sbatch --export=ALL,SFT_OUTPUT_DIR=/anvil/scratch/.../cs_500k_j15-25 scripts/anvil/cs_rl.slurm
#   sbatch --export=ALL,SFT_CHECKPOINT=/anvil/scratch/.../checkpoint-31250 scripts/anvil/cs_rl.slurm
#   sbatch --export=ALL,NUM_NODES= scripts/anvil/cs_rl.slurm
#
# Expects:
#   data_rl/cs/train/train_rl.json
#   data_rl/cs/eval/test.json

#SBATCH -A cis260760-ai
#SBATCH -p ai
#SBATCH --job-name=llmco-cs-rl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH -o /anvil/scratch/x-malizadeh/MetaRL/logs/cs-rl-%j.out

set -euo pipefail

LLMCO="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$LLMCO"

VENV="${LLMCO_VENV:-/anvil/scratch/x-malizadeh/MetaRL/envs/llmco-py312}"
if [ ! -x "$VENV/bin/python" ]; then
  echo "ERROR: LLMCO venv not found: $VENV" >&2
  echo "Create it: bash scripts/anvil/setup_llmco_venv.sh" >&2
  exit 1
fi
source "$VENV/bin/activate"

export HF_HOME="${HF_HOME:-/anvil/scratch/x-malizadeh/MetaRL/cache/huggingface}"
export TMPDIR="${TMPDIR:-/anvil/scratch/x-malizadeh/MetaRL/tmp}"
mkdir -p "$HF_HOME" "$TMPDIR" /anvil/scratch/x-malizadeh/MetaRL/logs

# --- SFT checkpoint (defaults to latest under SFT_OUTPUT_DIR) ---
SFT_OUTPUT_DIR="${SFT_OUTPUT_DIR:-/anvil/scratch/x-malizadeh/MetaRL/runs/sft/cs_500k_j15-25}"
if [ -z "${SFT_CHECKPOINT:-}" ]; then
  SFT_CHECKPOINT="$(ls -d "$SFT_OUTPUT_DIR"/checkpoint-* 2>/dev/null | sort -V | tail -1 || true)"
fi
NUM_EPOCHS="${NUM_EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
NUM_GENERATIONS="${NUM_GENERATIONS:-8}"
SAVE_STEPS="${SAVE_STEPS:-50}"
NUM_NODES="${NUM_NODES:-}"
SEED="${SEED:-42}"

OUTPUT_DIR="${OUTPUT_DIR:-/anvil/scratch/x-malizadeh/MetaRL/runs/rl/cs_rl_ep${NUM_EPOCHS}_job${SLURM_JOB_ID}}"

echo "HOST=$(hostname) JOB_ID=$SLURM_JOB_ID START=$(date -Is)"
echo "SFT_CHECKPOINT=$SFT_CHECKPOINT"
echo "NUM_EPOCHS=$NUM_EPOCHS REWARDS=optimality+feasibility OUTPUT_DIR=$OUTPUT_DIR"

python - <<'PY'
import json, sys
from pathlib import Path
train = Path("data_rl/cs/train/train_rl.json")
eval_ = Path("data_rl/cs/eval/test.json")
for p in (train, eval_):
    if not p.is_file():
        sys.exit(f"Missing {p}")
with train.open() as f:
    n_train = len(json.load(f))
with eval_.open() as f:
    n_eval = len(json.load(f))
print(f"RL train: {n_train:,}  RL eval: {n_eval:,}")
PY

if [ -z "${SFT_CHECKPOINT:-}" ] || [ ! -d "$SFT_CHECKPOINT" ]; then
  echo "ERROR: SFT checkpoint not found under: $SFT_OUTPUT_DIR" >&2
  echo "Set SFT_CHECKPOINT or SFT_OUTPUT_DIR explicitly." >&2
  exit 1
fi

RL_ARGS=(
  --problem cs
  --model_name "$SFT_CHECKPOINT"
  --num_train_epochs "$NUM_EPOCHS"
  --per_device_train_batch_size "$BATCH_SIZE"
  --gradient_accumulation_steps "$GRAD_ACCUM"
  --learning_rate "$LEARNING_RATE"
  --num_generations "$NUM_GENERATIONS"
  --save_step "$SAVE_STEPS"
  --seed "$SEED"
  --report_to none
  --output_dir "$OUTPUT_DIR"
)
if [ -n "${NUM_NODES:-}" ]; then
  RL_ARGS+=(--num_nodes "$NUM_NODES")
fi

python rl_train.py "${RL_ARGS[@]}"

echo "RL complete $(date -Is)"
echo "Checkpoints: $OUTPUT_DIR"
ls -la "$OUTPUT_DIR" | head -20
