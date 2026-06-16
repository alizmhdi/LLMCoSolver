#!/bin/bash -l
# SFT training for CS from scratch (base Qwen2.5-7B + LoRA) on generated data.
#
# Submit from login node:
#   cd src/problems/cluster_scheduling/solvers/LLMCO
#   sbatch scripts/anvil/cs_sft.slurm
#
# Override hyperparameters:
#   sbatch --export=ALL,NUM_EPOCHS=3,BATCH_SIZE=4,OUTPUT_DIR=$SCRATCH/MetaRL/runs/my_sft scripts/anvil/cs_sft.slurm
#
# Expects:
#   data/cs/train/train_cs.json   (100k SFT train)
#   data/cs/eval/test.json        (10k SFT eval; test.raw.json also fine)

#SBATCH -A cis260760-ai
#SBATCH -p ai
#SBATCH --job-name=llmco-cs-sft
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH -o /anvil/scratch/x-malizadeh/MetaRL/logs/cs-sft-%j.out

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

# --- training config (override via sbatch --export=ALL,VAR=value) ---
BASE_MODEL="${BASE_MODEL:-unsloth/Qwen2.5-7B}"
NUM_EPOCHS="${NUM_EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
LORA_R="${LORA_R:-64}"
LORA_ALPHA="${LORA_ALPHA:-64}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-20000}"
SAVE_STEPS="${SAVE_STEPS:-5000}"
NUM_NODES="${NUM_NODES:-20}"
SEED="${SEED:-42}"

OUTPUT_DIR="${OUTPUT_DIR:-/anvil/scratch/x-malizadeh/MetaRL/runs/sft/cs_ep${NUM_EPOCHS}_n${NUM_NODES}_job${SLURM_JOB_ID}}"

echo "HOST=$(hostname) JOB_ID=$SLURM_JOB_ID START=$(date -Is)"
echo "BASE_MODEL=$BASE_MODEL NUM_EPOCHS=$NUM_EPOCHS BATCH_SIZE=$BATCH_SIZE"
echo "OUTPUT_DIR=$OUTPUT_DIR"

# Sanity-check training data
python - <<'PY'
import json, sys
from pathlib import Path
train = Path("data/cs/train/train_cs.json")
eval_ = Path("data/cs/eval/test.json")
if not train.is_file():
    sys.exit(f"Missing {train}")
if not eval_.is_file():
    sys.exit(f"Missing {eval_}")
with train.open() as f:
    n_train = len(json.load(f))
with eval_.open() as f:
    n_eval = len(json.load(f))
print(f"Train examples: {n_train:,}  Eval examples: {n_eval:,}")
PY

python main_train.py \
  --problem cs \
  --model_name "$BASE_MODEL" \
  --num_train_epochs "$NUM_EPOCHS" \
  --per_device_train_batch_size "$BATCH_SIZE" \
  --gradient_accumulation_steps "$GRAD_ACCUM" \
  --learning_rate "$LEARNING_RATE" \
  --lora_r "$LORA_R" \
  --lora_alpha "$LORA_ALPHA" \
  --max_seq_length "$MAX_SEQ_LENGTH" \
  --save_step "$SAVE_STEPS" \
  --num_nodes "$NUM_NODES" \
  --seed "$SEED" \
  --report_to none \
  --output_dir "$OUTPUT_DIR"

echo "SFT complete $(date -Is)"
echo "Checkpoints: $OUTPUT_DIR"
ls -la "$OUTPUT_DIR" | head -20
