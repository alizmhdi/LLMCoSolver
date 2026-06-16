#!/bin/bash -l
# Evaluate LLMCO CS checkpoint on GPU (Anvil ai partition).
#
# Submit:
#   cd src/problems/cluster_scheduling/solvers/LLMCO
#   sbatch scripts/anvil/cs_eval.slurm
#
# Override checkpoint or sample count:
#   sbatch --export=ALL,MODEL_ID=path/to/checkpoint,NUM_SAMPLES=1000 scripts/anvil/cs_eval.slurm
#
# Regenerate eval JSON from test.raw.json (default: use existing data/cs/ as-is):
#   sbatch --export=ALL,PREPARE_DATASET=true scripts/anvil/cs_eval.slurm

#SBATCH -A cis260760-ai
#SBATCH -p ai
#SBATCH --job-name=llmco-cs-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH -o /anvil/scratch/x-malizadeh/MetaRL/logs/cs-eval-%j.out

set -euo pipefail

DEFAULT_LLMCO="/anvil/scratch/x-malizadeh/MetaRL/repo/src/problems/cluster_scheduling/solvers/LLMCO"
if [ -n "${LLMCO_DIR:-}" ]; then
  LLMCO="$LLMCO_DIR"
elif [ -f "${SLURM_SUBMIT_DIR:-}/eval.py" ]; then
  LLMCO="$SLURM_SUBMIT_DIR"
else
  LLMCO="$DEFAULT_LLMCO"
fi

if [ ! -f "$LLMCO/eval.py" ]; then
  echo "ERROR: LLMCO root not found (expected eval.py): $LLMCO" >&2
  exit 1
fi
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

MODEL_ID="${MODEL_ID:-output_alpha64_r64_cs_gamma_train_embed_tok_False_seq20000_b4_ep10_n20_100k/checkpoint-25000}"
NUM_SAMPLES="${NUM_SAMPLES:-100}"
EVAL_METHOD="${EVAL_METHOD:-vanilla}"

echo "HOST=$(hostname) LLMCO=$LLMCO MODEL_ID=$MODEL_ID NUM_SAMPLES=$NUM_SAMPLES EVAL_METHOD=$EVAL_METHOD"

PREPARE_DATASET="${PREPARE_DATASET:-false}"
if [ "$PREPARE_DATASET" = true ]; then
  if [ ! -f data/cs/eval/test.raw.json ]; then
    cp data/cs/eval/test.json data/cs/eval/test.raw.json
  fi
  python scripts/prepare_cs_eval_dataset.py \
    --input data/cs/eval/test.raw.json \
    --output-json data/cs/eval/test.json \
    --instances-pkl data/cs/instances.pkl
else
  echo "Using existing dataset under $LLMCO/data/cs/"
  for path in data/cs/eval/test.json data/cs/instances.pkl; do
    if [ ! -f "$path" ]; then
      echo "ERROR: missing $path (set PREPARE_DATASET=true to regenerate)" >&2
      exit 1
    fi
  done
fi

python eval.py \
  --model_id "$MODEL_ID" \
  --problem cs \
  --eval_method "$EVAL_METHOD" \
  --num_samples "$NUM_SAMPLES" \
  --num_nodes 20

echo "Evaluation complete $(date -Is)"
