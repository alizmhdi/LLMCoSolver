#!/bin/bash -l
# Regenerate CS eval splits only (SFT + RL), without touching train data.
#
# Submit from login node:
#   cd src/problems/cluster_scheduling/solvers/LLMCO
#   sbatch scripts/anvil/cs_generate_eval.slurm
#
# Override eval size / seed / job range:
#   sbatch --export=ALL,N_EVAL=1000,EVAL_SEED=54321,N_JOB_MIN=15,N_JOB_MAX=25 \
#     scripts/anvil/cs_generate_eval.slurm

#SBATCH -A cis260760-ai
#SBATCH -p ai
#SBATCH --job-name=cs-gen-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH -o /anvil/scratch/x-malizadeh/MetaRL/logs/cs-gen-eval-%j.out

set -euo pipefail

echo "JOB_ID=$SLURM_JOB_ID HOST=$(hostname) START=$(date -Is)"

module load gurobi/9.5.1
export GRB_LICENSE_FILE="$GUROBI_HOME/license/gurobi.lic"

LLMCO="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$LLMCO"

VENV="${LLMCO_VENV:-/anvil/scratch/x-malizadeh/MetaRL/envs/llmco-py312}"
source "$VENV/bin/activate"

python -c "import gurobipy as gp; m=gp.Model(); m.dispose(); print('Gurobi license OK')"

N_EVAL="${N_EVAL:-1000}"
EVAL_SEED="${EVAL_SEED:-54321}"
N_JOB_MIN="${N_JOB_MIN:-15}"
N_JOB_MAX="${N_JOB_MAX:-25}"

mkdir -p data_rl/cs/eval data/cs/eval

COMMON=(
  --n_job_range "$N_JOB_MIN" "$N_JOB_MAX"
  --num_gpus 100
  --throughput_range 1 10
  --gpu_range 1 20
)

run_step() {
  local label="$1"
  shift
  echo ""
  echo "========== $label $(date -Is) =========="
  python Envs/CSEnv/CSEnv.py "$@"
  echo "========== $label done $(date -Is) =========="
}

echo "N_EVAL=$N_EVAL EVAL_SEED=$EVAL_SEED N_JOB_RANGE=$N_JOB_MIN-$N_JOB_MAX"

run_step "RL eval (${N_EVAL}, seed=${EVAL_SEED})" \
  --n_instance "$N_EVAL" \
  --output data_rl/cs/eval/test.json \
  --rl_data \
  "${COMMON[@]}" \
  --seed "$EVAL_SEED"

run_step "SFT eval (${N_EVAL}, seed=${EVAL_SEED})" \
  --n_instance "$N_EVAL" \
  --output data/cs/eval/test.json \
  "${COMMON[@]}" \
  --seed "$EVAL_SEED"

python - <<'PY'
import json
from pathlib import Path
for name, path in {
    "sft_eval": Path("data/cs/eval/test.json"),
    "rl_eval": Path("data_rl/cs/eval/test.json"),
}.items():
    with path.open() as f:
        n = len(json.load(f))
    print(f"  {name}: {n:,} examples ({path})")
PY

echo "Eval regeneration complete $(date -Is)"
