#!/bin/bash -l
# Generate all CS datasets (RL + SFT) using Anvil's site Gurobi license.
#
# Submit from login node:
#   cd src/problems/cluster_scheduling/solvers/LLMCO
#   sbatch scripts/anvil/cs_generate_all.slurm
#
# Total: 121,000 Gurobi solves (100k SFT train is the long pole).

#SBATCH -A cis260760-ai
#SBATCH -p ai
#SBATCH --job-name=cs-gen-all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH -o /anvil/scratch/x-malizadeh/MetaRL/logs/cs-gen-all-%j.out

set -euo pipefail

echo "JOB_ID=$SLURM_JOB_ID HOST=$(hostname) START=$(date -Is)"

module load gurobi/9.5.1
export GRB_LICENSE_FILE="$GUROBI_HOME/license/gurobi.lic"

LLMCO="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$LLMCO"

VENV="${LLMCO_VENV:-/anvil/scratch/x-malizadeh/MetaRL/envs/llmco-py312}"
if [ ! -x "$VENV/bin/python" ]; then
  echo "ERROR: LLMCO venv not found: $VENV" >&2
  echo "Create it: bash scripts/anvil/setup_llmco_venv.sh" >&2
  exit 1
fi
source "$VENV/bin/activate"

python -c "import gurobipy as gp; m=gp.Model(); m.dispose(); print('Gurobi license OK')"

mkdir -p data_rl/cs/train data_rl/cs/eval data/cs/train data/cs/eval

COMMON=(
  --n_job_range 20 20
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

# RL train (10k + pickle)
run_step "RL train" \
  --n_instance 10000 \
  --output data_rl/cs/train/train_rl.json \
  --rl_data --save_pkl \
  "${COMMON[@]}" \
  --seed 42

# RL eval/test (1k)
run_step "RL eval" \
  --n_instance 1000 \
  --output data_rl/cs/eval/test.json \
  --rl_data \
  "${COMMON[@]}" \
  --seed 123

# SFT train (100k)
run_step "SFT train" \
  --n_instance 100000 \
  --output data/cs/train/train_cs.json \
  "${COMMON[@]}" \
  --seed 42

# SFT eval/test (10k)
run_step "SFT eval" \
  --n_instance 10000 \
  --output data/cs/eval/test.json \
  "${COMMON[@]}" \
  --seed 123

echo ""
echo "ALL DATASETS COMPLETE $(date -Is)"
echo "Outputs:"
ls -lh data_rl/cs/train/train_rl.json data_rl/cs/eval/test.json \
       data/cs/train/train_cs.json data/cs/eval/test.json instances.pkl 2>/dev/null || true
