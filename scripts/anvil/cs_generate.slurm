#!/bin/bash -l
# Generate CS datasets with Anvil's site Gurobi license (token server).
#
# Submit from login node:
#   cd src/problems/cluster_scheduling/solvers/LLMCO
#   sbatch scripts/anvil/cs_generate.slurm
#
# Uses RCAC license: module load gurobi + GRB_LICENSE_FILE=$GUROBI_HOME/license/gurobi.lic
# Do NOT use grbgetkey — that installs a personal license file.
#
# Your allocation (cis260760-ai) requires partition ai and at least 1 GPU,
# even though this job is CPU-only (Gurobi MIP + JSON writing).

#SBATCH -A cis260760-ai
#SBATCH -p ai
#SBATCH --job-name=cs-gen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH -o /anvil/scratch/x-malizadeh/MetaRL/logs/cs-gen-%j.out

set -euo pipefail

echo "HOST=$(hostname)"
echo "Using Anvil Gurobi token server license"

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

# --- edit flags below or override at submit time with sbatch --export=ALL,CS_ARGS='...' ---
N_INSTANCE="${N_INSTANCE:-10000}"
OUTPUT="${OUTPUT:-data/cs/raw/eval.json}"

python Envs/CSEnv/CSEnv.py \
  --n_instance "$N_INSTANCE" \
  --output "$OUTPUT" \
  --n_job_range 20 20 \
  --num_gpus 100 \
  --throughput_range 0 10 \
  --gpu_range 1 20 \
  --seed 123

echo "Done -> $OUTPUT"
