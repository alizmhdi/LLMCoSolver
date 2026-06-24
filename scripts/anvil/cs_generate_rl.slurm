#!/bin/bash -l
# Regenerate CS RL splits only (train + eval), without touching SFT data.
#
# Submit from login node:
#   cd src/problems/cluster_scheduling/solvers/LLMCO
#   sbatch scripts/anvil/cs_generate_rl.slurm
#
# Override sizes / seeds / job range (use seeds different from SFT):
#   sbatch --export=ALL,N_TRAIN=5000,N_EVAL=300,TRAIN_SEED=13579,EVAL_SEED=24680,N_JOB_MIN=15,N_JOB_MAX=25 \
#     scripts/anvil/cs_generate_rl.slurm

#SBATCH -A cis260760-ai
#SBATCH -p ai
#SBATCH --job-name=cs-gen-rl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH -o /anvil/scratch/x-malizadeh/MetaRL/logs/cs-gen-rl-%j.out

set -euo pipefail

echo "JOB_ID=$SLURM_JOB_ID HOST=$(hostname) START=$(date -Is)"

module load gurobi/9.5.1
export GRB_LICENSE_FILE="$GUROBI_HOME/license/gurobi.lic"

LLMCO="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$LLMCO"

VENV="${LLMCO_VENV:-/anvil/scratch/x-malizadeh/MetaRL/envs/llmco-py312}"
source "$VENV/bin/activate"

python -c "import gurobipy as gp; m=gp.Model(); m.dispose(); print('Gurobi license OK')"

N_TRAIN="${N_TRAIN:-5000}"
N_EVAL="${N_EVAL:-300}"
TRAIN_SEED="${TRAIN_SEED:-13579}"
EVAL_SEED="${EVAL_SEED:-24680}"
N_JOB_MIN="${N_JOB_MIN:-15}"
N_JOB_MAX="${N_JOB_MAX:-25}"

mkdir -p data_rl/cs/train data_rl/cs/eval

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

echo "N_TRAIN=$N_TRAIN N_EVAL=$N_EVAL TRAIN_SEED=$TRAIN_SEED EVAL_SEED=$EVAL_SEED N_JOB_RANGE=$N_JOB_MIN-$N_JOB_MAX"

run_step "RL train (${N_TRAIN}, seed=${TRAIN_SEED})" \
  --n_instance "$N_TRAIN" \
  --output data_rl/cs/train/train_rl.json \
  --rl_data \
  "${COMMON[@]}" \
  --seed "$TRAIN_SEED"

run_step "RL eval (${N_EVAL}, seed=${EVAL_SEED})" \
  --n_instance "$N_EVAL" \
  --output data_rl/cs/eval/test.json \
  --rl_data \
  "${COMMON[@]}" \
  --seed "$EVAL_SEED"

python - <<'PY'
import json, pickle, re
from pathlib import Path

train_path = Path("data_rl/cs/train/train_rl.json")
eval_path = Path("data_rl/cs/eval/test.json")
rows = json.loads(train_path.read_text())
pat = re.compile(r"Schedule: \[(.*?)\], Objective: (\d+)")
instances = []
for row in rows:
    tp, gp, cap = row["instance"]
    m = pat.search(row["output"])
    if not m:
        continue
    sched = [int(x.strip()) for x in m.group(1).split(",") if x.strip()]
    obj = float(m.group(2))
    instances.append((tp, gp, cap, sched, obj))
out = Path("data_rl/cs/train/instances.pkl")
with out.open("wb") as f:
    pickle.dump(instances, f, pickle.HIGHEST_PROTOCOL)
print(f"Wrote {out} ({len(instances):,} instances)")

for name, path in {
    "rl_train": train_path,
    "rl_eval": eval_path,
}.items():
    with path.open() as f:
        n = len(json.load(f))
    print(f"  {name}: {n:,} examples ({path})")
PY

echo "RL dataset regeneration complete $(date -Is)"
