#!/bin/bash -l
# Generate all CS datasets (RL + SFT) using Anvil's site Gurobi license.
#
# Submit from login node:
#   cd src/problems/cluster_scheduling/solvers/LLMCO
#   sbatch scripts/anvil/cs_generate_all.slurm
#
# Defaults: 200k train + 10k eval per split (SFT and RL).
# Train/eval use different seeds. Large train sets are generated in 100k
# chunks (fits the 48h QOS wall limit) then merged.
#
# Override:
#   sbatch --export=ALL,N_TRAIN=500000,N_EVAL=500000,N_JOB_MIN=15,N_JOB_MAX=25 \
#     scripts/anvil/cs_generate_all.slurm

#SBATCH -A cis260760-ai
#SBATCH -p ai
#SBATCH --job-name=cs-gen-all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
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

N_TRAIN="${N_TRAIN:-200000}"
N_EVAL="${N_EVAL:-1000}"
N_JOB_MIN="${N_JOB_MIN:-15}"
N_JOB_MAX="${N_JOB_MAX:-25}"
TRAIN_SEED="${TRAIN_SEED:-4242}"
EVAL_SEED="${EVAL_SEED:-8989}"
CHUNK_SIZE="${CHUNK_SIZE:-100000}"

mkdir -p data_rl/cs/train data_rl/cs/eval data/cs/train data/cs/eval \
         data_rl/cs/train/chunks data/cs/train/chunks

COMMON=(
  --n_job_range "$N_JOB_MIN" "$N_JOB_MAX"
  --num_gpus 100
  --throughput_range 1 10
  --gpu_range 1 20
)

META_FILE="data/cs/dataset_meta_job${SLURM_JOB_ID}.json"
python - <<PY
import json
meta = {
    "job_id": ${SLURM_JOB_ID},
    "n_train": ${N_TRAIN},
    "n_eval": ${N_EVAL},
    "n_job_min": ${N_JOB_MIN},
    "n_job_max": ${N_JOB_MAX},
    "train_seed_base": ${TRAIN_SEED},
    "eval_seed": ${EVAL_SEED},
    "chunk_size": ${CHUNK_SIZE},
    "sft_train": "data/cs/train/train_cs.json",
    "sft_eval": "data/cs/eval/test.json",
    "rl_train": "data_rl/cs/train/train_rl.json",
    "rl_eval": "data_rl/cs/eval/test.json",
}
with open("${META_FILE}", "w") as f:
    json.dump(meta, f, indent=2)
print("Wrote metadata -> ${META_FILE}")
PY

run_step() {
  local label="$1"
  shift
  echo ""
  echo "========== $label $(date -Is) =========="
  python Envs/CSEnv/CSEnv.py "$@"
  echo "========== $label done $(date -Is) =========="
}

merge_json_chunks() {
  local out="$1"
  shift
  python - "$out" "$@" <<'PY'
import json, sys
out_path = sys.argv[1]
chunk_paths = sys.argv[2:]
merged = []
for path in chunk_paths:
    with open(path) as f:
        merged.extend(json.load(f))
with open(out_path, "w") as f:
    json.dump(merged, f, indent=4)
print(f"Merged {len(chunk_paths)} chunk(s) -> {out_path} ({len(merged):,} examples)")
PY
}

echo "N_TRAIN=$N_TRAIN N_EVAL=$N_EVAL N_JOB_RANGE=$N_JOB_MIN-$N_JOB_MAX TRAIN_SEED=$TRAIN_SEED EVAL_SEED=$EVAL_SEED CHUNK_SIZE=$CHUNK_SIZE"

# --- eval splits (fast) ---
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

# --- train splits in chunks ---
remaining="$N_TRAIN"
chunk_idx=0
rl_chunks=()
sft_chunks=()

while [ "$remaining" -gt 0 ]; do
  if [ "$remaining" -gt "$CHUNK_SIZE" ]; then
    n_chunk="$CHUNK_SIZE"
  else
    n_chunk="$remaining"
  fi
  chunk_seed=$((TRAIN_SEED + chunk_idx))

  rl_chunk="data_rl/cs/train/chunks/train_rl_${chunk_idx}.json"
  sft_chunk="data/cs/train/chunks/train_cs_${chunk_idx}.json"
  rl_chunks+=("$rl_chunk")
  sft_chunks+=("$sft_chunk")

  run_step "RL train chunk ${chunk_idx} (${n_chunk}, seed=${chunk_seed})" \
    --n_instance "$n_chunk" \
    --output "$rl_chunk" \
    --rl_data \
    "${COMMON[@]}" \
    --seed "$chunk_seed"

  run_step "SFT train chunk ${chunk_idx} (${n_chunk}, seed=${chunk_seed})" \
    --n_instance "$n_chunk" \
    --output "$sft_chunk" \
    "${COMMON[@]}" \
    --seed "$chunk_seed"

  remaining=$((remaining - n_chunk))
  chunk_idx=$((chunk_idx + 1))
done

merge_json_chunks data_rl/cs/train/train_rl.json "${rl_chunks[@]}"
merge_json_chunks data/cs/train/train_cs.json "${sft_chunks[@]}"

# RL pickle from merged train (optional convenience artifact)
python - <<'PY'
import json, pickle, re
from pathlib import Path
path = Path("data_rl/cs/train/train_rl.json")
rows = json.loads(path.read_text())
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
PY

echo ""
echo "ALL DATASETS COMPLETE $(date -Is)"
echo "Metadata: $META_FILE"
ls -lh data_rl/cs/train/train_rl.json data_rl/cs/eval/test.json \
       data/cs/train/train_cs.json data/cs/eval/test.json \
       data_rl/cs/train/instances.pkl 2>/dev/null || true

python - <<'PY'
import json
from pathlib import Path
paths = {
    "sft_train": Path("data/cs/train/train_cs.json"),
    "sft_eval": Path("data/cs/eval/test.json"),
    "rl_train": Path("data_rl/cs/train/train_rl.json"),
    "rl_eval": Path("data_rl/cs/eval/test.json"),
}
for name, path in paths.items():
    with path.open() as f:
        n = len(json.load(f))
    print(f"  {name}: {n:,} examples ({path})")
PY
