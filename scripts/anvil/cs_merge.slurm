#!/bin/bash -l
# Merge LoRA checkpoint (SFT or RL) into a full model under models/merged/.
#
# Submit:
#   cd src/problems/cluster_scheduling/solvers/LLMCO
#   sbatch --export=ALL,MODEL_DIR=/anvil/scratch/.../checkpoint-3850 \
#     scripts/anvil/cs_merge.slurm
#
# Override output:
#   sbatch --export=ALL,MODEL_DIR=...,OUTPUT_DIR=/anvil/scratch/.../models/merged/my_run \
#     scripts/anvil/cs_merge.slurm

#SBATCH -A cis260760-ai
#SBATCH -p ai
#SBATCH --job-name=llmco-cs-merge
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH -o /anvil/scratch/x-malizadeh/MetaRL/logs/cs-merge-%j.out

set -euo pipefail

LLMCO="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$LLMCO"

VENV="${LLMCO_VENV:-/anvil/scratch/x-malizadeh/MetaRL/envs/llmco-py312}"
source "$VENV/bin/activate"

export HF_HOME="${HF_HOME:-/anvil/scratch/x-malizadeh/MetaRL/cache/huggingface}"
export TMPDIR="${TMPDIR:-/anvil/scratch/x-malizadeh/MetaRL/tmp}"
mkdir -p "$HF_HOME" "$TMPDIR" /anvil/scratch/x-malizadeh/MetaRL/logs

METARL_ROOT="/anvil/scratch/x-malizadeh/MetaRL"

if [ -z "${MODEL_DIR:-}" ]; then
  MODEL_DIR="$(ls -d "$METARL_ROOT"/runs/rl/*/checkpoint-* 2>/dev/null | sort -V | tail -1 || true)"
fi

OUTPUT_DIR="${OUTPUT_DIR:-$METARL_ROOT/models/merged/$(basename "$MODEL_DIR")}"

if [ -z "${MODEL_DIR:-}" ] || [ ! -d "$MODEL_DIR" ]; then
  echo "ERROR: checkpoint not found (MODEL_DIR=${MODEL_DIR:-unset})" >&2
  exit 1
fi

echo "HOST=$(hostname) JOB_ID=$SLURM_JOB_ID START=$(date -Is)"
echo "MODEL_DIR=$MODEL_DIR"
echo "OUTPUT_DIR=$OUTPUT_DIR"

export MODEL_DIR OUTPUT_DIR
python merge_model.py

echo "Merge complete $(date -Is)"
