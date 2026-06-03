# LLMCO for Traffic Engineering (MetaRL Integration)

This document covers the end-to-end workflow for using
[LLMCoSolver](https://github.com/alizmhdi/LLMCoSolver) (NeurIPS 2025) as a
Traffic Engineering solver inside MetaRL.

The pipeline has three stages:

1. **Generate training data** — solve random Traffic Matrices (TMs) with the
   Gurobi LP and convert the continuous solution to a discrete routing.
2. **Fine-tune the LLM** — run Supervised Fine-Tuning (SFT) with
   `LLMCO/main_train.py`, optionally followed by GRPO RL with
   `LLMCO/rl_train.py`.
3. **Run MetaRL adversarial evaluation** — serve the fine-tuned model with
   vLLM and plug `LLMCO` in as the MetaRL target solver.

---

## Prerequisites

| Requirement | Notes |
|---|---|
| `metarl` conda environment | `conda env create -f environment.yml` |
| Gurobi licence | Required for the LP data-generation step |
| GPU with ≥ 24 GB VRAM | For SFT / RL fine-tuning; CPU is fine for inference tests |
| vLLM | Installed in the environment used to serve the model |

All shell commands below assume your working directory is **`MetaRL/src/`**
unless otherwise noted.

---

## Step 1 — Generate Training Data

The script `utils/generate_te_llmco_data.py`
samples random TMs, solves each one with the Gurobi LP (`PathOptimalSolver`),
discretizes the result to one path index per OD pair, and serializes the
dataset as Alpaca-style JSON.

### Output layout

```
LLMCO/
└── data_rl/
    └── te/
        ├── train/train_rl.json    ← 2 000 instances by default
        └── eval/test.json         ← 200 instances by default
```

Each file is a JSON list of objects:

```json
{
  "instruction": "You are given a network ...",
  "input": "### Network:\n ... ### Demands:\n ... ### Paths:\n ...",
  "output": "<routing> 2,0,1,3 </routing>\nTotal Flow: 48776.1310"
}
```

### Basic usage

```bash
conda run -n metarl python utils/generate_te_llmco_data.py \
    --topo B4.json \
    --objective total_flow \
    --min_demand_element 25 \
    --max_demand_element 5000 \
    --num_train 2000 \
    --num_eval 200
```

### All CLI flags

| Flag | Default | Description |
|---|---|---|
| `--topo` | `B4.json` | Topology file (resolved by `GraphUtils`) |
| `--num_path` | `4` | Max k-shortest paths per OD pair |
| `--edge_disjoint` | off | Use edge-disjoint paths |
| `--objective` | `total_flow` | `total_flow` or `min_max_link_util` |
| `--min_demand_element` | `25.0` | Minimum TM element value |
| `--max_demand_element` | `5000.0` | Maximum TM element value |
| `--num_train` | `2000` | Number of training instances |
| `--num_eval` | `200` | Number of evaluation instances |
| `--seed` | `42` | NumPy random seed |
| `--output_dir` | *(LLMCO/data_rl/te/)* | Override output root directory |
| `--skip_infeasible` | `True` | Skip TMs for which the LP is infeasible |

---

## Step 2 — Fine-Tune the LLM

All training scripts live inside the `LLMCO/` submodule.  Change into that
directory before running them.

```bash
cd problems/traffic_engineering/solvers/LLMCO/
```

The training pipeline uses [Unsloth](https://github.com/unslothai/unsloth)
for fast LoRA fine-tuning and the `trl` library.

### 2a — Supervised Fine-Tuning (SFT)

```bash
conda run -n metarl python main_train.py \
    --problem te \
    --model_name unsloth/Qwen2.5-7B \
    --lora_r 64 \
    --lora_alpha 64 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --output_dir output_te_sft
```

The checkpoint is saved to `LLMCO/output_te_sft/`.

Key flags:

| Flag | Default | Description |
|---|---|---|
| `--problem` | `jssp` | Set to `te` for Traffic Engineering |
| `--model_name` | `unsloth/Qwen2.5-7B` | Base model (HuggingFace or local path) |
| `--lora_r` | `64` | LoRA rank |
| `--lora_alpha` | `64` | LoRA scaling factor |
| `--load_in_4bit` | off | Enable 4-bit quantization to reduce VRAM |
| `--num_train_epochs` | `1` | Training epochs |
| `--per_device_train_batch_size` | `4` | Batch size per GPU |
| `--gradient_accumulation_steps` | `4` | Gradient accumulation steps |
| `--learning_rate` | `2e-4` | Learning rate |
| `--max_seq_length` | `20000` | Maximum sequence length |
| `--output_dir` | *(auto)* | Checkpoint output directory |
| `--eval_steps` | `1000` | Evaluate every N steps |
| `--save_step` | `5000` | Save checkpoint every N steps |

### 2b — RL Fine-Tuning with GRPO (optional)

> **Note:** `rl_train.py` currently has hard-coded reward functions for the
> original LLMCO problems (TSP, CVRP, etc.).  TE reward functions must be
> added to `rewards.py` and a `te` branch added to `rl_train.py` before this
> step can be used for TE.

```bash
conda run -n metarl python rl_train.py \
    --problem te \
    --model_name output_te_sft/checkpoint-XXXXX \
    --lora_r 64 \
    --lora_alpha 64 \
    --num_generations 8 \
    --per_device_train_batch_size 8 \
    --learning_rate 1e-6 \
    --output_dir output_te_rl
```

---

## Step 3 — Serve the Model with vLLM

The `LLMCOSolver` calls the vLLM OpenAI-compatible API.  Start the server
before running MetaRL.

### Serving the base model (no fine-tuning)

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
    --port 8001 \
    --max-model-len 8192
```

### Serving a fine-tuned LoRA checkpoint

```bash
vllm serve unsloth/Qwen2.5-7B \
    --enable-lora \
    --lora-modules te-lora=problems/traffic_engineering/solvers/LLMCO/output_te_sft/checkpoint-XXXXX \
    --port 8001 \
    --max-model-len 8192
```

Then pass `--llmco_model te-lora` to the MetaRL command below.

---

## Step 4 — Run MetaRL Adversarial Evaluation

From **`MetaRL/src/`**:

```bash
python main.py adversarial MetaRL TE LLMCO \
    --topo B4.json \
    --objective total_flow \
    --llmco_vllm_url http://localhost:8001/v1 \
    --llmco_model Qwen/Qwen2.5-7B-Instruct \
    --llmco_num_samples 8 \
    --llmco_temperature 0.7 \
    --llmco_max_tokens 512 \
    --num_actors 4 \
    --timesteps 200 \
    --min_demand_element 25 \
    --max_demand_element 5000 \
    --min_action -2 \
    --max_action 2 \
    --reward_freq 1
```

### LLMCO-specific MetaRL flags

| Flag | Default | Description |
|---|---|---|
| `--llmco_vllm_url` | `http://localhost:8001/v1` | vLLM server base URL |
| `--llmco_model` | `Qwen/Qwen2.5-7B-Instruct` | Model name as registered in vLLM |
| `--llmco_num_samples` | `8` | Completions per `solve()` call (best-of-n) |
| `--llmco_temperature` | `0.7` | Sampling temperature |
| `--llmco_max_tokens` | `512` | Max tokens per completion |
| `--llmco_verbose` | `False` | Print per-call diagnostics (e.g. `1/8 valid routings`) |

### Quick smoke test (1 step, no GPU)

```bash
CUDA_VISIBLE_DEVICES="" python main.py adversarial MetaRL TE LLMCO \
    --topo B4.json \
    --objective total_flow \
    --num_actors 1 \
    --timesteps 1 \
    --min_demand_element 25 \
    --max_demand_element 5000 \
    --min_action -2 \
    --max_action 2 \
    --reward_freq 1 \
    --llmco_verbose True
```

Expected output (values will vary):

```
[LLMCO] 1/8 valid routings; best total_flow=48776.1310
============================================================
Best input found  (actor 0, step 1)
  normalized gap       : 0.248
  opt_obj              : 95915.5
  target_obj           : 48776.1
```

---

## File Map

| File | Purpose |
|---|---|
| `LLMCO/` | LLMCoSolver submodule (NeurIPS 2025) |
| `LLMCO/Envs/TEEnv/TEEnv.py` | TE environment for LLMCO — prompt building, routing evaluation, LP→discrete conversion |
| `LLMCO/main_train.py` | SFT training entry-point |
| `LLMCO/rl_train.py` | GRPO RL training entry-point |
| `LLMCO/data_rl/te/` | Generated training / evaluation data (created by step 1) |
| `utils/generate_te_llmco_data.py` | MetaRL data-generation script (Gurobi LP → Alpaca JSON) |
| `llmco_solver.py` | MetaRL solver adapter (`LLMCOSolver`) |
| `registry.py` | Registers `LLMCO` in MetaRL's solver registry |
| `../../descriptor.py` | Registers `LLMCO` CLI flags in MetaRL's TE descriptor |
