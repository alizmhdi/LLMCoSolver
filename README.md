# LLMCoSolver: Large Language Models as End-to-end Combinatorial Optimization Solvers

[![NeurIPS 2025](https://img.shields.io/badge/NeurIPS-2025-blue.svg)](https://openreview.net/forum?id=qr5uMEs6iR)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official implementation of the paper **"Large Language Models as End-to-end Combinatorial Optimization Solvers"** presented at The Thirty-ninth Annual Conference on Neural Information Processing Systems (NeurIPS 2025).

## 📖 TL; DR

A framework for training Large Language Models (LLMs) to solve combinatorial optimization problems using supervised fine-tuning (SFT) followed by reinforcement learning (RL).

## 📰 Paper

**Title:** Large Language Models as End-to-end Combinatorial Optimization Solvers

**Authors:** Xia Jiang, Yaoxin Wu, Minshuo Li, Zhiguang Cao, Yingqian Zhang

**Conference:** The Thirty-ninth Annual Conference on Neural Information Processing Systems (NeurIPS 2025)

**Paper Link:** [Arxiv](https://arxiv.org/abs/2509.16865)

## 🚀 Overview

It now supports training and evaluation on multiple combinatorial optimization problems:
- **TSP** (Traveling Salesman Problem)
- **CVRP** (Capacitated Vehicle Routing Problem) 
- **OP** (Orienteering Problem)
- **CS** (GPU Cluster Scheduling)
- **MVC** (Minimum Vertex Cover)
- **MIS** (Maximum Independent Set)
- **PFSP** (Permutation Flow Shop Problem)
- **JSSP** (Job Shop Scheduling Problem)

## Environment setup

LLMCO pins `unsloth==2025.3.19`, which requires **Python 3.9–3.12** (`>=3.9,<3.13`). Python 3.13+ will fail with:

```
ERROR: No matching distribution found for unsloth==2025.3.19
```

Create the virtualenv with Python 3.12:

```bash
cd src/problems/cluster_scheduling/solvers/LLMCO
rm -rf .venv
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

If `python3.12` is not on your PATH, use the full path (e.g. `/home/linuxbrew/.linuxbrew/bin/python3.12`).

## 🔔 Data Format

You can generate your own data through the problem-specific environments under /Envs/, or use the data generated in the original paper: 

- **SFT DATA**: https://drive.google.com/drive/folders/1bE1coGUa00gfuMkPXnfvldi1-WHGNnEb?usp=sharing
- **RL DATA**: https://drive.google.com/drive/folders/1VN9crftdW7DTsMQupbc06u6PzRT-Bwnx?usp=sharing

Place your training and evaluation data in the following structure:
```
data/
├── <problem_name>/
│   ├── train/           # Training data (HF Dataset on disk)
│   ├── eval/            # Evaluation data  
│   └── instances.pkl    # Problem instances (optional, for vanilla eval)
data_rl/
├── <problem_name>/
│   ├── train/train_rl.json
│   └── eval/test.json
```

### Cluster Scheduling (CS) data generation

CS is a 0/1 knapsack-style problem: maximize total job throughput subject to a GPU capacity constraint. Gurobi is required to generate optimal labels.

```bash
cd src/problems/cluster_scheduling/solvers/LLMCO

# RL training data (includes instance field for reward computation)
python Envs/CSEnv/CSEnv.py \
  --n_instance 10000 \
  --output data_rl/cs/train/train_rl.json \
  --rl_data --save_pkl \
  --n_job_range 10 20 \
  --num_gpus 50 \
  --throughput_range 0 50 \
  --gpu_range 1 20

python Envs/CSEnv/CSEnv.py \
  --n_instance 1000 \
  --output data_rl/cs/eval/test.json \
  --rl_data \
  --n_job_range 10 20 \
  --num_gpus 50

# SFT data (no instance field)
python Envs/CSEnv/CSEnv.py \
  --n_instance 10000 \
  --output data/cs/raw/train.json \
  --n_job_range 10 20

python Envs/CSEnv/CSEnv.py \
  --n_instance 1000 \
  --output data/cs/raw/eval.json \
  --n_job_range 10 20

# Convert JSON to HuggingFace datasets for SFT
python scripts/export_sft_dataset.py --input data/cs/raw/train.json --output data/cs/train
python scripts/export_sft_dataset.py --input data/cs/raw/eval.json --output data/cs/eval
```

Generation parameters:

| Flag | Default | Meaning |
|------|---------|---------|
| `--n_job_range` | 20 20 | Min/max jobs per instance |
| `--num_gpus` | 50 | Cluster GPU capacity N |
| `--throughput_range` | 0 50 | Min/max **integer** throughput per job (inclusive) |
| `--gpu_range` | 1 20 | Min/max GPUs per job |

Use `--num_nodes` when training/filtering to match job count (CS sets `num_nodes` = number of jobs).


## 💻 Training Pipeline

The training consists of three main stages:

### 1. Supervised Fine-Tuning (SFT)

First, train the model using supervised learning on problem-specific data:

```bash
python main_train.py --problem <problem_name> [options]
```

**Key parameters:**
- `--problem`: Problem type (tsp, cvrp, op, cs, mvc, mis, pfsp, jssp)
- `--model_name`: Base model to fine-tune (default: unsloth/Qwen2.5-7B)
- `--max_seq_length`: Maximum sequence length (default: 20000)
- `--per_device_train_batch_size`: Batch size per device (default: 4)
- `--num_train_epochs`: Number of training epochs (default: 1)
- `--learning_rate`: Learning rate (default: 2e-4)
- `--lora_r`: LoRA rank (default: 64)
- `--lora_alpha`: LoRA alpha (default: 64)

**Example:**
```bash
python main_train.py --problem cvrp --num_train_epochs 1 --per_device_train_batch_size 4
```

### 2. Reinforcement Learning (RL)

After SFT, improve the model using reinforcement learning (GRPO):

```bash
python rl_train.py --problem <problem_name> --model_name <sft_checkpoint_path> [options]
```

**Key parameters:**
- `--model_name`: Path to SFT checkpoint (e.g., `output_alpha64_r64_cvrp_gamma_train_embed_tok_False_seq20000_b4_ep1/checkpoint-31250`)
- `--num_generations`: Number of generations for GRPO (default: 8)
- `--beta`: KL coefficient (default: 0.05)
- `--learning_rate`: Learning rate (default: 1e-6)
- `--max_prompt_length`: Maximum prompt length (default: 20000)
- `--max_completion_length`: Maximum completion length (default: 1000)

**Example:**
```bash
python rl_train.py --problem cvrp --model_name output_alpha64_r64_cvrp_gamma_train_embed_tok_False_seq20000_b4_ep1/checkpoint-31250
```

### 3. Model Merging

After training, merge the LoRA weights with the base model:

1. Edit `cmd.sh` to specify your model checkpoint path:
   ```bash
   MODEL_DIR="./path/to/your/checkpoint"
   ```

2. Run the merge script:
   ```bash
   bash cmd.sh
   ```

This creates a `saved_models/` directory with the merged model.

## 🧪 Evaluation

Evaluate the trained model using two methods:

### Vanilla Evaluation
```bash
python eval.py --model_id saved_models --problem <problem_name> --eval_method vanilla --num_samples 100
```

### Best-of-N Evaluation
```bash
python eval.py --model_id saved_models --problem <problem_name> --eval_method best_of_n --num_samples 100 --best_of_n 8 --temperature 0.7
```

**Evaluation parameters:**
- `--model_id`: Path to the merged model (default: saved_models)
- `--eval_method`: Evaluation method (vanilla or best_of_n)
- `--num_samples`: Number of test instances to evaluate
- `--best_of_n`: Number of solutions to generate per instance (for best_of_n)
- `--temperature`: Sampling temperature
- `--batch_size`: Batch size for evaluation

### Output Metrics

The evaluation provides:
- **Feasibility Rate**: Percentage of valid solutions
- **Optimality Gap**: Average gap from optimal/reference solutions  

## 📊 Quick Start Example

Here's a complete example for training on CVRP:

```bash
# 1. Supervised Fine-Tuning
python main_train.py --problem cvrp --num_train_epochs 1

# 2. Reinforcement Learning  
python rl_train.py --problem cvrp --model_name output_alpha64_r64_cvrp_gamma_train_embed_tok_False_seq20000_b4_ep1/checkpoint-31250

# 3. Merge Model (edit MODEL_DIR in cmd.sh first)
bash cmd.sh

# 4. Evaluate
python eval.py --model_id saved_models --problem cs --eval_method best_of_n \
  --dataset_method get_dataset --num_samples 100 --best_of_n 8
```

### CS quick start

```bash
# 1. Generate data (see CS data generation above)

# 2. SFT
python main_train.py --problem cs --num_nodes 20 --num_train_epochs 1

# 3. RL
python rl_train.py --problem cs --model_name <sft_checkpoint> --num_nodes 20

# 4. Merge + eval
bash cmd.sh
python eval.py --model_id saved_models --problem cs --eval_method best_of_n \
  --dataset_method get_dataset --num_samples 100
```


## 🤝 Contributing

We welcome contributions to this project. Please feel free to submit issues and pull requests.

## 📜 Citation

If you find this work useful in your research, please consider citing:

```bibtex
@inproceedings{
jiang2025large,
title={Large Language Models as End-to-end Combinatorial Optimization Solvers},
author={Xia Jiang, Yaoxin Wu, Minshuo Li, Zhiguang Cao, Yingqian Zhang},
booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
year={2025},
url={https://arxiv.org/abs/2509.16865}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

