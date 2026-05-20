# Chain-of-Thought Hurts When the Answer Speaks for Itself

> **Answer backtrackability determines when chain-of-thought supervision
> helps or hurts LLM fine-tuning.**

This repository contains code, datasets, and evaluation scripts for the paper:

> **Chain-of-Thought Hurts When the Answer Speaks for Itself**
> Anonymous Authors. Under review at EMNLP 2025.

---

## Overview

We introduce *answer backtrackability* — the degree to which a question
can be reconstructed from its answer alone — and show it reliably predicts
whether CoT or No-CoT fine-tuning will be more effective.

**Key results (IID, Qwen-2.5-7B unless noted):**

| Dataset | Model | CoT SFT | No-CoT SFT | Winner |
|---------|-------|---------|------------|--------|
| GSM8K | 1.5B | 47.0% | 12.0% | **CoT** |
| GSM8K | 7B | **77.0%** | 17.5% | **CoT** |
| GSM8K | 8B (LLaMA) | **70.0%** | 22.5% | **CoT** |
| MATH-500 | 1.5B | **24.0%** | 10.5% | **CoT** |
| MATH-500 | 7B | **47.0%** | 24.0% | **CoT** |
| MATH-500 | 8B (LLaMA) | **31.5%** | 16.0% | **CoT** |
| MathInstruct | 1.5B | **49.2%** | 45.8% | **CoT** |
| MathInstruct | 7B | **63.2%** | 58.9% | **CoT** |
| MathInstruct | 8B (LLaMA) | **59.3%** | 54.8% | **CoT** |
| NFA (overall) | 1.5B | 83.5% | **86.5%** | **No-CoT** |
| NFA (overall) | 7B | 83.4% | **88.6%** | **No-CoT** |
| NFA (overall) | 8B (LLaMA) | 90.3% | **93.0%** | **No-CoT** |
| NFA (T5, hardest) | 7B | 27.0% | **52.0%** | **No-CoT** |
| DFA (overall) | 1.5B | 79.9% | **82.2%** | **No-CoT** |
| DFA (overall) | 7B | 88.5% | **91.6%** | **No-CoT** |
| DFA (overall) | 8B (LLaMA) | 90.2% | **95.8%** | **No-CoT** |
| DFA (T4, hardest) | 7B | 71.0% | **80.5%** | **No-CoT** |
| DFA (T4, hardest) | 8B (LLaMA) | 74.0% | **89.5%** | **No-CoT** |

No-CoT wins on every high-backtrackability setting (NFA, DFA) and CoT
wins on every low-backtrackability setting (GSM8K, MATH, MathInstruct)
across all 18 model-task pairs without exception.

---

## Repository Structure

```
cot-vs-nocot/
├── datasets/                        # NFA and DFA benchmark datasets
│   ├── nfa/                         # nfa_v2_{cot,nocot}_{train,val,test,ood_test,full}.jsonl
│   └── dfa/                         # dfa_{cot,nocot}_{train,val,test,ood_test,full}.jsonl
├── data_generation/
│   ├── nfa_parallel_dataset_gen_v2.py
│   └── dfa_parallel_dataset_gen.py
├── training/
│   ├── nfa/{finetune_nfa_v2_cot.py, finetune_nfa_v2_nocot.py}
│   ├── dfa/{finetune_dfa_cot.py, finetune_dfa_nocot.py}
│   └── math/{finetune_gsm8k_{cot,nocot}.py, finetune_math_{cot,nocot}.py,
│              finetune_mathinstruct_{cot,nocot}.py}
├── evaluation/
│   ├── evaluate_nfa_v2.py
│   ├── evaluate_dfa.py
│   ├── eval_gsm8K.py
│   ├── eval_math.py
│   └── eval_mathinstruct.py
├── slurm/
│   ├── nfa/{submit_nfa_cot_v2.sh, submit_nfa_nocot_v2.sh}
│   ├── dfa/{submit_dfa_cot.sh, submit_dfa_nocot.sh}
│   └── math/{submit_gsm8k.sh, submit_math.sh, run_mathinstruct.sh}
├── analysis/
│   ├── compare_runs.py
│   └── summarise_results.py
├── requirements.txt
└── LICENSE
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Datasets

Datasets are included in `datasets/`. To regenerate from scratch:

```bash
python3 data_generation/nfa_parallel_dataset_gen_v2.py \
    --n 25000 --seed 42 --out_prefix nfa_v2 --test_per_tier 200

python3 data_generation/dfa_parallel_dataset_gen.py \
    --n 25000 --seed 42 --out_prefix dfa --test_per_tier 200
```

### 3. Fine-tune

Edit `MODEL_PATH` at the top of each script before running.

```bash
# NFA
python3 training/nfa/finetune_nfa_v2_cot.py
python3 training/nfa/finetune_nfa_v2_nocot.py

# DFA
python3 training/dfa/finetune_dfa_cot.py
python3 training/dfa/finetune_dfa_nocot.py

# Math
python3 training/math/finetune_gsm8k_cot.py
python3 training/math/finetune_mathinstruct_cot.py
```

### 4. Evaluate

```bash
# NFA CoT IID
python3 evaluation/evaluate_nfa_v2.py \
    --model_path /path/to/model \
    --lora_path  lora_output_nfa_v2_cot \
    --data_file  datasets/nfa/nfa_v2_cot_test.jsonl \
    --mode cot --out eval_results/nfa_cot_iid.json

# DFA No-CoT IID
python3 evaluation/evaluate_dfa.py \
    --model_path /path/to/model \
    --lora_path  lora_output_dfa_nocot \
    --data_file  datasets/dfa/dfa_nocot_test.jsonl \
    --mode nocot --out eval_results/dfa_nocot_iid.json

# GSM8K
python3 evaluation/eval_gsm8K.py \
    --model_path /path/to/model \
    --lora_path  lora_output_gsm8k_cot \
    --dataset gsm8k --mode gsm8k_cot \
    --out eval_results/gsm8k_cot.json
```

### 5. Aggregate results

```bash
python3 analysis/summarise_results.py --results_dir eval_results/
```

---

## Datasets

| Dataset | Task | Tiers | Train | Test | OOD Test |
|---------|------|-------|-------|------|----------|
| NFA v2 | Regex → NFA (Thompson's construction) | 5 | ~22,500 | 1,000 | 1,000 |
| DFA | Regex → minimised DFA (Thompson + subset + Hopcroft) | 4 | ~22,500 | 641 | 641 |

Both datasets provide paired CoT and No-CoT versions with identical
questions and ground-truth answers.

---

## Models

All experiments use LoRA fine-tuning (r=32, alpha=64) on:

| Model | Size | HuggingFace ID |
|-------|------|----------------|
| Qwen-2.5-Instruct | 1.5B | `Qwen/Qwen2.5-1.5B-Instruct` |
| Qwen-2.5-Instruct | 7B | `Qwen/Qwen2.5-7B-Instruct` |
| LLaMA-3.1-Instruct | 8B | `meta-llama/Llama-3.1-8B-Instruct` |

---

## Hardware

All experiments run on a single NVIDIA H100 NVL GPU (80 GB).
Total compute: approximately 150--200 GPU hours.

---

## SLURM

```bash
sbatch slurm/nfa/submit_nfa_cot_v2.sh
sbatch slurm/nfa/submit_nfa_nocot_v2.sh
sbatch slurm/dfa/submit_dfa_cot.sh
sbatch slurm/dfa/submit_dfa_nocot.sh
sbatch slurm/math/submit_gsm8k.sh
sbatch slurm/math/submit_math.sh
sbatch slurm/math/run_mathinstruct.sh
```

Edit `BASE`, `MODEL_PATH`, and venv path variables at the top of each script.

---

## Citation

```
Anonymous Authors (2025).
Chain-of-Thought Hurts When the Answer Speaks for Itself.
Under review at EMNLP 2025.
```

---

## License

MIT License. See [LICENSE](LICENSE).
