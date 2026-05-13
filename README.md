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

| Dataset | CoT SFT | No-CoT SFT | Winner |
|---------|---------|------------|--------|
| GSM8K | 77.0% | 17.5% | **CoT** |
| NFA construction (T5) | 27.0% | 52.0% | **No-CoT** |
| DFA construction (T4) | 82.9% | 95.4% | **No-CoT** |

---

## Repository Structure

```
cot-vs-nocot/
├── datasets/                        # NFA and DFA benchmark datasets
│   ├── nfa/                         # (files hosted on HuggingFace -- see below)
│   └── dfa/
├── data_generation/
│   ├── nfa_parallel_dataset_gen_v2.py
│   └── dfa_parallel_dataset_gen.py
├── training/
│   ├── nfa/
│   │   ├── finetune_nfa_v2_cot.py
│   │   └── finetune_nfa_v2_nocot.py
│   ├── dfa/
│   │   ├── finetune_dfa_cot.py
│   │   └── finetune_dfa_nocot.py
│   └── math/
│       ├── finetune_gsm8k_cot.py
│       ├── finetune_gsm8k_nocot.py
│       ├── finetune_math_cot.py
│       ├── finetune_math_nocot.py
│       ├── finetune_mathinstruct_cot.py
│       └── finetune_mathinstruct_nocot.py
├── evaluation/
│   ├── evaluate_nfa_v2.py           # NFA language equivalence evaluator
│   ├── evaluate_dfa.py              # DFA language equivalence evaluator
│   ├── eval_gsm8K.py                # GSM8K and MATH-500 evaluator
│   ├── eval_math.py                 # Hendrycks MATH evaluator
│   └── eval_mathinstruct.py         # MathInstruct evaluator
├── slurm/
│   ├── nfa/
│   │   ├── submit_nfa_cot_v2.sh
│   │   └── submit_nfa_nocot_v2.sh
│   ├── dfa/
│   │   ├── submit_dfa_cot.sh
│   │   └── submit_dfa_nocot.sh
│   └── math/
│       ├── submit_gsm8k.sh
│       ├── submit_math.sh
│       └── run_mathinstruct.sh
├── analysis/
│   ├── compare_runs.py              # Plot CoT vs No-CoT loss curves
│   └── summarise_results.py        # Aggregate eval JSONs into result tables
├── paper/
│   ├── main.tex
│   ├── appendix.tex
│   ├── references.bib
│   └── figures/figure1.tex
├── requirements.txt
└── LICENSE
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download datasets

Datasets are hosted anonymously on HuggingFace Hub for the review period:

```python
from datasets import load_dataset

# NFA dataset (5 tiers, 25,000 examples)
nfa_cot   = load_dataset("[anonymised]", split="train")
nfa_nocot = load_dataset("[anonymised]", split="train")

# DFA dataset (4 tiers, 25,000 examples)
dfa_cot   = load_dataset("[anonymised]", split="train")
dfa_nocot = load_dataset("[anonymised]", split="train")
```

Or regenerate from scratch (~30-60 minutes on CPU):

```bash
# NFA dataset (25,000 examples, 5 tiers)
python3 data_generation/nfa_parallel_dataset_gen_v2.py \
    --n 25000 --seed 42 --out_prefix nfa_v2 --test_per_tier 200

# DFA dataset (25,000 examples, 4 tiers)
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
# NFA -- CoT fine-tuned (IID)
python3 evaluation/evaluate_nfa_v2.py \
    --model_path /path/to/model \
    --lora_path  lora_output_nfa_v2_cot \
    --data_file  nfa_v2_cot_test.jsonl \
    --mode       cot \
    --out        eval_results/nfa_cot_iid.json

# NFA -- No-CoT fine-tuned (OOD)
python3 evaluation/evaluate_nfa_v2.py \
    --model_path /path/to/model \
    --lora_path  lora_output_nfa_v2_nocot \
    --data_file  nfa_v2_nocot_ood_test.jsonl \
    --mode       nocot --ood \
    --out        eval_results/nfa_nocot_ood.json

# DFA
python3 evaluation/evaluate_dfa.py \
    --model_path /path/to/model \
    --lora_path  lora_output_dfa_cot \
    --data_file  dfa_cot_test.jsonl \
    --mode       cot \
    --out        eval_results/dfa_cot_iid.json

# GSM8K
python3 evaluation/eval_gsm8K.py \
    --model_path /path/to/model \
    --lora_path  lora_output_gsm8k_cot \
    --dataset    gsm8k \
    --mode       gsm8k_cot \
    --out        eval_results/gsm8k_cot.json

# Hendrycks MATH
python3 evaluation/eval_math.py \
    --model_path /path/to/model \
    --lora_path  lora_output_math_cot \
    --mode       math_cot \
    --out        eval_results/math_cot.json

# MathInstruct
python3 evaluation/eval_mathinstruct.py \
    --model_path /path/to/model \
    --lora_path  lora_output_mathinstruct_cot \
    --mode       mathinstruct_cot \
    --out        eval_results/mathinstruct_cot.json
```

### 5. Aggregate results

```bash
python3 analysis/summarise_results.py --results_dir eval_results/
```

---

## Datasets

| Dataset | Task | Tiers | Train | Test | OOD Test |
|---------|------|-------|-------|------|----------|
| NFA v2 | Regex → NFA via Thompson's construction | 5 | ~22,500 | 1,000 | 1,000 |
| DFA | Regex → minimised DFA (Thompson + subset + Hopcroft) | 4 | ~22,500 | 800 | 800 |

Both datasets provide paired CoT and No-CoT versions with identical
questions — only the presence or absence of intermediate reasoning steps differs.

**NFA answer format** — markdown table with State / Role / symbol / epsilon columns.

**DFA answer format** — markdown table with State / Role / symbol columns
(no epsilon column; `--` for dead/trap state transitions).

Correctness is evaluated via **language equivalence**: both ground-truth and
predicted automata are simulated on all strings up to a tier-dependent maximum
length. This metric is robust to state relabelling.

---

## Models

All experiments use LoRA fine-tuning (r=32, alpha=64) on:

| Model | Size | HuggingFace ID |
|-------|------|----------------|
| Qwen-2.5-Instruct | 1.5B | `Qwen/Qwen2.5-1.5B-Instruct` |
| Qwen-2.5-Instruct | 7B | `Qwen/Qwen2.5-7B-Instruct` |
| LLaMA-3.1-Instruct | 8B | `meta-llama/Llama-3.1-8B-Instruct` |

Fine-tuned checkpoints available at [anonymised] for the review period.

---

## Hardware

All experiments run on a single NVIDIA H100 NVL GPU (80 GB).
Total compute: approximately 150--200 GPU hours.

---

## SLURM

For cluster users, job scripts are provided in `slurm/`. Submit CoT and
No-CoT jobs in parallel -- they coordinate via a sentinel file to avoid
double dataset generation.

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
