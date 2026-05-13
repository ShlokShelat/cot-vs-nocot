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
├── datasets/           # NFA and DFA benchmark datasets (see HuggingFace link below)
├── data_generation/    # Scripts to regenerate datasets from scratch
├── training/           # LoRA fine-tuning scripts (NFA, DFA, Math)
├── evaluation/         # Evaluation scripts with language equivalence metric
├── slurm/              # SLURM job scripts for H100 cluster
├── analysis/           # Result aggregation and loss curve plotting
└── paper/              # LaTeX source
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

# NFA dataset
nfa_cot   = load_dataset("[anonymised]", split="train")
nfa_nocot = load_dataset("[anonymised]", split="train")

# DFA dataset
dfa_cot   = load_dataset("[anonymised]", split="train")
dfa_nocot = load_dataset("[anonymised]", split="train")
```

Or regenerate from scratch (takes ~30-60 minutes on CPU):

```bash
# NFA dataset (25,000 examples, 5 tiers)
python3 data_generation/nfa_parallel_dataset_gen_v2.py     --n 25000 --seed 42 --out_prefix nfa_v2 --test_per_tier 200

# DFA dataset (25,000 examples, 4 tiers)
python3 data_generation/dfa_parallel_dataset_gen.py     --n 25000 --seed 42 --out_prefix dfa --test_per_tier 200
```

### 3. Fine-tune

Edit `MODEL_PATH` at the top of each script to point to your local model.

```bash
# NFA -- CoT condition
python3 training/nfa/finetune_nfa_v2_cot.py

# NFA -- No-CoT condition
python3 training/nfa/finetune_nfa_v2_nocot.py
```

### 4. Evaluate

```bash
# NFA CoT fine-tuned (IID)
python3 evaluation/evaluate_nfa_v2.py     --model_path /path/to/model     --lora_path  lora_output_nfa_v2_cot     --data_file  nfa_v2_cot_test.jsonl     --mode       cot     --out        eval_results/cot_iid.json

# NFA No-CoT fine-tuned (OOD)
python3 evaluation/evaluate_nfa_v2.py     --model_path /path/to/model     --lora_path  lora_output_nfa_v2_nocot     --data_file  nfa_v2_nocot_ood_test.jsonl     --mode       nocot     --ood     --out        eval_results/nocot_ood.json

# DFA evaluation
python3 evaluation/evaluate_dfa.py     --model_path /path/to/model     --lora_path  lora_output_dfa_cot     --data_file  dfa_cot_test.jsonl     --mode       cot     --out        eval_results/dfa_cot_iid.json

# Math evaluation
python3 evaluation/eval_qwen_three_benchmarks.py     --dataset    gsm8k     --model_path /path/to/model     --lora_path  lora_output_gsm8k_cot     --mode       cot     --out        eval_results/gsm8k_cot.json
```

### 5. Aggregate results

```bash
python3 analysis/summarise_results.py --results_dir eval_results/
```

---

## Datasets

| Dataset | Task | Tiers | Train | Test | OOD Test |
|---------|------|-------|-------|------|----------|
| NFA v2  | Regex → NFA via Thompson's construction | 5 | ~22,500 | 1,000 | 1,000 |
| DFA     | Regex → minimised DFA (Thompson + subset + Hopcroft) | 4 | ~22,500 | 800 | 800 |

Both datasets provide paired CoT and No-CoT versions with identical
questions. The only difference between the two versions is the presence
or absence of intermediate reasoning steps in the assistant response.

### Answer format

NFA and DFA answers are markdown transition tables:

```
| State | Role   | a      | b      | ε       |
|-------|--------|--------|--------|---------|
| q1    | start  | ∅      | ∅      | {q2,q7} |
| q10   | accept | ∅      | ∅      | ∅       |
```

Correctness is evaluated via **language equivalence** — both automata
are simulated on all strings up to a tier-dependent maximum length.
This metric is robust to state relabelling.

---

## Models

All experiments use LoRA fine-tuning (r=32, alpha=64) on three
open-source instruction-tuned models:

| Model | Size |
|-------|------|
| Qwen-2.5-Instruct | 1.5B |
| Qwen-2.5-Instruct | 7B |
| LLaMA-3.1-Instruct | 8B |

Fine-tuned checkpoints are available at [anonymised] for the review period.

---

## Hardware

All experiments run on a single NVIDIA H100 NVL GPU (80 GB).
Total compute: approximately 150--200 GPU hours across all experiments.

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
