# Evaluation

## NFA

```bash
# Baseline (zero-shot)
python3 evaluate_nfa_v2.py \
    --model_path /path/to/model \
    --data_file  nfa_v2_nocot_test.jsonl \
    --mode       baseline \
    --out        eval_results/nfa_baseline_iid.json

# CoT fine-tuned -- IID
python3 evaluate_nfa_v2.py \
    --model_path /path/to/model \
    --lora_path  lora_output_nfa_v2_cot \
    --data_file  nfa_v2_cot_test.jsonl \
    --mode       cot \
    --out        eval_results/nfa_cot_iid.json

# No-CoT fine-tuned -- OOD
python3 evaluate_nfa_v2.py \
    --model_path /path/to/model \
    --lora_path  lora_output_nfa_v2_nocot \
    --data_file  nfa_v2_nocot_ood_test.jsonl \
    --mode       nocot --ood \
    --out        eval_results/nfa_nocot_ood.json
```

## DFA

```bash
# Baseline
python3 evaluate_dfa.py \
    --model_path /path/to/model \
    --data_file  dfa_nocot_test.jsonl \
    --mode       baseline \
    --out        eval_results/dfa_baseline_iid.json

# CoT fine-tuned -- IID
python3 evaluate_dfa.py \
    --model_path /path/to/model \
    --lora_path  lora_output_dfa_cot \
    --data_file  dfa_cot_test.jsonl \
    --mode       cot \
    --out        eval_results/dfa_cot_iid.json

# No-CoT fine-tuned -- OOD
python3 evaluate_dfa.py \
    --model_path /path/to/model \
    --lora_path  lora_output_dfa_nocot \
    --data_file  dfa_nocot_ood_test.jsonl \
    --mode       nocot --ood \
    --out        eval_results/dfa_nocot_ood.json
```

## Math

```bash
# GSM8K and MATH-500
python3 eval_gsm8K.py \
    --model_path /path/to/model \
    --lora_path  lora_output_gsm8k_cot \
    --dataset    gsm8k \
    --mode       gsm8k_cot \
    --out        eval_results/gsm8k_cot.json

# Hendrycks MATH
python3 eval_math.py \
    --model_path /path/to/model \
    --lora_path  lora_output_math_cot \
    --mode       math_cot \
    --out        eval_results/math_cot.json

# MathInstruct
python3 eval_mathinstruct.py \
    --model_path /path/to/model \
    --lora_path  lora_output_mathinstruct_cot \
    --mode       mathinstruct_cot \
    --out        eval_results/mathinstruct_cot.json
```

## Correctness Metric

NFA and DFA predictions are evaluated via **language equivalence**:
both the ground-truth and predicted automata are simulated on all
strings up to a tier-dependent maximum length. A prediction is correct
if and only if the two automata agree on every string.
This metric is robust to state relabelling.

| Tier | Dataset | Max string length |
|------|---------|-------------------|
| T1 | NFA | 6 |
| T2 | NFA | 7 |
| T3 | NFA | 8 |
| T4 | NFA | 9 |
| T5 | NFA | 10 |
| T1 | DFA | 6 |
| T2 | DFA | 7 |
| T3 | DFA | 8 |
| T4 | DFA | 9 |
