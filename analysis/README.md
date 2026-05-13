# Analysis

## Aggregate all evaluation results into a summary table

```bash
python3 summarise_results.py --results_dir eval_results/

# Filter to a specific dataset
python3 summarise_results.py --results_dir eval_results/ --filter nfa

# Save to file
python3 summarise_results.py --results_dir eval_results/ --out summary.txt
```

## Plot CoT vs No-CoT training and evaluation loss curves

```bash
python3 compare_runs.py \
    --cot_dir   lora_output_nfa_v2_cot \
    --nocot_dir lora_output_nfa_v2_nocot \
    --title     "NFA Dataset" \
    --out       figures/nfa_loss_curves.png
```
