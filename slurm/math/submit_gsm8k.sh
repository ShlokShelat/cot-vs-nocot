#!/bin/bash
#SBATCH --job-name=gsm8k_cot_nocot_[model_tag]
#SBATCH --output=[BASE_DIR]/cot_vs_nocot/logs/gsm8k_[model_tag]_%j.out
#SBATCH --error=[BASE_DIR]/cot_vs_nocot/logs/gsm8k_[model_tag]_%j.err
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --mem=120G
#SBATCH --time=168:00:00
#SBATCH --qos=limit_gpu_qos

BASE=[BASE_DIR]/cot_vs_nocot
mkdir -p $BASE/logs $BASE/eval_results \
         $BASE/lora_output_gsm8k_cot_[model_tag] \
         $BASE/lora_output_gsm8k_nocot_[model_tag]

source [BASE_DIR]/[ENV_NAME]/venv/bin/activate

echo "========================================================"
echo "Job  : $SLURM_JOB_ID  Node: $HOSTNAME"
echo "GPU  : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "Start: $(date)"
echo "========================================================"

cd $BASE

# ── STEP 1: CoT fine-tuning ───────────────────────────────────────────────────
echo "========================================================"
echo "GSM8K CoT fine-tuning ([BASE_MODEL_NAME])..."
echo "========================================================"
python3 -u finetune_gsm8k_cot_[model_tag].py
COT_EXIT=$?
echo "CoT training exit: $COT_EXIT at $(date)"
[ $COT_EXIT -ne 0 ] && echo "GSM8K CoT training FAILED" && exit 1

# ── STEP 2: No-CoT fine-tuning ───────────────────────────────────────────────
echo "========================================================"
echo "GSM8K No-CoT fine-tuning ([BASE_MODEL_NAME])..."
echo "========================================================"
python3 -u finetune_gsm8k_nocot_[model_tag].py
NOCOT_EXIT=$?
echo "No-CoT training exit: $NOCOT_EXIT at $(date)"
[ $NOCOT_EXIT -ne 0 ] && echo "GSM8K No-CoT training FAILED" && exit 1

# ── STEP 3: Baseline eval (no LoRA) ──────────────────────────────────────────
echo "========================================================"
echo "Evaluating GSM8K baseline (no LoRA, [BASE_MODEL_NAME])..."
echo "========================================================"
python3 -u eval_[model_tag]_three_benchmarks.py \
    --model_path [BASE_DIR]/[MODEL_SNAPSHOT_PATH] \
    --dataset    gsm8k \
    --split      test \
    --mode       baseline \
    --n_samples  200 \
    --out        eval_results/gsm8k_baseline_[model_tag].json
echo "Baseline eval done: $(date)"

# ── STEP 4: CoT model eval ───────────────────────────────────────────────────
echo "========================================================"
echo "Evaluating GSM8K CoT fine-tuned model ([BASE_MODEL_NAME])..."
echo "========================================================"
python3 -u eval_[model_tag]_three_benchmarks.py \
    --model_path [BASE_DIR]/[MODEL_SNAPSHOT_PATH] \
    --lora_path  lora_output_gsm8k_cot_[model_tag] \
    --dataset    gsm8k \
    --split      test \
    --mode       gsm8k_cot \
    --n_samples  200 \
    --out        eval_results/gsm8k_cot_[model_tag].json
echo "CoT eval done: $(date)"

# ── STEP 5: No-CoT model eval ────────────────────────────────────────────────
echo "========================================================"
echo "Evaluating GSM8K No-CoT fine-tuned model ([BASE_MODEL_NAME])..."
echo "========================================================"
python3 -u eval_[model_tag]_three_benchmarks.py \
    --model_path [BASE_DIR]/[MODEL_SNAPSHOT_PATH] \
    --lora_path  lora_output_gsm8k_nocot_[model_tag] \
    --dataset    gsm8k \
    --split      test \
    --mode       gsm8k_nocot \
    --n_samples  200 \
    --out        eval_results/gsm8k_nocot_[model_tag].json
echo "No-CoT eval done: $(date)"

# ── SUMMARY ──────────────────────────────────────────────────────────────────
echo "========================================================"
echo "GSM8K RESULTS SUMMARY ([BASE_MODEL_NAME])"
echo "========================================================"
python3 -u - << 'PYEOF'
import json, glob
for path in sorted(glob.glob("eval_results/gsm8k_*[model_tag].json")):
    with open(path) as f:
        data = json.load(f)
    s = data["summary"]
    print(f"[{s['mode'].upper()}]  Accuracy: {s['accuracy_pct']:.1f}%  ({s['correct']}/{s['total']})")
PYEOF

echo "All done: $(date)"
