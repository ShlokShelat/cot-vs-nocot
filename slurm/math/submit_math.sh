#!/bin/bash
#SBATCH --job-name=math_cot_nocot_[model_tag]
#SBATCH --output=[BASE_DIR]/cot_vs_nocot/logs/math_[model_tag]_%j.out
#SBATCH --error=[BASE_DIR]/cot_vs_nocot/logs/math_[model_tag]_%j.err
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
         $BASE/lora_output_math_cot_[model_tag] \
         $BASE/lora_output_math_nocot_[model_tag]

source [BASE_DIR]/[ENV_NAME]/venv/bin/activate

echo "========================================================"
echo "Job  : $SLURM_JOB_ID  Node: $HOSTNAME"
echo "GPU  : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "Start: $(date)"
echo "========================================================"

cd $BASE

BASE_MODEL=[BASE_DIR]/[MODEL_SNAPSHOT_PATH]

export MATH_COT_DIR=$BASE/lora_output_math_cot_[model_tag]
export MATH_NOCOT_DIR=$BASE/lora_output_math_nocot_[model_tag]

# ── STEP 1: CoT fine-tuning ───────────────────────────────────────────────────
echo "========================================================"
echo "MATH CoT fine-tuning ([BASE_MODEL_NAME])..."
echo "========================================================"
python3 -u finetune_math_cot_[model_tag].py
COT_EXIT=$?
echo "CoT training exit: $COT_EXIT at $(date)"
[ $COT_EXIT -ne 0 ] && echo "MATH CoT training FAILED" && exit 1

# ── STEP 2: No-CoT fine-tuning ───────────────────────────────────────────────
echo "========================================================"
echo "MATH No-CoT fine-tuning ([BASE_MODEL_NAME])..."
echo "========================================================"
python3 -u finetune_math_nocot_[model_tag].py
NOCOT_EXIT=$?
echo "No-CoT training exit: $NOCOT_EXIT at $(date)"
[ $NOCOT_EXIT -ne 0 ] && echo "MATH No-CoT training FAILED" && exit 1

# ── STEP 3: Baseline eval (no LoRA) ──────────────────────────────────────────
echo "========================================================"
echo "Evaluating MATH baseline (no LoRA, [BASE_MODEL_NAME])..."
echo "========================================================"
python3 -u eval_math_[model_tag].py \
    --model_path $BASE_MODEL \
    --mode       baseline \
    --n_samples  200 \
    --out        $BASE/eval_results/math_baseline_[model_tag].json
BASE_EXIT=$?
echo "Baseline eval done: $(date)"
[ $BASE_EXIT -ne 0 ] && echo "Baseline eval FAILED" && exit 1

# ── STEP 4: CoT model eval ───────────────────────────────────────────────────
echo "========================================================"
echo "Evaluating MATH CoT fine-tuned model ([BASE_MODEL_NAME])..."
echo "========================================================"
python3 -u eval_math_[model_tag].py \
    --model_path $BASE_MODEL \
    --lora_path  $BASE/lora_output_math_cot_[model_tag] \
    --mode       math_cot \
    --n_samples  200 \
    --out        $BASE/eval_results/math_cot_[model_tag].json
COT_EVAL_EXIT=$?
echo "CoT eval done: $(date)"
[ $COT_EVAL_EXIT -ne 0 ] && echo "CoT eval FAILED" && exit 1

# ── STEP 5: No-CoT model eval ────────────────────────────────────────────────
echo "========================================================"
echo "Evaluating MATH No-CoT fine-tuned model ([BASE_MODEL_NAME])..."
echo "========================================================"
python3 -u eval_math_[model_tag].py \
    --model_path $BASE_MODEL \
    --lora_path  $BASE/lora_output_math_nocot_[model_tag] \
    --mode       math_nocot \
    --n_samples  200 \
    --out        $BASE/eval_results/math_nocot_[model_tag].json
NOCOT_EVAL_EXIT=$?
echo "No-CoT eval done: $(date)"
[ $NOCOT_EVAL_EXIT -ne 0 ] && echo "No-CoT eval FAILED" && exit 1

# ── SUMMARY ──────────────────────────────────────────────────────────────────
echo "========================================================"
echo "MATH RESULTS SUMMARY ([BASE_MODEL_NAME])"
echo "========================================================"
python3 -u - << PYEOF
import json, glob, os

results_dir = "$BASE/eval_results"

print(f"\n{'MODE':<20s}  {'ACCURACY':>10s}  {'CORRECT':>10s}")
print("-" * 45)
for path in sorted(glob.glob(os.path.join(results_dir, "math_*_[model_tag].json"))):
    with open(path) as f:
        data = json.load(f)
    s = data["summary"]
    print(f"[{s['mode'].upper():<18s}]  {s['accuracy_pct']:>8.1f}%  "
          f"({s['correct']}/{s['total']})")

# Per-subject breakdown for each mode
print("\n\nPER-SUBJECT BREAKDOWN")
print("=" * 80)
for path in sorted(glob.glob(os.path.join(results_dir, "math_*_[model_tag].json"))):
    with open(path) as f:
        data = json.load(f)
    s = data["summary"]
    print(f"\n[{s['mode'].upper()}]")
    for subj, v in s.get("subject_accuracy", {}).items():
        print(f"  {subj:<30s}  {v['accuracy_pct']:5.1f}%  ({v['correct']}/{v['total']})")

# Per-level breakdown for each mode
print("\n\nPER-LEVEL BREAKDOWN")
print("=" * 80)
for path in sorted(glob.glob(os.path.join(results_dir, "math_*_[model_tag].json"))):
    with open(path) as f:
        data = json.load(f)
    s = data["summary"]
    print(f"\n[{s['mode'].upper()}]")
    for lvl, v in sorted(s.get("level_accuracy", {}).items()):
        print(f"  {lvl:<12s}  {v['accuracy_pct']:5.1f}%  ({v['correct']}/{v['total']})")
PYEOF

echo ""
echo "All done: $(date)"
