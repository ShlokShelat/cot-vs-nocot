#!/bin/bash
#SBATCH --job-name=mathinstruct_cot_nocot_[model_tag]
#SBATCH --output=[BASE_DIR]/cot_vs_nocot/logs/mathinstruct_[model_tag]_%j.out
#SBATCH --error=[BASE_DIR]/cot_vs_nocot/logs/mathinstruct_[model_tag]_%j.err
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
         $BASE/lora_output_mathinstruct_cot_[model_tag] \
         $BASE/lora_output_mathinstruct_nocot_[model_tag]

source [BASE_DIR]/[ENV_NAME]/venv/bin/activate

echo "========================================================"
echo "Job  : $SLURM_JOB_ID  Node: $HOSTNAME"
echo "GPU  : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "Start: $(date)"
echo "========================================================"

cd $BASE

export MATHINSTRUCT_COT_DIR=$BASE/lora_output_mathinstruct_cot_[model_tag]
export MATHINSTRUCT_NOCOT_DIR=$BASE/lora_output_mathinstruct_nocot_[model_tag]

# ── STEP 1: CoT fine-tuning ───────────────────────────────────────────────────
echo "========================================================"
echo "MathInstruct CoT fine-tuning ([BASE_MODEL_NAME])..."
echo "========================================================"
python3 -u finetune_mathinstruct_cot_[model_tag].py
COT_EXIT=$?
echo "CoT training exit: $COT_EXIT at $(date)"
[ $COT_EXIT -ne 0 ] && echo "MathInstruct CoT training FAILED" && exit 1

# ── STEP 2: No-CoT fine-tuning ───────────────────────────────────────────────
echo "========================================================"
echo "MathInstruct No-CoT fine-tuning ([BASE_MODEL_NAME])..."
echo "========================================================"
python3 -u finetune_mathinstruct_nocot_[model_tag].py
NOCOT_EXIT=$?
echo "No-CoT training exit: $NOCOT_EXIT at $(date)"
[ $NOCOT_EXIT -ne 0 ] && echo "MathInstruct No-CoT training FAILED" && exit 1

# ── STEP 3: Baseline eval (no LoRA) ──────────────────────────────────────────
echo "========================================================"
echo "Evaluating MathInstruct baseline (no LoRA, [BASE_MODEL_NAME])..."
echo "========================================================"
python3 -u eval_mathinstruct_[model_tag].py \
    --model_path [BASE_DIR]/[MODEL_SNAPSHOT_PATH] \
    --mode       baseline_[model_tag] \
    --n_samples  5000 \
    --out        $BASE/eval_results/mathinstruct_[model_tag]_baseline.json
BASE_EXIT=$?
echo "Baseline eval done: $(date)"
[ $BASE_EXIT -ne 0 ] && echo "Baseline eval FAILED" && exit 1

# ── STEP 4: CoT model eval ───────────────────────────────────────────────────
echo "========================================================"
echo "Evaluating MathInstruct CoT fine-tuned model ([BASE_MODEL_NAME])..."
echo "========================================================"
python3 -u eval_mathinstruct_[model_tag].py \
    --model_path [BASE_DIR]/[MODEL_SNAPSHOT_PATH] \
    --lora_path  $BASE/lora_output_mathinstruct_cot_[model_tag] \
    --mode       mathinstruct_cot_[model_tag] \
    --n_samples  5000 \
    --out        $BASE/eval_results/mathinstruct_[model_tag]_cot.json
COT_EVAL_EXIT=$?
echo "CoT eval done: $(date)"
[ $COT_EVAL_EXIT -ne 0 ] && echo "CoT eval FAILED" && exit 1

# ── STEP 5: No-CoT model eval ────────────────────────────────────────────────
echo "========================================================"
echo "Evaluating MathInstruct No-CoT fine-tuned model ([BASE_MODEL_NAME])..."
echo "========================================================"
python3 -u eval_mathinstruct_[model_tag].py \
    --model_path [BASE_DIR]/[MODEL_SNAPSHOT_PATH] \
    --lora_path  $BASE/lora_output_mathinstruct_nocot_[model_tag] \
    --mode       mathinstruct_nocot_[model_tag] \
    --n_samples  5000 \
    --out        $BASE/eval_results/mathinstruct_[model_tag]_nocot.json
NOCOT_EVAL_EXIT=$?
echo "No-CoT eval done: $(date)"
[ $NOCOT_EVAL_EXIT -ne 0 ] && echo "No-CoT eval FAILED" && exit 1

# ── SUMMARY ──────────────────────────────────────────────────────────────────
echo "========================================================"
echo "MATHINSTRUCT RESULTS SUMMARY ([BASE_MODEL_NAME])"
echo "========================================================"
python3 -u - << PYEOF
import json, glob, os

results_dir = "$BASE/eval_results"

print(f"\n{'MODE':<30s}  {'ACCURACY':>10s}  {'CORRECT':>10s}")
print("-" * 55)
for path in sorted(glob.glob(os.path.join(results_dir, "mathinstruct_[model_tag]_*.json"))):
    with open(path) as f:
        data = json.load(f)
    s = data["summary"]
    print(f"[{s['mode'].upper():<28s}]  {s['accuracy_pct']:>8.1f}%  "
          f"({s['correct']}/{s['total']})")

# CoT vs PoT breakdown for each mode
print("\n\nCOT vs POT BREAKDOWN")
print("=" * 80)
for path in sorted(glob.glob(os.path.join(results_dir, "mathinstruct_[model_tag]_*.json"))):
    with open(path) as f:
        data = json.load(f)
    s = data["summary"]
    print(f"\n[{s['mode'].upper()}]")
    for t, v in sorted(s.get("type_accuracy", {}).items()):
        print(f"  {t:<6s}  {v['accuracy_pct']:5.1f}%  ({v['correct']}/{v['total']})")

# Per-source breakdown for each mode
print("\n\nPER-SOURCE BREAKDOWN")
print("=" * 80)
for path in sorted(glob.glob(os.path.join(results_dir, "mathinstruct_[model_tag]_*.json"))):
    with open(path) as f:
        data = json.load(f)
    s = data["summary"]
    print(f"\n[{s['mode'].upper()}]")
    for src, v in sorted(s.get("source_accuracy", {}).items(),
                         key=lambda x: -x[1]["total"]):
        print(f"  {src:<45s}  {v['accuracy_pct']:5.1f}%  ({v['correct']}/{v['total']})")
PYEOF

echo ""
echo "All done: $(date)"
