#!/bin/bash
#SBATCH --job-name=nfa_v2_cot_[model]
#SBATCH --output=[FILE_PATH]/cot_vs_nocot/new_dataset_nfa/logs/nfa_v2_cot_[model]_%j.out
#SBATCH --error=[FILE_PATH]/cot_vs_nocot/new_dataset_nfa/logs/nfa_v2_cot_[model]_%j.err
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --mem=120G
#SBATCH --time=168:00:00
#SBATCH --qos=limit_gpu_qos

BASE=[FILE_PATH]/cot_vs_nocot/new_dataset_nfa
mkdir -p $BASE/logs $BASE/lora_output_nfa_v2_cot_[model_tag] $BASE/eval_results

source [FILE_PATH]/qwen/venv/bin/activate

echo "========================================================"
echo "Job  : $SLURM_JOB_ID  Node: $HOSTNAME"
echo "GPU  : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "Model: [MODEL_NAME]"
echo "Mode : NFA-V2-COT"
echo "Start: $(date)"
echo "========================================================"

cd $BASE

# ── STEP 1: Generate dataset ──────────────────────────────────────────────────
SENTINEL=$BASE/.datagen_done
if [ ! -f "$SENTINEL" ]; then
    echo "Generating 5-tier NFA v2 dataset..."
    python3 -u [COMPANION_SCRIPT] \
        --n 25000 --seed 42 --out_prefix nfa_v2 --test_per_tier 200
    [ $? -ne 0 ] && echo "Dataset generation FAILED" && exit 1
    touch $SENTINEL
    echo "Dataset done: $(date)"
else
    echo "Dataset already exists, skipping."
fi

# ── STEP 2: CoT fine-tuning ───────────────────────────────────────────────────
echo "========================================================"
echo "CoT fine-tuning ([MODEL_NAME])..."
echo "========================================================"
python3 -u finetune_nfa_v2_cot_[model].py
[ $? -ne 0 ] && echo "CoT training FAILED" && exit 1
echo "CoT training done: $(date)"

# ── STEP 3: Baseline eval — IID ──────────────────────────────────────────────
echo "========================================================"
echo "Evaluating BASELINE IID..."
echo "========================================================"
python3 -u evaluate_nfa_v2_[model].py \
    --model_path [FILE_PATH]/[MODEL_PROVIDER]--[MODEL_NAME]/snapshots/[MODEL_SNAPSHOT] \
    --data_file  nfa_v2_nocot_test.jsonl \
    --mode       baseline \
    --out        eval_results/results_baseline_iid_[model_tag].json
echo "Baseline IID done: $(date)"

# ── STEP 4: Baseline eval — OOD ──────────────────────────────────────────────
echo "========================================================"
echo "Evaluating BASELINE OOD..."
echo "========================================================"
python3 -u evaluate_nfa_v2_[model].py \
    --model_path [FILE_PATH]/[MODEL_PROVIDER]--[MODEL_NAME]/snapshots/[MODEL_SNAPSHOT] \
    --data_file  nfa_v2_nocot_ood_test.jsonl \
    --mode       baseline \
    --ood \
    --out        eval_results/results_baseline_ood_[model_tag].json
echo "Baseline OOD done: $(date)"

# ── STEP 5: CoT fine-tuned eval — IID ────────────────────────────────────────
echo "========================================================"
echo "Evaluating CoT IID..."
echo "========================================================"
python3 -u evaluate_nfa_v2_[model].py \
    --model_path [FILE_PATH]/[MODEL_PROVIDER]--[MODEL_NAME]/snapshots/[MODEL_SNAPSHOT] \
    --lora_path  lora_output_nfa_v2_cot_[model_tag] \
    --data_file  nfa_v2_cot_test.jsonl \
    --mode       cot \
    --out        eval_results/results_cot_iid_[model_tag].json
echo "CoT IID done: $(date)"

# ── STEP 6: CoT fine-tuned eval — OOD ────────────────────────────────────────
echo "========================================================"
echo "Evaluating CoT OOD..."
echo "========================================================"
python3 -u evaluate_nfa_v2_[model].py \
    --model_path [FILE_PATH]/[MODEL_PROVIDER]--[MODEL_NAME]/snapshots/[MODEL_SNAPSHOT] \
    --lora_path  lora_output_nfa_v2_cot_[model_tag] \
    --data_file  nfa_v2_cot_ood_test.jsonl \
    --mode       cot \
    --ood \
    --out        eval_results/results_cot_ood_[model_tag].json
echo "CoT OOD done: $(date)"

# ── SUMMARY ──────────────────────────────────────────────────────────────────
echo "========================================================"
echo "SUMMARY (CoT job — [MODEL_NAME])"
echo "========================================================"
python3 -u - << 'PYEOF'
import json, glob, os
tier_labels = {"1":"trivial","2":"easy","3":"medium","4":"hard","5":"very hard"}
for path in sorted(glob.glob("eval_results/results_*_iid_[model_tag].json") +
                   glob.glob("eval_results/results_*_ood_[model_tag].json")):
    if not os.path.exists(path): continue
    with open(path) as f: data = json.load(f)
    s = data["summary"]
    tag = "OOD" if s.get("ood") else "IID"
    print(f"\n[{s['mode'].upper()} | {tag}]  Overall: {s['accuracy']*100:.1f}%  ({s['n_correct']}/{s['n_total']})")
    for t, stats in sorted(s.get("per_tier",{}).items()):
        label = tier_labels.get(t,"")
        print(f"  Tier {t} ({label:9s}): {stats['accuracy']*100:.1f}%  ({stats['correct']}/{stats['total']})")
PYEOF

echo "All done: $(date)"
