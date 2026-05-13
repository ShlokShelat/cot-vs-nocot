#!/bin/bash
#SBATCH --job-name=nfa_v2_nocot_llama
#SBATCH --output=[BASE_DIR]/cot_vs_nocot/new_dataset_nfa/logs/nfa_v2_nocot_llama_%j.out
#SBATCH --error=[BASE_DIR]/cot_vs_nocot/new_dataset_nfa/logs/nfa_v2_nocot_llama_%j.err
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --mem=120G
#SBATCH --time=168:00:00
#SBATCH --qos=limit_gpu_qos

BASE=[BASE_DIR]/cot_vs_nocot/new_dataset_nfa
mkdir -p $BASE/logs $BASE/lora_output_nfa_v2_nocot_[MODEL_NAME] $BASE/eval_results

source [BASE_DIR]/[ENV_NAME]/venv/bin/activate

echo "========================================================"
echo "Job  : $SLURM_JOB_ID  Node: $HOSTNAME"
echo "GPU  : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "Model: [BASE_MODEL_NAME]"
echo "Mode : NFA-V2-NOCOT"
echo "Start: $(date)"
echo "========================================================"

cd $BASE

# ── STEP 1: Wait for dataset ──────────────────────────────────────────────────
SENTINEL=$BASE/.datagen_done
WAIT=0
while [ ! -f "$SENTINEL" ] && [ $WAIT -lt 3600 ]; do
    sleep 30; WAIT=$((WAIT+30))
    echo "  Waiting for dataset... (${WAIT}s elapsed)"
done
if [ ! -f "$SENTINEL" ]; then
    echo "Dataset not found — generating now..."
    python3 -u nfa_parallel_dataset_gen_v2.py \
        --n 25000 --seed 42 --out_prefix nfa_v2 --test_per_tier 200
    [ $? -ne 0 ] && echo "Dataset generation FAILED" && exit 1
    touch $SENTINEL
fi

# ── STEP 2: No-CoT fine-tuning ───────────────────────────────────────────────
echo "========================================================"
echo "No-CoT fine-tuning ([BASE_MODEL_NAME])..."
echo "========================================================"
python3 -u finetune_nfa_v2_nocot_[model_tag].py
[ $? -ne 0 ] && echo "No-CoT training FAILED" && exit 1
echo "No-CoT training done: $(date)"

# ── STEP 3: Baseline eval — IID ──────────────────────────────────────────────
echo "========================================================"
echo "Evaluating BASELINE IID..."
echo "========================================================"
python3 -u evaluate_nfa_v2_[model_tag]_nocot.py \
    --model_path [BASE_DIR]/[MODEL_SNAPSHOT_PATH] \
    --data_file  nfa_v2_nocot_test.jsonl \
    --mode       baseline \
    --out        eval_results/results_baseline_iid_[model_tag]_nocot.json
echo "Baseline IID done: $(date)"

# ── STEP 4: Baseline eval — OOD ──────────────────────────────────────────────
echo "========================================================"
echo "Evaluating BASELINE OOD..."
echo "========================================================"
python3 -u evaluate_nfa_v2_[model_tag]_nocot.py \
    --model_path [BASE_DIR]/[MODEL_SNAPSHOT_PATH] \
    --data_file  nfa_v2_nocot_ood_test.jsonl \
    --mode       baseline \
    --ood \
    --out        eval_results/results_baseline_ood_[model_tag]_nocot.json
echo "Baseline OOD done: $(date)"

# ── STEP 5: No-CoT fine-tuned eval — IID ─────────────────────────────────────
echo "========================================================"
echo "Evaluating No-CoT IID..."
echo "========================================================"
python3 -u evaluate_nfa_v2_[model_tag]_nocot.py \
    --model_path [BASE_DIR]/[MODEL_SNAPSHOT_PATH] \
    --lora_path  lora_output_nfa_v2_nocot_[model_tag] \
    --data_file  nfa_v2_nocot_test.jsonl \
    --mode       nocot \
    --out        eval_results/results_nocot_iid_[model_tag].json
echo "No-CoT IID done: $(date)"

# ── STEP 6: No-CoT fine-tuned eval — OOD ─────────────────────────────────────
echo "========================================================"
echo "Evaluating No-CoT OOD..."
echo "========================================================"
python3 -u evaluate_nfa_v2_[model_tag]_nocot.py \
    --model_path [BASE_DIR]/[MODEL_SNAPSHOT_PATH] \
    --lora_path  lora_output_nfa_v2_nocot_[model_tag] \
    --data_file  nfa_v2_nocot_ood_test.jsonl \
    --mode       nocot \
    --ood \
    --out        eval_results/results_nocot_ood_[model_tag].json
echo "No-CoT OOD done: $(date)"

# ── SUMMARY ──────────────────────────────────────────────────────────────────
echo "========================================================"
echo "SUMMARY (No-CoT job — [BASE_MODEL_NAME])"
echo "========================================================"
python3 -u - << 'PYEOF'
import json, glob, os
tier_labels = {"1":"trivial","2":"easy","3":"medium","4":"hard","5":"very hard"}
for path in sorted(glob.glob("eval_results/results_*_iid_[model_tag]*.json") +
                   glob.glob("eval_results/results_*_ood_[model_tag]*.json")):
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
