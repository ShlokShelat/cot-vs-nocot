#!/bin/bash
#SBATCH --job-name=dfa_nocot
#SBATCH --output=/home/[USER]/cot_vs_nocot/new_dataset_nfa/logs/dfa_nocot_%j.out
#SBATCH --error=/home/[USER]/cot_vs_nocot/new_dataset_nfa/logs/dfa_nocot_%j.err
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --mem=120G
#SBATCH --time=168:00:00
#SBATCH --qos=limit_gpu_qos

BASE=/home/[USER]/cot_vs_nocot/new_dataset_nfa
mkdir -p $BASE/logs $BASE/lora_output_dfa_nocot $BASE/eval_results

source /home/[USER]/qwen/venv/bin/activate

echo "========================================================"
echo "Job  : $SLURM_JOB_ID  Node: $HOSTNAME"
echo "GPU  : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "Mode : DFA-NOCOT"
echo "Start: $(date)"
echo "========================================================"

cd $BASE

# ── STEP 1: Generate DFA dataset ─────────────────────────────────────────────
SENTINEL=$BASE/.dfa_datagen_done
WAIT=0
while [ ! -f "$SENTINEL" ] && [ $WAIT -lt 3600 ]; do
    sleep 30; WAIT=$((WAIT+30))
    echo "  Waiting for DFA dataset sentinel... (${WAIT}s elapsed)"
done
if [ ! -f "$SENTINEL" ]; then
    echo "Dataset not ready after 1h -- generating now..."
    python3 -u dfa_parallel_dataset_gen.py \
        --n 25000 --seed 42 --out_prefix dfa --test_per_tier 200
    [ $? -ne 0 ] && echo "Dataset generation FAILED" && exit 1
    touch $SENTINEL
fi

# ── STEP 2: CoT fine-tuning ───────────────────────────────────────────────────
echo "========================================================"
echo "DFA No-CoT fine-tuning..."
echo "========================================================"
python3 -u finetune_dfa_nocot.py
[ $? -ne 0 ] && echo "CoT training FAILED" && exit 1
echo "CoT training done: $(date)"

# ── STEP 3: Baseline eval IID ─────────────────────────────────────────────────
echo "========================================================"
echo "Evaluating BASELINE IID..."
echo "========================================================"
python3 -u evaluate_dfa.py \
    --model_path /path/to/[MODEL_NAME] \
    --data_file  dfa_nocot_test.jsonl \
    --mode       baseline \
    --out        eval_results/dfa_baseline_iid.json
echo "Baseline IID done: $(date)"

# ── STEP 4: Baseline eval OOD ─────────────────────────────────────────────────
echo "========================================================"
echo "Evaluating BASELINE OOD..."
echo "========================================================"
python3 -u evaluate_dfa.py \
    --model_path /path/to/[MODEL_NAME] \
    --data_file  dfa_nocot_ood_test.jsonl \
    --mode       baseline \
    --ood \
    --out        eval_results/dfa_baseline_ood.json
echo "Baseline OOD done: $(date)"

# ── STEP 5: No-CoT fine-tuned eval IID ───────────────────────────────────────
echo "========================================================"
echo "Evaluating No-CoT IID..."
echo "========================================================"
python3 -u evaluate_dfa.py \
    --model_path /path/to/[MODEL_NAME] \
    --lora_path  lora_output_dfa_nocot \
    --data_file  dfa_nocot_test.jsonl \
    --mode       nocot \
    --out        eval_results/dfa_nocot_iid.json
echo "CoT IID done: $(date)"

# ── STEP 6: No-CoT fine-tuned eval OOD ───────────────────────────────────────
echo "========================================================"
echo "Evaluating No-CoT OOD..."
echo "========================================================"
python3 -u evaluate_dfa.py \
    --model_path /path/to/[MODEL_NAME] \
    --lora_path  lora_output_dfa_nocot \
    --data_file  dfa_nocot_ood_test.jsonl \
    --mode       nocot \
    --ood \
    --out        eval_results/dfa_nocot_ood.json
echo "CoT OOD done: $(date)"

# ── SUMMARY ──────────────────────────────────────────────────────────────────
echo "========================================================"
echo "SUMMARY (DFA No-CoT job)"
echo "========================================================"
python3 -u - << 'PYEOF'
import json, glob, os
tier_labels = {"1":"simple","2":"easy","3":"medium","4":"hard"}
for path in sorted(glob.glob("eval_results/dfa_*_iid.json") +
                   glob.glob("eval_results/dfa_*_ood.json")):
    if not os.path.exists(path): continue
    with open(path) as f: data = json.load(f)
    s = data["summary"]
    tag = "OOD" if s.get("ood") else "IID"
    print(f"\n[{s['mode'].upper()} | {tag}]  Overall: {s['accuracy']*100:.1f}%  ({s['n_correct']}/{s['n_total']})")
    for t, stats in sorted(s.get("per_tier",{}).items()):
        label = tier_labels.get(t,"")
        print(f"  Tier {t} ({label:8s}): {stats['accuracy']*100:.1f}%  ({stats['correct']}/{stats['total']})")
PYEOF

echo "All done: $(date)"
