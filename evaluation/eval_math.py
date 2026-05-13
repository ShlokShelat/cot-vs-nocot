"""
eval_math.py
===================
Evaluation script for Hendrycks MATH dataset.
Supports baseline (no LoRA) and fine-tuned (with LoRA) evaluation.
Evaluates on the official lighteval/MATH test split (5000 examples).
Reports overall accuracy and per-subject / per-level breakdowns.

Dataset: DigitalLearningGmbH/MATH-lighteval (pre-downloaded to disk)

Usage:
  python3 eval_math_llama.py \\
      --model_path [FILE_PATH]/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659 \\
      --mode       baseline \\
      --n_samples  200 \\
      --out        eval_results/math_baseline_llama8b.json

  python3 eval_math_llama.py \\
      --model_path [FILE_PATH]/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659 \\
      --lora_path  lora_output_math_cot_llama8b \\
      --mode       math_cot \\
      --n_samples  200 \\
      --out        eval_results/math_cot_llama8b.json
"""

import re
import json
import time
import random
import argparse
import os
from collections import defaultdict
from datetime import datetime

import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

MATH_PATH = "[FILE_PATH]/datasets/lighteval_MATH"

# ===================
# ANSWER UTILS
# ===================

def extract_boxed_answer(text):
    """
    Extract content from the last \\boxed{...} in text.
    Handles nested braces correctly.
    """
    pattern = r"\\boxed\{"
    matches = list(re.finditer(pattern, text))
    if not matches:
        return None
    last_match = matches[-1]
    start = last_match.end()
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    return text[start:i - 1].strip()


def extract_final_answer(model_output):
    """
    Extract the answer from model output.
    Priority:
      1. Content after our explicit 'FINAL ANSWER:' marker
      2. Last \\boxed{} in the output (model may naturally use LaTeX)
      3. Last number found in output
    """
    # 1. Explicit marker
    m = re.search(r"FINAL ANSWER:\s*(.+?)(?:\n|$)", model_output, re.IGNORECASE)
    if m:
        candidate = m.group(1).strip()
        boxed = extract_boxed_answer(candidate)
        return boxed if boxed else candidate

    # 2. Last \\boxed{} in the raw output
    boxed = extract_boxed_answer(model_output)
    if boxed:
        return boxed

    # 3. Fallback: last number
    numbers = re.findall(r"-?\d+\.?\d*", model_output)
    if numbers:
        return numbers[-1]

    return model_output.strip()[-200:]


def _frac_to_float(s):
    """Convert a simple 'a/b' fraction string to float. Returns None if not a fraction."""
    m = re.match(r"^(-?\d+)/(-?\d+)$", s)
    if m:
        denom = int(m.group(2))
        if denom == 0:
            return None
        return int(m.group(1)) / denom
    return None


def normalize_answer(answer):
    """
    Normalize a LaTeX math answer for comparison.
    Strips whitespace, LaTeX wrappers, and common formatting differences.
    """
    answer = str(answer).strip()
    answer = re.sub(r"\$+", "", answer)
    answer = re.sub(r"\\left|\\right", "", answer)
    answer = re.sub(r"\s+", "", answer)
    answer = answer.lower()
    try:
        f = float(answer)
        if not (f != f or abs(f) == float("inf")):
            if f == int(f):
                answer = str(int(f))
            else:
                answer = str(f)
    except ValueError:
        pass
    return answer


def answers_match(predicted, ground_truth):
    """
    Check if predicted answer matches ground truth.
    Tries: exact normalized string match, numeric match, fraction match.
    """
    pred = normalize_answer(predicted)
    gt   = normalize_answer(ground_truth)

    if pred == gt:
        return True

    try:
        return abs(float(pred) - float(gt)) < 1e-6
    except ValueError:
        pass

    pf = _frac_to_float(pred)
    gf = _frac_to_float(gt)
    if pf is not None and gf is not None:
        return abs(pf - gf) < 1e-6
    if pf is not None:
        try:
            return abs(pf - float(gt)) < 1e-6
        except ValueError:
            pass
    if gf is not None:
        try:
            return abs(gf - float(pred)) < 1e-6
        except ValueError:
            pass

    return False


# ===================
# MODEL LOADING
# ===================

def load_model_and_tokenizer(model_path, lora_path=None):
    print(f"\nLoading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    if lora_path:
        from peft import PeftModel
        print(f"Loading LoRA from {lora_path}...")
        model = PeftModel.from_pretrained(model, lora_path)
        model = model.merge_and_unload()
        print("LoRA merged.")

    model.eval()
    print("Model ready.\n")
    return tokenizer, model


# ===================
# INFERENCE
# ===================

SYSTEM_PROMPT = (
    "You are a helpful math assistant. "
    "Solve the problem step by step, then give the final answer clearly at the end. "
    "End your response with: FINAL ANSWER: <your answer>"
)

MAX_INPUT_LEN  = 2048
MAX_NEW_TOKENS = 1024


@torch.no_grad()
def generate_answer(tokenizer, model, problem, max_new_tokens):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": problem},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=MAX_INPUT_LEN
    ).to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    new_ids = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True)


# ===================
# DATASET LOADER
# ===================

def load_math_test(n_samples, seed):
    """
    Load the MATH test split from disk.
    Shuffle with fixed seed then optionally subsample.
    Returns list of dicts with: problem, ground_truth, subject, level.
    """
    print(f"Loading MATH test split from {MATH_PATH}/test ...")
    ds = load_from_disk(f"{MATH_PATH}/test")
    ds = ds.shuffle(seed=seed)
    if n_samples and n_samples < len(ds):
        ds = ds.select(range(n_samples))
    print(f"  Loaded {len(ds)} examples")
    return [
        {
            "problem":      ex["problem"],
            "ground_truth": extract_boxed_answer(ex["solution"]) or ex["solution"],
            "subject":      ex.get("type", ""),
            "level":        ex.get("level", ""),
        }
        for ex in ds
    ]


# ===================
# EVALUATION LOOP
# ===================

def evaluate(args):
    random.seed(args.seed)

    examples = load_math_test(args.n_samples, args.seed)

    print(f"Evaluating {len(examples)} examples | dataset=math | mode={args.mode}")

    tokenizer, model = load_model_and_tokenizer(args.model_path, args.lora_path)

    results = []
    correct = 0

    subject_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    level_stats   = defaultdict(lambda: {"correct": 0, "total": 0})

    for i, ex in enumerate(tqdm(examples, desc=f"Eval [{args.mode}]")):
        t0           = time.time()
        model_output = generate_answer(tokenizer, model, ex["problem"], MAX_NEW_TOKENS)
        elapsed      = time.time() - t0
        predicted    = extract_final_answer(model_output)
        is_correct   = answers_match(predicted, ex["ground_truth"])

        if is_correct:
            correct += 1

        subject = ex["subject"]
        level   = ex["level"]
        subject_stats[subject]["total"]   += 1
        level_stats[level]["total"]       += 1
        if is_correct:
            subject_stats[subject]["correct"] += 1
            level_stats[level]["correct"]     += 1

        results.append({
            "index":        i,
            "problem":      ex["problem"][:300],
            "ground_truth": ex["ground_truth"],
            "model_output": model_output[:600],
            "predicted":    predicted,
            "correct":      is_correct,
            "subject":      subject,
            "level":        level,
            "time_sec":     round(elapsed, 2),
        })

        if (i + 1) % 10 == 0 or i == 0:
            acc  = correct / (i + 1) * 100
            mark = "CORRECT" if is_correct else "WRONG"
            print(f"  [{i+1:4d}/{len(examples)}]  Acc: {acc:5.1f}%  |  {mark}  "
                  f"({elapsed:.1f}s)  [{subject} | {level}]")

    accuracy = correct / len(examples) * 100

    subject_accuracy = {
        subj: {
            "correct": v["correct"],
            "total":   v["total"],
            "accuracy_pct": round(v["correct"] / v["total"] * 100, 2) if v["total"] else 0,
        }
        for subj, v in sorted(subject_stats.items())
    }
    level_accuracy = {
        lvl: {
            "correct": v["correct"],
            "total":   v["total"],
            "accuracy_pct": round(v["correct"] / v["total"] * 100, 2) if v["total"] else 0,
        }
        for lvl, v in sorted(level_stats.items())
    }

    summary = {
        "mode":             args.mode,
        "dataset":          "DigitalLearningGmbH/MATH-lighteval",
        "model_path":       args.model_path,
        "lora_path":        args.lora_path,
        "n_samples":        len(examples),
        "correct":          correct,
        "total":            len(examples),
        "accuracy_pct":     round(accuracy, 2),
        "subject_accuracy": subject_accuracy,
        "level_accuracy":   level_accuracy,
        "timestamp":        datetime.now().isoformat(),
    }

    print(f"\n{'=' * 60}")
    print(f"  Dataset  : DigitalLearningGmbH/MATH-lighteval")
    print(f"  Mode     : {args.mode}")
    print(f"  Accuracy : {accuracy:.2f}%  ({correct}/{len(examples)})")
    print(f"{'=' * 60}")

    print("\n  By subject:")
    for subj, v in subject_accuracy.items():
        print(f"    {subj:<30s}  {v['accuracy_pct']:5.1f}%  ({v['correct']}/{v['total']})")

    print("\n  By difficulty level:")
    for lvl, v in sorted(level_accuracy.items()):
        print(f"    {lvl:<12s}  {v['accuracy_pct']:5.1f}%  ({v['correct']}/{v['total']})")

    print(f"{'=' * 60}\n")

    os.makedirs(
        os.path.dirname(args.out) if os.path.dirname(args.out) else ".",
        exist_ok=True,
    )
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)
    print(f"Saved → {args.out}")


# ===================
# ENTRY POINT
# ===================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="[FILE_PATH]/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659")
    ap.add_argument("--lora_path",  default=None,
                    help="Path to LoRA adapter directory. Omit for baseline.")
    ap.add_argument("--mode",       default="baseline",
                    help="Label written to results JSON: baseline | math_cot | math_nocot")
    ap.add_argument("--n_samples",  type=int, default=None,
                    help="Number of test examples to evaluate. None = full 5000.")
    ap.add_argument("--seed",       type=int, default=42)
    ap.add_argument("--out",        default="eval_results/math_results_llama8b.json")
    args = ap.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
