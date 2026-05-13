"""
eval_mathinstruct.py
==========================
Evaluation script for TIGER-Lab/MathInstruct dataset.
Model   : meta-llama/Llama-3.1-8B-Instruct
Supports baseline (no LoRA) and fine-tuned (with LoRA) evaluation.

Split strategy (fixed seed=42, MUST match finetune scripts exactly):
  shuffle(seed=42)
  [0      :   5000] → held-out test set   ← evaluated here
  [5000   : 262000] → training set        (never evaluated here)

Reports overall accuracy and per-source / per-type (CoT vs PoT) breakdowns.

Usage:
  # Baseline (no LoRA)
  python3 eval_mathinstruct_llama.py \
      --model_path [FILE_PATH]/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659 \
      --mode       baseline_llama \
      --n_samples  5000 \
      --out        eval_results/mathinstruct_llama_baseline.json

  # CoT fine-tuned
  python3 eval_mathinstruct_llama.py \
      --model_path [FILE_PATH]/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659 \
      --lora_path  lora_output_mathinstruct_cot_llama \
      --mode       mathinstruct_cot_llama \
      --n_samples  5000 \
      --out        eval_results/mathinstruct_llama_cot.json

  # No-CoT fine-tuned
  python3 eval_mathinstruct_llama.py \
      --model_path [FILE_PATH]/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659 \
      --lora_path  lora_output_mathinstruct_nocot_llama \
      --mode       mathinstruct_nocot_llama \
      --n_samples  5000 \
      --out        eval_results/mathinstruct_llama_nocot.json
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

# Fixed constants — must match finetune scripts exactly
MATHINSTRUCT_PATH = "[FILE_PATH]/datasets/MathInstruct/train"
_TEST_HOLD_OUT    = 5000   # first N examples after shuffle(seed=42) = test set


# ===================
# ANSWER UTILS
# ===================

def extract_answer_from_output(output_str):
    """
    Extract the final answer from a MathInstruct output field.
    Used to get the ground-truth answer from the dataset's output field.

    Priority:
      1. "The answer is X"   (CoT entries)
      2. \\boxed{X}          (MATH-sourced entries)
      3. Last print() arg    (PoT entries)
      4. Last number         (fallback)
    """
    output_str = output_str.strip()

    # 1. "The answer is X"
    m = re.search(
        r"[Tt]he answer is\s*:?\s*\(?([A-Za-z0-9\s\+\-\*/\.\,\/\\\$\%\^\{\}\_\(\)]+?)\)?\.?\s*$",
        output_str,
        re.MULTILINE,
    )
    if m:
        return m.group(1).strip()

    # 2. \boxed{}
    pattern = r"\\boxed\{"
    matches = list(re.finditer(pattern, output_str))
    if matches:
        last_match = matches[-1]
        start = last_match.end()
        depth = 1
        i = start
        while i < len(output_str) and depth > 0:
            if output_str[i] == "{":
                depth += 1
            elif output_str[i] == "}":
                depth -= 1
            i += 1
        return output_str[start:i - 1].strip()

    # 3. PoT: last print() argument
    pot_match = re.findall(r"print\((.+?)\)", output_str)
    if pot_match:
        return pot_match[-1].strip()

    # 4. Fallback: last number
    numbers = re.findall(r"-?\d+\.?\d*", output_str)
    if numbers:
        return numbers[-1]

    return output_str[-100:].strip()


def _extract_boxed(text):
    """Extract content from last \\boxed{...} in text. Returns None if absent."""
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
      2. "The answer is X" pattern
      3. Last \\boxed{} in the output
      4. Last number found
    """
    # 1. Explicit FINAL ANSWER marker (our fine-tuning format)
    m = re.search(r"FINAL ANSWER:\s*(.+?)(?:\n|$)", model_output, re.IGNORECASE)
    if m:
        candidate = m.group(1).strip()
        boxed = _extract_boxed(candidate)
        return boxed if boxed else candidate

    # 2. "The answer is X"
    m2 = re.search(
        r"[Tt]he answer is\s*:?\s*\(?([A-Za-z0-9\s\+\-\*/\.\,\/\\\$\%\^\{\}\_\(\)]+?)\)?\.?\s*$",
        model_output,
        re.MULTILINE,
    )
    if m2:
        return m2.group(1).strip()

    # 3. Last \boxed{}
    boxed = _extract_boxed(model_output)
    if boxed:
        return boxed

    # 4. Fallback: last number
    numbers = re.findall(r"-?\d+\.?\d*", model_output)
    if numbers:
        return numbers[-1]

    return model_output.strip()[-200:]


def _frac_to_float(s):
    """Convert simple 'a/b' fraction string to float. Returns None if not a fraction."""
    m = re.match(r"^(-?\d+)/(-?\d+)$", s)
    if m:
        denom = int(m.group(2))
        if denom == 0:
            return None
        return int(m.group(1)) / denom
    return None


def normalize_answer(answer):
    """Normalize a math answer for comparison."""
    answer = str(answer).strip()
    answer = re.sub(r"\$+", "", answer)
    answer = re.sub(r"\\left|\\right", "", answer)
    answer = re.sub(r"\s+", "", answer)
    answer = answer.lower()
    # Strip surrounding parentheses for multiple-choice like "(A)" → "A"
    answer = re.sub(r"^\(([a-e])\)$", r"\1", answer)
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
    Tries: exact normalized match, numeric match, fraction match.
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
MAX_NEW_TOKENS = 512   # MathInstruct answers are shorter than MATH competition


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

def load_mathinstruct_test(n_samples, seed):
    """
    Load the held-out test set from disk.

    Carving logic (MUST match finetune scripts exactly):
      1. Load full dataset from disk (262k)
      2. shuffle(seed=42)
      3. [0 : _TEST_HOLD_OUT]   → test set  (5,000 examples)
      4. [_TEST_HOLD_OUT : end] → train set  (never loaded here)

    n_samples: how many of the 5000 test examples to evaluate.
               Pass None or 5000 to use all 5000.
    """
    print(f"Loading MathInstruct from {MATHINSTRUCT_PATH}...")
    full_ds = load_from_disk(MATHINSTRUCT_PATH)
    print(f"  Full dataset: {len(full_ds)} examples")

    # Same shuffle as finetune scripts — seed MUST match
    full_ds = full_ds.shuffle(seed=seed)

    # Take only the held-out test portion
    test_pool = full_ds.select(range(_TEST_HOLD_OUT))

    # Optionally subsample
    if n_samples and n_samples < len(test_pool):
        test_pool = test_pool.select(range(n_samples))

    print(f"  Test examples to evaluate: {len(test_pool)}")

    examples = []
    for ex in test_pool:
        gt = extract_answer_from_output(ex["output"])
        source     = ex.get("source", "unknown")
        entry_type = "PoT" if "/PoT/" in source else "CoT"
        examples.append({
            "problem":      ex["instruction"],
            "ground_truth": gt,
            "source":       source,
            "entry_type":   entry_type,
            "full_output":  ex["output"],
        })

    return examples


# ===================
# EVALUATION LOOP
# ===================

def evaluate(args):
    random.seed(args.seed)

    examples = load_mathinstruct_test(args.n_samples, args.seed)

    print(f"Evaluating {len(examples)} examples | dataset=MathInstruct | mode={args.mode}")

    tokenizer, model = load_model_and_tokenizer(args.model_path, args.lora_path)

    results      = []
    correct      = 0
    source_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    type_stats   = defaultdict(lambda: {"correct": 0, "total": 0})

    for i, ex in enumerate(tqdm(examples, desc=f"Eval [{args.mode}]")):
        t0           = time.time()
        model_output = generate_answer(tokenizer, model, ex["problem"], MAX_NEW_TOKENS)
        elapsed      = time.time() - t0
        predicted    = extract_final_answer(model_output)
        is_correct   = answers_match(predicted, ex["ground_truth"])

        if is_correct:
            correct += 1

        source     = ex["source"]
        entry_type = ex["entry_type"]

        source_stats[source]["total"] += 1
        type_stats[entry_type]["total"] += 1
        if is_correct:
            source_stats[source]["correct"] += 1
            type_stats[entry_type]["correct"] += 1

        results.append({
            "index":        i,
            "problem":      ex["problem"][:300],
            "ground_truth": ex["ground_truth"],
            "model_output": model_output[:600],
            "predicted":    predicted,
            "correct":      is_correct,
            "source":       source,
            "entry_type":   entry_type,
            "time_sec":     round(elapsed, 2),
        })

        if (i + 1) % 10 == 0 or i == 0:
            acc  = correct / (i + 1) * 100
            mark = "CORRECT" if is_correct else "WRONG"
            print(f"  [{i+1:4d}/{len(examples)}]  Acc: {acc:5.1f}%  |  {mark}  "
                  f"({elapsed:.1f}s)  [{entry_type} | {source}]")

    accuracy = correct / len(examples) * 100

    source_accuracy = {
        src: {
            "correct":      v["correct"],
            "total":        v["total"],
            "accuracy_pct": round(v["correct"] / v["total"] * 100, 2) if v["total"] else 0,
        }
        for src, v in sorted(source_stats.items())
    }

    type_accuracy = {
        t: {
            "correct":      v["correct"],
            "total":        v["total"],
            "accuracy_pct": round(v["correct"] / v["total"] * 100, 2) if v["total"] else 0,
        }
        for t, v in sorted(type_stats.items())
    }

    summary = {
        "mode":            args.mode,
        "dataset":         "TIGER-Lab/MathInstruct",
        "model":           "meta-llama/Llama-3.1-8B-Instruct",
        "test_split":      f"held-out indices 0-{_TEST_HOLD_OUT-1} after shuffle seed={args.seed}",
        "model_path":      args.model_path,
        "lora_path":       args.lora_path,
        "n_samples":       len(examples),
        "correct":         correct,
        "total":           len(examples),
        "accuracy_pct":    round(accuracy, 2),
        "source_accuracy": source_accuracy,
        "type_accuracy":   type_accuracy,
        "timestamp":       datetime.now().isoformat(),
    }

    print(f"\n{'=' * 65}")
    print(f"  Dataset  : TIGER-Lab/MathInstruct")
    print(f"  Model    : meta-llama/Llama-3.1-8B-Instruct")
    print(f"  Test set : held-out indices 0-{_TEST_HOLD_OUT-1} (seed={args.seed}, no leakage)")
    print(f"  Mode     : {args.mode}")
    print(f"  Accuracy : {accuracy:.2f}%  ({correct}/{len(examples)})")
    print(f"{'=' * 65}")

    print("\n  By entry type (CoT vs PoT):")
    for t, v in type_accuracy.items():
        print(f"    {t:<6s}  {v['accuracy_pct']:5.1f}%  ({v['correct']}/{v['total']})")

    print("\n  By source (sorted by volume):")
    for src, v in sorted(source_accuracy.items(), key=lambda x: -x[1]["total"]):
        print(f"    {src:<45s}  {v['accuracy_pct']:5.1f}%  ({v['correct']}/{v['total']})")

    print(f"{'=' * 65}\n")

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
    ap.add_argument("--model_path",
                    default="[FILE_PATH]/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659")
    ap.add_argument("--lora_path",  default=None,
                    help="Path to LoRA adapter directory. Omit for baseline.")
    ap.add_argument("--mode",       default="baseline_llama",
                    help="Label for results JSON: baseline_llama | mathinstruct_cot_llama | mathinstruct_nocot_llama")
    ap.add_argument("--n_samples",  type=int, default=None,
                    help="Number of test examples to evaluate. None = all 5000.")
    ap.add_argument("--seed",       type=int, default=42,
                    help="Must match the seed used in fine-tuning scripts (default: 42).")
    ap.add_argument("--out",        default="eval_results/mathinstruct_llama_results.json")
    args = ap.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
