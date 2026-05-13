"""
eval_llama_three_benchmarks.py
==============================
Unified evaluation script for MATH-500 and GSM8K.
Supports baseline (no LoRA) and fine-tuned (with LoRA) evaluation.

Called by submit_gsm8k_llama.sh like:
  python3 eval_llama_three_benchmarks.py \
      --model_path [FILE_PATH]/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659 \
      --lora_path  lora_output_gsm8k_cot_llama8b \   # omit for baseline
      --dataset    gsm8k \
      --split      test \
      --mode       gsm8k_cot \
      --n_samples  200 \
      --out        eval_results/gsm8k_cot_llama8b.json
"""

import re
import json
import time
import random
import argparse
import os
import torch
from datetime import datetime
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# ===================
# ANSWER UTILS
# ===================

def extract_final_answer(model_output: str) -> str:
    # Look for our explicit marker first
    m = re.search(r"FINAL ANSWER:\s*(.+?)(?:\n|$)", model_output, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # Fallback: last number in output
    numbers = re.findall(r"-?\d+\.?\d*", model_output)
    if numbers:
        return numbers[-1]
    return model_output.strip()[-100:]


def normalize_answer(answer: str) -> str:
    answer = str(answer).strip().lower()
    answer = answer.replace("$", "").replace(",", "").replace(" ", "")
    answer = answer.replace("\\", "").replace("{", "").replace("}", "")
    return answer


def answers_match(predicted: str, ground_truth: str) -> bool:
    pred = normalize_answer(predicted)
    gt   = normalize_answer(ground_truth)
    if pred == gt:
        return True
    try:
        return abs(float(pred) - float(gt)) < 1e-6
    except ValueError:
        return False


def extract_gsm8k_answer(answer_str: str) -> str:
    m = re.search(r"####\s*(-?\d+\.?\d*)", answer_str)
    return m.group(1).strip() if m else answer_str.strip()


# ===================
# MODEL LOADING
# ===================

def load_model_and_tokenizer(model_path: str, lora_path: str = None):
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

@torch.no_grad()
def generate_answer(tokenizer, model, problem: str, max_new_tokens: int) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": problem},
    ]
    text   = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                       max_length=2048).to(model.device)
    out    = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    new_ids = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True)


# ===================
# DATASET LOADERS
# ===================

def load_math_test(n_samples: int, seed: int):
    """
    Returns the held-out 50 test examples — the same slice NOT used in training.
    Training used indices 0..449 after shuffle(seed=42).
    Test uses indices 450..499 after the same shuffle.
    """
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    ds = ds.shuffle(seed=seed)
    test_ds = ds.select(range(450, len(ds)))   # last 50, matches finetune split
    if n_samples and n_samples < len(test_ds):
        test_ds = test_ds.select(range(n_samples))
    return [
        {
            "problem":      ex["problem"],
            "ground_truth": ex["answer"],
            "subject":      ex.get("subject", ""),
            "level":        ex.get("level", ""),
        }
        for ex in test_ds
    ]


def load_gsm8k_test(n_samples: int, seed: int):
    ds = load_dataset("openai/gsm8k", "main", split="test")
    ds = ds.shuffle(seed=seed)
    if n_samples and n_samples < len(ds):
        ds = ds.select(range(n_samples))
    return [
        {
            "problem":      ex["question"],
            "ground_truth": extract_gsm8k_answer(ex["answer"]),
        }
        for ex in ds
    ]


# ===================
# EVALUATION LOOP
# ===================

def evaluate(args):
    random.seed(args.seed)

    # Load examples
    if args.dataset == "math":
        examples     = load_math_test(args.n_samples, args.seed)
        max_new_tokens = 1024
    elif args.dataset == "gsm8k":
        examples     = load_gsm8k_test(args.n_samples, args.seed)
        max_new_tokens = 512
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    print(f"Evaluating {len(examples)} examples | dataset={args.dataset} | mode={args.mode}")

    tokenizer, model = load_model_and_tokenizer(args.model_path, args.lora_path)

    results = []
    correct = 0

    for i, ex in enumerate(tqdm(examples, desc=f"Eval [{args.mode}]")):
        t0           = time.time()
        model_output = generate_answer(tokenizer, model, ex["problem"], max_new_tokens)
        elapsed      = time.time() - t0
        predicted    = extract_final_answer(model_output)
        is_correct   = answers_match(predicted, ex["ground_truth"])
        if is_correct:
            correct += 1

        results.append({
            "index":        i,
            "problem":      ex["problem"][:200],
            "ground_truth": ex["ground_truth"],
            "predicted":    predicted,
            "correct":      is_correct,
            "time_sec":     round(elapsed, 2),
            **{k: v for k, v in ex.items()
               if k not in ("problem", "ground_truth")},
        })

        if (i + 1) % 10 == 0 or i == 0:
            acc  = correct / (i + 1) * 100
            mark = "CORRECT" if is_correct else "WRONG"
            print(f"  [{i+1:4d}/{len(examples)}]  Acc: {acc:5.1f}%  |  {mark}  ({elapsed:.1f}s)")

    accuracy = correct / len(examples) * 100

    summary = {
        "mode":        args.mode,
        "dataset":     args.dataset,
        "model_path":  args.model_path,
        "lora_path":   args.lora_path,
        "n_samples":   len(examples),
        "correct":     correct,
        "total":       len(examples),
        "accuracy_pct": round(accuracy, 2),
        "timestamp":   datetime.now().isoformat(),
    }

    print(f"\n{'='*60}")
    print(f"  Dataset  : {args.dataset.upper()}")
    print(f"  Mode     : {args.mode}")
    print(f"  Accuracy : {accuracy:.2f}%  ({correct}/{len(examples)})")
    print(f"{'='*60}")

    os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else ".", exist_ok=True)
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
    ap.add_argument("--lora_path",   default=None,
                    help="Path to LoRA adapter. Omit for baseline.")
    ap.add_argument("--dataset",     required=True, choices=["math", "gsm8k"])
    ap.add_argument("--split",       default="test")
    ap.add_argument("--mode",        default="baseline",
                    help="Label: baseline | math_cot | math_nocot | gsm8k_cot | gsm8k_nocot")
    ap.add_argument("--n_samples",   type=int, default=None,
                    help="Number of test examples. None = full test set.")
    ap.add_argument("--seed",        type=int, default=42)
    ap.add_argument("--out",         default="eval_results/results.json")
    args = ap.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
