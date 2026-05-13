"""
evaluate_nfa_v2.py
===================
Evaluates base or LoRA fine-tuned model on 5-tier NFA v2 dataset.

Correctness = LANGUAGE EQUIVALENCE across all strings up to length 6.
Reports overall accuracy AND per-tier accuracy breakdown.

Usage:
  # Baseline:
  python3 evaluate_nfa_v2.py \
      --model_path /path/to/Qwen2.5-7B-Instruct \
      --data_file  nfa_v2_nocot_test.jsonl \
      --mode       baseline \
      --out        eval_results/results_baseline.json

  # CoT fine-tuned:
  python3 evaluate_nfa_v2.py \
      --model_path /path/to/Qwen2.5-7B-Instruct \
      --lora_path  lora_output_nfa_v2_cot \
      --data_file  nfa_v2_cot_test.jsonl \
      --mode       cot \
      --out        eval_results/results_cot.json

  # No-CoT fine-tuned:
  python3 evaluate_nfa_v2.py \
      --model_path /path/to/Qwen2.5-7B-Instruct \
      --lora_path  lora_output_nfa_v2_nocot \
      --data_file  nfa_v2_nocot_test.jsonl \
      --mode       nocot \
      --out        eval_results/results_nocot.json
"""

import argparse
import json
import re
import itertools
import random
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm


# ===================
# NFA DATA STRUCTURE
# ===================

@dataclass
class ParsedNFA:
    states:      list
    start:       Optional[str]
    accept:      set
    alphabet:    set
    transitions: dict   # {(state, symbol): [state, ...]}
    eps_trans:   dict   # {state: [state, ...]}


# ===================
# NFA SIMULATION
# ===================

def _eps_closure(nfa: ParsedNFA, states: set) -> frozenset:
    stack, closure = list(states), set(states)
    while stack:
        s = stack.pop()
        for t in nfa.eps_trans.get(s, []):
            if t not in closure:
                closure.add(t)
                stack.append(t)
    return frozenset(closure)


def _nfa_accepts(nfa: ParsedNFA, string: str) -> bool:
    if nfa.start is None:
        return False
    current = _eps_closure(nfa, {nfa.start})
    for c in string:
        nxt = set()
        for s in current:
            nxt.update(nfa.transitions.get((s, c), []))
        current = _eps_closure(nfa, nxt)
    return bool(current & nfa.accept)


# Tier-aware eval length — harder tiers need longer strings to distinguish NFAs
TIER_EVAL_LEN = {1: 6, 2: 7, 3: 8, 4: 9, 5: 10}


def language_equivalent(gt: ParsedNFA, pred: ParsedNFA, max_len: int = 6) -> bool:
    alphabet = gt.alphabet | (pred.alphabet or set())
    for length in range(max_len + 1):
        for combo in itertools.product(sorted(alphabet), repeat=length):
            s = "".join(combo)
            if _nfa_accepts(gt, s) != _nfa_accepts(pred, s):
                return False
    return True


# ===================
# NFA TABLE PARSER
# ===================

def _parse_state_set(cell: str) -> list:
    cell = cell.strip()
    if cell in ("∅", "", "--", "—"):
        return []
    cell = re.sub(r"^\{|\}$", "", cell).strip()
    if not cell:
        return []
    return [p.strip() for p in cell.split(",") if p.strip()]


def parse_nfa_table(text: str) -> Optional[ParsedNFA]:
    lines = text.split("\n")

    # Find header row containing State and Role
    header_idx = None
    for i, line in enumerate(lines):
        if re.search(r"\|\s*[Ss]tate\s*\|", line) and re.search(r"\|\s*[Rr]ole\s*\|", line):
            header_idx = i
            break
    if header_idx is None:
        return None

    cols = [c.strip() for c in lines[header_idx].split("|") if c.strip()]
    if len(cols) < 3:
        return None

    symbol_cols = []
    eps_col_idx = None
    for i, col in enumerate(cols):
        if col.lower() in ("state", "role"):
            continue
        if "eps" in col.lower() or "ε" in col or "epsilon" in col.lower():
            eps_col_idx = i
        else:
            symbol_cols.append((i, col))

    alphabet = {sym for _, sym in symbol_cols}

    data_lines = []
    for line in lines[header_idx + 2:]:
        stripped = line.strip()
        if not stripped or not stripped.startswith("|"):
            break
        data_lines.append(stripped)

    if not data_lines:
        return None

    states    = []
    start     = None
    accept    = set()
    trans     = defaultdict(list)
    eps_trans = defaultdict(list)

    for line in data_lines:
        parts = [p.strip() for p in line.split("|")][1:-1]
        if len(parts) < len(cols):
            continue
        state_name = parts[0]
        if not state_name:
            continue
        role = parts[1].lower()
        states.append(state_name)
        if "start"  in role: start = state_name
        if "accept" in role: accept.add(state_name)
        for col_idx, sym in symbol_cols:
            if col_idx < len(parts):
                targets = _parse_state_set(parts[col_idx])
                if targets: trans[(state_name, sym)].extend(targets)
        if eps_col_idx is not None and eps_col_idx < len(parts):
            targets = _parse_state_set(parts[eps_col_idx])
            if targets: eps_trans[state_name].extend(targets)

    if not states or start is None:
        return None

    return ParsedNFA(
        states=states, start=start, accept=accept,
        alphabet=alphabet,
        transitions=dict(trans),
        eps_trans=dict(eps_trans),
    )


def extract_nfa_table(text: str) -> str:
    """Extract last table containing State+Role header — works for both CoT and No-CoT."""
    lines = text.split("\n")
    table_blocks, current_block, in_table = [], [], False
    for line in lines:
        if line.strip().startswith("|"):
            in_table = True
            current_block.append(line)
        else:
            if in_table and current_block:
                table_blocks.append("\n".join(current_block))
                current_block = []
            in_table = False
    if in_table and current_block:
        table_blocks.append("\n".join(current_block))
    for block in reversed(table_blocks):
        if "State" in block and "Role" in block:
            return block
    return ""


# ===================
# GROUND TRUTH EXTRACTION
# ===================

def gt_nfa_from_entry(entry: dict) -> Optional[ParsedNFA]:
    for msg in entry.get("messages", []):
        if msg["role"] == "assistant":
            table_text = extract_nfa_table(msg["content"])
            if table_text:
                return parse_nfa_table(table_text)
    return None


def alphabet_from_entry(entry: dict) -> set:
    return set(entry.get("metadata", {}).get("alphabet", []))


def tier_from_entry(entry: dict) -> int:
    return entry.get("metadata", {}).get("tier", 0)


# ===================
# MODEL INFERENCE
# ===================

def build_prompt(entry: dict, tokenizer) -> str:
    prompt_msgs = [m for m in entry["messages"] if m["role"] != "assistant"]
    return tokenizer.apply_chat_template(
        prompt_msgs, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def generate(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    inputs = tokenizer(
        prompt, return_tensors="pt",
        truncation=True, max_length=4096,
    ).to(model.device)
    out_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    new_ids = out_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True)


# ===================
# EVALUATION LOOP
# ===================

def evaluate(args):
    random.seed(42)

    print(f"Loading {args.data_file}...")
    with open(args.data_file) as f:
        entries = [json.loads(l) for l in f if l.strip()]

    if args.n_samples and args.n_samples < len(entries):
        # Stratified sample — preserve tier distribution
        by_tier = defaultdict(list)
        for e in entries:
            by_tier[tier_from_entry(e)].append(e)
        sampled = []
        per_tier = max(1, args.n_samples // len(by_tier))
        for tier_entries in by_tier.values():
            random.shuffle(tier_entries)
            sampled.extend(tier_entries[:per_tier])
        entries = sampled

    print(f"Evaluating {len(entries)} examples | mode={args.mode}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    print(f"Loading model from {args.model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    if args.lora_path:
        from peft import PeftModel
        print(f"Loading LoRA from {args.lora_path}...")
        model = PeftModel.from_pretrained(model, args.lora_path)
        model = model.merge_and_unload()
        print("LoRA merged.")

    model.eval()

    # Max new tokens: CoT needs more space
    max_new_tokens = 2048 if args.mode == "cot" else 1024

    results         = []
    n_correct       = 0
    n_pred_fail     = 0
    n_gt_fail       = 0
    per_tier        = defaultdict(lambda: {"correct": 0, "total": 0})

    for entry in tqdm(entries, desc=f"Eval [{args.mode}]"):
        regex    = entry.get("metadata", {}).get("regex", "")
        alphabet = alphabet_from_entry(entry)
        tier     = tier_from_entry(entry)

        # Ground truth NFA
        gt_nfa = gt_nfa_from_entry(entry)
        if gt_nfa is None:
            n_gt_fail += 1
            results.append({"regex": regex, "tier": tier,
                            "correct": False, "error": "gt_parse_fail"})
            per_tier[tier]["total"] += 1
            continue

        # Generate prediction
        prompt   = build_prompt(entry, tokenizer)
        response = generate(model, tokenizer, prompt, max_new_tokens)

        # Parse prediction
        table_text = extract_nfa_table(response)
        pred_nfa   = parse_nfa_table(table_text) if table_text else None

        if pred_nfa is None:
            n_pred_fail += 1
            results.append({
                "regex": regex, "tier": tier,
                "correct": False, "error": "pred_parse_fail",
                "response_snippet": response[:300],
            })
            per_tier[tier]["total"] += 1
            continue

        # Language equivalence — stricter check for harder tiers
        pred_nfa.alphabet |= alphabet
        tier_max_len = TIER_EVAL_LEN.get(tier, args.eval_len)
        correct = language_equivalent(gt_nfa, pred_nfa, max_len=tier_max_len)
        if correct:
            n_correct += 1
            per_tier[tier]["correct"] += 1

        per_tier[tier]["total"] += 1
        results.append({
            "regex":       regex,
            "alphabet":    list(alphabet),
            "tier":        tier,
            "correct":     correct,
            "error":       None,
            "gt_states":   len(gt_nfa.states),
            "pred_states": len(pred_nfa.states),
        })

    # Compute per-tier accuracy
    per_tier_summary = {}
    for t in sorted(per_tier.keys()):
        tot = per_tier[t]["total"]
        cor = per_tier[t]["correct"]
        per_tier_summary[str(t)] = {
            "correct":  cor,
            "total":    tot,
            "accuracy": round(cor / tot, 4) if tot > 0 else 0.0,
        }

    total     = len(entries)
    accuracy  = n_correct / total if total > 0 else 0.0

    summary = {
        "mode":              args.mode,
        "ood":               args.ood,
        "model_path":        args.model_path,
        "lora_path":         args.lora_path,
        "data_file":         args.data_file,
        "n_total":           total,
        "n_correct":         n_correct,
        "n_pred_parse_fail": n_pred_fail,
        "n_gt_parse_fail":   n_gt_fail,
        "accuracy":          round(accuracy, 4),
        "per_tier":          per_tier_summary,
        "tier_eval_lengths": TIER_EVAL_LEN,
    }

    # Print results
    print("\n" + "=" * 65)
    print(f"  mode     : {args.mode}")
    print(f"  total    : {total}")
    print(f"  correct  : {n_correct}  ({accuracy*100:.1f}%)")
    print(f"  pred parse fails : {n_pred_fail}")
    print(f"  gt parse fails   : {n_gt_fail}")
    print(f"\n  Per-tier breakdown:")
    tier_labels = {
        "1": "[ 4- 8 states] trivial",
        "2": "[ 8-14 states] easy",
        "3": "[14-22 states] medium",
        "4": "[22-32 states] hard",
        "5": "[32-50 states] very hard",
    }
    for t, stats in sorted(per_tier_summary.items()):
        label = tier_labels.get(t, f"tier {t}")
        print(f"    Tier {t} {label}: "
              f"{stats['accuracy']*100:.1f}%  ({stats['correct']}/{stats['total']})")
    print("=" * 65)

    os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)
    print(f"Saved → {args.out}")


# ===================
# ENTRY POINT
# ===================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path",     default="/path/to/Qwen2.5-7B-Instruct")
    ap.add_argument("--lora_path",      default=None)
    ap.add_argument("--data_file",      required=True)
    ap.add_argument("--mode",           default="baseline",
                    help="baseline | cot | nocot")
    ap.add_argument("--n_samples",      type=int, default=None,
                    help="Stratified subsample. None = full test set.")
    ap.add_argument("--eval_len",       type=int, default=6,
                    help="Baseline max string length. Overridden per-tier by TIER_EVAL_LEN.")
    ap.add_argument("--ood",            action="store_true",
                    help="Mark this run as OOD evaluation (stored in summary).")
    ap.add_argument("--out",            default="eval_results/results.json")
    args = ap.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
