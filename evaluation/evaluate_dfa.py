"""
evaluate_dfa.py
================
Evaluates base or LoRA fine-tuned models on the DFA dataset.

Correctness = LANGUAGE EQUIVALENCE: both ground-truth and predicted DFAs
are simulated on all strings up to a tier-dependent maximum length.
Prediction is correct iff the two DFAs agree on every string.
This metric is robust to state relabelling.

Supports all three evaluation modes:
  baseline : zero-shot unmodified model
  cot      : CoT fine-tuned model (reads dfa_cot_test.jsonl)
  nocot    : No-CoT fine-tuned model (reads dfa_nocot_test.jsonl)

Reports:
  - Overall accuracy
  - Per-tier accuracy (T1-T4)
  - Parse failure rate (separate from language equivalence failures)
  - IID and OOD results (pass --ood flag for OOD test file)

Usage:
  # Baseline
  python3 evaluate_dfa.py \
      --model_path <path/to/model> \
      --data_file  dfa_nocot_test.jsonl \
      --mode       baseline \
      --out        eval_results/dfa_baseline_iid.json

  # CoT fine-tuned IID
  python3 evaluate_dfa.py \
      --model_path <path/to/model> \
      --lora_path  <path/to/lora_adapter> \
      --data_file  dfa_cot_test.jsonl \
      --mode       cot \
      --out        eval_results/dfa_cot_iid.json

  # No-CoT fine-tuned OOD
  python3 evaluate_dfa.py \
      --model_path <path/to/model> \
      --lora_path  <path/to/lora_adapter> \
      --data_file  dfa_nocot_ood_test.jsonl \
      --mode       nocot \
      --ood \
      --out        eval_results/dfa_nocot_ood.json
"""

import argparse, json, re, itertools, random, os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm


# ===================
#  DFA DATA STRUCTURE
# ===================

@dataclass
class ParsedDFA:
    states:      list
    start:       Optional[str]
    accept:      set
    alphabet:    set
    transitions: dict   # {(state, symbol): state}


# ===================
#  DFA SIMULATION
# ===================

def dfa_accepts(dfa: ParsedDFA, string: str) -> bool:
    if dfa.start is None: return False
    cur = dfa.start
    for c in string:
        cur = dfa.transitions.get((cur, c))
        if cur is None: return False
    return cur in dfa.accept


# Tier-aware evaluation lengths
TIER_EVAL_LEN = {1: 6, 2: 7, 3: 8, 4: 9}


def language_equivalent(gt: ParsedDFA, pred: ParsedDFA, max_len: int = 6) -> bool:
    alphabet = gt.alphabet | (pred.alphabet or set())
    for length in range(max_len + 1):
        for combo in itertools.product(sorted(alphabet), repeat=length):
            s = "".join(combo)
            if dfa_accepts(gt, s) != dfa_accepts(pred, s):
                return False
    return True


# ===================
#  DFA TABLE PARSER
# ===================

def _parse_state_set(cell: str) -> Optional[str]:
    """Parse a single DFA transition cell. Returns state name or None (dead)."""
    cell = cell.strip()
    if cell in ("--", "—", "", "dead", "trap", "∅"): return None
    # Strip D prefix if present, return as-is otherwise
    return cell


def parse_dfa_table(text: str) -> Optional[ParsedDFA]:
    """
    Parse a markdown DFA transition table from model output.
    Expects header: | State | Role | sym1 | sym2 | ...
    No epsilon column (DFA has no epsilon transitions).
    Dead/trap transitions marked as --.
    Robust to state relabelling: uses whatever state names appear in the table.
    """
    lines = text.split("\n")

    # Find header row with State and Role columns
    header_idx = None
    for i, line in enumerate(lines):
        if re.search(r"\|\s*[Ss]tate\s*\|", line) and re.search(r"\|\s*[Rr]ole\s*\|", line):
            header_idx = i
            break
    if header_idx is None: return None

    cols = [c.strip() for c in lines[header_idx].split("|") if c.strip()]
    if len(cols) < 3: return None

    # Identify symbol columns (everything that is not State or Role)
    symbol_cols = []
    for i, col in enumerate(cols):
        if col.lower() in ("state", "role"): continue
        # Skip epsilon columns if present (should not be in DFA tables)
        if "eps" in col.lower() or "epsilon" in col.lower() or col in ("ε","\u03b5"): continue
        symbol_cols.append((i, col))

    if not symbol_cols: return None
    alphabet = {sym for _, sym in symbol_cols}

    # Parse data rows (skip header and separator)
    data_lines = []
    for line in lines[header_idx + 2:]:
        stripped = line.strip()
        if not stripped or not stripped.startswith("|"): break
        data_lines.append(stripped)

    if not data_lines: return None

    states = []; start = None; accept = set(); trans = {}

    for line in data_lines:
        parts = [p.strip() for p in line.split("|")][1:-1]
        if len(parts) < len(cols): continue
        state_name = parts[0]
        if not state_name: continue
        role = parts[1].lower()
        states.append(state_name)
        if "start"  in role: start = state_name
        if "accept" in role: accept.add(state_name)
        for col_idx, sym in symbol_cols:
            if col_idx < len(parts):
                target = _parse_state_set(parts[col_idx])
                if target is not None:
                    trans[(state_name, sym)] = target

    if not states or start is None: return None

    return ParsedDFA(
        states=states, start=start, accept=accept,
        alphabet=alphabet, transitions=trans,
    )


def extract_dfa_table(text: str) -> str:
    """
    Extract the last markdown table containing State and Role columns.
    For CoT outputs: skips intermediate tables (subset construction etc.)
      and returns the final minimised DFA table.
    For No-CoT outputs: returns the only table present.
    """
    lines = text.split("\n")
    table_blocks = []; current = []; in_table = False
    for line in lines:
        if line.strip().startswith("|"):
            in_table = True; current.append(line)
        else:
            if in_table and current:
                table_blocks.append("\n".join(current)); current = []
            in_table = False
    if in_table and current:
        table_blocks.append("\n".join(current))

    # Return last block that has both State and Role headers
    for block in reversed(table_blocks):
        if "State" in block and "Role" in block:
            return block
    return ""


# ===================
#  GROUND TRUTH EXTRACTION
# ===================

def gt_dfa_from_entry(entry: dict) -> Optional[ParsedDFA]:
    for msg in entry.get("messages", []):
        if msg["role"] == "assistant":
            table_text = extract_dfa_table(msg["content"])
            if table_text:
                return parse_dfa_table(table_text)
    return None


def alphabet_from_entry(entry: dict) -> set:
    return set(entry.get("metadata", {}).get("alphabet", []))


def tier_from_entry(entry: dict) -> int:
    return entry.get("metadata", {}).get("tier", 0)


# ===================
#  MODEL INFERENCE
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
#  EVALUATION LOOP
# ===================

def evaluate(args):
    random.seed(42)

    print(f"Loading {args.data_file}...")
    with open(args.data_file) as f:
        entries = [json.loads(l) for l in f if l.strip()]

    if args.n_samples and args.n_samples < len(entries):
        # Stratified sample preserving tier distribution
        by_tier = defaultdict(list)
        for e in entries: by_tier[tier_from_entry(e)].append(e)
        sampled = []
        per_tier = max(1, args.n_samples // len(by_tier))
        for tier_entries in by_tier.values():
            random.shuffle(tier_entries)
            sampled.extend(tier_entries[:per_tier])
        entries = sampled

    print(f"Evaluating {len(entries)} examples | mode={args.mode} | ood={args.ood}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model from {args.model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)

    if args.lora_path:
        from peft import PeftModel
        print(f"Loading LoRA from {args.lora_path}...")
        model = PeftModel.from_pretrained(model, args.lora_path)
        model = model.merge_and_unload()
        print("LoRA merged.")

    model.eval()

    # CoT outputs are longer (8 steps + table)
    max_new_tokens = 4096 if args.mode == "cot" else 1024

    results    = []
    n_correct  = 0
    n_pred_fail= 0
    n_gt_fail  = 0
    per_tier   = defaultdict(lambda: {"correct":0, "total":0})

    for entry in tqdm(entries, desc=f"Eval [{args.mode}]"):
        regex    = entry.get("metadata", {}).get("regex", "")
        alphabet = alphabet_from_entry(entry)
        tier     = tier_from_entry(entry)

        # Ground truth DFA
        gt_dfa = gt_dfa_from_entry(entry)
        if gt_dfa is None:
            n_gt_fail += 1
            results.append({"regex":regex,"tier":tier,"correct":False,"error":"gt_parse_fail"})
            per_tier[tier]["total"] += 1
            continue

        # Generate prediction
        prompt   = build_prompt(entry, tokenizer)
        response = generate(model, tokenizer, prompt, max_new_tokens)

        # Parse prediction
        table_text = extract_dfa_table(response)
        pred_dfa   = parse_dfa_table(table_text) if table_text else None

        if pred_dfa is None:
            n_pred_fail += 1
            results.append({
                "regex":regex, "tier":tier, "correct":False, "error":"pred_parse_fail",
                "response_snippet": response[:300],
            })
            per_tier[tier]["total"] += 1
            continue

        # Language equivalence check (tier-dependent string length)
        pred_dfa.alphabet |= alphabet
        tier_max_len = TIER_EVAL_LEN.get(tier, args.eval_len)
        correct = language_equivalent(gt_dfa, pred_dfa, max_len=tier_max_len)

        if correct:
            n_correct += 1
            per_tier[tier]["correct"] += 1
        per_tier[tier]["total"] += 1

        results.append({
            "regex":      regex,
            "alphabet":   list(alphabet),
            "tier":       tier,
            "correct":    correct,
            "error":      None,
            "gt_states":  len(gt_dfa.states),
            "pred_states": len(pred_dfa.states),
        })

    # Compute per-tier accuracy
    per_tier_summary = {}
    for t in sorted(per_tier.keys()):
        tot = per_tier[t]["total"]; cor = per_tier[t]["correct"]
        per_tier_summary[str(t)] = {
            "correct": cor, "total": tot,
            "accuracy": round(cor/tot,4) if tot>0 else 0.0,
        }

    total    = len(entries)
    accuracy = n_correct/total if total>0 else 0.0

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
        "accuracy":          round(accuracy,4),
        "per_tier":          per_tier_summary,
        "tier_eval_lengths": TIER_EVAL_LEN,
    }

    # Print results
    print("\n"+"="*65)
    print(f"  mode     : {args.mode}")
    print(f"  ood      : {args.ood}")
    print(f"  total    : {total}")
    print(f"  correct  : {n_correct}  ({accuracy*100:.1f}%)")
    print(f"  pred parse fails : {n_pred_fail}")
    print(f"  gt parse fails   : {n_gt_fail}")
    print(f"\n  Per-tier breakdown:")
    tier_labels = {
        "1": "[ 2    states] simple",
        "2": "[ 2- 4 states] easy",
        "3": "[ 3- 6 states] medium",
        "4": "[ 4-18 states] hard",
    }
    for t, stats in sorted(per_tier_summary.items()):
        label = tier_labels.get(t, f"tier {t}")
        print(f"    Tier {t} {label}: "
              f"{stats['accuracy']*100:.1f}%  ({stats['correct']}/{stats['total']})")
    print("="*65)

    os.makedirs(
        os.path.dirname(args.out) if os.path.dirname(args.out) else ".",
        exist_ok=True)
    with open(args.out,"w") as f:
        json.dump({"summary":summary,"results":results}, f, indent=2)
    print(f"Saved -> {args.out}")


# ===================
#  ENTRY POINT
# ===================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True,
                    help="Path to base model.")
    ap.add_argument("--lora_path",  default=None,
                    help="Path to LoRA adapter. Omit for baseline.")
    ap.add_argument("--data_file",  required=True,
                    help="Test JSONL file (dfa_cot_test.jsonl or dfa_nocot_test.jsonl)")
    ap.add_argument("--mode",       default="baseline",
                    help="baseline | cot | nocot")
    ap.add_argument("--n_samples",  type=int, default=None,
                    help="Stratified subsample. None = full test set.")
    ap.add_argument("--eval_len",   type=int, default=6,
                    help="Fallback max string length if tier not found in TIER_EVAL_LEN.")
    ap.add_argument("--ood",        action="store_true",
                    help="Mark this as OOD evaluation (stored in summary JSON).")
    ap.add_argument("--out",        default="eval_results/dfa_results.json")
    args = ap.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
