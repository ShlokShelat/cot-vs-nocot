"""
evaluate_nfa_v2.py
===================
Evaluates base or LoRA fine-tuned model on 5-tier NFA dataset.

Correctness = STRUCTURAL EQUIVALENCE:
Both NFAs are converted to DFAs via subset construction, minimised
using Hopcroft's algorithm, and compared as canonical forms up to
state relabelling. Dead-equivalent classes (states from which no
accepting state is reachable) are identified and excluded from the
canonical form, so a partial DFA (with implicit dead transitions) is
correctly treated as equivalent to a full DFA with explicit dead states.
Exact decision over the full infinite language; no string-length bounds.
"""

import argparse, json, re, random, os
from collections import defaultdict, deque
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
# STRUCTURAL EQUIVALENCE
# ===================

def _eps_closure(states, eps_trans):
    stack, cl = list(states), set(states)
    while stack:
        s = stack.pop()
        for t in eps_trans.get(s, []):
            if t not in cl: cl.add(t); stack.append(t)
    return frozenset(cl)


def _nfa_to_dfa(nfa):
    """Convert NFA to DFA via subset construction."""
    alphabet = sorted(nfa.alphabet)
    sc = _eps_closure({nfa.start}, nfa.eps_trans)
    dfa_trans = {}; dfa_accept = set()
    idx = {sc: 0}; queue = deque([sc])
    if nfa.accept in sc: dfa_accept.add(0)
    while queue:
        cur = queue.popleft(); ci = idx[cur]
        for c in alphabet:
            nxt_raw = set()
            for s in cur: nxt_raw.update(nfa.transitions.get((s, c), []))
            nxt = _eps_closure(nxt_raw, nfa.eps_trans)
            if not nxt: continue
            if nxt not in idx:
                i = len(idx); idx[nxt] = i
                if nfa.accept in nxt: dfa_accept.add(i)
                queue.append(nxt)
            dfa_trans[(ci, c)] = idx[nxt]
    return dfa_trans, 0, dfa_accept


def _canonical_form(dfa_trans, start, accept_states, alphabet):
    """
    Minimise a DFA using Hopcroft's algorithm and return its canonical form.
    Dead-equivalent classes (no path to accepting state) are excluded,
    so partial DFAs (implicit dead transitions) compare equal to full
    DFAs with explicit dead states.
    """
    DEAD = "__dead__"
    all_live = set()
    for (s, _), t in dfa_trans.items():
        all_live.add(s); all_live.add(t)
    all_live.add(start); all_live.update(accept_states)
    all_states = all_live | {DEAD}

    full = {}
    for s in all_states:
        for c in alphabet:
            full[(s, c)] = dfa_trans.get((s, c), DEAD)

    # Initial partition: accepting | non-accepting-live | {DEAD}
    accepting    = frozenset(s for s in all_live if s in accept_states)
    non_acc_live = frozenset(s for s in all_live if s not in accept_states)
    dead_block   = frozenset({DEAD})
    P = set()
    if accepting:    P.add(accepting)
    if non_acc_live: P.add(non_acc_live)
    P.add(dead_block)
    W = set(P)

    while W:
        A = W.pop()
        for c in alphabet:
            X = frozenset(s for s in all_states if full.get((s, c)) in A)
            if not X: continue
            new_P = set()
            for Y in P:
                inter = Y & X; diff = Y - X
                if inter and diff:
                    new_P.add(inter); new_P.add(diff)
                    if Y in W: W.discard(Y); W.add(inter); W.add(diff)
                    else: W.add(inter if len(inter) <= len(diff) else diff)
                else: new_P.add(Y)
            P = new_P

    # Live classes: groups not containing DEAD
    live_P = [g for g in P if DEAD not in g]
    if not live_P: return frozenset(), frozenset()

    state_to_class = {}
    for gi, g in enumerate(live_P):
        for s in g: state_to_class[s] = gi

    reps = {gi: next(iter(g)) for gi, g in enumerate(live_P)}
    n = len(live_P)

    # Build class-level transition graph
    class_trans = {}
    for gi in range(n):
        for c in alphabet:
            tgt = full.get((reps[gi], c), DEAD)
            class_trans[(gi, c)] = state_to_class.get(tgt)  # None = DEAD

    # Find classes that can reach an accepting class (reverse reachability)
    accepting_classes = {gi for gi in range(n)
                        if any(s in accept_states for s in live_P[gi])}
    can_reach = set(accepting_classes)
    changed = True
    while changed:
        changed = False
        for gi in range(n):
            if gi in can_reach: continue
            for c in alphabet:
                tgt = class_trans.get((gi, c))
                if tgt is not None and tgt in can_reach:
                    can_reach.add(gi); changed = True; break

    dead_classes = set(range(n)) - can_reach  # functionally dead

    start_class = state_to_class.get(start)
    if start_class is None: return frozenset(), frozenset()

    # BFS renaming from start, skipping dead classes
    old_to_new = {}
    queue = deque([start_class]); visited = {start_class}; ctr = 0
    while queue:
        gi = queue.popleft()
        if gi in dead_classes: continue
        old_to_new[gi] = ctr; ctr += 1
        for c in sorted(alphabet):
            tgt = class_trans.get((gi, c))
            if tgt is None or tgt in dead_classes: continue
            if tgt not in visited: visited.add(tgt); queue.append(tgt)

    canon_trans = set(); canon_accept = set()
    for gi in old_to_new:
        new_gi = old_to_new[gi]
        if gi in accepting_classes: canon_accept.add(new_gi)
        for c in sorted(alphabet):
            tgt = class_trans.get((gi, c))
            if tgt is None or tgt in dead_classes: continue
            if tgt in old_to_new:
                canon_trans.add((new_gi, c, old_to_new[tgt]))

    return frozenset(canon_trans), frozenset(canon_accept)


def structural_equivalent(gt: ParsedNFA, pred: ParsedNFA) -> bool:
    """
    Check structural equivalence:
    1. Convert both NFAs to DFAs (subset construction)
    2. Minimise with Hopcroft's algorithm
    3. Compare canonical forms (dead-equivalent classes excluded)
    Exact over the full infinite language; robust to state relabelling
    and to partial vs fully-specified automata.
    """
    alphabet = sorted(gt.alphabet | pred.alphabet)
    gt_t,   gt_s,   gt_a   = _nfa_to_dfa(gt)
    pred_t, pred_s, pred_a = _nfa_to_dfa(pred)
    return (_canonical_form(gt_t,   gt_s,   gt_a,   alphabet) ==
            _canonical_form(pred_t, pred_s, pred_a, alphabet))


# ===================
# NFA TABLE PARSER
# ===================

def _parse_state_set(cell: str) -> list:
    cell = cell.strip()
    if cell in ("∅", "", "--", "—"): return []
    cell = re.sub(r"^\{|\}$", "", cell).strip()
    if not cell: return []
    return [p.strip() for p in cell.split(",") if p.strip()]


def parse_nfa_table(text: str) -> Optional[ParsedNFA]:
    lines = text.split("\n")
    header_idx = None
    for i, line in enumerate(lines):
        if re.search(r"\|\s*[Ss]tate\s*\|", line) and re.search(r"\|\s*[Rr]ole\s*\|", line):
            header_idx = i; break
    if header_idx is None: return None

    cols = [c.strip() for c in lines[header_idx].split("|") if c.strip()]
    if len(cols) < 3: return None

    symbol_cols = []; eps_col_idx = None
    for i, col in enumerate(cols):
        if col.lower() in ("state", "role"): continue
        if "eps" in col.lower() or "ε" in col or "epsilon" in col.lower():
            eps_col_idx = i
        else:
            symbol_cols.append((i, col))

    alphabet = {sym for _, sym in symbol_cols}
    data_lines = []
    for line in lines[header_idx + 2:]:
        stripped = line.strip()
        if not stripped or not stripped.startswith("|"): break
        data_lines.append(stripped)
    if not data_lines: return None

    states = []; start = None; accept = set()
    trans = defaultdict(list); eps_trans = defaultdict(list)

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
                targets = _parse_state_set(parts[col_idx])
                if targets: trans[(state_name, sym)].extend(targets)
        if eps_col_idx is not None and eps_col_idx < len(parts):
            targets = _parse_state_set(parts[eps_col_idx])
            if targets: eps_trans[state_name].extend(targets)

    if not states or start is None: return None
    return ParsedNFA(states=states, start=start, accept=accept,
                     alphabet=alphabet, transitions=dict(trans),
                     eps_trans=dict(eps_trans))


def extract_nfa_table(text: str) -> str:
    lines = text.split("\n")
    table_blocks, current_block, in_table = [], [], False
    for line in lines:
        if line.strip().startswith("|"):
            in_table = True; current_block.append(line)
        else:
            if in_table and current_block:
                table_blocks.append("\n".join(current_block)); current_block = []
            in_table = False
    if in_table and current_block:
        table_blocks.append("\n".join(current_block))
    for block in reversed(table_blocks):
        if "State" in block and "Role" in block: return block
    return ""


# ===================
# GROUND TRUTH / METADATA
# ===================

def gt_nfa_from_entry(entry: dict) -> Optional[ParsedNFA]:
    for msg in entry.get("messages", []):
        if msg["role"] == "assistant":
            table_text = extract_nfa_table(msg["content"])
            if table_text: return parse_nfa_table(table_text)
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
        prompt, return_tensors="pt", truncation=True, max_length=4096,
    ).to(model.device)
    out_ids = model.generate(
        **inputs, max_new_tokens=max_new_tokens,
        do_sample=False, pad_token_id=tokenizer.eos_token_id,
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
        by_tier = defaultdict(list)
        for e in entries: by_tier[tier_from_entry(e)].append(e)
        sampled = []
        per_tier_n = max(1, args.n_samples // len(by_tier))
        for tier_entries in by_tier.values():
            random.shuffle(tier_entries); sampled.extend(tier_entries[:per_tier_n])
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
    max_new_tokens = 2048 if args.mode == "cot" else 1024

    results = []; n_correct = 0; n_pred_fail = 0; n_gt_fail = 0
    per_tier = defaultdict(lambda: {"correct": 0, "total": 0})

    for entry in tqdm(entries, desc=f"Eval [{args.mode}]"):
        regex    = entry.get("metadata", {}).get("regex", "")
        alphabet = alphabet_from_entry(entry)
        tier     = tier_from_entry(entry)

        gt_nfa = gt_nfa_from_entry(entry)
        if gt_nfa is None:
            n_gt_fail += 1
            results.append({"regex": regex, "tier": tier,
                            "correct": False, "error": "gt_parse_fail"})
            per_tier[tier]["total"] += 1; continue

        prompt   = build_prompt(entry, tokenizer)
        response = generate(model, tokenizer, prompt, max_new_tokens)
        table_text = extract_nfa_table(response)
        pred_nfa   = parse_nfa_table(table_text) if table_text else None

        if pred_nfa is None:
            n_pred_fail += 1
            results.append({"regex": regex, "tier": tier,
                            "correct": False, "error": "pred_parse_fail",
                            "response_snippet": response[:300]})
            per_tier[tier]["total"] += 1; continue

        pred_nfa.alphabet |= alphabet
        gt_nfa.alphabet   |= alphabet

        correct = structural_equivalent(gt_nfa, pred_nfa)
        if correct: n_correct += 1; per_tier[tier]["correct"] += 1
        per_tier[tier]["total"] += 1

        results.append({"regex": regex, "alphabet": list(alphabet),
                        "tier": tier, "correct": correct, "error": None,
                        "gt_states": len(gt_nfa.states),
                        "pred_states": len(pred_nfa.states)})

    per_tier_summary = {}
    for t in sorted(per_tier.keys()):
        tot = per_tier[t]["total"]; cor = per_tier[t]["correct"]
        per_tier_summary[str(t)] = {
            "correct": cor, "total": tot,
            "accuracy": round(cor / tot, 4) if tot > 0 else 0.0,
        }

    total    = len(entries)
    accuracy = n_correct / total if total > 0 else 0.0

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
        "eval_method":       "structural_equivalence_hopcroft",
    }

    print("\n" + "=" * 65)
    print(f"  model    : {args.model_path.split('/')[-1]}")
    print(f"  mode     : {args.mode}")
    print(f"  ood      : {args.ood}")
    print(f"  total    : {total}")
    print(f"  correct  : {n_correct}  ({accuracy*100:.1f}%)")
    print(f"  pred parse fails : {n_pred_fail}")
    print(f"  gt parse fails   : {n_gt_fail}")
    print(f"\n  Per-tier breakdown:")
    tier_labels = {"1":"[ 4- 8 states] trivial","2":"[ 8-14 states] easy",
                   "3":"[14-22 states] medium","4":"[22-32 states] hard",
                   "5":"[32-50 states] very hard"}
    for t, stats in sorted(per_tier_summary.items()):
        print(f"    Tier {t} {tier_labels.get(t,f'tier {t}')}: "
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
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--lora_path",  default=None)
    ap.add_argument("--data_file",  required=True)
    ap.add_argument("--mode",       default="baseline", help="baseline | cot | nocot")
    ap.add_argument("--n_samples",  type=int, default=None)
    ap.add_argument("--ood",        action="store_true")
    ap.add_argument("--out",        default="eval_results/results.json")
    args = ap.parse_args()
    evaluate(args)

if __name__ == "__main__":
    main()
