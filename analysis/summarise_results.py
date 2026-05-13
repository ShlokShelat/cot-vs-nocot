"""
summarise_results.py
====================
Aggregates all evaluation JSON files produced by evaluate_nfa_v2.py,
evaluate_dfa.py, and eval_qwen_three_benchmarks.py into a clean
per-tier accuracy summary table printed to stdout.

Usage:
  python3 summarise_results.py --results_dir eval_results/

  # Filter to a specific dataset
  python3 summarise_results.py --results_dir eval_results/ --filter nfa

  # Save summary to a file
  python3 summarise_results.py --results_dir eval_results/ --out summary.txt
"""

import argparse, json, glob, os, sys
from collections import defaultdict


# ===================
#  HELPERS
# ===================

TIER_LABELS = {
    "1": "T1",
    "2": "T2",
    "3": "T3",
    "4": "T4",
    "5": "T5",
}


def pct(val):
    """Convert fraction or percentage to a display string."""
    if val is None: return "  --  "
    if val <= 1.0:  val *= 100
    return f"{val:5.1f}%"


def load_summary(path):
    with open(path) as f:
        data = json.load(f)
    # Support both top-level summary dict and nested {"summary": ...}
    if "summary" in data:
        return data["summary"]
    return data


def extract_fields(s):
    """Extract normalised fields from a summary dict."""
    mode     = s.get("mode", "?")
    ood_tag  = "OOD" if s.get("ood") else "IID"
    acc      = s.get("accuracy", s.get("accuracy_pct", None))
    n_cor    = s.get("n_correct",  s.get("correct",  "?"))
    n_tot    = s.get("n_total",    s.get("total",    "?"))
    per_tier = s.get("per_tier",  {})
    lora     = s.get("lora_path", None)
    dataset  = s.get("data_file", s.get("dataset", "?"))
    return mode, ood_tag, acc, n_cor, n_tot, per_tier, lora, dataset


# ===================
#  MAIN
# ===================

def main():
    ap = argparse.ArgumentParser(
        description="Aggregate evaluation JSON files into a summary table."
    )
    ap.add_argument("--results_dir", default="eval_results/",
                    help="Directory containing evaluation JSON files")
    ap.add_argument("--filter",      default=None,
                    help="Only show files whose name contains this string")
    ap.add_argument("--out",         default=None,
                    help="Save summary to this file (default: stdout only)")
    args = ap.parse_args()

    pattern = os.path.join(args.results_dir, "*.json")
    files   = sorted(glob.glob(pattern))

    if not files:
        print(f"No JSON files found in {args.results_dir}")
        sys.exit(1)

    if args.filter:
        files = [f for f in files if args.filter in os.path.basename(f)]
        if not files:
            print(f"No files matching filter '{args.filter}' in {args.results_dir}")
            sys.exit(1)

    lines = []
    lines.append("=" * 75)
    lines.append(f"  RESULTS SUMMARY  --  {len(files)} file(s) from {args.results_dir}")
    lines.append("=" * 75)

    # Group by dataset type (nfa / dfa / math)
    groups = defaultdict(list)
    for path in files:
        name = os.path.basename(path).lower()
        if "nfa" in name:
            groups["NFA"].append(path)
        elif "dfa" in name:
            groups["DFA"].append(path)
        elif any(k in name for k in ("gsm","math","mathinstruct")):
            groups["MATH"].append(path)
        else:
            groups["OTHER"].append(path)

    for group_name, group_files in sorted(groups.items()):
        lines.append(f"\n{'─'*75}")
        lines.append(f"  {group_name}")
        lines.append(f"{'─'*75}")

        # Header
        has_tiers = any(
            load_summary(f).get("per_tier") for f in group_files
        )
        if has_tiers:
            tier_keys = set()
            for f in group_files:
                tier_keys |= set(load_summary(f).get("per_tier", {}).keys())
            tier_keys = sorted(tier_keys)
            header = f"  {'File':<38} {'Mode':<12} {'Tag':<5} {'Overall':>8}"
            for t in tier_keys:
                header += f"  {TIER_LABELS.get(t, f'T{t}'):>6}"
            lines.append(header)
            lines.append("  " + "-"*73)
        else:
            lines.append(f"  {'File':<38} {'Mode':<12} {'Tag':<5} {'Overall':>8}")
            lines.append("  " + "-"*55)

        for path in sorted(group_files):
            try:
                s = load_summary(path)
            except Exception as e:
                lines.append(f"  [ERROR reading {os.path.basename(path)}: {e}]")
                continue

            mode, ood_tag, acc, n_cor, n_tot, per_tier, lora, dataset = extract_fields(s)
            fname   = os.path.basename(path)[:38]
            overall = pct(acc)

            row = f"  {fname:<38} {mode:<12} {ood_tag:<5} {overall:>8}"

            if has_tiers:
                for t in tier_keys:
                    ts = per_tier.get(t, {})
                    ta = ts.get("accuracy", None)
                    row += f"  {pct(ta):>6}"

            lines.append(row)

            # Print parse failure info if present
            pred_fail = s.get("n_pred_parse_fail", 0)
            gt_fail   = s.get("n_gt_parse_fail",   0)
            if pred_fail or gt_fail:
                lines.append(f"  {'':38} pred_parse_fail={pred_fail}  gt_parse_fail={gt_fail}")

    lines.append("")
    lines.append("=" * 75)
    lines.append("")

    output = "\n".join(lines)
    print(output)

    if args.out:
        with open(args.out, "w") as f:
            f.write(output)
        print(f"Summary saved -> {args.out}")


if __name__ == "__main__":
    main()
