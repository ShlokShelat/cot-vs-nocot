"""
compare_runs.py
===============
Plots CoT vs No-CoT training and evaluation loss curves from the
training_metrics.csv files produced by the MetricsLogger callback
during fine-tuning.

Usage:
  python3 compare_runs.py \
      --cot_dir   lora_output_cot \
      --nocot_dir lora_output_nocot \
      --title     "Dataset A (7B)" \
      --out       figures/loss_curves.png
"""

import argparse, os, csv
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# ===================
#  DATA LOADER
# ===================

def load_metrics(directory):
    """
    Load training_metrics.csv from a LoRA output directory.
    Returns (train_steps, train_losses, eval_steps, eval_losses).
    """
    path = os.path.join(directory, "training_metrics.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No training_metrics.csv found in {directory}.\n"
            f"Make sure the MetricsLogger callback ran during training."
        )

    train_steps  = []; train_losses = []
    eval_steps   = []; eval_losses  = []

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            step = int(row["step"]) if row["step"] else None
            if step is None:
                continue
            if row.get("train_loss", "").strip():
                train_steps.append(step)
                train_losses.append(float(row["train_loss"]))
            if row.get("eval_loss", "").strip():
                eval_steps.append(step)
                eval_losses.append(float(row["eval_loss"]))

    return train_steps, train_losses, eval_steps, eval_losses


# ===================
#  PLOTTING
# ===================

def plot_curves(cot_dir, nocot_dir, title, out):
    cot_ts,   cot_tl,   cot_es,   cot_el   = load_metrics(cot_dir)
    nocot_ts, nocot_tl, nocot_es, nocot_el = load_metrics(nocot_dir)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)

    BLUE  = "#2563EB"
    RED   = "#DC2626"
    ALPHA = 0.85

    # ── Left: training loss ─────────────────────────────────────
    ax = axes[0]
    ax.plot(cot_ts,   cot_tl,   color=BLUE, linewidth=1.4,
            alpha=ALPHA, label="CoT")
    ax.plot(nocot_ts, nocot_tl, color=RED,  linewidth=1.4,
            alpha=ALPHA, linestyle="--", label="No-CoT")
    ax.set_xlabel("Training step", fontsize=11)
    ax.set_ylabel("Loss",          fontsize=11)
    ax.set_title("Training Loss",  fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{int(x):,}"))

    # ── Right: evaluation loss ───────────────────────────────────
    ax = axes[1]
    if cot_es:
        ax.plot(cot_es, cot_el, color=BLUE, linewidth=1.4,
                alpha=ALPHA, marker="o", markersize=3.5, label="CoT")
    if nocot_es:
        ax.plot(nocot_es, nocot_el, color=RED, linewidth=1.4,
                alpha=ALPHA, linestyle="--", marker="s",
                markersize=3.5, label="No-CoT")
    ax.set_xlabel("Training step", fontsize=11)
    ax.set_ylabel("Loss",          fontsize=11)
    ax.set_title("Evaluation Loss", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{int(x):,}"))

    plt.tight_layout()
    os.makedirs(os.path.dirname(out) if os.path.dirname(out) else ".", exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved -> {out}")


# ===================
#  ENTRY POINT
# ===================

def main():
    ap = argparse.ArgumentParser(
        description="Plot CoT vs No-CoT training/eval loss curves."
    )
    ap.add_argument("--cot_dir",   required=True,
                    help="Path to CoT LoRA output directory (contains training_metrics.csv)")
    ap.add_argument("--nocot_dir", required=True,
                    help="Path to No-CoT LoRA output directory")
    ap.add_argument("--title",     default="CoT vs No-CoT Training Curves",
                    help="Plot title")
    ap.add_argument("--out",       default="figures/loss_curves.png",
                    help="Output path for the saved figure")
    args = ap.parse_args()

    plot_curves(args.cot_dir, args.nocot_dir, args.title, args.out)


if __name__ == "__main__":
    main()
