"""
02_plot_results.py
Member 4 - Hybrid Model and Integration

Figures for the report/PPT from the outputs of 01_late_fusion.py:
  method_comparison.png    macro-F1 (mean +/- std over all outer folds) per method
  confusion_matrices.png   API vs. network vs. best hybrid (first repeat, all samples)
  f1_by_family.png         per-family F1 of the same three models

Usage (from the repo root; needs pandas, matplotlib, scikit-learn):
    python scripts/member4_hybrid/02_plot_results.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                     # write files only, no window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg  # noqa: E402

LABELS = {"api_only": "API only", "network_only": "Network only", "mean": "Mean",
          "product": "Product", "weighted": "Weighted", "stacking": "Stacking"}


def main():
    summary = pd.read_csv(cfg.CV_RESULTS)
    preds = pd.read_csv(cfg.PREDICTIONS)
    per_family = pd.read_csv(cfg.PER_FAMILY, index_col="family")
    families = list(per_family.index)
    # Best fusion method = highest mean macro-F1 among the non-baseline methods.
    fusion = summary[~summary["method"].isin(["api_only", "network_only"])]
    best = fusion.sort_values("macro_f1_mean", ascending=False).iloc[0]["method"]
    shown = ["api_only", "network_only", best]
    cfg.FIG_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Macro-F1 per method, baselines grey, best hybrid highlighted.
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#9e9e9e" if m in ("api_only", "network_only") else
              ("#1f77b4" if m == best else "#9ecae1") for m in summary["method"]]
    ax.bar([LABELS[m] for m in summary["method"]], summary["macro_f1_mean"],
           yerr=summary["macro_f1_std"], capsize=4, color=colors)
    for i, (v, s) in enumerate(zip(summary["macro_f1_mean"], summary["macro_f1_std"])):
        ax.text(i, v + s + 0.015, "%.3f" % v, ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Macro-F1 (mean ± std, %d folds)" % int(summary["folds_total"].iloc[0]))
    ax.set_ylim(0, 1)
    ax.set_title("API vs. network vs. hybrid (late fusion)")
    fig.tight_layout()
    fig.savefig(cfg.FIG_DIR / "method_comparison.png", dpi=200)
    plt.close(fig)

    # 2) Row-normalized confusion matrices of the three models side by side.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, m in zip(axes, shown):
        cm = confusion_matrix(preds["family"], preds["%s_pred" % m], labels=families)
        cm_norm = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(len(families)):
            for j in range(len(families)):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=9,
                        color="white" if cm_norm[i, j] > 0.5 else "black")
        ax.set_xticks(range(len(families)), families, rotation=45, ha="right")
        ax.set_yticks(range(len(families)), families)
        ax.set_xlabel("Predicted")
        ax.set_title(LABELS[m] + (" (best hybrid)" if m == best else ""))
    axes[0].set_ylabel("True")
    fig.tight_layout()
    fig.savefig(cfg.FIG_DIR / "confusion_matrices.png", dpi=200)
    plt.close(fig)

    # 3) Per-family F1, grouped bars.
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(families))
    width = 0.27
    for k, m in enumerate(shown):
        ax.bar(x + (k - 1) * width, per_family[m], width, label=LABELS[m])
    ax.set_xticks(x, families)
    ax.set_ylabel("F1 (first repeat)")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_title("F1 per family")
    fig.tight_layout()
    fig.savefig(cfg.FIG_DIR / "f1_by_family.png", dpi=200)
    plt.close(fig)

    print("best hybrid:", best)
    print("Saved figures to", cfg.FIG_DIR)


if __name__ == "__main__":
    main()
