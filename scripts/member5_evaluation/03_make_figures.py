"""
03_make_figures.py
Member 5 - Evaluation, Experiments and Reproducibility

Figures for the report and slides, drawn from the tables of 02_compare_all.py.
They add what Member 4's figures (reports/hybrid/figures/) do not show:

  macro_f1_with_ci.png      macro-F1 of every method with its 95% bootstrap CI
  delta_vs_api.png          paired gain over the API model, with 95% CI (is the gain real?)
  label_noise.png           accuracy on high- vs. low-confidence labels (AVClass agreement)
  confusion_api_net_hybrid.png  the three headline confusion matrices, normalised per row

Usage (from the repo root, after 02_compare_all.py):
    python scripts/member5_evaluation/03_make_figures.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                      # no window needed, works on any machine
import matplotlib.pyplot as plt           # noqa: E402
import numpy as np                        # noqa: E402
import pandas as pd                       # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg  # noqa: E402

# One colour per source, used the same way in every figure.
COLOR = {"api": "#2a78d6", "network": "#eb6834", "hybrid": "#1baf7a"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"
LABEL = {"api_lr": "API (LR)", "api_rf": "API (RF)", "network": "Network (RF)", "mean": "Hybrid: mean",
         "product": "Hybrid: product", "weighted": "Hybrid: weighted", "stacking": "Hybrid: stacking"}


def source(method):
    return "api" if method.startswith("api") else "network" if method == "network" else "hybrid"


def style(ax):
    """Quiet axes: light grid behind the data, no box."""
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=INK, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def ci_plot(table, value, low, high, xlabel, title, path, zero_line=False):
    """Horizontal dot + whisker chart, one row per method, best at the top."""
    t = table.dropna(subset=[value]).sort_values(value)
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(t) + 1.2))
    y = np.arange(len(t))
    for i, (_, r) in enumerate(t.iterrows()):
        c = COLOR[source(r["method"])]
        ax.plot([r[low], r[high]], [i, i], color=c, linewidth=2, solid_capstyle="round")
        ax.plot(r[value], i, "o", color=c, markersize=8, markeredgecolor="white", markeredgewidth=1.5)
        ax.text(r[high] + 0.006, i, f"{r[value]:+.3f}" if zero_line else f"{r[value]:.3f}",
                va="center", fontsize=9, color=INK)
    if zero_line:
        ax.axvline(0, color=MUTED, linewidth=1)
    ax.set_yticks(y, [LABEL[m] for m in t["method"]], fontsize=10)
    ax.set_xlabel(xlabel, color=INK)
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def label_noise_plot(noise, path):
    groups = list(dict.fromkeys(noise["group"]))
    methods = list(dict.fromkeys(noise["method"]))
    fig, ax = plt.subplots(figsize=(7, 3.6))
    width = 0.26
    for j, m in enumerate(methods):
        sub = noise[noise["method"] == m].set_index("group").loc[groups]
        x = np.arange(len(groups)) + (j - 1) * (width + 0.02)
        err = [sub["accuracy"] - sub["accuracy_ci_low"], sub["accuracy_ci_high"] - sub["accuracy"]]
        ax.bar(x, sub["accuracy"], width, color=COLOR[source(m)], label=LABEL.get(m, m), zorder=2)
        ax.errorbar(x, sub["accuracy"], yerr=err, fmt="none", ecolor=INK, elinewidth=1, capsize=3, zorder=3)
        for xi, v in zip(x, sub["accuracy"]):
            ax.text(xi, 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=8, color="white", zorder=4)
    n = noise.drop_duplicates("group").set_index("group")["n"]
    ax.set_xticks(np.arange(len(groups)), [f"{g}\nn = {n[g]}" for g in groups], fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy (95% CI)", color=INK)
    ax.set_title("All models are more accurate when AVClass agrees with the MalwareBazaar label",
                 loc="left", fontsize=10, color=INK)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.legend(frameon=False, fontsize=9, ncol=3, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def confusion_plot(cms, path):
    methods = ["api_lr", "network", "weighted"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3))
    for ax, m in zip(axes, methods):
        cm = (cms[cms["method"] == m].pivot(index="true", columns="pred", values="count")
              .loc[cfg.FAMILIES, cfg.FAMILIES].to_numpy())
        norm = cm / cm.sum(axis=1, keepdims=True)            # row = true family, sums to 1
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)        # same scale in all three panels
        for i in range(len(cfg.FAMILIES)):
            for j in range(len(cfg.FAMILIES)):
                ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=9,
                        color="white" if norm[i, j] > 0.55 else INK)
        ax.set_xticks(range(5), cfg.FAMILIES, rotation=35, ha="right", fontsize=8)
        ax.set_yticks(range(5), cfg.FAMILIES if ax is axes[0] else [""] * 5, fontsize=8)
        ax.set_title(LABEL[m], fontsize=10, color=INK)
        ax.set_xlabel("predicted", fontsize=8, color=MUTED)
    axes[0].set_ylabel("true family", fontsize=8, color=MUTED)
    fig.suptitle("Confusion matrices (counts; colour = share of the true family), n = 269",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    cfg.FIG_DIR.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(cfg.RESULTS_CSV)
    table = table[table["method"].isin(LABEL)]           # methods with per-sample predictions

    ci_plot(table, "macro_f1", "macro_f1_ci_low", "macro_f1_ci_high",
            "Macro-F1 (dot) with 95% bootstrap CI (line)",
            "Weighted late fusion scores highest; network alone is clearly weakest",
            cfg.FIG_DIR / "macro_f1_with_ci.png")
    ci_plot(table[table["method"] != "api_lr"], "delta_macro_f1_vs_api", "delta_ci_low", "delta_ci_high",
            "Change in macro-F1 vs. the API model (paired bootstrap, 95% CI)",
            "Weighted fusion gains +0.037 over API, but its CI still touches zero",
            cfg.FIG_DIR / "delta_vs_api.png", zero_line=True)
    label_noise_plot(pd.read_csv(cfg.LABEL_NOISE_CSV), cfg.FIG_DIR / "label_noise.png")
    confusion_plot(pd.read_csv(cfg.CONFUSION_CSV), cfg.FIG_DIR / "confusion_api_net_hybrid.png")
    print("Saved figures to", cfg.FIG_DIR)


if __name__ == "__main__":
    main()
