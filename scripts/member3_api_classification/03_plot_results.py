#!/usr/bin/env python3
"""Step 3 - Figures for the report: confusion matrix + F1 per family (from out-of-fold predictions)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

import config as cfg

df = pd.read_csv(str(cfg.PREDICTIONS))
kind = df["best_model"].iloc[0]
pred = df[kind + "_pred"]
labels = sorted(df["family"].unique())
cfg.FIG_DIR.mkdir(parents=True, exist_ok=True)

cm = confusion_matrix(df["family"], pred, labels=labels)
fig, ax = plt.subplots(figsize=(6, 5))
ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
ax.set_yticks(range(len(labels)), labels)
for i in range(len(labels)):
    for j in range(len(labels)):
        ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("API-call classifier (%s), 5-fold CV" % kind)
fig.tight_layout(); fig.savefig(str(cfg.FIG_DIR / "api_confusion_matrix.png"), dpi=200); plt.close(fig)

f1 = f1_score(df["family"], pred, labels=labels, average=None)
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(labels, f1)
ax.set_ylim(0, 1); ax.set_ylabel("F1-score"); ax.set_title("F1 per family - API-call classifier")
plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
fig.tight_layout(); fig.savefig(str(cfg.FIG_DIR / "api_f1_by_family.png"), dpi=200)
print("figures saved in", cfg.FIG_DIR)
