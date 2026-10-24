
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from config import (
    REPORT_DIR,
    CONFUSION_MATRIX,
    PREDICTIONS,
    CV_RESULTS,
)

# Save figures inside the Git repository.
OUT = REPORT_DIR / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# 1. Confusion matrix
cm_df = pd.read_csv(CONFUSION_MATRIX, index_col=0)

cm = cm_df.values
labels = cm_df.index.tolist()

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(cm, interpolation="nearest")
fig.colorbar(im, ax=ax, label="Number of samples")

ax.set(
    xticks=np.arange(len(labels)),
    yticks=np.arange(len(labels)),
    xticklabels=labels,
    yticklabels=labels,
    xlabel="Predicted family",
    ylabel="True family",
    title="Network Malware Family Classification",
)
plt.setp(ax.get_xticklabels(), rotation=35, ha="right")

threshold = cm.max() / 2 if cm.size else 0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(
            j, i, str(cm[i, j]),
            ha="center", va="center",
            color="white" if cm[i, j] > threshold else "black",
        )

fig.tight_layout()
fig.savefig(
    OUT / "confusion_matrix.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)

# 2. F1-score by family
pred = pd.read_csv(PREDICTIONS)

families = sorted(pred["family"].unique())
f1_values = []

for family in families:
    true_positive = (
        (pred["family"] == family) &
        (pred["predicted_family"] == family)
    ).sum()

    predicted_positive = (
        pred["predicted_family"] == family
    ).sum()

    actual_positive = (
        pred["family"] == family
    ).sum()

    precision = (
        true_positive / predicted_positive
        if predicted_positive else 0
    )
    recall = (
        true_positive / actual_positive
        if actual_positive else 0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0
    )
    f1_values.append(f1)

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(families, f1_values)
ax.set_ylim(0, 1)
ax.set_ylabel("F1-score")
ax.set_xlabel("Malware family")
ax.set_title("F1-score by Malware Family")

for bar, value in zip(bars, f1_values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.02,
        "%.2f" % value,
        ha="center",
        va="bottom",
    )

plt.xticks(rotation=30, ha="right")
fig.tight_layout()
fig.savefig(
    OUT / "f1_by_family.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)

# 3. Report where the cross-validation results are stored.
if CV_RESULTS.exists():
    cv_df = pd.read_csv(CV_RESULTS)
    print("Cross-validation results:", CV_RESULTS)
    print("Available columns:", list(cv_df.columns))
else:
    print("Cross-validation results not found:", CV_RESULTS)

print("\nFigures saved in:", OUT)
print("- confusion_matrix.png")
print("- f1_by_family.png")
