
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "network"
OUT = ROOT / "reports" / "network"
FIGURES = OUT / "figures"

PREDICTIONS = DATA / "network_oof_predictions_compatible_api.csv"

OUT.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(PREDICTIONS)

families = [
    "agenttesla",
    "amadey",
    "asyncrat",
    "formbook",
    "lokibot",
]

# Check required columns.
required = {"family", "network_pred"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {sorted(missing)}")

# Use the OOF predictions to evaluate the network classifier.
y_true = df["family"].astype(str).str.lower()
y_pred = df["network_pred"].astype(str).str.lower()

labels = [f for f in families if f in set(y_true)]

accuracy = accuracy_score(y_true, y_pred)
macro_f1 = f1_score(
    y_true, y_pred, labels=labels, average="macro", zero_division=0
)

report = classification_report(
    y_true,
    y_pred,
    labels=labels,
    output_dict=True,
    zero_division=0,
)

pd.DataFrame(report).transpose().to_csv(
    OUT / "network_cv_results.csv",
    index_label="family",
)

cm = confusion_matrix(y_true, y_pred, labels=labels)
pd.DataFrame(
    cm, index=labels, columns=labels
).to_csv(OUT / "network_final_confusion_matrix.csv", index_label="actual")

fig, ax = plt.subplots(figsize=(9, 7))
display = ConfusionMatrixDisplay(
    confusion_matrix=cm, display_labels=labels
)
display.plot(ax=ax, xticks_rotation=45, colorbar=False)
ax.set_title("Network Classifier — OOF Confusion Matrix")
fig.tight_layout()
fig.savefig(FIGURES / "confusion_matrix.png", dpi=160)
plt.close(fig)

f1_values = [report[f]["f1-score"] for f in labels]
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(labels, f1_values)
ax.set_ylabel("F1 score")
ax.set_ylim(0, 1)
ax.set_title("Network Classifier — F1 by Family (OOF)")
ax.tick_params(axis="x", rotation=30)
fig.tight_layout()
fig.savefig(FIGURES / "f1_by_family.png", dpi=160)
plt.close(fig)

with open(OUT / "RESULTS.md", "w", encoding="utf-8") as f:
    f.write("# API-Compatible Network Classifier Results\n\n")
    f.write("Evaluation uses the saved out-of-fold (OOF) predictions.\n\n")
    f.write(f"- Samples evaluated: **{len(df)}**\n")
    f.write(f"- Accuracy: **{accuracy:.4f}**\n")
    f.write(f"- Macro-F1: **{macro_f1:.4f}**\n\n")
    f.write("## Per-family results\n\n")
    f.write("| Family | Precision | Recall | F1 | Support |\n")
    f.write("|---|---:|---:|---:|---:|\n")
    for family in labels:
        r = report[family]
        f.write(
            f"| {family} | {r['precision']:.4f} | "
            f"{r['recall']:.4f} | {r['f1-score']:.4f} | "
            f"{int(r['support'])} |\n"
        )
    f.write("\n## Figures\n\n")
    f.write("- `figures/confusion_matrix.png`\n")
    f.write("- `figures/f1_by_family.png`\n")

print(f"Samples evaluated: {len(df)}")
print(f"OOF accuracy: {accuracy:.4f}")
print(f"OOF macro-F1: {macro_f1:.4f}")
print(f"Reports saved to: {OUT}")

