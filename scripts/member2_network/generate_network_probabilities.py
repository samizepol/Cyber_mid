"""
generate_network_probabilities.py
Member 3 - Network-Traffic Classification (Step 2)

Alternative (official-split) version of the network model: trains the random forest on the
TRAIN split only and outputs class probabilities for every sample, so the test rows can be
used as a held-out set. The final project uses the cross-validated version instead
(train_network_compatible_api.py), because LokiBot has no test samples.

Usage:  python generate_network_probabilities.py
"""
import os
import sys
import joblib
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Dataset and output paths
BASE = "/mnt/data/Cyber_mid_repo"
DATASET = os.path.join(
    BASE, "data/network/network_dataset_labeled_270.csv"
)
API_FILE = "/mnt/data/api_oof_predictions.csv"
OUTPUT = os.path.join(
    BASE, "data/network/network_probabilities_aligned.csv"
)
MODEL_OUTPUT = os.path.join(
    BASE, "data/network/network_model_aligned.joblib"
)

FAMILIES = [
    "agenttesla", "amadey", "asyncrat", "formbook", "lokibot"
]

# Load data
df = pd.read_csv(DATASET)
api = pd.read_csv(API_FILE)

# Check that every sample has one matching API record
assert df["sha256"].is_unique, "Duplicate network SHA-256 values"
assert api["sha256"].is_unique, "Duplicate API SHA-256 values"

check = df[["sha256", "family", "split"]].merge(
    api[["sha256", "family", "split"]],
    on="sha256",
    how="outer",
    suffixes=("_net", "_api"),
    indicator=True,
    validate="one_to_one"
)

assert (check["_merge"] == "both").all(), "Sample hashes do not match"
assert (check["family_net"] == check["family_api"]).all(), \
    "Family labels do not match"
assert (check["split_net"] == check["split_api"]).all(), \
    "Split assignments do not match"

# Use only valid PCAPs for the network model
df = df[
    (df["pcap_status"] == "ok") &
    df["family"].isin(FAMILIES)
].copy()

# Numeric network features only; exclude identifiers and labels
excluded = {
    "sample_id", "sha256", "family", "split",
    "pcap_file", "pcap_status"
}
features = [
    c for c in df.select_dtypes(include=[np.number]).columns
    if c not in excluded
]

if not features:
    raise ValueError("No numeric network features found")

X = df[features].replace([np.inf, -np.inf], np.nan)
y = df["family"].astype(str)

train_mask = df["split"] == "train"
val_mask = df["split"] == "val"
test_mask = df["split"] == "test"

# Train only on the training split
model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("classifier", RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ))
])

model.fit(X.loc[train_mask], y.loc[train_mask])

# Predict probabilities for all valid rows.
# Hybrid evaluation should use the test rows only.
proba = model.predict_proba(X)
model_classes = list(model.named_steps["classifier"].classes_)

# Ensure fixed five-family probability columns.
result = df[["sample_id", "sha256", "family", "split"]].copy()

for family in FAMILIES:
    if family in model_classes:
        result["network_proba_" + family] = proba[
            :, model_classes.index(family)
        ]
    else:
        result["network_proba_" + family] = 0.0

result["network_pred"] = [
    FAMILIES[i] for i in np.argmax(
        result[["network_proba_" + f for f in FAMILIES]].values,
        axis=1
    )
]

# Save aligned probabilities and model separately
result.to_csv(OUTPUT, index=False)
joblib.dump(
    {"model": model, "features": features, "families": FAMILIES},
    MODEL_OUTPUT
)

# Report test performance; LokiBot may be absent from this existing test split.
test_result = result[result["split"] == "test"]
print("Valid PCAP samples:", len(df))
print("Features:", features)
print("Train samples:", int(train_mask.sum()))
print("Validation samples:", int(val_mask.sum()))
print("Test samples:", int(test_mask.sum()))
print("Test accuracy:", round(
    accuracy_score(test_result["family"], test_result["network_pred"]), 4
))
print("Test macro-F1:", round(
    f1_score(
        test_result["family"], test_result["network_pred"],
        labels=FAMILIES, average="macro", zero_division=0
    ), 4
))
print("\nTest classification report:")
print(classification_report(
    test_result["family"], test_result["network_pred"],
    labels=FAMILIES, zero_division=0
))
print("\nProbabilities saved to:", OUTPUT)
print("Model saved to:", MODEL_OUTPUT)

