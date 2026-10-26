
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, f1_score, classification_report

BASE = "/mnt/data/Cyber_mid_repo"
DATA = os.path.join(
    BASE, "data/network/network_dataset_labeled_270.csv"
)
API_OOF = "/mnt/data/api_oof_predictions.csv"

OUT_DIR = os.path.join(BASE, "data/network")
OOF_OUT = os.path.join(OUT_DIR, "network_oof_predictions_compatible_api.csv")
MODEL_OUT = os.path.join(OUT_DIR, "network_model_compatible_api.joblib")
CV_OUT = os.path.join(OUT_DIR, "network_cv_results_compatible_api.csv")

FAMILIES = [
    "agenttesla", "amadey", "asyncrat", "formbook", "lokibot"
]
FEATURES = [
    "packet_count", "total_bytes", "duration_seconds",
    "average_packet_bytes", "tcp_packets", "udp_packets",
    "dns_packets", "http_packets", "tls_packets",
    "icmp_packets", "other_packets"
]

# Load datasets
df = pd.read_csv(DATA)
api = pd.read_csv(API_OOF)

# Validate identity, labels, and split compatibility
assert df["sha256"].is_unique, "Duplicate network hashes"
assert api["sha256"].is_unique, "Duplicate API hashes"

check = df[["sha256", "family", "split"]].merge(
    api[["sha256", "family", "split"]],
    on="sha256",
    how="outer",
    suffixes=("_network", "_api"),
    indicator=True,
    validate="one_to_one"
)
assert (check["_merge"] == "both").all(), "API/network samples differ"
assert (check["family_network"] == check["family_api"]).all(), \
    "API/network family labels differ"
assert (check["split_network"] == check["split_api"]).all(), \
    "API/network split assignments differ"

# Exclude the one empty PCAP and keep the same five family names
df = df[
    (df["pcap_status"] == "ok") &
    df["family"].isin(FAMILIES)
].copy().reset_index(drop=True)

X = df[FEATURES].replace([np.inf, -np.inf], np.nan)
y = df["family"].astype(str)

# Stratified five-fold OOF predictions, matching API model's CV approach
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("classifier", RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ))
])

# Each sample is predicted by a model that did not train on that sample
proba = cross_val_predict(
    model, X, y,
    cv=cv,
    method="predict_proba",
    n_jobs=1
)

# Fit a separate final model on all valid network samples for deployment
final_model = model.fit(X, y)
classes = list(final_model.named_steps["classifier"].classes_)

# Save probabilities in fixed family order
out = df[["sample_id", "sha256", "family", "split"]].copy()

for family in FAMILIES:
    if family in classes:
        out["network_proba_" + family] = proba[:, classes.index(family)]
    else:
        out["network_proba_" + family] = 0.0

prob_cols = ["network_proba_" + f for f in FAMILIES]
out["network_pred"] = np.array(FAMILIES)[
    out[prob_cols].to_numpy().argmax(axis=1)
]

# Report OOF performance
print("Valid PCAP samples:", len(df))
print("Families:", sorted(y.unique()))
print("OOF accuracy:", round(
    accuracy_score(y, out["network_pred"]), 4
))
print("OOF macro-F1:", round(
    f1_score(y, out["network_pred"], labels=FAMILIES,
             average="macro", zero_division=0), 4
))
print("\nOOF classification report:")
print(classification_report(
    y, out["network_pred"], labels=FAMILIES, zero_division=0
))

# Save to new files only
out.to_csv(OOF_OUT, index=False)
pd.DataFrame([{
    "method": "StratifiedKFold_5",
    "random_state": 42,
    "samples": len(df),
    "oof_accuracy": accuracy_score(y, out["network_pred"]),
    "oof_macro_f1": f1_score(
        y, out["network_pred"], labels=FAMILIES,
        average="macro", zero_division=0
    )
}]).to_csv(CV_OUT, index=False)

joblib.dump({
    "model": final_model,
    "features": FEATURES,
    "families": FAMILIES
}, MODEL_OUT)

print("\nSaved OOF probabilities:", OOF_OUT)
print("Saved CV results:", CV_OUT)
print("Saved final model:", MODEL_OUT)

