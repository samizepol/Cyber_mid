"""
final_evaluation_network.py
Member 3 - Network-Traffic Classification (Step 2)

Out-of-fold evaluation of the reduced-feature random forest (experiment B of
compare_network_features.py) on the first labelled network dataset. Saves the per-sample
predictions (PREDICTIONS) and the confusion matrix (CONFUSION_MATRIX).
Note: this is the earlier experiment; the final model used in the report and the hybrid is
train_network_compatible_api.py.

Usage:  python final_evaluation_network.py
"""
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)

from config import DATASET, CONFUSION_MATRIX, PREDICTIONS

df = pd.read_csv(DATASET)
df = df[(df["status"] == "ok") & df["family"].notna()].copy()
df = df.reset_index(drop=True)

# Identifier and label columns are never used as features
excluded = {
    "analysis_id", "pcap_path", "sha256", "family",
    "label_agreement", "sample_id", "status"
}

features = [
    col for col in df.select_dtypes(include="number").columns
    if col not in excluded
]

X = df[features].replace([np.inf, -np.inf], np.nan)
y = df["family"].astype(str)

# Features removed in experiment B (see compare_network_features.py).
drop_features = [
    "ipv6_packets",
    "icmpv6_packets",
    "dhcp_bootp_packets",
    "ipv4_packets",
    "tcp_ack_packets",
    "average_packet_bytes",
    "igmp_packets",
    "smb_packets",
    "netbios_datagram_packets",
    "ssdp_packets",
    "unique_tcp_flows",
    "unique_ip_pairs",
    "unique_dst_ips",
    "tcp_fin_packets",
    "unique_udp_flows"
]

reduced_features = [c for c in features if c not in drop_features]
X_reduced = X[reduced_features].copy()

# Five stratified, reproducible folds.
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Out-of-fold predictions: every sample is predicted
# by a model that was not trained on that sample.
model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("classifier", RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42
    ))
])

predictions = cross_val_predict(
    model, X_reduced, y,
    cv=cv,
    method="predict"
)

labels = sorted(y.unique())

print("=== EVALUACIÓN FINAL DEL MODELO NETWORK ===")
print("Muestras:", len(df))
print("Características:", len(reduced_features))
print("Familias:", len(labels))

print("\nAccuracy global:", round(accuracy_score(y, predictions), 4))
print("F1 macro global:", round(f1_score(
    y, predictions, average="macro"
), 4))

print("\nInforme por familia:")
print(classification_report(
    y, predictions,
    labels=labels,
    zero_division=0
))

cm = confusion_matrix(y, predictions, labels=labels)
print("\nOrden de familias:", labels)
print("Matriz de confusión:")
print(cm)

# Save the predictions and metrics to document the result.
output = df[["analysis_id", "sha256", "family"]].copy()
output["predicted_family"] = predictions
output["correct"] = output["family"] == output["predicted_family"]
output.to_csv(PREDICTIONS, index=False)

pd.DataFrame(
    cm, index=labels, columns=labels
).to_csv(CONFUSION_MATRIX)

print("\nArchivos guardados:")
print(PREDICTIONS)
print(CONFUSION_MATRIX)
