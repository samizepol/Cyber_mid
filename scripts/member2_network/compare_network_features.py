
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from config import DATASET

df = pd.read_csv(DATASET)
df = df[(df["status"] == "ok") & df["family"].notna()].copy()

excluded = {
    "analysis_id", "pcap_path", "sha256", "family",
    "label_agreement", "sample_id", "status"
}

all_features = [
    c for c in df.select_dtypes(include="number").columns
    if c not in excluded
]

# Modelo B: retirar constantes y algunas características redundantes.
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

reduced_features = [c for c in all_features if c not in drop_features]

X_all = df[all_features].replace([np.inf, -np.inf], np.nan)
X_reduced = df[reduced_features].replace([np.inf, -np.inf], np.nan)
y = df["family"].astype(str)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def make_model():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=42
        ))
    ])

scoring = {
    "accuracy": "accuracy",
    "f1_macro": "f1_macro"
}

for name, X in [
    ("A - Todas las características", X_all),
    ("B - Características reducidas", X_reduced)
]:
    results = cross_validate(
        make_model(), X, y,
        cv=cv,
        scoring=scoring,
        error_score="raise"
    )

    print("\n" + name)
    print("Número de características:", X.shape[1])
    print("Accuracy por partición:",
          np.round(results["test_accuracy"], 3))
    print("Accuracy promedio:",
          round(results["test_accuracy"].mean(), 4))
    print("F1 macro por partición:",
          np.round(results["test_f1_macro"], 3))
    print("F1 macro promedio:",
          round(results["test_f1_macro"].mean(), 4))
    print("Desviación estándar F1:",
          round(results["test_f1_macro"].std(), 4))
