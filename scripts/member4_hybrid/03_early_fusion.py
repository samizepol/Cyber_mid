"""
03_early_fusion.py
Member 4 - Hybrid Model and Integration

Feature-level hybrid: instead of combining two finished predictions (01_late_fusion.py),
one classifier is trained on the API features and the network information together.

Methods (all evaluated on exactly the same folds as 01_late_fusion.py)
  api_features          Member 2's tfidf_lr model (TF-IDF over API 1-3-grams + sequence
                        statistics, logistic regression), re-trained inside our folds.
                        Baseline for this script.
  api_features+net_proba
                        the same features plus the network classifier's out-of-fold
                        probabilities (log, 5 columns) -> API evidence in full detail,
                        network evidence as a compact score.
  network_features      Random Forest on the per-pcap network statistics only
                        (needs NETWORK_FEATURES).
  api_features+net_features
                        true early fusion: TF-IDF + sequence statistics + all network
                        statistics in one feature vector (needs NETWORK_FEATURES).

NETWORK_FEATURES (data/network/network_dataset_labeled_270.csv) is not in the repo. If it is
missing or does not cover every sample, the two network-feature methods are skipped and
the script says so.

Usage (from the repo root; needs pandas, scikit-learn, joblib):
    python scripts/member4_hybrid/03_early_fusion.py
    python scripts/member4_hybrid/03_early_fusion.py --network-features <path.csv> --repeats 2
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg  # noqa: E402

# Same sequence statistics as Member 2 (02_train_api_classifier.py).
STATS = ["log_calls", "log_collapsed", "n_procs", "n_unique_apis", "api_entropy"]
# Columns of the network feature file that are identifiers / labels, not features.
NET_NON_FEATURES = {"analysis_id", "pcap_path", "status", "sha256", "family",
                    "label_agreement", "sample_id", "split"}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load(network_features_path):
    """One row per sample in BOTH the API sequences and the network OOF file."""
    seq = pd.read_json(cfg.API_SEQUENCES, lines=True)
    seq["text"] = seq["seq"].apply(" ".join)
    seq["log_calls"] = np.log1p(seq["n_calls"])
    seq["log_collapsed"] = np.log1p(seq["n_collapsed"])
    seq = seq.drop(columns=["seq"])

    net = pd.read_csv(cfg.NETWORK_OOF)
    families = sorted(c[len(cfg.NETWORK_PREFIX):] for c in net.columns if c.startswith(cfg.NETWORK_PREFIX))
    proba_cols = [cfg.NETWORK_PREFIX + f for f in families]
    df = seq.merge(net[["sample_id", "family"] + proba_cols], on="sample_id", suffixes=("", "_net"))
    assert (df["family"] == df["family_net"]).all(), "label mismatch API vs network"
    # log of the probabilities, like the stacking input in 01_late_fusion.py
    net_proba_cols = ["log_" + c for c in proba_cols]
    df[net_proba_cols] = np.log(df[proba_cols].to_numpy(float) + cfg.EPS)

    net_feature_cols = []
    path = Path(network_features_path)
    if path.exists():
        nf = pd.read_csv(path)
        net_feature_cols = [c for c in nf.columns
                            if c not in NET_NON_FEATURES and pd.api.types.is_numeric_dtype(nf[c])]
        covered = df["sample_id"].isin(nf["sample_id"]).sum()
        if covered < len(df):
            print("network features: %s covers only %d / %d samples -> network-feature methods skipped"
                  % (path.name, covered, len(df)))
            net_feature_cols = []
        else:
            df = df.merge(nf[["sample_id"] + net_feature_cols], on="sample_id", how="left")
            print("network features: %d columns from %s" % (len(net_feature_cols), path.name))
    else:
        print("network features: %s not found -> network-feature methods skipped" % path)

    # Same row order as 01_late_fusion.py -> same folds.
    df = df.sort_values("sample_id").reset_index(drop=True)
    print("samples: %d | families: %s" % (len(df), families))
    return df, families, net_proba_cols, net_feature_cols


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def tfidf():
    # Identical to Member 2's vectorizer so the api_features baseline matches their model.
    return TfidfVectorizer(token_pattern=r"\S+", lowercase=False, ngram_range=(1, 3),
                           min_df=2, max_features=20000, sublinear_tf=True)


def net_scaler():
    # Packet/byte counts are heavy-tailed -> log1p, then impute and standardize.
    return make_pipeline(FunctionTransformer(np.log1p), SimpleImputer(strategy="median"), StandardScaler())


def make_model(method, net_proba_cols, net_feature_cols):
    blocks = [("ngrams", tfidf(), "text"), ("stats", StandardScaler(), STATS)]
    if method == "network_features":
        return make_pipeline(
            ColumnTransformer([("net", SimpleImputer(strategy="median"), net_feature_cols)]),
            RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                   random_state=cfg.SEED, n_jobs=1))
    if method == "api_features+net_proba":
        blocks.append(("net_proba", StandardScaler(), net_proba_cols))
    elif method == "api_features+net_features":
        blocks.append(("net", net_scaler(), net_feature_cols))
    elif method != "api_features":
        raise ValueError(method)
    clf = LogisticRegression(C=10, max_iter=3000, class_weight="balanced")   # = Member 2's tfidf_lr
    return Pipeline([("features", ColumnTransformer(blocks)), ("clf", clf)])


def run_fold(fold_id, tr, te, df, methods, net_proba_cols, net_feature_cols):
    y = df["family"].to_numpy()
    rows = []
    for m in methods:
        model = make_model(m, net_proba_cols, net_feature_cols).fit(df.iloc[tr], y[tr])
        pred = model.predict(df.iloc[te])
        rows.append({"repeat": fold_id // cfg.N_SPLITS, "fold": fold_id % cfg.N_SPLITS, "method": m,
                     "accuracy": accuracy_score(y[te], pred),
                     "macro_f1": f1_score(y[te], pred, average="macro")})
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--network-features", default=str(cfg.NETWORK_FEATURES))
    ap.add_argument("--repeats", type=int, default=cfg.N_REPEATS)
    ap.add_argument("--out-dir", default=str(cfg.REPORT_DIR))
    args = ap.parse_args()

    df, families, net_proba_cols, net_feature_cols = load(args.network_features)
    methods = ["api_features", "api_features+net_proba"]
    if net_feature_cols:
        methods += ["network_features", "api_features+net_features"]

    cv = RepeatedStratifiedKFold(n_splits=cfg.N_SPLITS, n_repeats=args.repeats, random_state=cfg.SEED)
    splits = list(cv.split(df, df["family"]))
    print("running %d methods x %d folds ..." % (len(methods), len(splits)))
    rows = Parallel(n_jobs=-1)(delayed(run_fold)(i, tr, te, df, methods, net_proba_cols, net_feature_cols)
                               for i, (tr, te) in enumerate(splits))
    folds = pd.DataFrame([r for fold_rows in rows for r in fold_rows])

    summary = (folds.groupby("method", sort=False)[["accuracy", "macro_f1"]]
               .agg(["mean", "std"]).round(4))
    summary.columns = ["accuracy_mean", "accuracy_std", "macro_f1_mean", "macro_f1_std"]
    summary = summary.reset_index()
    # Paired comparison with the API-features baseline on the same test fold.
    wide = folds.pivot_table(index=["repeat", "fold"], columns="method", values="macro_f1")
    delta = wide.sub(wide["api_features"], axis=0)
    summary["delta_macro_f1_vs_api_mean"] = summary["method"].map(delta.mean()).round(4)
    summary["delta_macro_f1_vs_api_std"] = summary["method"].map(delta.std()).round(4)
    summary["folds_better_than_api"] = summary["method"].map((delta > 0).sum())
    summary["folds_total"] = len(wide)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    folds.to_csv(out_dir / cfg.EARLY_FOLD_RESULTS.name, index=False)
    summary.to_csv(out_dir / cfg.EARLY_CV_RESULTS.name, index=False)
    print("\n== Early fusion, repeated %dx%d-fold CV (n=%d) ==" % (args.repeats, cfg.N_SPLITS, len(df)))
    print(summary.to_string(index=False))
    print("\nSaved:", out_dir / cfg.EARLY_CV_RESULTS.name, out_dir / cfg.EARLY_FOLD_RESULTS.name, sep="\n  ")


if __name__ == "__main__":
    main()
