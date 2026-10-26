#!/usr/bin/env python3
"""Step 2 - Train and evaluate the API-call classifier.

Two candidate models on API n-gram TF-IDF + sequence statistics:
  tfidf_lr : Logistic Regression  (linear baseline, gives calibrated probabilities)
  tfidf_rf : Random Forest

Evaluation (same protocol as the network module, so the hybrid comparison is fair):
  A) Stratified 5-fold CV (seed 42) over ALL ok samples -> out-of-fold probabilities
     for every sample = what Member 4 needs for the hybrid model.
  B) Official split: fit on train+val, report on the fixed test split (reference only,
     the test set is small).
The vectorizer is fitted inside each fold, so there is no leakage.

  python 02_train_api_classifier.py
  python 02_train_api_classifier.py --families agenttesla,amadey,asyncrat,formbook,lokibot
"""
import argparse

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import config as cfg

STATS = ["log_calls", "log_collapsed", "n_procs", "n_unique_apis", "api_entropy"]


def load():
    df = pd.read_json(str(cfg.SEQUENCES), lines=True)
    df["text"] = df["seq"].apply(" ".join)
    df["log_calls"] = np.log1p(df["n_calls"])
    df["log_collapsed"] = np.log1p(df["n_collapsed"])
    return df.drop(columns=["seq"])


def make_model(kind):
    features = ColumnTransformer([
        ("ngrams", TfidfVectorizer(token_pattern=r"\S+", lowercase=False, ngram_range=(1, 3),
                                   min_df=2, max_features=20000, sublinear_tf=True), "text"),
        ("stats", StandardScaler(), STATS),
    ])
    if kind == "tfidf_lr":
        clf = LogisticRegression(C=10, max_iter=3000, class_weight="balanced")
    else:
        clf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                     random_state=cfg.SEED, n_jobs=-1)
    return Pipeline([("features", features), ("clf", clf)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", default="", help="comma-separated; default = all in the file")
    args = ap.parse_args()

    df = load()
    if args.families:
        df = df[df["family"].isin(args.families.split(","))]
    df = df.reset_index(drop=True)
    y = df["family"].astype(str)
    labels = sorted(y.unique())
    print("samples: %d | families: %s" % (len(df), labels))
    print(y.value_counts().to_string(), "\n")
    cfg.REPORT_DIR.mkdir(parents=True, exist_ok=True)

    cv = StratifiedKFold(n_splits=cfg.N_SPLITS, shuffle=True, random_state=cfg.SEED)
    results, oof = [], {}
    for kind in ("tfidf_lr", "tfidf_rf"):
        proba = cross_val_predict(make_model(kind), df[["text"] + STATS], y, cv=cv, method="predict_proba")
        pred = np.array(labels)[proba.argmax(axis=1)]
        oof[kind] = proba
        results.append({"model": kind, "accuracy": accuracy_score(y, pred),
                        "macro_f1": f1_score(y, pred, average="macro"),
                        "weighted_f1": f1_score(y, pred, average="weighted")})
        print("== %s (5-fold CV) ==  acc=%.4f  macroF1=%.4f" % (kind, results[-1]["accuracy"], results[-1]["macro_f1"]))
        print(classification_report(y, pred, zero_division=0))
        print("confusion (rows=true, cols=pred, order=%s)\n%s\n" % (labels, confusion_matrix(y, pred, labels=labels)))

    res = pd.DataFrame(results)
    res.to_csv(str(cfg.CV_RESULTS), index=False)
    best = res.sort_values("macro_f1", ascending=False).iloc[0]["model"]
    print("Best by CV macro-F1:", best)

    # Out-of-fold predictions of BOTH models -> input for the hybrid model (Member 4) and Member 5.
    out = df[["sample_id", "sha256", "family", "split"]].copy()
    for kind, proba in oof.items():
        for i, fam in enumerate(labels):
            out["%s_proba_%s" % (kind, fam)] = proba[:, i].round(5)
        out["%s_pred" % kind] = np.array(labels)[proba.argmax(axis=1)]
    out["best_model"] = best
    out.to_csv(str(cfg.PREDICTIONS), index=False)

    # B) official split reference + final model
    train, test = df[df["split"].isin(["train", "val"])], df[df["split"] == "test"]
    final = make_model(best).fit(train[["text"] + STATS], train["family"].astype(str))
    if len(test):
        tp = final.predict(test[["text"] + STATS])
        print("\n== Official split: fit train+val (%d), test (%d) ==  acc=%.4f macroF1=%.4f" % (
            len(train), len(test), accuracy_score(test["family"], tp),
            f1_score(test["family"], tp, average="macro")))
        print(classification_report(test["family"], tp, zero_division=0))
        proba = final.predict_proba(test[["text"] + STATS])
        tdf = test[["sample_id", "family"]].copy()
        tdf["pred"] = tp
        for i, fam in enumerate(final.classes_):
            tdf["proba_" + fam] = proba[:, i].round(5)
        tdf.to_csv(str(cfg.TEST_PREDICTIONS), index=False)

    # Final model trained on everything (for the demo / hybrid at inference time).
    full = make_model(best).fit(df[["text"] + STATS], y)
    joblib.dump({"model": full, "labels": labels, "stats": STATS, "kind": best}, str(cfg.MODEL_OUTPUT))
    print("\nSaved:", cfg.CV_RESULTS, cfg.PREDICTIONS, cfg.TEST_PREDICTIONS, cfg.MODEL_OUTPUT, sep="\n  ")


if __name__ == "__main__":
    main()
