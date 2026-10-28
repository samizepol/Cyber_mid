"""
01_late_fusion.py
Member 4 - Hybrid Model and Integration

Combines the API-call classifier (Member 2) and the network classifier (Member 3) at the
decision level: both models already give one probability per family for every sample
(out-of-fold, so no sample was predicted by a model that saw it during training).
This script learns how to combine the two probability vectors and compares the result with
both single-source classifiers on exactly the same samples and the same folds.

Fusion methods
  api_only / network_only  single-source baselines (their OOF probabilities as they are)
  mean                     (p_api + p_net) / 2, nothing is learned
  product                  p_api * p_net, renormalized (both sources must agree)
  weighted                 w * p_api + (1 - w) * p_net, w chosen on the training folds
  stacking                 logistic regression on [log p_api, log p_net] (10 inputs),
                           trained on the training folds

Evaluation
  Repeated stratified 5-fold CV (10 repeats) over the samples present in BOTH files.
  Methods that learn something (weighted, stacking) only see the training folds of each
  outer split. Baselines are scored on the same test folds, so all rows are comparable.
  McNemar's exact test compares every fusion method with the API baseline (first repeat).

Usage (from the repo root; needs pandas, scikit-learn):
    python scripts/member4_hybrid/01_late_fusion.py
"""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg  # noqa: E402


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load():
    """Join both OOF files on sample_id and return (meta, P_api, P_net, families)."""
    api = pd.read_csv(cfg.API_OOF)
    net = pd.read_csv(cfg.NETWORK_OOF)

    api_model = api["best_model"].iloc[0]                    # e.g. tfidf_lr
    api_prefix = cfg.API_PREFIX.format(model=api_model)
    families = sorted(c[len(cfg.NETWORK_PREFIX):] for c in net.columns if c.startswith(cfg.NETWORK_PREFIX))
    api_families = sorted(c[len(api_prefix):] for c in api.columns if c.startswith(api_prefix))
    assert families == api_families, "API and network models use different families"

    df = api.merge(net, on="sample_id", suffixes=("", "_net"), how="inner")
    assert (df["family"] == df["family_net"]).all(), "label mismatch between API and network file"
    assert (df["split"] == df["split_net"]).all(), "split mismatch between API and network file"
    missing = sorted(set(api["sample_id"]) ^ set(net["sample_id"]))
    print("API model: %s | samples in both files: %d | dropped (only in one file): %s"
          % (api_model, len(df), missing or "none"))

    # Fixed row order -> identical folds on every machine.
    df = df.sort_values("sample_id").reset_index(drop=True)
    p_api = df[[api_prefix + f for f in families]].to_numpy(float)
    p_net = df[[cfg.NETWORK_PREFIX + f for f in families]].to_numpy(float)
    # Rows should already sum to 1; renormalize against rounding in the CSVs.
    p_api /= p_api.sum(axis=1, keepdims=True)
    p_net /= p_net.sum(axis=1, keepdims=True)
    return df[["sample_id", "sha256", "family", "split"]], p_api, p_net, families


# ---------------------------------------------------------------------------
# Fusion methods: fit on the training rows, return probabilities for the test rows
# ---------------------------------------------------------------------------

def macro_f1(y, proba, families):
    return f1_score(y, np.asarray(families)[proba.argmax(axis=1)], average="macro")


def log_features(p_api, p_net):
    return np.hstack([np.log(p_api + cfg.EPS), np.log(p_net + cfg.EPS)])


def fuse(method, tr, te, p_api, p_net, y, families):
    """Return (test-row probabilities, learned parameter or None)."""
    if method == "api_only":
        return p_api[te], None
    if method == "network_only":
        return p_net[te], None
    if method == "mean":
        return (p_api[te] + p_net[te]) / 2, None
    if method == "product":
        p = p_api[te] * p_net[te] + 1e-12
        return p / p.sum(axis=1, keepdims=True), None
    if method == "weighted":
        # Pick the API weight with the best macro-F1 on the TRAINING rows only.
        scores = [macro_f1(y[tr], w * p_api[tr] + (1 - w) * p_net[tr], families) for w in cfg.WEIGHT_GRID]
        w = cfg.WEIGHT_GRID[int(np.argmax(scores))]
        return w * p_api[te] + (1 - w) * p_net[te], w
    if method == "stacking":
        meta = LogisticRegression(max_iter=2000, C=1.0)
        meta.fit(log_features(p_api[tr], p_net[tr]), y[tr])
        assert list(meta.classes_) == families
        return meta.predict_proba(log_features(p_api[te], p_net[te])), None
    raise ValueError(method)


METHODS = ["api_only", "network_only", "mean", "product", "weighted", "stacking"]


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def mcnemar_exact(correct_a, correct_b):
    """Two-sided exact McNemar test; b/c = samples only A / only B gets right."""
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    n = b + c
    if n == 0:
        return b, c, 1.0
    k = min(b, c)
    p = 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return b, c, min(p, 1.0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    meta, p_api, p_net, families = load()
    y = meta["family"].to_numpy()
    print("families:", families)
    print(meta["family"].value_counts().to_string(), "\n")

    # How complementary are the two sources? (uses the OOF predictions directly)
    ok_api = np.asarray(families)[p_api.argmax(1)] == y
    ok_net = np.asarray(families)[p_net.argmax(1)] == y
    print("complementarity: both right %d | only API %d | only network %d | both wrong %d"
          % ((ok_api & ok_net).sum(), (ok_api & ~ok_net).sum(), (~ok_api & ok_net).sum(), (~ok_api & ~ok_net).sum()))
    print("oracle upper bound (either model right): %.4f\n" % (ok_api | ok_net).mean())

    cv = RepeatedStratifiedKFold(n_splits=cfg.N_SPLITS, n_repeats=cfg.N_REPEATS, random_state=cfg.SEED)
    rows = []
    first_repeat = {m: np.zeros_like(p_api) for m in METHODS}   # OOF fused probabilities, repeat 0
    weights = []

    for fold_id, (tr, te) in enumerate(cv.split(p_api, y)):
        repeat = fold_id // cfg.N_SPLITS
        for m in METHODS:
            proba, param = fuse(m, tr, te, p_api, p_net, y, families)
            pred = np.asarray(families)[proba.argmax(axis=1)]
            rows.append({"repeat": repeat, "fold": fold_id % cfg.N_SPLITS, "method": m,
                         "accuracy": accuracy_score(y[te], pred),
                         "macro_f1": f1_score(y[te], pred, average="macro"),
                         "param": param})
            if m == "weighted":
                weights.append(param)
            if repeat == 0:
                first_repeat[m][te] = proba

    cfg.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    folds = pd.DataFrame(rows)
    folds.to_csv(cfg.FOLD_RESULTS, index=False)

    summary = (folds.groupby("method", sort=False)[["accuracy", "macro_f1"]]
               .agg(["mean", "std"]).round(4))
    summary.columns = ["accuracy_mean", "accuracy_std", "macro_f1_mean", "macro_f1_std"]
    summary = summary.reset_index()
    # Paired comparison with the API baseline on the SAME test fold: the fold-to-fold std
    # above is mostly fold difficulty, the paired difference removes it.
    wide = folds.pivot_table(index=["repeat", "fold"], columns="method", values="macro_f1")
    delta = wide.sub(wide["api_only"], axis=0)
    summary["delta_macro_f1_vs_api_mean"] = summary["method"].map(delta.mean()).round(4)
    summary["delta_macro_f1_vs_api_std"] = summary["method"].map(delta.std()).round(4)
    summary["folds_better_than_api"] = summary["method"].map((delta > 0).sum())
    summary["folds_total"] = len(wide)
    summary.to_csv(cfg.CV_RESULTS, index=False)
    print("== Repeated %dx%d-fold CV (n=%d) ==" % (cfg.N_REPEATS, cfg.N_SPLITS, len(y)))
    print(summary.to_string(index=False))
    print("\nweighted: API weight chosen per fold: mean %.2f, min %.2f, max %.2f"
          % (np.mean(weights), np.min(weights), np.max(weights)))

    # Per-sample OOF predictions of the first repeat -> confusion matrices, Member 5.
    out = meta.copy()
    for m in METHODS:
        for i, fam in enumerate(families):
            out["%s_proba_%s" % (m, fam)] = first_repeat[m][:, i].round(5)
        out["%s_pred" % m] = np.asarray(families)[first_repeat[m].argmax(axis=1)]
    out.to_csv(cfg.PREDICTIONS, index=False)

    # Per-family F1 (first repeat) for API, network and every fusion method.
    per_family = pd.DataFrame({
        m: f1_score(y, out["%s_pred" % m], labels=families, average=None) for m in METHODS
    }, index=families).round(4)
    per_family.index.name = "family"
    per_family.to_csv(cfg.PER_FAMILY)
    print("\n== F1 per family (first repeat) ==")
    print(per_family.to_string())

    # McNemar: does a fusion method make significantly different errors than the API model?
    tests = []
    correct_api = out["api_only_pred"].to_numpy() == y
    for m in METHODS[1:]:
        correct_m = out["%s_pred" % m].to_numpy() == y
        b, c, p = mcnemar_exact(correct_api, correct_m)
        tests.append({"method": m, "only_api_right": b, "only_method_right": c, "p_value": round(p, 4)})
    tests = pd.DataFrame(tests)
    tests.to_csv(cfg.MCNEMAR, index=False)
    print("\n== McNemar exact test vs. api_only (first repeat) ==")
    print(tests.to_string(index=False))

    print("\nSaved:", cfg.CV_RESULTS, cfg.FOLD_RESULTS, cfg.PREDICTIONS, cfg.PER_FAMILY, cfg.MCNEMAR, sep="\n  ")


if __name__ == "__main__":
    main()
