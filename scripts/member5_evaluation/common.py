"""Helpers shared by the Member 5 scripts: loading the aligned predictions and metrics."""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

import config as cfg


def load_predictions():
    """Return one DataFrame, one row per sample present in ALL prediction files.

    Columns: sample_id, family, split, label_agreement and one '<method>' column per entry
    of cfg.METHODS holding that method's out-of-fold predicted family.
    """
    api = pd.read_csv(cfg.API_OOF)
    hyb = pd.read_csv(cfg.HYBRID_OOF)
    meta = pd.read_csv(cfg.SAMPLES_CSV, usecols=["sample_id", "label_agreement", "analysis_status"])

    df = hyb[["sample_id", "family", "split"]].merge(
        api[["sample_id", "family", "split", "tfidf_lr_pred", "tfidf_rf_pred"]],
        on="sample_id", how="inner", suffixes=("", "_api"), validate="one_to_one")
    # Labels and splits must be identical in every file, otherwise the comparison is meaningless.
    assert (df["family"] == df["family_api"]).all(), "family labels differ between API and hybrid files"
    assert (df["split"] == df["split_api"]).all(), "split assignments differ between API and hybrid files"

    for name, (col, source, _) in cfg.METHODS.items():
        df[name] = (api if source == "api" else hyb).set_index("sample_id")[col].reindex(df["sample_id"]).to_numpy()
    df = df.merge(meta, on="sample_id", how="left", validate="one_to_one")
    assert (df["analysis_status"] == "ok").all(), "a predicted sample is not an ok Cuckoo run"
    keep = ["sample_id", "family", "split", "label_agreement"] + list(cfg.METHODS)
    return df[keep].sort_values("sample_id").reset_index(drop=True)


def _encode(labels):
    """Family names -> integer codes 0..4 in cfg.FAMILIES order."""
    index = {f: i for i, f in enumerate(cfg.FAMILIES)}
    return np.array([index[v] for v in labels])


def _scores_from_codes(t, p):
    """Same numbers as sklearn's accuracy_score / f1_score(zero_division=0), but computed
    from one confusion matrix with numpy - fast enough for thousands of bootstrap resamples."""
    k = len(cfg.FAMILIES)
    cm = np.bincount(t * k + p, minlength=k * k).reshape(k, k)
    tp = np.diag(cm).astype(float)
    support, predicted = cm.sum(axis=1), cm.sum(axis=0)
    denom = support + predicted
    f1 = np.divide(2 * tp, denom, out=np.zeros(k), where=denom > 0)   # F1 = 2TP / (2TP + FP + FN)
    return {"accuracy": tp.sum() / len(t),
            "macro_f1": f1.mean(),
            "weighted_f1": (f1 * support).sum() / support.sum()}


def scores(y_true, y_pred):
    """Accuracy, macro-F1 and weighted-F1 for one set of predictions (family names)."""
    return _scores_from_codes(_encode(y_true), _encode(y_pred))


def bootstrap_ci(y_true, y_pred, metric, n=cfg.N_BOOTSTRAP, seed=cfg.SEED, baseline=None):
    """Percentile 95% CI of a metric by resampling samples with replacement.

    With `baseline` (predictions of another model on the SAME samples) the CI is for the
    paired difference metric(y_pred) - metric(baseline): both are computed on the same
    resample, so the sample-to-sample difficulty cancels out.
    """
    t, p = _encode(y_true), _encode(y_pred)
    b = None if baseline is None else _encode(baseline)
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n):
        idx = rng.integers(0, len(t), len(t))
        value = _scores_from_codes(t[idx], p[idx])[metric]
        if b is not None:
            value -= _scores_from_codes(t[idx], b[idx])[metric]
        stats.append(value)
    return np.percentile(stats, [2.5, 97.5])


def _self_test():
    """The fast metrics must match scikit-learn exactly (run: python common.py)."""
    rng = np.random.default_rng(0)
    for _ in range(200):
        t = rng.choice(cfg.FAMILIES, 50)
        p = rng.choice(cfg.FAMILIES[:3], 50)          # some families never predicted
        s = scores(t, p)
        assert np.isclose(s["accuracy"], accuracy_score(t, p))
        assert np.isclose(s["macro_f1"], f1_score(t, p, labels=cfg.FAMILIES, average="macro", zero_division=0))
        assert np.isclose(s["weighted_f1"], f1_score(t, p, labels=cfg.FAMILIES, average="weighted", zero_division=0))
    print("fast metrics == scikit-learn: OK")


if __name__ == "__main__":
    _self_test()
