"""
02_compare_all.py
Member 5 - Evaluation, Experiments and Reproducibility

Builds the single source of truth for the report and the slides: every classifier
(API, network, every hybrid method) evaluated with the SAME protocol on the SAME samples.

Protocol (docs/evaluation.md)
  - 269 samples = the ok Cuckoo runs that have both an API trace and a network row.
  - Every prediction is out-of-fold (5-fold stratified CV, seed 42): no sample is predicted
    by a model that saw it in training.
  - Main metric: macro-F1 (each family counts the same). Also accuracy and weighted-F1.
  - 95% confidence intervals: bootstrap over samples (2000 resamples, seed 42).
  - Paired difference vs. the API model with its own bootstrap CI (same resamples).
  - Repeated-CV mean +/- std (10x5 folds) copied from Member 4 where it exists.

Outputs (reports/final/)
  results_table.csv / .md     master table, one row per method
  per_family_metrics.csv      precision, recall, F1, support per family and method
  confusion_matrices.csv      long format: method, true family, predicted family, count
  test_split_reference.csv    official fixed test split (API model only, 4 families)
  label_noise.csv             metrics on high- vs. low-confidence labels (AVClass agreement)

Usage (from the repo root):
    python scripts/member5_evaluation/02_compare_all.py
"""
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg  # noqa: E402
from common import bootstrap_ci, load_predictions, scores  # noqa: E402


def master_table(df):
    """One row per method: pooled OOF metrics + CIs + paired delta vs. API + repeated-CV numbers."""
    y = df["family"].to_numpy()
    repeated = pd.read_csv(cfg.HYBRID_CV).set_index("method")
    rows = []
    for name, (_, _, description) in cfg.METHODS.items():
        pred = df[name].to_numpy()
        s = scores(y, pred)
        f1_lo, f1_hi = bootstrap_ci(y, pred, "macro_f1")
        acc_lo, acc_hi = bootstrap_ci(y, pred, "accuracy")
        row = {"method": name, "description": description, "n": len(y),
               "accuracy": s["accuracy"], "accuracy_ci_low": acc_lo, "accuracy_ci_high": acc_hi,
               "macro_f1": s["macro_f1"], "macro_f1_ci_low": f1_lo, "macro_f1_ci_high": f1_hi,
               "weighted_f1": s["weighted_f1"]}
        if name != "api_lr":
            d_lo, d_hi = bootstrap_ci(y, pred, "macro_f1", baseline=df["api_lr"].to_numpy())
            row.update(delta_macro_f1_vs_api=s["macro_f1"] - scores(y, df["api_lr"])["macro_f1"],
                       delta_ci_low=d_lo, delta_ci_high=d_hi)
        rep = cfg.REPEATED_CV_NAME.get(name)
        if rep in repeated.index:
            row.update(repeated_cv_macro_f1_mean=repeated.loc[rep, "macro_f1_mean"],
                       repeated_cv_macro_f1_std=repeated.loc[rep, "macro_f1_std"],
                       folds_better_than_api=None if rep == "api_only" else
                       f"{repeated.loc[rep, 'folds_better_than_api']}/{repeated.loc[rep, 'folds_total']}")
        rows.append(row)

    # Early fusion has no per-sample predictions, only Member 4's repeated-CV summary.
    for _, r in pd.read_csv(cfg.EARLY_CV).iterrows():
        if r["method"] == "api_features":
            continue                     # same model as api_lr, retrained on Member 4's folds
        rows.append({"method": "early_" + r["method"].replace("api_features+", ""),
                     "description": "Early fusion: API features + " + r["method"].split("+")[-1].replace("_", " "),
                     "repeated_cv_macro_f1_mean": r["macro_f1_mean"], "repeated_cv_macro_f1_std": r["macro_f1_std"],
                     "folds_better_than_api": f"{r['folds_better_than_api']}/{r['folds_total']} (vs. its own API baseline)"})
    return pd.DataFrame(rows)


def to_markdown(table):
    """Readable version of the master table for the Word report."""
    def ci(v, lo, hi):
        return "" if pd.isna(v) else f"{v:.3f} [{lo:.3f}, {hi:.3f}]"

    def delta(v, lo, hi):
        return "" if pd.isna(v) else f"{v:+.3f} [{lo:+.3f}, {hi:+.3f}]"

    def rep(m, s):
        return "" if pd.isna(m) else f"{m:.3f} ± {s:.3f}"

    lines = ["| Method | Accuracy [95% CI] | Macro-F1 [95% CI] | Δ macro-F1 vs. API [95% CI] "
             "| Macro-F1, 10×5-fold CV | Folds better than API |",
             "|---|---|---|---|---|---|"]
    for _, r in table.iterrows():
        lines.append("| {} | {} | {} | {} | {} | {} |".format(
            r["method"], ci(r.get("accuracy"), r.get("accuracy_ci_low"), r.get("accuracy_ci_high")),
            ci(r.get("macro_f1"), r.get("macro_f1_ci_low"), r.get("macro_f1_ci_high")),
            delta(r.get("delta_macro_f1_vs_api"), r.get("delta_ci_low"), r.get("delta_ci_high")),
            rep(r.get("repeated_cv_macro_f1_mean"), r.get("repeated_cv_macro_f1_std")),
            "" if pd.isna(r.get("folds_better_than_api")) else r["folds_better_than_api"]))
    notes = [
        "",
        f"n = {int(table['n'].max())} samples (ok Cuckoo runs with both an API trace and a network row), 5 families.",
        "Accuracy / macro-F1: pooled out-of-fold predictions (5-fold stratified CV, seed 42; late fusion: first repeat).",
        "95% CI: bootstrap over samples, 2000 resamples. Δ: paired bootstrap on the same resamples.",
        "10×5-fold CV: mean ± std over 50 folds from Member 4 (reports/hybrid/).",
        "Early fusion has no per-sample predictions, so only its repeated-CV numbers exist.",
    ]
    return "\n".join(lines + notes) + "\n"


def per_family(df):
    rows, cms = [], []
    for name in cfg.METHODS:
        p, r, f, n = precision_recall_fscore_support(df["family"], df[name], labels=cfg.FAMILIES, zero_division=0)
        rows += [{"method": name, "family": fam, "precision": p[i], "recall": r[i], "f1": f[i], "support": n[i]}
                 for i, fam in enumerate(cfg.FAMILIES)]
        cm = confusion_matrix(df["family"], df[name], labels=cfg.FAMILIES)
        cms += [{"method": name, "true": t, "pred": pr, "count": cm[i, j]}
                for i, t in enumerate(cfg.FAMILIES) for j, pr in enumerate(cfg.FAMILIES)]
    return pd.DataFrame(rows), pd.DataFrame(cms)


def test_split_reference():
    """Official fixed split (fit on train+val, predict test). Only the API model has it, and
    LokiBot has no analysed test sample, so it covers 4 families - a sanity check, not the
    headline result."""
    t = pd.read_csv(cfg.API_TEST)
    s = scores(t["family"], t["pred"])
    present = sorted(t["family"].unique())
    s["macro_f1_present_families"] = f1_score(t["family"], t["pred"], labels=present, average="macro")
    return pd.DataFrame([{"method": "api_lr (official split)", "n": len(t), "families": ", ".join(present),
                          "accuracy": s["accuracy"], "macro_f1_4_families": s["macro_f1_present_families"]}])


def label_noise(df):
    """Do the models do better when MalwareBazaar and AVClass agree on the family?"""
    groups = {"high confidence (agree/partial)": df["label_agreement"].isin(cfg.HIGH_CONFIDENCE),
              "low confidence (disagree/no_avclass)": df["label_agreement"].isin(cfg.LOW_CONFIDENCE)}
    rows = []
    for name in ["api_lr", "network", "weighted"]:
        for group, mask in groups.items():
            sub = df[mask]
            s = scores(sub["family"], sub[name])
            lo, hi = bootstrap_ci(sub["family"], sub[name], "accuracy")
            rows.append({"method": name, "group": group, "n": len(sub), "accuracy": s["accuracy"],
                         "accuracy_ci_low": lo, "accuracy_ci_high": hi, "macro_f1": s["macro_f1"]})
    return pd.DataFrame(rows)


def main():
    df = load_predictions()
    print(f"samples: {len(df)} | families: {df['family'].value_counts().to_dict()}\n")
    cfg.OUT_DIR.mkdir(parents=True, exist_ok=True)

    table = master_table(df)
    table.round(4).to_csv(cfg.RESULTS_CSV, index=False)
    cfg.RESULTS_MD.write_text(to_markdown(table), encoding="utf-8")
    print(to_markdown(table))

    fam, cms = per_family(df)
    fam.round(4).to_csv(cfg.PER_FAMILY_CSV, index=False)
    cms.to_csv(cfg.CONFUSION_CSV, index=False)
    print("== F1 per family ==")
    print(fam.pivot(index="family", columns="method", values="f1")[list(cfg.METHODS)].round(3).to_string(), "\n")

    test = test_split_reference()
    test.round(4).to_csv(cfg.TEST_SPLIT_CSV, index=False)
    print("== Official test split (reference only) ==")
    print(test.round(4).to_string(index=False), "\n")

    noise = label_noise(df)
    noise.round(4).to_csv(cfg.LABEL_NOISE_CSV, index=False)
    print("== Label noise: accuracy by AVClass agreement ==")
    print(noise.round(3).to_string(index=False))

    print("\nSaved:", cfg.RESULTS_CSV, cfg.RESULTS_MD, cfg.PER_FAMILY_CSV, cfg.CONFUSION_CSV,
          cfg.TEST_SPLIT_CSV, cfg.LABEL_NOISE_CSV, sep="\n  ")


if __name__ == "__main__":
    main()
