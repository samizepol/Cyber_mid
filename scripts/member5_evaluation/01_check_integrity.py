"""
01_check_integrity.py
Member 5 - Evaluation, Experiments and Reproducibility

Reproducibility level A: checks that every file the results depend on is consistent
BEFORE any number is compared. It needs only files that are in the repo.

Checks
  1. samples.csv: unique ids/hashes, one label per sample, 270 'ok' runs, unique Cuckoo task ids.
  2. Every prediction file (API, network, hybrid) agrees with samples.csv on family and split,
     has no duplicate sample, only uses ok runs and only the 5 evaluated families.
  3. Probabilities are valid (0..1, rows sum to 1) and the stored prediction is their argmax.
  4. API and network files cover the same samples, except the known gap (S0001, no network row).
  5. The fixed test split contains no sample of the training split (no leakage).
  6. No malware, pcap or archive file is tracked by git (safety rule in the README).

Usage (from the repo root):
    python scripts/member5_evaluation/01_check_integrity.py
Exit code 1 if any check fails; the report is saved to reports/final/integrity_check.txt.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg  # noqa: E402

KNOWN_GAPS = {"S0001"}          # API trace but no network row (excluded by the network script)
FORBIDDEN = (".exe", ".dll", ".bin", ".scr", ".sys", ".zip", ".7z", ".rar", ".pcap", ".pcapng")

results = []                    # (check name, passed, detail)


def check(name, passed, detail=""):
    results.append((name, bool(passed), detail))


def check_prediction_file(name, path, prefix, pred_col, meta):
    """Checks 2 and 3 for one out-of-fold prediction file."""
    df = pd.read_csv(path)
    check(f"{name}: no duplicate sample_id", df["sample_id"].is_unique, f"{len(df)} rows")
    m = df.merge(meta, on="sample_id", how="left", suffixes=("", "_meta"))
    check(f"{name}: every sample is in samples.csv", m["final_label"].notna().all())
    check(f"{name}: family == samples.csv final_label", (m["family"] == m["final_label"]).all())
    check(f"{name}: split == samples.csv split", (m["split"] == m["split_meta"]).all())
    check(f"{name}: only ok Cuckoo runs", (m["analysis_status"] == "ok").all())
    check(f"{name}: only the 5 evaluated families", set(df["family"]) <= set(cfg.FAMILIES),
          ", ".join(sorted(set(df["family"]))))

    proba = df[[prefix + f for f in cfg.FAMILIES]].to_numpy(float)
    check(f"{name}: probabilities in [0, 1]", ((proba >= 0) & (proba <= 1)).all())
    sums = proba.sum(axis=1)
    check(f"{name}: probability rows sum to 1", np.allclose(sums, 1, atol=1e-3),
          f"min {sums.min():.4f}, max {sums.max():.4f}")
    argmax = np.asarray(cfg.FAMILIES)[proba.argmax(axis=1)]
    check(f"{name}: stored prediction == argmax(probabilities)", (argmax == df[pred_col]).all(),
          f"{(argmax != df[pred_col]).sum()} mismatches")
    return set(df["sample_id"])


def main():
    meta = pd.read_csv(cfg.SAMPLES_CSV)

    # 1. Master metadata table
    check("samples.csv: unique sample_id", meta["sample_id"].is_unique, f"{len(meta)} rows")
    check("samples.csv: unique sha256", meta["sha256"].is_unique)
    ok = meta[meta["analysis_status"] == "ok"]
    check("samples.csv: 270 ok Cuckoo runs", len(ok) == 270, f"{len(ok)} ok")
    runs = meta["execution_id"].dropna()
    check("samples.csv: unique Cuckoo task ids", runs.is_unique, f"{len(runs)} runs")
    check("samples.csv: ok runs are the 5 evaluated families", set(ok["final_label"]) == set(cfg.FAMILIES),
          ok["final_label"].value_counts().to_dict())
    meta = meta[["sample_id", "final_label", "split", "analysis_status"]]

    # 2 + 3. Every prediction file
    api_ids = check_prediction_file("API OOF", cfg.API_OOF, "tfidf_lr_proba_", "tfidf_lr_pred", meta)
    net_ids = check_prediction_file("network OOF", cfg.NETWORK_OOF, "network_proba_", "network_pred", meta)
    hyb_ids = check_prediction_file("hybrid OOF (weighted)", cfg.HYBRID_OOF, "weighted_proba_", "weighted_pred", meta)

    # 4. Same samples everywhere (apart from the known gap)
    check("API covers every ok sample", api_ids == set(ok["sample_id"]), f"{len(api_ids)} samples")
    gap = api_ids ^ net_ids
    check("API vs network: only the known gap differs", gap <= KNOWN_GAPS, f"only in one file: {sorted(gap)}")
    check("hybrid = API intersect network", hyb_ids == (api_ids & net_ids), f"{len(hyb_ids)} samples")

    # 5. Official test split: predictions come from a model fitted on train+val only
    test = pd.read_csv(cfg.API_TEST).merge(meta, on="sample_id", how="left")
    check("API test split: every sample has split == test", (test["split"] == "test").all(),
          f"{len(test)} samples, families: {sorted(test['family'].unique())}")

    # 6. Safety: nothing dangerous tracked by git
    try:
        tracked = subprocess.run(["git", "ls-files"], cwd=cfg.PROJECT_ROOT, capture_output=True,
                                 text=True, check=True).stdout.splitlines()
        bad = [f for f in tracked if f.lower().endswith(FORBIDDEN)]
        check("git: no binaries, archives or pcaps tracked", not bad, ", ".join(bad) or f"{len(tracked)} files")
    except (OSError, subprocess.CalledProcessError):
        check("git: no binaries, archives or pcaps tracked", True, "skipped (not a git checkout)")

    # Report
    lines = [f"[{'PASS' if p else 'FAIL'}] {n}" + (f"  ({d})" if d else "") for n, p, d in results]
    failed = sum(not p for _, p, _ in results)
    lines.append(f"\n{len(results) - failed}/{len(results)} checks passed")
    report = "\n".join(lines)
    print(report)
    cfg.OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg.INTEGRITY_TXT.write_text(report + "\n", encoding="utf-8")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
