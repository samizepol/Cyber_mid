"""
04_check_reproducibility.py
Member 5 - Evaluation, Experiments and Reproducibility

Re-runs the other members' training scripts in a TEMPORARY copy of the repo and checks
that they regenerate the committed numbers. The committed files are never overwritten.

Levels (see REPRODUCE.md)
  A  integrity of all input files ............. 01_check_integrity.py (run separately)
  B  hybrid late fusion from the committed OOF predictions ......... always possible
  C  API classifier + early fusion ......... needs data/api/api_sequences.jsonl (Drive)
  D  network classifier ......... needs data/network/network_dataset_labeled_270.csv
  E  Cuckoo sandbox -> traces / pcaps ......... lab VM only, documented, not re-run here
A level whose input file is missing is reported as SKIPPED, not as failed.

Usage (from the repo root):
    python scripts/member5_evaluation/04_check_reproducibility.py
Exit code 1 if a level that could run gives different numbers.
"""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg  # noqa: E402

ROOT = cfg.PROJECT_ROOT


def copy_repo(dest):
    """Copy everything the scripts need (code, data, reports) - not .git or virtualenvs."""
    shutil.copytree(ROOT, dest, ignore=shutil.ignore_patterns(".git", ".venv", "venv", "__pycache__"))


def run(script, cwd, *args):
    """Run one script of the copied repo; return True if it exited cleanly."""
    proc = subprocess.run([sys.executable, str(cwd / script), *args], cwd=cwd,
                          capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout[-2000:], proc.stderr[-2000:], sep="\n")
    return proc.returncode == 0


def compare(level, label, committed, regenerated, key, columns):
    """Compare numeric columns of two result tables row by row (matched on `key`)."""
    a = pd.read_csv(committed).set_index(key)
    b = pd.read_csv(regenerated).set_index(key)
    rows = []
    for idx in a.index:
        for col in columns:
            old = float(a.loc[idx, col])
            new = float(b.loc[idx, col]) if idx in b.index else float("nan")
            diff = abs(old - new)
            rows.append({"level": level, "check": f"{label}: {idx} {col}", "committed": old,
                         "regenerated": new, "abs_diff": diff, "status": "PASS" if diff <= cfg.REPRO_TOLERANCE else "FAIL"})
    return rows


def skipped(level, label, reason):
    return [{"level": level, "check": label, "status": "SKIPPED", "note": reason}]


def failed(level, label):
    return [{"level": level, "check": label, "status": "FAIL", "note": "script exited with an error"}]


def main():
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp) / "repo"
        copy_repo(work)

        # ---- B: late fusion from the committed out-of-fold predictions -----------------
        if run("scripts/member4_hybrid/01_late_fusion.py", work):
            rows += compare("B", "late fusion", cfg.HYBRID_CV, work / "reports/hybrid/hybrid_cv_results.csv",
                            "method", ["accuracy_mean", "macro_f1_mean", "delta_macro_f1_vs_api_mean"])
        else:
            rows += failed("B", "late fusion")

        # ---- C: API classifier from the cleaned API sequences ---------------------------
        if cfg.API_SEQUENCES.exists():
            if run("scripts/member3_api_classification/02_train_api_classifier.py", work):
                rows += compare("C", "API classifier", cfg.API_CV, work / "reports/api/api_cv_results.csv",
                                "model", ["accuracy", "macro_f1"])
            else:
                rows += failed("C", "API classifier")
            if run("scripts/member4_hybrid/03_early_fusion.py", work):
                rows += compare("C", "early fusion", cfg.EARLY_CV, work / "reports/hybrid/early_cv_results.csv",
                                "method", ["accuracy_mean", "macro_f1_mean"])
            else:
                rows += failed("C", "early fusion")
        else:
            rows += skipped("C", "API classifier + early fusion", f"missing {cfg.API_SEQUENCES.relative_to(ROOT)}")

        # ---- D: network classifier from the per-pcap statistics -------------------------
        if cfg.NETWORK_FEATURES.exists():
            if run("scripts/member2_network/train_network_compatible_api.py", work):
                rows += compare("D", "network classifier", cfg.NETWORK_CV,
                                work / "data/network/network_cv_results_compatible_api.csv",
                                "method", ["oof_accuracy", "oof_macro_f1"])
            else:
                rows += failed("D", "network classifier")
        else:
            rows += skipped("D", "network classifier", f"missing {cfg.NETWORK_FEATURES.relative_to(ROOT)}")

    rows += skipped("E", "Cuckoo analysis (traces, pcaps)", "lab VM only - see REPRODUCE.md")

    report = pd.DataFrame(rows)
    cfg.OUT_DIR.mkdir(parents=True, exist_ok=True)
    report.to_csv(cfg.REPRO_CSV, index=False)
    with pd.option_context("display.width", 160, "display.max_colwidth", 70):
        print(report.fillna("").to_string(index=False))
    counts = report["status"].value_counts().to_dict()
    print("\nSummary:", counts, "| tolerance", cfg.REPRO_TOLERANCE)
    print("Saved:", cfg.REPRO_CSV)
    sys.exit(1 if counts.get("FAIL") else 0)


if __name__ == "__main__":
    main()
