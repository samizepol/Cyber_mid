"""
run_all.py
Member 5 - Evaluation, Experiments and Reproducibility

One command that rebuilds every final table and figure from the files in the repo,
in the right order, and stops at the first failure.

    python scripts/member5_evaluation/run_all.py            # evaluation + reproducibility check
    python scripts/member5_evaluation/run_all.py --fast     # skip the reproducibility re-runs

Order
  1. 01_check_integrity.py         inputs are consistent (level A)
  2. 02_compare_all.py             master results table, per-family metrics, label noise
  3. 03_make_figures.py            final figures (reports/final/figures)
  4. 04_check_reproducibility.py   re-runs B (+ C, D when the feature files exist)
Member 4's own figures (reports/hybrid/figures) are committed as they are; regenerate them
with scripts/member4_hybrid/02_plot_results.py if needed.
"""
import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

STEPS = [
    HERE / "01_check_integrity.py",
    HERE / "02_compare_all.py",
    HERE / "03_make_figures.py",
    HERE / "04_check_reproducibility.py",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="skip 04_check_reproducibility.py")
    args = ap.parse_args()
    steps = STEPS[:-1] if args.fast else STEPS
    for i, script in enumerate(steps, 1):
        print(f"\n===== [{i}/{len(steps)}] {script.relative_to(ROOT)} =====", flush=True)
        if subprocess.run([sys.executable, str(script)], cwd=ROOT).returncode != 0:
            sys.exit(f"FAILED at {script.name}")
    print("\nAll steps passed. Final tables and figures: reports/final/")


if __name__ == "__main__":
    main()
