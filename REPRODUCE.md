# Reproducing the results

Everything after the Cuckoo sandbox runs on any computer with Python and needs **no
malware, traces or pcaps**. The full chain has five levels. The higher a level, the more
input it needs.

| Level | What is re-run | Input needed | Where |
|---|---|---|---|
| A | Integrity checks of every input file | repo only | any computer |
| B | Hybrid late fusion (Member 4) from the committed OOF predictions | repo only | any computer |
| C | API classifier + early fusion | `data/api/api_sequences.jsonl` (21 MB, Drive) | any computer |
| D | Network classifier | `data/network/network_dataset_labeled_270.csv` (Drive) | any computer |
| E | Cuckoo analysis → API traces + pcaps → feature extraction | malware samples | **lab VM only** |

## 1. Setup (once)

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements-eval.txt
```

For levels C and D, copy the two feature files from the shared Drive (README link) to the
paths in the table. They contain only API names and packet statistics, no malware
artifacts, and are git-ignored.

## 2. One command

```bash
python scripts/member5_evaluation/run_all.py
```

The script runs these steps in order and stops at the first failure:

1. `01_check_integrity.py`: level A, 37 checks (labels, splits, duplicates, probabilities, no malware in git).
2. `02_compare_all.py`: master results table, per-family metrics, confusion matrices, test split, label noise.
3. `03_make_figures.py`: final figures.
4. `04_check_reproducibility.py`: levels B–D. Each training script is re-run in a temporary copy
   of the repo and its metrics are compared with the committed ones (tolerance 0.001).
   Committed files are never overwritten. A level whose input file is missing shows `SKIPPED`.

Outputs go to `reports/final/`:

| File | Content |
|---|---|
| `results_table.md` / `.csv` | Master table: every method, accuracy and macro-F1 with 95% CI, gain vs. API |
| `per_family_metrics.csv` | Precision / recall / F1 per family and method |
| `confusion_matrices.csv` | Confusion matrix of every method (long format) |
| `test_split_reference.csv` | Official fixed test split (API model, 4 families) |
| `label_noise.csv` | Metrics on high- vs. low-confidence labels |
| `integrity_check.txt` | Result of level A |
| `reproducibility_check.csv` | Result of levels B–E |
| `figures/` | `macro_f1_with_ci.png`, `delta_vs_api.png`, `label_noise.png`, `confusion_api_net_hybrid.png` |

## 3. Last verified run

| Level | Result |
|---|---|
| A | 37/37 checks pass |
| B | 18/18 metrics match; max difference 0.0004 (stacking, a scikit-learn version difference; all other methods identical) |
| C | Logistic regression (the reported API model) and early fusion match exactly. The comparison random forest gives macro-F1 0.661 vs. 0.675 committed |
| D | Macro-F1 0.557 vs. 0.559 committed (accuracy 0.554 vs. 0.558). Gap comes from the original training environment; random forests are sensitive to library versions. Reported results use the committed predictions (level B) |
| E | Not re-run (lab VM only); documented below |

## 4. Level E: lab VM (documentation only)

Done once on the isolated Ubuntu/Cuckoo machine, never on the school network
(`scripts/member1_dataset/`, `requirements-lab.txt`):

1. `05_download_samples.py`: download the 540 selected samples and verify SHA-256.
2. `06_run_cuckoo_batch.py`: run each sample in Cuckoo. One task id per sample
   (`execution_id`) links its API trace and its pcap from the **same run**.
3. `scripts/member3_api_classification/01_extract_sequences.py`: API traces → `api_sequences.jsonl`.
4. `scripts/member2_network/extract_network_features_*.py`: pcaps → per-sample network statistics.

Re-running level E gives different raw traffic (C2 servers go offline, timing changes), so
levels C and D start from the saved feature files instead.
