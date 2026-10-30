# API-Call Classification (Member 2)

Classifies malware families from Cuckoo API-call sequences (midterm Step 1).

## Run (from the repo root, inside the lab VM or wherever the traces are)
```bash
pip install pandas scikit-learn joblib matplotlib
export API_TRACE_DIR=~/malware_lab/api_traces      # folder with <sha256>_<taskid>.json
python scripts/member3_api_classification/01_extract_sequences.py
python scripts/member3_api_classification/02_train_api_classifier.py
python scripts/member3_api_classification/03_plot_results.py
```

## Pipeline
1. `01_extract_sequences.py` - reads `samples.csv` (status `ok`), loads each trace, lowercases API names, merges processes, collapses consecutive repeats, truncates to 5000 calls -> `data/api/api_sequences.jsonl`.
2. `02_train_api_classifier.py` - TF-IDF over API 1-3-grams + sequence statistics; Logistic Regression vs Random Forest; stratified 5-fold CV (seed 42, vectorizer fitted per fold) + official-split reference on the test set.
3. `03_plot_results.py` - confusion matrix and F1 per family.

## Outputs
| File | Use |
|---|---|
| `reports/api/api_cv_results.csv` | Metrics table for the report |
| `reports/api/api_oof_predictions.csv` | Out-of-fold probabilities per sample -> **Member 4 (hybrid)** and Member 5 |
| `reports/api/api_test_predictions.csv` | Official test split predictions |
| `reports/api/figures/*.png` | Report/PPT figures |
| `data/api/api_family_classifier.joblib` | Final model trained on all samples |

## Limitations
Small dataset (~270 samples), noisy labels (MalwareBazaar vs AVClass), single sandbox configuration, very different sequence lengths. Do not commit traces or reports that contain malware artifacts.
