# Cyber_mid — Hybrid Malware Classification (API call sequence + network traffic)

Drive link: https://drive.google.com/drive/folders/1COakXlKHOAAIpoxLr_v5rP6TFiM0rznB?usp=drive_link

Midterm project: classify malware using Cuckoo API call sequences (Step 1),
network traffic / pcap (Step 2), and a hybrid of both (Step 3).

## Results

All models are evaluated on the same 269 samples (5 families) with 5-fold cross-validation.
Full table with 95% confidence intervals: `reports/final/results_table.md`.

| Model | Macro-F1 |
|---|---|
| Step 1: API call sequences (TF-IDF + logistic regression) | 0.688 |
| Step 2: Network traffic (random forest) | 0.559 |
| **Step 3: Hybrid, weighted late fusion** | **0.725** |

## Repository structure
```
data/
  metadata/samples.csv     # Master metadata table (sample IDs, hashes, labels, execution IDs)
  labels/                  # AVClass outputs
  vt_reports/              # VirusTotal JSON reports (text only, safe)
  network/                 # Network OOF predictions + model (feature CSV via Drive)
  api/                     # api_sequences.jsonl (via Drive, git-ignored)
scripts/
  member1_dataset/             # Dataset & label management (MalwareBazaar, VirusTotal, AVClass, Cuckoo)
  member3_api_classification/  # Step 1: API-call classifier
  member2_network/             # Step 2: network-traffic classifier
  member4_hybrid/              # Step 3: late and early fusion
  member5_evaluation/          # Shared evaluation, final tables/figures, reproducibility checks
reports/
  api/  network/  hybrid/      # Each module's own results and figures
  final/                       # Master results table and final figures (Member 5)
docs/
  dataset.md               # Dataset documentation and limitations
  evaluation.md            # Evaluation protocol: samples, splits, metrics, CIs
REPRODUCE.md               # How to re-run everything after the Cuckoo stage
```

## Safety rules
- **Never commit malware binaries, zips, or pcaps.** Only hashes and metadata go in this repo.
- Malware is downloaded, stored, and executed **only on the Ubuntu/Cuckoo VM machine**.
- Never use the school network for any malware activity.
- API keys go in `.env` (see `.env.example`); `.env` is gitignored.

## Setup
There are three requirements files:

| File | Machine | Scripts |
|---|---|---|
| `requirements.txt` | Main computer (Windows) | 01–04: metadata, VirusTotal, AVClass |
| `requirements-lab.txt` | Ubuntu/Cuckoo lab machine | 05–07: kept minimal for its older Python |
| `requirements-eval.txt` | Any computer | Classifiers, hybrid, evaluation (see `REPRODUCE.md`) |

Main computer:
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Lab machine:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-lab.txt
```
