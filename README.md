# Cyber_mid — Hybrid Malware Classification (API call sequence + network traffic)

Drive link: https://drive.google.com/drive/folders/1COakXlKHOAAIpoxLr_v5rP6TFiM0rznB?usp=drive_link

Midterm project: classify malware using Cuckoo API call sequences (Step 1),
network traffic / pcap (Step 2), and a hybrid of both (Step 3).

## Repository structure
```
data/
  metadata/samples.csv     # Master metadata table (sample IDs, hashes, labels, execution IDs)
  labels/                  # AVClass outputs
  vt_reports/              # VirusTotal JSON reports (text only, safe)
scripts/member1_dataset/   # Dataset & label management scripts
docs/dataset.md            # Dataset documentation and limitations
```

## Safety rules
- **Never commit malware binaries, zips, or pcaps.** Only hashes and metadata go in this repo.
- Malware is downloaded, stored, and executed **only on the Ubuntu/Cuckoo VM machine**.
- Never use the school network for any malware activity.
- API keys go in `.env` (see `.env.example`); `.env` is gitignored.

## Setup
There are two requirements files, one per machine:

| File | Machine | Scripts |
|---|---|---|
| `requirements.txt` | Main computer (Windows) | 01–04: metadata, VirusTotal, AVClass |
| `requirements-lab.txt` | Ubuntu/Cuckoo lab machine | 05–07: kept minimal for its older Python |

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
