"""Shared paths/settings for the API-call classifier (Member 2). Override with env vars per machine."""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]            # Cyber_mid/
SAMPLES_CSV = PROJECT_ROOT / "data" / "metadata" / "samples.csv"

# Folder with the API traces written by 06_run_cuckoo_batch.py (<sha256>_<taskid>.json).
# They are NOT in the repo: copy them from the lab machine / shared Drive.
TRACE_DIR = Path(os.path.expanduser(os.getenv("API_TRACE_DIR", "~/malware_lab/api_traces")))

DATA_DIR = PROJECT_ROOT / "data" / "api"                      # derived features (text only)
REPORT_DIR = PROJECT_ROOT / "reports" / "api"
FIG_DIR = REPORT_DIR / "figures"

SEQUENCES = DATA_DIR / "api_sequences.jsonl"
PREDICTIONS = REPORT_DIR / "api_oof_predictions.csv"          # hand this to Member 4 (hybrid)
TEST_PREDICTIONS = REPORT_DIR / "api_test_predictions.csv"
CV_RESULTS = REPORT_DIR / "api_cv_results.csv"
MODEL_OUTPUT = DATA_DIR / "api_family_classifier.joblib"

MAX_SEQ_LEN = 5000      # keep the first N calls after collapsing repeats (counts range 100 - 480,000)
SEED = 42               # same seed/folds as the network module
N_SPLITS = 5
