"""Shared paths/settings for the hybrid model (Member 4)."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]            # Cyber_mid/

# Inputs: out-of-fold class probabilities of the two single-source classifiers.
API_OOF = PROJECT_ROOT / "reports" / "api" / "api_oof_predictions.csv"
NETWORK_OOF = PROJECT_ROOT / "data" / "network" / "network_oof_predictions_compatible_api.csv"
API_PREFIX = "{model}_proba_"     # API file has one block per model; {model} = value of best_model
NETWORK_PREFIX = "network_proba_"

# Inputs for early (feature-level) fusion. Not in the repo: copy them from the Drive.
API_SEQUENCES = PROJECT_ROOT / "data" / "api" / "api_sequences.jsonl"             # Member 2
NETWORK_FEATURES = PROJECT_ROOT / "data" / "network" / "network_dataset_labeled_270.csv"  # Member 3

REPORT_DIR = PROJECT_ROOT / "reports" / "hybrid"
FIG_DIR = REPORT_DIR / "figures"
CV_RESULTS = REPORT_DIR / "hybrid_cv_results.csv"            # mean/std per method -> report table
FOLD_RESULTS = REPORT_DIR / "hybrid_fold_results.csv"        # every outer fold (for Member 5)
PREDICTIONS = REPORT_DIR / "hybrid_oof_predictions.csv"      # per-sample probabilities, first repeat
PER_FAMILY = REPORT_DIR / "hybrid_per_family_f1.csv"
MCNEMAR = REPORT_DIR / "hybrid_mcnemar.csv"
FINAL_MODEL = REPORT_DIR / "hybrid_final_model.json"        # fusion weight fitted on all samples
EARLY_CV_RESULTS = REPORT_DIR / "early_cv_results.csv"
EARLY_FOLD_RESULTS = REPORT_DIR / "early_fold_results.csv"

SEED = 42
N_SPLITS = 5            # outer CV folds used to train/evaluate the fusion itself
N_REPEATS = 10          # repeated with different shuffles -> mean +/- std
WEIGHT_GRID = [round(i * 0.05, 2) for i in range(21)]     # w for API, 1 - w for network
EPS = 1e-3              # probabilities of exactly 0 (Random Forest) -> log(0) avoided
