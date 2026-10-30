"""Shared paths/settings for evaluation, comparison and reproducibility (Member 5).

Everything here reads files that are already in the repo (out-of-fold predictions and
result tables written by Members 1-4). No malware, traces or pcaps are needed.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]            # Cyber_mid/
SCRIPTS = PROJECT_ROOT / "scripts"

# ----- Inputs written by the other members -----
SAMPLES_CSV = PROJECT_ROOT / "data" / "metadata" / "samples.csv"                        # Member 1
API_OOF = PROJECT_ROOT / "reports" / "api" / "api_oof_predictions.csv"                  # API classifier
API_TEST = PROJECT_ROOT / "reports" / "api" / "api_test_predictions.csv"
API_CV = PROJECT_ROOT / "reports" / "api" / "api_cv_results.csv"
NETWORK_OOF = PROJECT_ROOT / "data" / "network" / "network_oof_predictions_compatible_api.csv"  # network classifier
NETWORK_CV = PROJECT_ROOT / "data" / "network" / "network_cv_results_compatible_api.csv"
HYBRID_OOF = PROJECT_ROOT / "reports" / "hybrid" / "hybrid_oof_predictions.csv"         # Member 4
HYBRID_CV = PROJECT_ROOT / "reports" / "hybrid" / "hybrid_cv_results.csv"
HYBRID_FOLDS = PROJECT_ROOT / "reports" / "hybrid" / "hybrid_fold_results.csv"
EARLY_CV = PROJECT_ROOT / "reports" / "hybrid" / "early_cv_results.csv"

# Optional feature files (not in git, shared via Drive). Only needed to RETRAIN models.
API_SEQUENCES = PROJECT_ROOT / "data" / "api" / "api_sequences.jsonl"
NETWORK_FEATURES = PROJECT_ROOT / "data" / "network" / "network_dataset_labeled_270.csv"

# ----- Outputs -----
OUT_DIR = PROJECT_ROOT / "reports" / "final"
FIG_DIR = OUT_DIR / "figures"
RESULTS_CSV = OUT_DIR / "results_table.csv"            # master table: one row per method
RESULTS_MD = OUT_DIR / "results_table.md"              # same, ready to paste into the report
PER_FAMILY_CSV = OUT_DIR / "per_family_metrics.csv"    # precision / recall / F1 per family and method
CONFUSION_CSV = OUT_DIR / "confusion_matrices.csv"     # long format: method, true, pred, count
TEST_SPLIT_CSV = OUT_DIR / "test_split_reference.csv"  # fixed official test split (API only)
LABEL_NOISE_CSV = OUT_DIR / "label_noise.csv"          # metrics by AVClass label agreement
INTEGRITY_TXT = OUT_DIR / "integrity_check.txt"
REPRO_CSV = OUT_DIR / "reproducibility_check.csv"

# ----- Protocol (see docs/evaluation.md) -----
FAMILIES = ["agenttesla", "amadey", "asyncrat", "formbook", "lokibot"]
SEED = 42
N_BOOTSTRAP = 2000          # resamples for 95% confidence intervals
REPRO_TOLERANCE = 1e-3      # max allowed |difference| between committed and regenerated metrics

# Methods in the master table: name -> (prediction column, source file key, description)
# The hybrid file holds the first-repeat OOF predictions of every late-fusion method;
# api_only / network_only in it are the unchanged OOF predictions of the base models.
METHODS = {
    "api_lr":   ("tfidf_lr_pred", "api", "API: TF-IDF 1-3-grams + stats, logistic regression"),
    "api_rf":   ("tfidf_rf_pred", "api", "API: TF-IDF 1-3-grams + stats, random forest"),
    "network":  ("network_only_pred", "hybrid", "Network: packet/protocol statistics, random forest"),
    "mean":     ("mean_pred", "hybrid", "Late fusion: mean of probabilities"),
    "product":  ("product_pred", "hybrid", "Late fusion: product of probabilities"),
    "weighted": ("weighted_pred", "hybrid", "Late fusion: w*API + (1-w)*network, w tuned on training folds (~0.7)"),
    "stacking": ("stacking_pred", "hybrid", "Late fusion: logistic-regression stacking"),
}
# Name used for the same method in Member 4's repeated-CV summary (hybrid_cv_results.csv)
REPEATED_CV_NAME = {"api_lr": "api_only", "network": "network_only", "mean": "mean",
                    "product": "product", "weighted": "weighted", "stacking": "stacking"}

# Label-agreement groups from Member 1 (AVClass vs. MalwareBazaar)
HIGH_CONFIDENCE = ["agree", "partial"]
LOW_CONFIDENCE = ["disagree", "no_avclass"]
