from pathlib import Path
import os

# Root directory of the Git repository
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Local data directory (not committed unless explicitly approved)
DATA_DIR = PROJECT_ROOT / "data" / "network"

# Local Cuckoo analyses directory.
# Override this environment variable on each machine if necessary.
CUCKOO_ROOT = Path(
    os.environ.get(
        "CUCKOO_ANALYSES_DIR",
        "/mnt/data/cuckoo-storage/analyses"
    )
)

# Reports and figures
REPORT_DIR = PROJECT_ROOT / "reports" / "network"

DATASET = DATA_DIR / "network_dataset_labeled.csv"
FEATURES_CLEAN = DATA_DIR / "network_features_clean.csv"
FEATURES_ADVANCED = DATA_DIR / "network_features_advanced.csv"
FEATURES_STEP2 = DATA_DIR / "network_features_step2.csv"
MODEL_OUTPUT = DATA_DIR / "network_family_classifier.joblib"

CV_RESULTS = REPORT_DIR / "network_cv_results.csv"
CONFUSION_MATRIX = REPORT_DIR / "network_final_confusion_matrix.csv"
PREDICTIONS = DATA_DIR / "network_final_predictions.csv"
