"""
02_select_samples.py
Member 1 - Dataset and Label Management

Selects the final set of samples the team will analyze from candidates.csv:
  - picks N samples per family at random (fixed seed -> reproducible for the whole team)
  - assigns a permanent sample_id (S0001, S0002, ...)
  - assigns a stratified train / val / test split per family
  - creates data/metadata/samples.csv, the MASTER metadata table, with empty columns
    that later scripts fill in (AVClass labels, Cuckoo execution IDs, analysis status...)

SAFE TO RUN ON THE MAIN OS: works only with metadata (CSV), no malware files.

Usage (from the repo root):
    python scripts/member1_dataset/02_select_samples.py
    python scripts/member1_dataset/02_select_samples.py --per-family 60 --seed 42

samples.csv is protected: the script refuses to overwrite it (because later steps add
AVClass labels and Cuckoo task IDs to it) unless you pass --force.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = REPO_ROOT / "data" / "metadata" / "candidates.csv"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "metadata" / "samples.csv"

# Split proportions inside each family (70% / 15% / 15%).
# With 60 samples per family this gives 42 train / 9 val / 9 test.
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}

# Final column order of the master metadata table.
# Columns marked "later" stay empty now and are filled by the next scripts / teammates.
MASTER_COLUMNS = [
    "sample_id",           # permanent ID used by the whole team (S0001...)
    "sha256", "sha1", "md5",
    "file_type", "file_size", "file_name",
    "source", "first_seen",
    "mb_signature",        # family name according to MalwareBazaar
    "avclass_family",      # later: 04_run_avclass.py
    "avclass_num_av",      # later: number of AV engines that agreed on the family
    "vt_detection_ratio",  # later: 03_get_vt_reports.py, e.g. "55/72"
    "label_agreement",     # later: MalwareBazaar vs AVClass -> agree / disagree / no_avclass
    "final_label",         # later: the label the models will use
    "split",               # train / val / test
    "download_date",       # later: when the binary was downloaded on the VM machine
    "sha256_verified",     # later: True if the downloaded file's hash matched
    "assigned_member",     # later: who runs it in Cuckoo
    "execution_id",        # later: Cuckoo task ID (links API trace + pcap of the SAME run)
    "analysis_date",       # later
    "analysis_status",     # later: ok / failed / no_network / evasive
    "api_call_count",      # later: filled after Cuckoo
    "pcap_packets",        # later: filled after Cuckoo
    "notes",
]


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------

def assign_splits(n):
    """
    Return a list of n split names following SPLIT_RATIOS, e.g. n=60 -> 42 train, 9 val, 9 test.
    Rounding leftovers go to 'train'.
    """
    n_val = round(n * SPLIT_RATIOS["val"])
    n_test = round(n * SPLIT_RATIOS["test"])
    n_train = n - n_val - n_test
    return ["train"] * n_train + ["val"] * n_val + ["test"] * n_test


def select_samples(candidates, per_family, seed):
    """
    For each family, randomly pick `per_family` samples and give them a split.
    Families with fewer candidates than requested keep all of them (with a warning).
    """
    selected = []
    for family, group in candidates.groupby("query_family", sort=True):
        if len(group) < per_family:
            print(f"  [!] {family}: only {len(group)} candidates (< {per_family}), keeping all")
            chosen = group.sample(frac=1, random_state=seed)  # shuffle all
        else:
            chosen = group.sample(n=per_family, random_state=seed)

        # The order after sample() is already random, so the first 70% -> train, etc.
        chosen = chosen.copy()
        chosen["split"] = assign_splits(len(chosen))
        selected.append(chosen)

    return pd.concat(selected, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description="Select final samples and build samples.csv.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="candidates.csv path.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT, help="samples.csv path.")
    parser.add_argument("--per-family", type=int, default=60, help="Samples to keep per family.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (keep it fixed!).")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite an existing samples.csv (DESTROYS labels/task IDs added later).")
    args = parser.parse_args()

    # Protect the master table from being overwritten by accident
    if args.out.exists() and not args.force:
        sys.exit(f"ERROR: {args.out} already exists. Use --force only if you really want to rebuild it.")

    candidates = pd.read_csv(args.input)
    print(f"Loaded {len(candidates)} candidates from {args.input}")

    # Safety checks on the input
    candidates = candidates.drop_duplicates(subset="sha256")
    candidates = candidates[candidates["file_type"] == "exe"]

    selected = select_samples(candidates, args.per_family, args.seed)

    # Sort by family then split so the table is easy to read, then give permanent IDs
    split_order = {"train": 0, "val": 1, "test": 2}
    selected = selected.sort_values(
        by=["query_family", "split"], key=lambda col: col.map(split_order) if col.name == "split" else col
    ).reset_index(drop=True)
    selected["sample_id"] = [f"S{i:04d}" for i in range(1, len(selected) + 1)]

    # Add the empty "later" columns and put everything in the final order
    for col in MASTER_COLUMNS:
        if col not in selected.columns:
            selected[col] = ""
    master = selected[MASTER_COLUMNS]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    master.to_csv(args.out, index=False, encoding="utf-8")

    # Also save a plain hash list: this is what the VT and download scripts will read
    hash_list = args.out.parent / "selected_hashes.txt"
    hash_list.write_text("\n".join(master["sha256"]) + "\n", encoding="utf-8")

    # Summary table: samples per family and per split (check the balance here)
    print("\nSelected samples per family and split:")
    summary = pd.crosstab(master["mb_signature"], master["split"], margins=True, margins_name="TOTAL")
    print(summary[["train", "val", "test", "TOTAL"]].to_string())
    print(f"\nWrote {len(master)} samples to {args.out}")
    print(f"Wrote hash list to {hash_list}")


if __name__ == "__main__":
    main()
