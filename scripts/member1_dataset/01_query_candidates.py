"""
01_query_candidates.py
Member 1 - Dataset and Label Management

Queries the MalwareBazaar API for candidate malware samples of each chosen family
and saves their METADATA ONLY (hashes, family signature, file type, dates) to a CSV.

SAFE TO RUN ON THE MAIN OS: this script never downloads malware binaries.
Binaries are downloaded later by 04_download_samples.py, on the VM machine only.

Usage (from the repo root):
    python scripts/member1_dataset/01_query_candidates.py
    python scripts/member1_dataset/01_query_candidates.py --families AgentTesla Formbook --limit 50

Requires MALWAREBAZAAR_AUTH_KEY in the .env file at the repo root.
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Repo root = two levels above this file (scripts/member1_dataset/ -> repo root)
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "data" / "metadata" / "candidates.csv"

API_URL = "https://mb-api.abuse.ch/api/v1/"

# Windows malware families that are common on MalwareBazaar AND usually generate
# network traffic (needed for Step 2 / pcap analysis). Edit this list as the team decides.
DEFAULT_FAMILIES = [
    "AgentTesla",
    "Formbook",
    "RedLineStealer",
    "RemcosRAT",
    "AsyncRAT",
    "njrat",
    "Loki",
    "Smoke Loader",  # MalwareBazaar spells this one with a space
    "Amadey",
]

# Columns written to the output CSV (subset of the MalwareBazaar response)
CSV_COLUMNS = [
    "sha256",
    "sha1",
    "md5",
    "mb_signature",
    "file_type",
    "file_type_mime",
    "file_size",
    "file_name",
    "first_seen",
    "last_seen",
    "imphash",
    "tags",
    "source",
    "query_family",
]

# Seconds to wait between API requests, to be polite to the free service
REQUEST_DELAY = 2


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------

def load_auth_key():
    """Read the MalwareBazaar Auth-Key from .env (never hard-code keys in code)."""
    load_dotenv(REPO_ROOT / ".env")
    key = os.getenv("MALWAREBAZAAR_AUTH_KEY", "").strip()
    if not key or key.startswith("your_"):
        sys.exit("ERROR: MALWAREBAZAAR_AUTH_KEY is missing in .env (see .env.example).")
    return key


def query_family(auth_key, family, limit):
    """
    Ask MalwareBazaar for the most recent samples tagged with a family signature.
    Returns a list of sample dictionaries (metadata only), or [] if none/error.
    """
    response = requests.post(
        API_URL,
        data={"query": "get_siginfo", "signature": family, "limit": limit},
        headers={"Auth-Key": auth_key},
        timeout=60,
    )

    if response.status_code == 401:
        sys.exit("ERROR: MalwareBazaar rejected the Auth-Key (HTTP 401). Check your .env.")
    response.raise_for_status()

    result = response.json()
    status = result.get("query_status")
    if status != "ok":
        # Common statuses: "no_results", "illegal_signature", "unknown_signature"
        print(f"  [!] {family}: query_status = {status}")
        return []
    return result.get("data", [])


def to_row(sample, family):
    """Convert one MalwareBazaar sample record into a CSV row (dict)."""
    tags = sample.get("tags") or []
    return {
        "sha256": sample.get("sha256_hash", ""),
        "sha1": sample.get("sha1_hash", ""),
        "md5": sample.get("md5_hash", ""),
        "mb_signature": sample.get("signature", ""),
        "file_type": sample.get("file_type", ""),
        "file_type_mime": sample.get("file_type_mime", ""),
        "file_size": sample.get("file_size", ""),
        "file_name": sample.get("file_name", ""),
        "first_seen": sample.get("first_seen", ""),
        "last_seen": sample.get("last_seen", ""),
        "imphash": sample.get("imphash", ""),
        "tags": ";".join(tags),
        "source": "MalwareBazaar",
        "query_family": family,
    }


def main():
    parser = argparse.ArgumentParser(description="Query MalwareBazaar for candidate sample metadata.")
    parser.add_argument("--families", nargs="+", default=DEFAULT_FAMILIES,
                        help="Family signatures to query (MalwareBazaar names).")
    parser.add_argument("--limit", type=int, default=100,
                        help="Max samples requested per family (MalwareBazaar max is 1000).")
    parser.add_argument("--file-type", default="exe",
                        help="Keep only this file type (default: exe = Windows PE). Use 'any' to keep all.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT,
                        help="Output CSV path.")
    args = parser.parse_args()

    auth_key = load_auth_key()

    rows = []
    seen_hashes = set()  # used to remove duplicate samples across families

    for family in args.families:
        print(f"Querying {family} (limit {args.limit}) ...")
        samples = query_family(auth_key, family, args.limit)

        kept = 0
        for sample in samples:
            sha256 = sample.get("sha256_hash")
            # Keep only Windows executables, which Cuckoo's Windows guest can run
            if args.file_type != "any" and sample.get("file_type") != args.file_type:
                continue
            # Skip duplicates (same hash returned for more than one family)
            if not sha256 or sha256 in seen_hashes:
                continue
            seen_hashes.add(sha256)
            rows.append(to_row(sample, family))
            kept += 1

        print(f"  -> {len(samples)} returned, {kept} kept")
        time.sleep(REQUEST_DELAY)

    # Write the candidate list (metadata only)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    # Summary per family, useful for checking class balance early
    print("\nSummary (kept samples per family):")
    for family in args.families:
        count = sum(1 for r in rows if r["query_family"] == family)
        print(f"  {family:<16} {count}")
    print(f"\nTotal: {len(rows)} candidates written to {args.out}")


if __name__ == "__main__":
    main()
