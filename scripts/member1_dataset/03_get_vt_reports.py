"""
03_get_vt_reports.py
Member 1 - Dataset and Label Management

Downloads the VirusTotal (API v3) file report of every selected sample. These reports
contain the labels given by ~70 antivirus engines, which AVClass (04_run_avclass.py)
uses to infer a normalized malware-family label.

SAFE TO RUN ON THE MAIN OS: it only sends a hash and receives a JSON report.
No malware file is downloaded or uploaded.

Free VirusTotal key limits: 4 requests/minute and 500 requests/day.
  -> the script waits 15.5 s between requests
  -> it stops by itself after --max-requests (default 480) or when the daily quota is hit
  -> it RESUMES where it stopped: just run it again the next day

Outputs (in data/vt_reports/):
  reports.jsonl    one trimmed VT report per line (the AVClass input format)
  not_found.txt    hashes VirusTotal does not know (HTTP 404)
And it fills the "vt_detection_ratio" column of data/metadata/samples.csv.

Usage (from the repo root):
    python scripts/member1_dataset/03_get_vt_reports.py
    python scripts/member1_dataset/03_get_vt_reports.py --max-requests 100

Requires VIRUSTOTAL_API_KEY in the .env file at the repo root.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SAMPLES = REPO_ROOT / "data" / "metadata" / "samples.csv"
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "vt_reports"

VT_URL = "https://www.virustotal.com/api/v3/files/{}"

REQUEST_DELAY = 15.5      # seconds between requests (free tier = 4 per minute)
RATE_LIMIT_WAIT = 65      # seconds to wait after a 429 before retrying once

# Report fields kept. AVClass only needs md5/sha1/sha256/tags/last_analysis_results;
# the others are useful for the report. Trimming keeps the repo small (~5 KB per sample).
KEEP_ATTRIBUTES = [
    "md5", "sha1", "sha256", "size", "type_description", "tags",
    "first_submission_date", "last_analysis_date",
    "last_analysis_stats", "last_analysis_results",
    "popular_threat_classification",
]
# Inside last_analysis_results, keep only these fields for each AV engine
KEEP_ENGINE_FIELDS = ["category", "engine_name", "result"]


class DailyQuotaExceeded(Exception):
    """Raised when VirusTotal keeps answering 429 (daily quota used up)."""


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------

def load_api_key():
    """Read the VirusTotal key from .env (never hard-code keys in code)."""
    load_dotenv(REPO_ROOT / ".env")
    key = os.getenv("VIRUSTOTAL_API_KEY", "").strip()
    if not key or key.startswith("your_"):
        sys.exit("ERROR: VIRUSTOTAL_API_KEY is missing in .env (see .env.example).")
    return key


def read_done_hashes(reports_path, not_found_path):
    """Return the set of hashes already processed in previous runs (for resuming)."""
    done = set()
    if reports_path.exists():
        with open(reports_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    done.add(json.loads(line)["data"]["attributes"]["sha256"])
    if not_found_path.exists():
        done.update(h.strip() for h in not_found_path.read_text(encoding="utf-8").splitlines() if h.strip())
    return done


def trim_report(report):
    """Keep only the fields we need, in the same {"data": {"attributes": ...}} shape AVClass expects."""
    attrs = report["data"]["attributes"]
    trimmed = {k: attrs[k] for k in KEEP_ATTRIBUTES if k in attrs}
    trimmed["last_analysis_results"] = {
        engine: {k: res.get(k) for k in KEEP_ENGINE_FIELDS}
        for engine, res in attrs.get("last_analysis_results", {}).items()
    }
    return {"data": {"id": report["data"]["id"], "type": "file", "attributes": trimmed}}


def fetch_report(api_key, sha256):
    """
    Get one VT report. Returns the report dict, or None if VT does not know the hash.
    Raises DailyQuotaExceeded if VT still answers 429 after waiting.
    """
    for attempt in range(2):
        response = requests.get(VT_URL.format(sha256), headers={"x-apikey": api_key}, timeout=60)

        if response.status_code == 200:
            return response.json()
        if response.status_code == 404:
            return None
        if response.status_code == 401:
            sys.exit("ERROR: VirusTotal rejected the API key (HTTP 401). Check your .env.")
        if response.status_code == 429:
            # First 429: probably the per-minute limit -> wait and retry once.
            # Second 429: the daily quota is used up -> stop for today.
            if attempt == 0:
                print(f"  [!] rate limited, waiting {RATE_LIMIT_WAIT}s ...")
                time.sleep(RATE_LIMIT_WAIT)
                continue
            raise DailyQuotaExceeded()
        response.raise_for_status()
    raise DailyQuotaExceeded()


def detection_ratio(attrs):
    """'malicious/total' like the VT website, e.g. '55/72' (total = engines that gave a verdict)."""
    stats = attrs.get("last_analysis_stats", {})
    total = sum(stats.get(k, 0) for k in ("malicious", "suspicious", "undetected", "harmless"))
    return f"{stats.get('malicious', 0)}/{total}"


def update_samples_csv(samples_path, reports_path, not_found_path):
    """Write vt_detection_ratio (and a note for unknown hashes) into samples.csv."""
    # dtype=str + keep_default_na=False keeps every other column exactly as it was
    samples = pd.read_csv(samples_path, dtype=str, keep_default_na=False)

    ratios = {}
    if reports_path.exists():
        with open(reports_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    attrs = json.loads(line)["data"]["attributes"]
                    ratios[attrs["sha256"]] = detection_ratio(attrs)
    not_found = set()
    if not_found_path.exists():
        not_found = set(not_found_path.read_text(encoding="utf-8").split())

    for i, sha256 in samples["sha256"].items():
        if sha256 in ratios:
            samples.at[i, "vt_detection_ratio"] = ratios[sha256]
        elif sha256 in not_found:
            samples.at[i, "vt_detection_ratio"] = "not_in_vt"

    samples.to_csv(samples_path, index=False, encoding="utf-8")
    return len(ratios), len(not_found)


def main():
    parser = argparse.ArgumentParser(description="Download VirusTotal reports for the selected samples.")
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES, help="samples.csv path.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Folder for the reports.")
    parser.add_argument("--max-requests", type=int, default=480,
                        help="Stop after this many requests in this run (daily limit is 500).")
    args = parser.parse_args()

    api_key = load_api_key()
    if not args.samples.exists():
        sys.exit(f"ERROR: {args.samples} not found. Run 02_select_samples.py first.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    reports_path = args.out_dir / "reports.jsonl"
    not_found_path = args.out_dir / "not_found.txt"

    # Work out what is still missing (this is how the script resumes)
    all_hashes = pd.read_csv(args.samples, dtype=str)["sha256"].tolist()
    done = read_done_hashes(reports_path, not_found_path)
    todo = [h for h in all_hashes if h not in done]
    batch = todo[: args.max_requests]

    print(f"{len(all_hashes)} samples | {len(done)} already done | {len(todo)} to do | "
          f"{len(batch)} in this run (~{len(batch) * REQUEST_DELAY / 60:.0f} min)")

    fetched = 0
    try:
        # Append mode + flush after every line: nothing is lost if you press Ctrl+C
        with open(reports_path, "a", encoding="utf-8") as rep_f, \
             open(not_found_path, "a", encoding="utf-8") as nf_f:
            for n, sha256 in enumerate(batch, start=1):
                report = fetch_report(api_key, sha256)
                if report is None:
                    nf_f.write(sha256 + "\n")
                    nf_f.flush()
                    print(f"  [{n}/{len(batch)}] {sha256[:16]}...  NOT IN VT")
                else:
                    trimmed = trim_report(report)
                    rep_f.write(json.dumps(trimmed) + "\n")
                    rep_f.flush()
                    print(f"  [{n}/{len(batch)}] {sha256[:16]}...  "
                          f"{detection_ratio(trimmed['data']['attributes'])}")
                fetched += 1
                if n < len(batch):
                    time.sleep(REQUEST_DELAY)
    except DailyQuotaExceeded:
        print("\n[!] VirusTotal daily quota reached. Run the script again tomorrow to continue.")
    except KeyboardInterrupt:
        print("\n[!] Stopped by user. Progress is saved; run again to continue.")

    # Always update samples.csv with whatever we have so far
    n_reports, n_not_found = update_samples_csv(args.samples, reports_path, not_found_path)
    remaining = len(all_hashes) - n_reports - n_not_found
    print(f"\nThis run: {fetched} requests.")
    print(f"Total: {n_reports} reports, {n_not_found} not in VT, {remaining} remaining.")
    if remaining == 0:
        print("All samples done -> next step: 04_run_avclass.py")


if __name__ == "__main__":
    main()
