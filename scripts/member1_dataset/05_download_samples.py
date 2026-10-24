#!/usr/bin/env python3
"""Download the samples listed in data/metadata/samples.csv from MalwareBazaar,
verify every file against the SHA-256 in the CSV, and fill sha256_verified /
download_date. Run ONLY on the Ubuntu/Cuckoo VM (never on the school network).

  python scripts/member1_dataset/05_download_samples.py            # all rows
  python scripts/member1_dataset/05_download_samples.py --limit 3   # quick test
"""
import argparse
import io
import os
import sys
import time
from datetime import date

import pyzipper
import requests

import lab_common as lc

MB_URL = os.getenv("MB_URL", "https://mb-api.abuse.ch/api/v1/")
MB_KEY = os.getenv("MB_AUTH_KEY", "").strip()
ZIP_PASSWORD = b"infected"


def fetch_zip(sha256):
    """Return zip bytes, or (None, reason)."""
    last = "unknown error"
    for attempt in range(3):
        try:
            r = requests.post(MB_URL, data={"query": "get_file", "sha256_hash": sha256},
                              headers={"Auth-Key": MB_KEY}, timeout=120)
            if r.status_code == 200 and r.content[:2] == b"PK":
                return r.content, ""
            try:
                last = "MalwareBazaar: %s" % r.json().get("query_status", r.text[:80])
            except ValueError:
                last = "HTTP %s" % r.status_code
            if r.status_code in (401, 403) or "not_found" in last:
                break                                  # retrying will not help
        except requests.RequestException as exc:
            last = "download failed: %s" % str(exc)[:120]
        time.sleep(5 * (attempt + 1))
    return None, last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="only process the first N pending rows")
    args = ap.parse_args()
    if not MB_KEY:
        sys.exit("MB_AUTH_KEY is missing in .env (get one at https://auth.abuse.ch/)")

    lc.SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    fields, rows = lc.read_rows()
    done = 0
    for row in rows:
        sha = row["sha256"].strip().lower()
        path = lc.sample_path(row)
        if path.exists() and lc.sha256_file(path) == sha:
            row["sha256_verified"] = "True"
            continue
        if args.limit and done >= args.limit:
            break
        done += 1
        print("[%s] downloading %s" % (row.get("sample_id", "?"), sha[:16]))
        blob, err = fetch_zip(sha)
        if blob is None:
            row["sha256_verified"] = "False"
            row["notes"] = err
        else:
            with pyzipper.AESZipFile(io.BytesIO(blob)) as zf:
                zf.setpassword(ZIP_PASSWORD)
                data = zf.read(zf.namelist()[0])
            path.write_bytes(data)
            os.chmod(str(path), 0o600)
            if lc.sha256_file(path) == sha:
                row["sha256_verified"] = "True"
                row["download_date"] = row["download_date"] or date.today().isoformat()
                row["notes"] = "Downloaded and SHA-256 verified"
            else:
                path.unlink()                      # never keep a file that does not match
                row["sha256_verified"] = "False"
                row["notes"] = "SHA-256 mismatch - file discarded"
        lc.write_rows(fields, rows)
        time.sleep(1)
    lc.write_rows(fields, rows)
    ok = sum(1 for r in rows if r["sha256_verified"] == "True")
    print("Verified: %d / %d  (files in %s)" % (ok, len(rows), lc.SAMPLE_DIR))


if __name__ == "__main__":
    main()
