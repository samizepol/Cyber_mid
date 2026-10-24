"""Shared helpers for the Cyber_mid dataset scripts (Python 3.6+)."""
import csv
import hashlib
import os
import tempfile
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]          # Cyber_mid/
load_dotenv(str(ROOT / ".env"))

CSV_PATH = ROOT / "data" / "metadata" / "samples.csv"
LAB_DIR = Path(os.path.expanduser(os.getenv("LAB_DIR", "~/malware_lab")))
SAMPLE_DIR = Path(os.path.expanduser(os.getenv("SAMPLE_DIR", str(LAB_DIR / "samples"))))
REPORT_DIR = LAB_DIR / "reports"
TRACE_DIR = LAB_DIR / "api_traces"

REQUIRED_COLUMNS = [
    "download_date", "sha256_verified", "assigned_member", "execution_id",
    "analysis_date", "analysis_status", "api_call_count", "pcap_packets", "notes",
]


def read_rows():
    """Return (fieldnames, rows). Missing result columns are appended."""
    with open(str(CSV_PATH), newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    for col in REQUIRED_COLUMNS:
        if col not in fields:
            fields.append(col)
    for row in rows:
        for col in fields:
            row.setdefault(col, "")
            if row[col] is None:
                row[col] = ""
    return fields, rows


def write_rows(fields, rows):
    """Atomic write: temp file in the same folder, then replace."""
    fd, tmp = tempfile.mkstemp(dir=str(CSV_PATH.parent), suffix=".tmp")
    with os.fdopen(fd, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, str(CSV_PATH))


def sha256_file(path):
    h = hashlib.sha256()
    with open(str(path), "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sample_path(row):
    name = (row.get("file_name") or row.get("sha256") or "").strip()
    return SAMPLE_DIR / name
