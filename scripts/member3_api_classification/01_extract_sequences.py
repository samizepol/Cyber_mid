#!/usr/bin/env python3
"""Step 1 - Extract and clean API-call sequences.

Reads data/metadata/samples.csv (analysis_status == ok) and the Cuckoo API traces
(<sha256>_<taskid>.json), cleans each sequence and writes data/api/api_sequences.jsonl.

Cleaning: lowercase names, drop empty names, merge all processes of the run in order,
collapse consecutive repeats (a loop calling the same API 10,000x is one token),
truncate to MAX_SEQ_LEN. Global statistics are computed BEFORE truncation.

  python 01_extract_sequences.py
  python 01_extract_sequences.py --trace-dir /path/to/api_traces
"""
import argparse
import csv
import json
import math
from collections import Counter
from itertools import groupby
from pathlib import Path

import config as cfg


def load_trace(path):
    """Return (list of API names in execution order, number of processes)."""
    data = json.loads(Path(path).read_text())
    apis, n_proc = [], 0
    for proc in data.get("processes", []):
        calls = [str(a).strip().lower() for a in proc.get("apis", []) if a]
        if calls:
            n_proc += 1
            apis.extend(calls)
    return apis, n_proc


def clean(apis):
    collapsed = [k for k, _ in groupby(apis)]               # remove consecutive duplicates
    return collapsed[:cfg.MAX_SEQ_LEN], len(collapsed)


def entropy(counter):
    total = sum(counter.values())
    return -sum(c / total * math.log2(c / total) for c in counter.values()) if total else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-dir", default=str(cfg.TRACE_DIR))
    args = ap.parse_args()
    trace_dir = Path(args.trace_dir)

    with open(str(cfg.SAMPLES_CSV), newline="", encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if r["analysis_status"] == "ok"]
    print("samples with analysis_status=ok:", len(rows))

    cfg.DATA_DIR.mkdir(parents=True, exist_ok=True)
    kept, missing = 0, []
    with open(str(cfg.SEQUENCES), "w", encoding="utf-8") as out:
        for r in rows:
            path = trace_dir / ("%s_%s.json" % (r["sha256"].lower(), r["execution_id"]))
            if not path.exists():
                missing.append(r["sample_id"])
                continue
            apis, n_proc = load_trace(path)
            if not apis:
                missing.append(r["sample_id"])
                continue
            seq, n_collapsed = clean(apis)
            out.write(json.dumps({
                "sample_id": r["sample_id"], "sha256": r["sha256"], "family": r["final_label"],
                "split": r["split"], "label_agreement": r["label_agreement"],
                "execution_id": r["execution_id"],
                "n_calls": len(apis), "n_collapsed": n_collapsed, "n_procs": n_proc,
                "n_unique_apis": len(set(apis)), "api_entropy": round(entropy(Counter(apis)), 4),
                "seq": seq}) + "\n")
            kept += 1
    print("sequences written: %d -> %s" % (kept, cfg.SEQUENCES))
    if missing:
        print("WARNING: %d sample(s) without a usable trace: %s" % (len(missing), ", ".join(missing[:20])))


if __name__ == "__main__":
    main()
