#!/usr/bin/env python3
"""Submit verified samples to Cuckoo 2.0.7 (REST API), wait for each analysis,
save report / pcap / API trace outside the repo, and fill samples.csv.

  python scripts/member1_dataset/run_cuckoo_batch.py --only S0001   # smoke test
  python scripts/member1_dataset/run_cuckoo_batch.py                # everything pending
  python scripts/member1_dataset/run_cuckoo_batch.py --retry-failed # redo status=failed
"""
import argparse
import json
import os
import struct
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import requests

import lab_common as lc

CUCKOO_URL = os.getenv("CUCKOO_URL", "http://127.0.0.1:8090").rstrip("/")
CUCKOO_TOKEN = os.getenv("CUCKOO_TOKEN", "").strip()
CUCKOO_CWD = Path(os.path.expanduser(os.getenv("CUCKOO_CWD", "~/.cuckoo")))
EVASIVE_MAX_CALLS = int(os.getenv("EVASIVE_MAX_CALLS", "100"))
FINAL = {"ok", "failed", "no_network", "evasive"}

lock = threading.Lock()
stop = threading.Event()
state = {"consecutive_failed": 0}


def headers():
    return {"Authorization": "Bearer " + CUCKOO_TOKEN} if CUCKOO_TOKEN else {}


def api_get(path, **kw):
    return requests.get(CUCKOO_URL + path, headers=headers(), timeout=kw.pop("timeout", 60), **kw)


def count_pcap_packets(data):
    """Count packets in a classic .pcap (no scapy needed)."""
    if len(data) < 24:
        return 0
    magic = data[:4]
    if magic in (b"\xd4\xc3\xb2\xa1", b"\x4d\x3c\xb2\xa1"):
        endian = "<"
    elif magic in (b"\xa1\xb2\xc3\xd4", b"\xa1\xb2\x3c\x4d"):
        endian = ">"
    else:
        return 0
    pos, n = 24, 0
    while pos + 16 <= len(data):
        incl = struct.unpack(endian + "I", data[pos + 8:pos + 12])[0]
        if pos + 16 + incl > len(data):
            break
        pos += 16 + incl
        n += 1
    return n


def submit(path, timeout):
    with open(str(path), "rb") as fh:
        r = requests.post(CUCKOO_URL + "/tasks/create/file", headers=headers(),
                          files={"file": (path.name, fh)},
                          data={"timeout": str(timeout), "priority": "1"}, timeout=300)
    r.raise_for_status()
    body = r.json()
    tid = body.get("task_id") or (body.get("task_ids") or [None])[0]
    if tid is None:
        raise RuntimeError("Cuckoo returned no task_id: %s" % r.text[:200])
    return int(tid)


def task_info(tid):
    r = api_get("/tasks/view/%d" % tid)
    r.raise_for_status()
    return r.json()["task"]


def wait_done(tid, timeout):
    deadline = time.time() + timeout + 900          # analysis + VM start + processing
    while time.time() < deadline and not stop.is_set():
        task = task_info(tid)
        status = task.get("status", "")
        if status == "reported":
            return "reported", task
        if status.startswith("failed"):
            return status, task
        time.sleep(10)
    return "timeout", {}


def get_report(tid):
    r = api_get("/tasks/report/%d" % tid, timeout=300)
    if r.status_code == 200:
        return r.json()
    local = CUCKOO_CWD / "storage" / "analyses" / str(tid) / "reports" / "report.json"
    return json.loads(local.read_text())


def get_pcap(tid):
    r = api_get("/pcap/get/%d" % tid, timeout=300)
    if r.status_code == 200:
        return r.content
    local = CUCKOO_CWD / "storage" / "analyses" / str(tid) / "dump.pcap"
    return local.read_bytes() if local.exists() else b""


def classify(report, api_calls, packets, task_status):
    errors = (report.get("debug") or {}).get("errors") or []
    if task_status != "reported" or api_calls == 0:
        return "failed", "; ".join(errors)[:200] or "no behavioral data"
    if api_calls < EVASIVE_MAX_CALLS:
        return "evasive", "few API calls (%d) - review manually" % api_calls
    if packets == 0:
        return "no_network", "no packets in pcap"
    return "ok", ("; ".join(errors)[:200] if errors else "")


def analyze(row, args):
    sha = row["sha256"].strip().lower()
    path = lc.sample_path(row)
    tid = None
    old = row["execution_id"].strip()
    if old.isdigit() and row["analysis_status"] not in FINAL:      # resume, but only if it is our sample
        try:
            if task_info(int(old)).get("sample", {}).get("sha256") == sha:
                tid = int(old)
        except (requests.RequestException, KeyError, ValueError):
            tid = None
    if tid is None:
        tid = submit(path, args.timeout)
        with lock:
            row["execution_id"] = str(tid)
            lc.write_rows(FIELDS, ROWS)
    print("[%s] task %d submitted" % (row.get("sample_id", "?"), tid))
    status, task = wait_done(tid, args.timeout)
    report, pcap = {}, b""
    if status == "reported":
        report, pcap = get_report(tid), get_pcap(tid)
    processes = (report.get("behavior") or {}).get("processes") or []
    api_calls = sum(len(p.get("calls", [])) for p in processes)
    packets = count_pcap_packets(pcap)
    result, note = classify(report, api_calls, packets, status)
    if status != "reported":
        note = ("task %s" % status) + ("; " + note if note else "")

    lc.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    lc.TRACE_DIR.mkdir(parents=True, exist_ok=True)
    stem = "%s_%d" % (sha, tid)
    if report:
        (lc.REPORT_DIR / (stem + "_report.json")).write_text(json.dumps(report))
    (lc.REPORT_DIR / (stem + ".pcap")).write_bytes(pcap)
    trace = [{"pid": p.get("pid"), "process_name": p.get("process_name"),
              "apis": [c.get("api") for c in p.get("calls", [])]} for p in processes]
    (lc.TRACE_DIR / (stem + ".json")).write_text(json.dumps(
        {"sha256": sha, "task_id": tid, "processes": trace}))

    with lock:
        row.update({"assigned_member": args.member, "execution_id": str(tid),
                    "analysis_date": date.today().isoformat(), "analysis_status": result,
                    "api_call_count": str(api_calls), "pcap_packets": str(packets),
                    "notes": ("task %d; %s" % (tid, note)).strip("; ")})
        lc.write_rows(FIELDS, ROWS)
        state["consecutive_failed"] = state["consecutive_failed"] + 1 if result == "failed" else 0
        if state["consecutive_failed"] >= args.max_fail:
            stop.set()
            print("\n!! %d failed analyses in a row - stopping. Fix the Cuckoo/VM setup, then "
                  "rerun with --retry-failed." % args.max_fail)
    print("[%s] %s  api=%d  packets=%d" % (row.get("sample_id", "?"), result, api_calls, packets))


def main():
    global FIELDS, ROWS
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="comma-separated sample_ids")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--retry-failed", action="store_true")
    ap.add_argument("--redo", default="", help="comma-separated sample_ids to rerun from scratch")
    ap.add_argument("--workers", type=int, default=1, help="parallel analyses (= number of VMs)")
    ap.add_argument("--timeout", type=int, default=120, help="Cuckoo analysis timeout (s)")
    ap.add_argument("--member", default=os.getenv("ASSIGNED_MEMBER", "member"))
    ap.add_argument("--max-fail", type=int, default=3, help="stop after N failures in a row")
    args = ap.parse_args()

    r = api_get("/cuckoo/status", timeout=15)                      # fail fast if API/token is wrong
    if r.status_code != 200:
        sys.exit("Cuckoo API check failed: HTTP %s (token? port 8090? is `cuckoo api` running?)"
                 % r.status_code)

    FIELDS, ROWS = lc.read_rows()
    only = {s for s in args.only.split(",") if s}
    redo = {s for s in args.redo.split(",") if s}
    todo = []
    for row in ROWS:
        sid = row.get("sample_id", "")
        if only and sid not in only:
            continue
        if row["sha256_verified"] != "True" or not lc.sample_path(row).exists():
            continue
        if sid in redo:
            row["analysis_status"] = row["execution_id"] = ""
        elif row["analysis_status"] in FINAL and not (args.retry_failed and row["analysis_status"] == "failed"):
            continue
        todo.append(row)
    if args.limit:
        todo = todo[:args.limit]
    print("%d sample(s) to analyze with %d worker(s)" % (len(todo), args.workers))

    def guarded(row):
        if stop.is_set():
            return
        try:
            analyze(row, args)
        except Exception as exc:                                    # keep the batch alive
            with lock:
                row["analysis_status"] = "failed"
                row["notes"] = ("error: %s" % exc)[:250]
                lc.write_rows(FIELDS, ROWS)
            print("[%s] ERROR %s" % (row.get("sample_id", "?"), exc))

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(guarded, todo))
    print("Finished. Reports/pcaps: %s   API traces: %s" % (lc.REPORT_DIR, lc.TRACE_DIR))


if __name__ == "__main__":
    main()
