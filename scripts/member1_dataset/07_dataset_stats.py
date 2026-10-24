#!/usr/bin/env python3
"""Compute execution statistics from samples.csv and write them into docs/dataset.md
between the AUTO-STATS markers (appended at the end if the markers do not exist yet)."""
import statistics
from collections import Counter
from datetime import date

import lab_common as lc

BEGIN, END = "<!-- BEGIN AUTO-STATS -->", "<!-- END AUTO-STATS -->"
DOC = lc.ROOT / "docs" / "dataset.md"


def table(counter, head):
    lines = ["| %s | n |" % head, "|---|---|"]
    lines += ["| %s | %d |" % (k or "(empty)", v) for k, v in sorted(counter.items())]
    return "\n".join(lines)


def numbers(rows, col):
    out = []
    for r in rows:
        try:
            out.append(int(r[col]))
        except (ValueError, KeyError):
            pass
    return out


def main():
    _, rows = lc.read_rows()
    n = len(rows)
    out = [BEGIN, "## Cuckoo execution results (auto-generated %s)" % date.today().isoformat(), ""]
    out.append("- Samples in metadata: **%d**" % n)
    out.append("- SHA-256 verified: **%d**" % sum(r["sha256_verified"] == "True" for r in rows))
    out.append("- Analyses run (execution_id set): **%d**\n" % sum(bool(r["execution_id"]) for r in rows))
    out.append(table(Counter(r["analysis_status"] for r in rows), "analysis_status") + "\n")
    out.append(table(Counter(r["assigned_member"] for r in rows), "assigned_member") + "\n")
    out.append("| metric | min | median | max | mean |\n|---|---|---|---|---|")
    for col in ("api_call_count", "pcap_packets"):
        v = numbers([r for r in rows if r["analysis_status"] in ("ok", "no_network", "evasive")], col)
        if v:
            out.append("| %s | %d | %d | %d | %.1f |" % (col, min(v), statistics.median(v), max(v), statistics.mean(v)))
    for label_col in ("label", "family", "avclass_label", "avclass_family"):
        if rows and label_col in rows[0]:
            out.append("\nSamples per %s (all rows):\n" % label_col)
            out.append(table(Counter(r[label_col] for r in rows), label_col))
            break
    out.append("\n" + END)
    block = "\n".join(out)

    text = DOC.read_text(encoding="utf-8") if DOC.exists() else "# Dataset documentation\n"
    if BEGIN in text and END in text:
        head, rest = text.split(BEGIN, 1)
        text = head + block + rest.split(END, 1)[1]
    else:
        text = text.rstrip() + "\n\n" + block + "\n"
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text(text, encoding="utf-8")
    print("Updated", DOC)


if __name__ == "__main__":
    main()
