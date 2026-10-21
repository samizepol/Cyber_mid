"""
04_run_avclass.py
Member 1 - Dataset and Label Management

Runs AVClass on the VirusTotal reports and compares its family label with the
MalwareBazaar signature of each sample, then fills these columns of samples.csv:

  avclass_family    most voted family according to AVClass ("" if none)
  avclass_num_av    number of AV engines that voted for that family
  label_agreement   agree    -> AVClass top family = MalwareBazaar family
                    partial  -> MalwareBazaar family is among AVClass families, but not the top one
                    disagree -> AVClass points to a different family only
                    no_avclass -> AVClass could not extract any family
  final_label       the label the models will use (see LABEL POLICY below)

LABEL POLICY (document this in docs/dataset.md):
  final_label = MalwareBazaar family (canonical lowercase name) for ALL samples, because
  MalwareBazaar signatures come from YARA rules / analyst review and AV labels are noisy
  (packers, generic names). label_agreement lets the team also train/evaluate on a
  "high-confidence" subset (agree + partial) and compare results.

SAFE TO RUN ON THE MAIN OS: AVClass only reads text labels from JSON reports.

Outputs:
  data/labels/avclass_output.txt          raw AVClass output (all tags, sha256)
  data/labels/label_agreement_summary.csv agreement per family (use it in the report)
  data/metadata/samples.csv               updated columns listed above

Usage (from the repo root):
    python scripts/member1_dataset/04_run_avclass.py
The script can be re-run safely: it only overwrites the four columns above.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SAMPLES = REPO_ROOT / "data" / "metadata" / "samples.csv"
DEFAULT_REPORTS = REPO_ROOT / "data" / "vt_reports" / "reports.jsonl"
LABELS_DIR = REPO_ROOT / "data" / "labels"

# MalwareBazaar signature -> (canonical final_label, set of AVClass names meaning the same family).
# AVClass uses its own aliases, e.g. njRAT = "bladabindi", RedLine = "reline",
# SmokeLoader = "dofoil" / "mokes" (Kaspersky's name). Found by inspecting AVClass output
# and avclass/data/default.tagging.
FAMILY_MAP = {
    "AgentTesla":     ("agenttesla",  {"agenttesla"}),
    "Amadey":         ("amadey",      {"amadey"}),
    "AsyncRAT":       ("asyncrat",    {"asyncrat"}),
    "Formbook":       ("formbook",    {"formbook", "xloader"}),
    "Loki":           ("lokibot",     {"lokibot", "loki"}),
    "RedLineStealer": ("redline",     {"reline", "redline"}),
    "RemcosRAT":      ("remcos",      {"remcos"}),
    "Smoke Loader":   ("smokeloader", {"dofoil", "mokes", "smokeloader"}),
    "njrat":          ("njrat",       {"bladabindi", "njrat"}),
}


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------

def run_avclass(reports_path, output_path):
    """Run AVClass (all tags, sha256 names) and save its output to a text file."""
    # Use the avclass installed in the same Python environment that runs this script
    scripts_dir = Path(sys.executable).parent
    avclass_exe = scripts_dir / ("avclass.exe" if sys.platform == "win32" else "avclass")
    cmd = [str(avclass_exe), "-f", str(reports_path), "-hash", "sha256", "-t", "-o", str(output_path)]
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        sys.exit(f"ERROR: AVClass failed:\n{result.stderr}")
    # AVClass prints a short summary (samples / no tags) on stderr
    summary = [l for l in result.stderr.replace("[-]", "\n[-]").splitlines() if "Samples:" in l]
    if summary:
        print(summary[-1].strip())


def parse_avclass_output(output_path):
    """
    Parse lines like:
      <sha256> \t <num_detections> \t FILE:msil|13,...,FAM:agenttesla|6,FAM:formbook|2
    Returns {sha256: [(family, votes), ...]} with families sorted by votes (AVClass order).
    Only FAM: tags are used. UNK: tags (tokens AVClass does not recognize) are ignored on
    purpose: most are generic AV heuristic names (e.g. "noon", "jalapeno", "tedy").
    """
    families = {}
    with open(output_path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if not parts[0]:
                continue
            tags = parts[2].split(",") if len(parts) > 2 and parts[2] else []
            fams = []
            for tag in tags:
                name, _, votes = tag.rpartition("|")
                if name.startswith("FAM:"):
                    fams.append((name[4:], int(votes)))
            families[parts[0]] = fams
    return families


def compare(mb_signature, avclass_fams):
    """Return (label_agreement, final_label) for one sample."""
    canonical, aliases = FAMILY_MAP.get(mb_signature, (mb_signature.lower(), {mb_signature.lower()}))
    if not avclass_fams:
        return "no_avclass", canonical
    if avclass_fams[0][0] in aliases:
        return "agree", canonical
    if any(fam in aliases for fam, _ in avclass_fams):
        return "partial", canonical
    return "disagree", canonical


def main():
    parser = argparse.ArgumentParser(description="Run AVClass and compare with MalwareBazaar labels.")
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES, help="samples.csv path.")
    parser.add_argument("--reports", type=Path, default=DEFAULT_REPORTS, help="VT reports (jsonl).")
    parser.add_argument("--labels-dir", type=Path, default=LABELS_DIR, help="Output folder.")
    args = parser.parse_args()

    args.labels_dir.mkdir(parents=True, exist_ok=True)
    avclass_out = args.labels_dir / "avclass_output.txt"

    # 1) Run AVClass and read its families
    run_avclass(args.reports, avclass_out)
    families = parse_avclass_output(avclass_out)

    # 2) Fill the label columns of samples.csv (other columns are kept as they are)
    samples = pd.read_csv(args.samples, dtype=str, keep_default_na=False)
    for i, row in samples.iterrows():
        fams = families.get(row["sha256"], [])
        agreement, final_label = compare(row["mb_signature"], fams)
        samples.at[i, "avclass_family"] = fams[0][0] if fams else ""
        samples.at[i, "avclass_num_av"] = str(fams[0][1]) if fams else "0"
        samples.at[i, "label_agreement"] = agreement
        samples.at[i, "final_label"] = final_label
    samples.to_csv(args.samples, index=False, encoding="utf-8")

    # 3) Agreement summary per family (counts + % "agree or partial")
    summary = pd.crosstab(samples["final_label"], samples["label_agreement"])
    for col in ["agree", "partial", "disagree", "no_avclass"]:
        if col not in summary.columns:
            summary[col] = 0
    summary = summary[["agree", "partial", "disagree", "no_avclass"]]
    summary.loc["ALL"] = summary.sum()            # totals row (added before the % columns)
    summary["total"] = summary.sum(axis=1)
    summary["agree_pct"] = (100 * summary["agree"] / summary["total"]).round(1)
    summary["agree_or_partial_pct"] = (100 * (summary["agree"] + summary["partial"]) / summary["total"]).round(1)
    summary_path = args.labels_dir / "label_agreement_summary.csv"
    summary.to_csv(summary_path)

    # 4) Most common AVClass family among DISAGREEING samples (explains the noise)
    dis = samples[samples["label_agreement"] == "disagree"]
    print("\nMost common AVClass families when it disagrees with MalwareBazaar:")
    print(dis["avclass_family"].value_counts().head(8).to_string())

    print("\nLabel agreement per family (MalwareBazaar vs AVClass):")
    print(summary.to_string())
    print(f"\nUpdated {args.samples}")
    print(f"Saved {avclass_out}")
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
