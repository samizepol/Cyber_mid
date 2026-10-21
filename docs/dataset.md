# Dataset Documentation (Member 1 — Dataset and Label Management)

This document describes how the malware dataset used in the midterm project
("Hybrid Malware Classification based on API call sequence and network traffic")
was built, labeled, split and validated, and what its limitations are.

All files referenced here are in this repository. **No malware binaries are stored in
the repository**: only hashes, metadata and text reports.

---

## 1. Sources

| Source | Used for | Access |
|---|---|---|
| **MalwareBazaar** (abuse.ch) | Candidate samples and family signatures | Free Auth-Key (auth.abuse.ch) |
| **VirusTotal API v3** | Antivirus labels (~70 engines per sample), input for AVClass | Free API key (4 req/min, 500 req/day) |
| **AVClass** (malicialab, v2.8.10) | Normalized family labels from AV labels | `pip install avclass-malicialab` |

MalwareBazaar was chosen as the main source because it provides recent samples, a
family signature for each sample (from YARA rules and analyst review), and an API for
querying metadata.

Reference datasets cited for comparison (not used for training): Mal-API-2019
(Cuckoo API call sequences), USTC-TFC2016 (malware traffic, used by Wang et al., 2017),
and Stratosphere / CTU-13 (malware pcaps).

---

## 2. Family selection

Nine Windows malware families were selected. Criteria:

1. **Windows PE executables**, so they run in the Cuckoo Windows guest.
2. **Network activity** (C2 communication, data exfiltration, payload download), needed for
   the network-traffic features of Step 2.
3. **Enough recent samples** on MalwareBazaar to select 60 per family.

| MalwareBazaar signature | `final_label` | Type | .exe candidates found |
|---|---|---|---|
| AgentTesla | agenttesla | Info-stealer / keylogger | 264 |
| Amadey | amadey | Loader / botnet | 893 |
| AsyncRAT | asyncrat | Remote access trojan | 644 |
| Formbook | formbook | Info-stealer | 484 |
| Loki | lokibot | Info-stealer | 690 |
| RedLineStealer | redline | Info-stealer | 879 |
| RemcosRAT | remcos | Remote access trojan | 178 |
| Smoke Loader | smokeloader | Loader / backdoor | 929 |
| njrat | njrat | Remote access trojan | 882 |

Candidates were collected with `scripts/member1_dataset/01_query_candidates.py`
(`--limit 1000` per family, keeping only `file_type = exe`, duplicates removed):
**5,843 candidates** in `data/metadata/candidates.csv`.

---

## 3. Labeling method (MalwareBazaar signature + AVClass)

1. A VirusTotal v3 report was downloaded for each selected sample
   (`03_get_vt_reports.py` → `data/vt_reports/reports.jsonl`). All 540 samples were
   known to VirusTotal (0 not found).
2. AVClass was run on these reports (`04_run_avclass.py`, options `-hash sha256 -t`) →
   `data/labels/avclass_output.txt`.
3. The top AVClass family (`FAM:` tag with the most AV votes) was compared with the
   MalwareBazaar signature, using an alias map because AVClass uses its own names
   (e.g. njRAT = `bladabindi`, RedLine = `reline`, SmokeLoader = `dofoil` / `mokes`).
   AVClass `UNK:` tags (unrecognized tokens, mostly generic heuristic names such as
   `noon`, `jalapeno`, `tedy`) were ignored.

**Agreement categories (`label_agreement` column):**

| Value | Meaning |
|---|---|
| `agree` | AVClass top family = MalwareBazaar family |
| `partial` | MalwareBazaar family appears among AVClass families, but not as the top one |
| `disagree` | AVClass only reports other families |
| `no_avclass` | AVClass could not extract any family |

**Label policy:** `final_label` = MalwareBazaar family for all samples, because MalwareBazaar
signatures come from YARA rules and analyst review, while AV labels are noisy (packer and
generic names). Samples with `agree` or `partial` form a **high-confidence subset** that the
team can use to compare model performance against the full dataset.

**Results** (`data/labels/label_agreement_summary.csv`):

| Family | agree | partial | disagree | no_avclass | agree % | agree+partial % |
|---|---|---|---|---|---|---|
| agenttesla | 28 | 10 | 15 | 7 | 46.7 | 63.3 |
| amadey | 37 | 0 | 6 | 17 | 61.7 | 61.7 |
| asyncrat | 21 | 18 | 16 | 5 | 35.0 | 65.0 |
| formbook | 28 | 5 | 13 | 14 | 46.7 | 55.0 |
| lokibot | 12 | 7 | 40 | 1 | 20.0 | 31.7 |
| njrat | 28 | 5 | 21 | 6 | 46.7 | 55.0 |
| redline | 13 | 0 | 32 | 15 | 21.7 | 21.7 |
| remcos | 35 | 2 | 12 | 11 | 58.3 | 61.7 |
| smokeloader | 36 | 4 | 15 | 5 | 60.0 | 66.7 |
| **ALL** | **238** | **51** | **170** | **81** | **44.1** | **53.5** |

When AVClass disagrees, its most frequent families are `agenttesla` (40), `formbook` (25),
`loveletter` (18) and `gamarue` (14). Many .NET info-stealers are distributed with the same
packers/crypters, so AV engines often name the packer or a related family instead of the
payload. `loveletter` (a 2000-era worm) is clearly a mislabel.

---

## 4. Metadata table schema

Master table: `data/metadata/samples.csv` (one row per sample, 540 rows).

| Column | Description | Filled by |
|---|---|---|
| `sample_id` | Permanent team ID (S0001–S0540) | 02 |
| `sha256`, `sha1`, `md5` | Hashes | 01 |
| `file_type`, `file_size`, `file_name` | File info from MalwareBazaar | 01 |
| `source`, `first_seen` | Origin and first-seen date | 01 |
| `mb_signature` | MalwareBazaar family | 01 |
| `vt_detection_ratio` | VirusTotal malicious / total engines | 03 |
| `avclass_family`, `avclass_num_av` | AVClass top family and its votes | 04 |
| `label_agreement` | agree / partial / disagree / no_avclass | 04 |
| `final_label` | Label used by the models | 04 |
| `split` | train / val / test | 02 |
| `download_date`, `sha256_verified` | Sample retrieval on the VM machine | VM stage |
| `assigned_member` | Team member who ran the analysis | VM stage |
| `execution_id` | **Cuckoo task ID** — links the API trace and the pcap of the same run | VM stage |
| `analysis_date`, `analysis_status` | ok / failed / no_network / evasive | VM stage |
| `api_call_count`, `pcap_packets` | Basic output statistics | VM stage |
| `notes` | Free text | any |

---

## 5. Train / validation / test split

- **60 samples per family × 9 families = 540 samples**, chosen at random with
  `random_state = 42` (`02_select_samples.py`), so the selection is reproducible.
- Stratified split per family: **42 train / 9 val / 9 test** (70 / 15 / 15 %),
  totals 378 / 81 / 81.
- The split was fixed **before** any Cuckoo analysis, so no sample can appear in both
  train and test and test samples were not chosen after seeing results.
- 60 per family includes a buffer of about 10 samples for failed or evasive
  analyses (target: ~50 valid samples per family).
- Low-agreement labels are spread similarly across splits
  (agree: 166 train / 33 val / 39 test).

---

## 6. Validation checks and results

| Check | Result |
|---|---|
| Duplicate SHA256 in candidates / selection | 0 |
| MalwareBazaar signature = queried family | 540 / 540 |
| File type = Windows PE (exe) | 540 / 540 (MalwareBazaar metadata) |
| VirusTotal report available | 540 / 540 |
| Samples per family | 60 each (balanced) |
| Selection reproducible (same seed → identical file) | Yes |
| Downloaded file hash matches `sha256` | *To be completed on the VM machine* |
| Cuckoo run has both API trace and non-empty pcap | *To be completed after Cuckoo runs* |
| Valid samples per family after Cuckoo | *To be completed* |

---

## 7. Limitations

1. **Label noise.** Only 44.1 % of samples have full MalwareBazaar–AVClass agreement
   (53.5 % including partial). AV engines often label the packer or a related family.
2. **Weak families.** LokiBot (31.7 %) and RedLine (21.7 %) have low agreement; a
   high-confidence-only subset would leave them with very few training samples.
3. **MalwareBazaar can also be wrong.** Example: S0003 (MalwareBazaar: AgentTesla) carries
   the AVClass unknown token `purelogstealer`, a newer family not in AVClass's taxonomy.
4. **AVClass taxonomy coverage.** Newer families appear only as `UNK:` tokens and were
   not used.
5. **Temporal bias.** Collection windows differ by family (Formbook: last 6 days;
   Smoke Loader: Feb 2025 – Jul 2026; Loki back to Mar 2025). Models may partly learn
   sample age, and older samples are more likely to have inactive C2 servers.
6. **Network behavior depends on the environment.** Inactive C2 servers, the sandbox
   network configuration and sandbox-evasion techniques can reduce or remove network
   traffic for some samples.
7. **Single source and small size.** 540 samples from one repository and 9 families;
   results may not generalize to other families or sources.
8. **Single sandbox configuration.** All samples are analyzed with one guest OS,
   timeout and network setting.
