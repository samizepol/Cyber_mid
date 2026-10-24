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
| `download_date`, `sha256_verified` | Sample retrieval on the VM machine | 05 |
| `assigned_member` | Team member who ran the analysis | 06 |
| `execution_id` | **Cuckoo task ID** — links the API trace and the pcap of the same run | 06 |
| `analysis_date`, `analysis_status` | ok / failed / no_network / evasive | 06 |
| `api_call_count`, `pcap_packets` | Basic output statistics | 06 |
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
| Downloaded file hash matches `sha256` | 540 / 540 verified |
| Cuckoo runs recorded (unique `execution_id`) | 285 runs, 0 duplicate task IDs (23–24 Oct 2026) |
| `ok` runs have both API calls and a non-empty pcap | 270 / 270 |

### 6.1 Cuckoo analysis results (all families)

| Family | ok | failed | evasive | not run |
|---|---|---|---|---|
| agenttesla | 59 | 1 | 0 | 0 |
| amadey | 55 | 2 | 3 | 0 |
| asyncrat | 54 | 3 | 3 | 0 |
| formbook | 58 | 1 | 1 | 0 |
| lokibot | 44 | 1 | 0 | 15 |
| njrat | 0 | 0 | 0 | 60 |
| redline | 0 | 0 | 0 | 60 |
| remcos | 0 | 0 | 0 | 60 |
| smokeloader | 0 | 0 | 0 | 60 |
| **Total** | **270** | **8** | **7** | **255** |

### 6.2 Final dataset used for classification

**Scope decision:** because of time constraints, the classifier uses the **5 families
that were fully analyzed**. njrat, redline, remcos and smokeloader were selected but not
analyzed. **LokiBot was excluded**: its 15 unanalyzed samples were all in the val/test
splits (0 test samples, 3 val samples), so it could not be evaluated fairly.

Only samples with `analysis_status = ok` are used: **5 families, 270 samples**.

| Family | Train | Val | Test | Total |
|---|---|---|---|---|
| agenttesla | 41 | 9 | 9 | 59 |
| amadey | 38 | 9 | 8 | 55 |
| asyncrat | 39 | 9 | 6 | 54 |
| formbook | 40 | 9 | 9 | 58 |
| lokibot | 42 | 9 | 9 | 60 |
| **Total** | **200** | **45** | **41** | **286** |

Output statistics of the final set (median, min–max):

| Family | API calls | PCAP packets |
|---|---|---|
| agenttesla | 11,243 (942 – 108,526) | 135 (103 – 287) |
| amadey | 22,481 (110 – 406,624) | 153 (109 – 469) |
| asyncrat | 15,892 (391 – 478,902) | 137 (94 – 1,459) |
| formbook | 8,561 (1,837 – 351,006) | 132 (110 – 789) |
| lokibot | 10,147 (395 – 402,894) | 162.5 (113 – 238) |

Label agreement in the final set: 126 agree, 40 partial, 90 disagree, 44 no_avclass
(55.3% agree or partial).

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
9. **Reduced scope.** Only 4 of the 9 selected families were used (226 samples). Four
   families were not analyzed in time and LokiBot was excluded because none of its test
   samples were analyzed. Results apply to these 4 families only.
10. **Small evaluation set.** The test split has 32 samples (asyncrat only 6), so test
    accuracy has high variance; results should be reported per family and, ideally,
    with cross-validation on train + val.
11. **Failed and evasive runs.** 8 runs failed and 7 were marked evasive (mostly amadey and
    asyncrat). Excluding them may bias the dataset toward samples that behave normally in
    the sandbox.
12. **Wide range of API-call counts.** API calls per sample vary from about 100 to
    almost 480,000, so models need sequence truncation or normalization.
