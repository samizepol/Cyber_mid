# Hybrid Model (Member 4)

Combines the API-call classifier and the network classifier (midterm Step 3) in two ways:
**late fusion** (combine the two models' predictions) and **early fusion** (one model on
the combined features), and compares both with the single-source classifiers.

## Run (from the repo root, any machine — no malware, traces or pcaps needed)
```bash
pip install pandas scikit-learn matplotlib
python scripts/member4_hybrid/01_late_fusion.py     # late fusion + final hybrid model
python scripts/member4_hybrid/03_early_fusion.py    # early fusion (~20 s)
python scripts/member4_hybrid/02_plot_results.py    # figures (run last)
```

## Inputs
| File | From | In repo |
|---|---|---|
| `reports/api/api_oof_predictions.csv` | Member 2 — out-of-fold probabilities, model in column `best_model` | yes |
| `data/network/network_oof_predictions_compatible_api.csv` | Member 3 — out-of-fold probabilities | yes |
| `data/api/api_sequences.jsonl` | Member 2 — cleaned API sequences (21 MB) | no, Drive |
| `data/network/network_dataset_labeled_270.csv` | Member 3 — per-pcap network statistics | **no, still missing** |

Files are joined on `sample_id`; family and split must agree (checked by the scripts).
Samples present in only one file are dropped and printed (currently `S0001`, no network
prediction). All scripts sort by `sample_id` and use the same repeated stratified 5-fold CV
(10 repeats, seed 42), so **every number below is measured on the same 50 test folds**.

## 1. Late fusion (`01_late_fusion.py`)
Both classifiers output one probability per family. The hybrid combines the two vectors:

| Method | Rule | Learned |
|---|---|---|
| `mean` | (p_api + p_net) / 2 | nothing |
| `product` | p_api · p_net, renormalized | nothing |
| `weighted` | w · p_api + (1 − w) · p_net | w ∈ {0, 0.05, …, 1}, best macro-F1 on training folds |
| `stacking` | logistic regression on [log p_api, log p_net] | 10 → 5 weights on training folds |

Learned methods only see the training folds of each outer split. The two single-source
baselines are scored on the same test folds. McNemar's exact test (first repeat) checks
whether a method's errors differ significantly from the API model's.

| Method | Accuracy | Macro-F1 | Δ Macro-F1 vs. API (paired) | Folds better than API |
|---|---|---|---|---|
| API only | 0.695 ± 0.054 | 0.685 ± 0.059 | — | — |
| Network only | 0.558 ± 0.052 | 0.556 ± 0.055 | −0.129 ± 0.058 | 0 / 50 |
| Mean | 0.710 ± 0.052 | 0.704 ± 0.058 | +0.018 ± 0.043 | 34 / 50 |
| Product | 0.717 ± 0.054 | 0.712 ± 0.057 | +0.026 ± 0.056 | 28 / 50 |
| **Weighted** | **0.730 ± 0.053** | **0.725 ± 0.058** | **+0.039 ± 0.032** | **45 / 50** |
| Stacking | 0.689 ± 0.051 | 0.684 ± 0.055 | −0.002 ± 0.047 | 24 / 50 |

- The learned API weight is stable (0.55–0.75 across folds, 0.70 on all samples): the
  network source gets about a third of the vote.
- Complementarity: of 269 samples, 123 are classified correctly by both models, 64 only by
  API, 27 only by network, 55 by neither. Picking the right model per sample would reach
  0.796 accuracy (upper bound for any late fusion).
- Weighted fusion helps most for LokiBot (F1 0.55 → 0.66), AgentTesla (0.61 → 0.69) and
  Formbook (0.64 → 0.68), and costs a little for Amadey (0.90 → 0.89) and AsyncRAT
  (0.73 → 0.71) (first repeat, `hybrid_per_family_f1.csv`).
- Stacking does not beat the API model: with ~215 training samples per fold, the 10-input
  meta-classifier overfits.
- McNemar (weighted vs. API, first repeat): 18 vs. 9 discordant samples, p = 0.12 — the gain
  is consistent across folds but not significant at 5 % on a single repeat.

**Final hybrid model** (`hybrid_final_model.json`): `p = 0.70 · p_api + 0.30 · p_net`,
predicted family = argmax(p), weight fitted on all 269 samples. Its expected performance is
the CV estimate above, not a score on the training data.

## 2. Early fusion (`03_early_fusion.py`)
One classifier on the combined features. The API part is Member 2's `tfidf_lr` model
(TF-IDF over API 1–3-grams + 5 sequence statistics, logistic regression), re-trained inside
our folds so that "with" and "without network" differ only in the network columns.

| Method | Features | Status |
|---|---|---|
| `api_features` | TF-IDF + sequence statistics | baseline |
| `api_features+net_proba` | + log of the network classifier's 5 OOF probabilities | done |
| `network_features` | network statistics only (Random Forest) | needs `network_dataset_labeled_270.csv` |
| `api_features+net_features` | TF-IDF + sequence statistics + all network statistics | needs `network_dataset_labeled_270.csv` |

| Method | Accuracy | Macro-F1 | Δ Macro-F1 vs. `api_features` (paired) | Folds better |
|---|---|---|---|---|
| api_features | 0.684 ± 0.058 | 0.673 ± 0.064 | — | — |
| api_features+net_proba | 0.688 ± 0.064 | 0.680 ± 0.067 | +0.008 ± 0.045 | 28 / 50 |

- Adding the network evidence as extra input columns barely helps (+0.008, better in only
  28 of 50 folds), while the same evidence combined at the decision level gives +0.039
  (late fusion, weighted).
- Inspection of one fold: the five network columns get large individual coefficients, but
  they are 5 of ~10,000 inputs and carry 0.2 % of the total coefficient mass. Our
  interpretation: in a single linear model the network evidence is diluted by the
  high-dimensional API features; late fusion gives it an explicit, tuned share (30 %).
- The `api_features` baseline (0.673) is slightly below Member 2's reported 0.689 because
  it is evaluated on our folds instead of theirs — within the fold-to-fold spread.
- The network-feature branch was tested with a synthetic feature file (random numbers):
  the network-only model then drops to chance level (0.19 macro-F1 for 5 classes), which
  shows that no label information leaks into the features. Real results follow as soon as
  `network_dataset_labeled_270.csv` is available — just re-run `03` and `02`.

## Conclusion
Combining both sources **does** improve the classification, but only with the right
fusion: the weighted late fusion raises macro-F1 from 0.685 to 0.725 and wins in 45 of 50
folds. Simply appending the network evidence to the API feature vector does not.
The network classifier alone is clearly weaker (0.556); its value lies in the 27 samples
it gets right where the API model fails.

## Outputs (`reports/hybrid/`)
| File | Content |
|---|---|
| `hybrid_cv_results.csv` | Late-fusion summary table |
| `hybrid_fold_results.csv` | Every outer fold and late-fusion method (for Member 5) |
| `hybrid_oof_predictions.csv` | Per-sample probabilities and predictions of every late-fusion method (first repeat) |
| `hybrid_per_family_f1.csv` | F1 per family and method |
| `hybrid_mcnemar.csv` | McNemar exact test vs. API |
| `hybrid_final_model.json` | Final hybrid model (fusion weight on all samples) |
| `early_cv_results.csv`, `early_fold_results.csv` | Early-fusion summary and per-fold results |
| `figures/method_comparison.png` | Macro-F1 per late-fusion method |
| `figures/confusion_matrices.png` | API vs. network vs. best hybrid |
| `figures/f1_by_family.png` | Per-family F1 of the same three |
| `figures/late_vs_early.png` | Late vs. early fusion on the same folds |

## Limitations
- `network_dataset_labeled_270.csv` (Member 3) is not available yet, so early fusion with
  the raw network statistics has no results so far.
- The two base models were cross-validated with different fold assignments (different row
  order). Every probability is still out-of-fold, so the fusion is leakage-free, but the
  base models' own CV numbers are not on identical folds.
- Small dataset (269 samples, 5 of 9 planned families), noisy labels, one sandbox setup.
- No time alignment of API calls and packets: the API sequences contain no timestamps, so
  the "API + network + time" representation suggested in the assignment would need the
  full Cuckoo reports.
