# Hybrid Model (Member 4)

Combines the API-call classifier and the network classifier (midterm Step 3).

## Run (from the repo root, any machine — no malware, traces or pcaps needed)
```bash
pip install pandas scikit-learn matplotlib
python scripts/member4_hybrid/01_late_fusion.py
python scripts/member4_hybrid/02_plot_results.py
```

## Inputs
| File | From |
|---|---|
| `reports/api/api_oof_predictions.csv` | Member 2 — out-of-fold probabilities, model in column `best_model` |
| `data/network/network_oof_predictions_compatible_api.csv` | Member 3 — out-of-fold probabilities |

Both files are joined on `sample_id`; family and split must agree (checked by the script).
Samples present in only one file are dropped and printed (currently `S0001`, no network run).

## Method: late fusion
Both classifiers output one probability per family. The hybrid combines the two vectors:

| Method | Rule | Learned |
|---|---|---|
| `mean` | (p_api + p_net) / 2 | nothing |
| `product` | p_api · p_net, renormalized | nothing |
| `weighted` | w · p_api + (1 − w) · p_net | w ∈ {0, 0.05, …, 1}, best macro-F1 on training folds |
| `stacking` | logistic regression on [log p_api, log p_net] | 10 → 5 weights on training folds |

Evaluation: repeated stratified 5-fold CV (10 repeats, seed 42) over the 269 samples in
both files. Learned methods only see the training folds; the two single-source baselines
are scored on the same test folds, so every row of the result table is directly comparable.
McNemar's exact test (first repeat) checks whether a method's errors differ significantly
from the API model's.

## Results (n = 269, 5 families, 50 outer folds)
| Method | Accuracy | Macro-F1 | Δ Macro-F1 vs. API (paired) | Folds better than API |
|---|---|---|---|---|
| API only | 0.695 ± 0.054 | 0.685 ± 0.059 | — | — |
| Network only | 0.558 ± 0.052 | 0.556 ± 0.055 | −0.129 ± 0.058 | 0 / 50 |
| Mean | 0.710 ± 0.052 | 0.704 ± 0.058 | +0.018 ± 0.043 | 34 / 50 |
| Product | 0.717 ± 0.054 | 0.712 ± 0.057 | +0.026 ± 0.056 | 28 / 50 |
| **Weighted** | **0.730 ± 0.053** | **0.725 ± 0.058** | **+0.039 ± 0.032** | **45 / 50** |
| Stacking | 0.689 ± 0.051 | 0.684 ± 0.055 | −0.002 ± 0.047 | 24 / 50 |

- The learned API weight is stable (0.55–0.75, mean 0.69): the network source gets about a
  third of the vote.
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

## Outputs (`reports/hybrid/`)
| File | Content |
|---|---|
| `hybrid_cv_results.csv` | Summary table above |
| `hybrid_fold_results.csv` | Every outer fold and method (for Member 5) |
| `hybrid_oof_predictions.csv` | Per-sample probabilities and predictions of every method (first repeat) |
| `hybrid_per_family_f1.csv` | F1 per family and method |
| `hybrid_mcnemar.csv` | McNemar exact test vs. API |
| `figures/method_comparison.png` | Macro-F1 per method |
| `figures/confusion_matrices.png` | API vs. network vs. best hybrid |
| `figures/f1_by_family.png` | Per-family F1 of the same three |

## Limitations
- Late fusion only: the feature files (API sequences, network features) are not in the
  repository, so feature-level (early) fusion is not implemented yet.
- The two base models were cross-validated with different fold assignments (different row
  order). Every probability is still out-of-fold, so the fusion is leakage-free, but the
  base models' own CV numbers are not on identical folds.
- Small dataset (269 samples, 5 of 9 planned families), noisy labels, one sandbox setup.
