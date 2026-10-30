# Evaluation Protocol (Member 5: Evaluation, Experiments and Reproducibility)

This document defines how every classifier in the project is evaluated, so the numbers
of the API classifier (Step 1), the network classifier (Step 2) and the hybrid models
(Step 3) can be compared directly. The final numbers are in `reports/final/` and
are produced by `scripts/member5_evaluation/` (see `REPRODUCE.md`).

---

## 1. Samples

| Stage | Samples | Source |
|---|---|---|
| Selected (9 families × 60) | 540 | `data/metadata/samples.csv` |
| Cuckoo runs recorded | 285 | `execution_id` filled |
| Runs with `analysis_status = ok` (5 families) | 270 | API traces for all 270 |
| **Evaluated: ok runs with API trace AND network row** | **269** | S0001 has no network row |

Families of the 269 evaluated samples: agenttesla 58, formbook 58, amadey 55,
asyncrat 54, lokibot 44. The other four selected families (njrat, redline, remcos,
smokeloader) were not analysed in time (`docs/dataset.md`).

**All methods are compared on the same 269 samples.** The API model was trained on 270
samples; its prediction for S0001 is simply left out of the comparison.

## 2. Split rules

1. **Main protocol: stratified 5-fold cross-validation, `shuffle=True`, `random_state=42`.**
   Every sample is predicted exactly once by a model that did not see it during training
   (out-of-fold, OOF predictions). All headline numbers come from OOF predictions.
2. **Fusion: repeated stratified 5-fold CV (10 repeats, seed 42) on the 269 OOF probability
   vectors** (Member 4). Anything that is learned (fusion weight, stacking model) is fitted on
   the training folds of each split only; baselines are scored on the same test folds.
3. **Official fixed split (train / val / test = 70 / 15 / 15 %, fixed before any analysis):
   reference only.** LokiBot has no analysed test sample (its 15 unanalysed samples were all in
   val/test), so the test split has only 32 samples from 4 families. It is reported as a
   sanity check for the API model, not as the main result.
4. **No leakage:** text vectorizers, scalers and imputers are fitted inside each training
   fold (scikit-learn pipelines); the test split was fixed before any Cuckoo run; the same
   sample never appears twice (checked by `01_check_integrity.py`).

Known caveat: the API and network models each used StratifiedKFold(5, seed 42), but over
different row orders, so their fold assignments are not identical. Every prediction is still
out-of-fold, so the comparison and the fusion are leakage-free; only the per-fold scores of
the two base models are not paired.

## 3. Metrics

| Metric | Why |
|---|---|
| **Macro-F1 (main metric)** | Mean of the per-family F1 scores: every family counts the same, even LokiBot with 44 samples. |
| Accuracy | Easy to read; classes are close to balanced, so it agrees with macro-F1. |
| Weighted-F1 | Reported in the CSV for completeness. |
| Per-family precision / recall / F1 | Shows which families are confused (`per_family_metrics.csv`). |
| Confusion matrix | Rows = true family, columns = predicted family (`confusion_matrices.csv`). |

## 4. Uncertainty and significance

With 269 samples, a few samples move macro-F1 by about 0.01, so every number is reported
with its uncertainty:

- **95% confidence interval:** percentile bootstrap over samples (2000 resamples, seed 42).
- **Gain over the API model:** paired bootstrap, where both models are scored on the same
  resample, giving the 95% CI of the difference. A CI that includes 0 means "not clearly better".
- **Repeated CV (Member 4):** mean ± std of macro-F1 over 50 folds, plus the number of folds
  in which a fusion method beats the API model.
- **McNemar's exact test (Member 4):** compares which samples each model gets right
  (`reports/hybrid/hybrid_mcnemar.csv`).

## 5. Label-noise analysis

Member 1 recorded whether AVClass agrees with the MalwareBazaar family
(`label_agreement`). Samples are split into **high confidence** (agree / partial, 149
samples) and **low confidence** (disagree / no_avclass, 120 samples), and each model is
scored on both groups (`label_noise.csv`, `figures/label_noise.png`). This shows how much of
the error comes from samples whose label itself is uncertain.

## 6. Results (summary)

From `reports/final/results_table.md` (n = 269):

| Model | Macro-F1 [95% CI] | Gain vs. API [95% CI] |
|---|---|---|
| API, TF-IDF + logistic regression | 0.688 [0.634, 0.740] | n/a |
| Network, random forest | 0.559 [0.497, 0.619] | −0.129 [−0.195, −0.066] |
| **Hybrid, weighted late fusion** | **0.725 [0.669, 0.776]** | **+0.037 [−0.003, +0.078]** |

- The weighted hybrid is the best model. It beats the API model in 45 of 50 repeated-CV
  folds, but with 269 samples the gain is not statistically significant (bootstrap CI
  touches 0; McNemar p = 0.12).
- The network model alone is clearly weaker, yet it adds information: it is right on 27
  samples the API model gets wrong.
- All models are 0.16–0.20 more accurate on high-confidence labels (weighted hybrid
  0.82 vs. 0.62), so label noise is a major limit on the measured performance.
- Official test split (API model, 32 samples, 4 families): accuracy 0.781, macro-F1 0.778.
