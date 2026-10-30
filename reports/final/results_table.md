| Method | Accuracy [95% CI] | Macro-F1 [95% CI] | Δ macro-F1 vs. API [95% CI] | Macro-F1, 10×5-fold CV | Folds better than API |
|---|---|---|---|---|---|
| api_lr | 0.695 [0.639, 0.751] | 0.688 [0.634, 0.740] |  | 0.685 ± 0.059 |  |
| api_rf | 0.684 [0.628, 0.736] | 0.677 [0.621, 0.729] | -0.011 [-0.051, +0.028] |  |  |
| network | 0.558 [0.498, 0.617] | 0.559 [0.497, 0.619] | -0.129 [-0.195, -0.066] | 0.556 ± 0.055 | 0/50 |
| mean | 0.710 [0.654, 0.766] | 0.707 [0.647, 0.762] | +0.019 [-0.029, +0.066] | 0.704 ± 0.058 | 34/50 |
| product | 0.717 [0.662, 0.770] | 0.715 [0.658, 0.768] | +0.027 [-0.028, +0.082] | 0.712 ± 0.057 | 28/50 |
| weighted | 0.729 [0.676, 0.781] | 0.725 [0.669, 0.776] | +0.037 [-0.003, +0.078] | 0.725 ± 0.058 | 45/50 |
| stacking | 0.688 [0.632, 0.743] | 0.682 [0.625, 0.737] | -0.006 [-0.043, +0.033] | 0.683 ± 0.055 | 24/50 |
| early_net_proba |  |  |  | 0.680 ± 0.067 | 28/50 (vs. its own API baseline) |

n = 269 samples (ok Cuckoo runs with both an API trace and a network row), 5 families.
Accuracy / macro-F1: pooled out-of-fold predictions (5-fold stratified CV, seed 42; late fusion: first repeat).
95% CI: bootstrap over samples, 2000 resamples. Δ: paired bootstrap on the same resamples.
10×5-fold CV: mean ± std over 50 folds from Member 4 (reports/hybrid/).
Early fusion has no per-sample predictions, so only its repeated-CV numbers exist.
