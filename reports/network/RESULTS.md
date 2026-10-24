# Network Classification Results

## Final reduced-feature model

| Metric | Result |
|---|---:|
| Samples | 101 |
| Features | 20 |
| Malware families | 5 |
| Accuracy | 0.6238 |
| Macro F1-score | 0.6761 |

## Per-family F1-score

| Family | F1-score |
|---|---:|
| AgentTesla | 0.53 |
| Amadey | 0.83 |
| AsyncRAT | 0.81 |
| FormBook | 0.44 |
| LokiBot | 0.77 |

## Main observation

The most frequent confusion was between FormBook and AgentTesla. Thirteen FormBook samples were predicted as AgentTesla, while nine AgentTesla samples were predicted as FormBook.

LokiBot has only eight evaluation samples, so its metrics have substantial uncertainty.

## Limitations

- Small dataset with uneven family counts.
- Feature selection was informed by exploratory analysis on the full dataset.
- Results are preliminary and require validation on additional independent samples.

See `figures/confusion_matrix.png` and `figures/f1_by_family.png`.
