# Network-Based Malware Family Classification

## Objective

This module explores malware family classification using network-traffic features extracted from PCAP files generated during sandbox analysis.

It is the network-analysis component (Step 2) of the Cyber_mid hybrid malware classification project.

## Pipeline

1. Extract basic network traffic statistics.
2. Extract advanced packet, protocol, timing, and flow features.
3. Train a Random Forest classifier.
4. Evaluate the classifier using stratified cross-validation.
5. Generate a confusion matrix and per-family F1-score visualization.

## Scripts

| Script | Purpose |
|---|---|
| `extract_network_features_step2.py` | Extract initial network features |
| `extract_network_features_advanced.py` | Extract additional network statistics |
| `train_network_classifier.py` | Train the baseline classifier |
| `train_network_compatible_api.py` | Final model: OOF probabilities on the 269 API-compatible samples (used by the hybrid) |
| `validate_network_model.py` | Evaluate the baseline using cross-validation |
| `compare_network_features.py` | Compare feature sets |
| `final_evaluation_network.py` | Generate out-of-fold predictions and metrics |
| `plot_network_results.py` | Generate evaluation figures |

## Experimental results

Final model (`train_network_compatible_api.py`): Random Forest on 11 packet/protocol
statistics, evaluated with stratified 5-fold cross-validation (seed 42, out-of-fold
predictions) on the **269 samples** that also have an API trace (5 families).

- Accuracy: 0.558
- Macro F1-score: 0.559

Per-family results are in `reports/network/RESULTS.md`. Amadey (F1 0.64) and LokiBot (0.60)
are recognised best; AgentTesla and FormBook are often confused with each other.

An earlier experiment on 101 samples (macro-F1 0.676) used a smaller, imbalanced subset and
feature selection informed by exploratory analysis; it is superseded by the 269-sample result.
The comparison with the API and hybrid models is in `reports/final/results_table.md`.

## Requirements and data

Use the project's lab requirements and Python environment instructions in the root README. Feature extraction requires access to the relevant PCAP files and compatible analysis tools.

The original PCAP files, malware binaries, sandbox reports, and labeled dataset are not included in this module. Obtain data only through the project's approved workflow.

## Safety

Never commit malware binaries, PCAP files, secrets, API keys, or sensitive sandbox artifacts. Keep original analysis data in the authorized lab environment.
