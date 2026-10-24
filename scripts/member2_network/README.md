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
| `validate_network_model.py` | Evaluate the baseline using cross-validation |
| `compare_network_features.py` | Compare feature sets |
| `final_evaluation_network.py` | Generate out-of-fold predictions and metrics |
| `plot_network_results.py` | Generate evaluation figures |

## Experimental results

The reduced-feature model was evaluated on 101 labeled samples across five malware families.

- Accuracy: 0.6238
- Macro F1-score: 0.6761

The model performed best on Amadey and AsyncRAT in this experiment. FormBook was more difficult to distinguish, particularly from AgentTesla.

These results are preliminary. The dataset is small and imbalanced, and feature selection was informed by earlier exploratory analysis. They should not be interpreted as a guarantee of performance on unseen malware.

## Requirements and data

Use the project's lab requirements and Python environment instructions in the root README. Feature extraction requires access to the relevant PCAP files and compatible analysis tools.

The original PCAP files, malware binaries, sandbox reports, and labeled dataset are not included in this module. Obtain data only through the project's approved workflow.

## Safety

Never commit malware binaries, PCAP files, secrets, API keys, or sensitive sandbox artifacts. Keep original analysis data in the authorized lab environment.
