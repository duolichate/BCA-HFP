# BCA-HFP

**Bidirectional Cross-Attention with High-Resolution Fingerprints for Zero-Shot Anticancer Drug Response Prediction**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🧬 Overview

BCA-HFP is a deep learning framework for predicting anticancer drug sensitivity under the **Leave-Drug-Out (LDO)** zero-shot setting—where test drugs are completely unseen during training. The model addresses two critical failure modes in existing methods:

1. **Hash collision** from low-dimensional fingerprint compression
2. **Capacity bottleneck** from shallow regression heads

## 🔬 Key Features

- **Bidirectional Cross-Attention**: Simultaneously models Drug→Gene and Gene→Drug interactions
- **High-Resolution Fingerprints**: 2048-bit Morgan fingerprints preserve chemical topology
- **Gated Dynamic Fusion**: Adaptive weighting of three modalities (Gene Pooled, Global Gene, Drug Global) with temperature-scaled Softmax
- **Deep MLP Regressor**: 3-layer perceptron (512→256→128→1) with BatchNorm and Dropout
- **Multi-level Interpretability**: Gate weights, 8-head attention heatmaps, and RDKit molecular mapping

## 📊 Performance

| Model              |   LDO R²   | LDO Pearson |  RMSE  |  MAE   |
| :----------------- | :--------: | :---------: | :----: | :----: |
| **BCA-HFP (Ours)** | **0.2478** | **0.5413**  | 2.0195 | 1.6816 |
| DeepDR             |  -0.1425   |   0.1778    | 2.4889 | 2.1072 |
| GraTransDRP        |   0.0823   |   0.4756    | 2.2306 | 1.8399 |
| GraphDRP           |  -0.2425   |   0.2349    | 2.5955 | 2.1312 |

## 📁 Data

The model uses GDSCv2 and CCLE datasets. Data preprocessing includes:

- Geneformer V1 for gene expression embedding → `(2048, 256)`
- FG-BERT for atomic-level drug embedding → `(n_atoms, 256)`
- RDKit for 2048-bit Morgan fingerprints

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/yourusername/BCA-HFP.git
cd BCA-HFP
conda env create -f environment.yml
conda activate bca-hfp

```
