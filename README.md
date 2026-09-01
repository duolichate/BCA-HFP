# BCA-HFP

**Bidirectional Cross-Attention with High-Resolution Fingerprints for Zero-Shot Anticancer Drug Response Prediction**

## Overview

BCA-HFP is a deep learning framework for predicting anticancer drug sensitivity under the **Leave-Drug-Out (LDO)** zero-shot setting, where test drugs are completely unseen during training.

The model addresses two critical failure modes in existing methods:

1. **Hash collision** from low-dimensional fingerprint compression
2. **Capacity bottleneck** from shallow regression heads

## Key Features

- **Bidirectional Cross-Attention**: Simultaneously models Drug→Gene and Gene→Drug interactions
- **High-Resolution Fingerprints**: 2048-bit Morgan fingerprints preserve chemical topology
- **Gated Dynamic Fusion**: Adaptive weighting of three modalities (gene pooled, global gene, drug global) with temperature-scaled Softmax
- **Deep MLP Regressor**: 3-layer perceptron (512→256→128→1) with BatchNorm and Dropout
- **Multi-level Interpretability**: Gate weights and 8-head attention heatmaps

## Performance

Results are reported under the **LDO** setting with **Geneformer V1** gene embeddings.

| Model              |   LDO R²   | LDO Pearson |  RMSE  |  MAE   |
| :----------------- | :--------: | :---------: | :----: | :----: |
| **BCA-HFP (Ours)** | **0.2478** | **0.5413**  | 2.0195 | 1.6816 |
| DeepDR             |  -0.1425   |   0.1778    | 2.4889 | 2.1072 |
| GraTransDRP        |   0.0823   |   0.4756    | 2.2306 | 1.8399 |
| GraphDRP           |  -0.2425   |   0.2349    | 2.5955 | 2.1312 |

## Repository Structure

```
bca_hfp/
├── config/          # Configuration
│   ├── __init__.py
│   └── base_config.py               # All hyperparameters and data paths
│
├── data/            # Data loading
│   ├── __init__.py
│   ├── dataset.py                   # Dataset classes and collate functions
│   └── loader.py                    # Feature loading and pooling functions
│
├── models/          # Model definitions
│   ├── __init__.py                  # Model factory
│   ├── attention.py                 # Bidirectional cross-attention layers
│   ├── gated_fusion.py              # Gated fusion layer
│   ├── main_model.py                # BCA-HFP main model (Regressor)
│   └── no_attention.py              # No-attention baseline model
│
├── training/        # Training module
│   ├── __init__.py
│   ├── trainer.py                   # Trainer
│   ├── metrics.py                   # Evaluation metrics (R², RMSE, Pearson)
│   ├── train_holdout.py             # Holdout training script
│   ├── train_cv.py                  # Cross-validation training script
│   └── extract_attention.py         # Attention weight extraction
│
├── utils/           # Utility functions
│   ├── __init__.py
│   └── reproducibility.py           # Random seed configuration
│
└── __init__.py
```

## Data

The model uses the GDSCv2 dataset. Data preprocessing includes:

- Geneformer V1 gene expression embeddings → `(2048, 256)`
- FG-BERT atomic-level drug embeddings → `(n_atoms, 256)`
- RDKit 2048-bit Morgan fingerprints

## Quick Start

### Installation

```bash
git clone https://github.com/yourusername/BCA-HFP.git
cd BCA-HFP/DrugResponse_BCAHFP
conda env create -f environment.yml
conda activate bca-hfp
```

### Train

```bash
# Holdout training (default: Geneformer V1)
python -m bca_hfp.training.train_holdout --dataset GDSC --gene_version V1 --model_type attention

# Cross-validation
python -m bca_hfp.training.train_cv --dataset GDSC --gene_version V1 --model_type attention
```

### Evaluation

```bash
# Evaluate on the full test set
python -m bca_hfp.analysis.predict_all --dataset GDSC --gene_version V1 --model_type attention
```

## License

MIT License
