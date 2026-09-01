# BCA-HFP

**Bidirectional Cross-Attention with High-Resolution Fingerprints Enables Zero-Shot Anticancer Drug Response Prediction**

## Overview

BCA-HFP is a deep learning framework for predicting anticancer drug sensitivity under the **Leave-Drug-Out (LDO)** zero-shot setting, where test drugs are completely unseen during training.

The model addresses two fundamental design flaws that drive zero-shot generalization failure:

1. **Hash collision** from low-dimensional fingerprint compression
2. **Capacity bottleneck** from shallow regression heads

## Key Features

- **Bidirectional Cross-Attention**: Simultaneously models Drug→Gene and Gene→Drug interactions via two parallel multi-head attention streams
- **High-Resolution Fingerprints**: 2048-bit Morgan fingerprints (radius 2) preserve chemical space topology
- **Gated Dynamic Fusion**: Adaptive weighting of three modalities — gene pooled (micro target interaction), global gene (macro transcriptome), and drug global (macro structure) — with temperature-scaled Softmax
- **Deep MLP Regressor**: 3-layer perceptron (512→256→128→1) with BatchNorm, ReLU, and Dropout
- **Multi-level Interpretability**: Gate weights and 8-head attention heatmaps that can be mapped to core drug pharmacophores and functionally relevant genes

## Performance

Results are reported on the **GDSCv2** dataset under the **LDO** split with **Geneformer v1** gene embeddings (random seed = 42).

| Model | Pearson r | R² | RMSE | MAE |
| :---- | :-------: | :------: | :---: | :---: |
| GraphDRP | 0.2349 | -0.2425 | 2.5955 | 2.1312 |
| GraTransDRP | 0.4756 | 0.0823 | 2.2306 | 1.8399 |
| DeepDR | 0.1778 | -0.1425 | 2.4889 | 2.1072 |
| BCA-HFP (attention-free) | 0.5221 | -0.0884 | 2.4293 | 1.9958 |
| **BCA-HFP** | **0.5413** | **0.2478** | **2.0195** | **1.6816** |

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
│   └── no_attention.py              # Attention-free baseline model
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

The model uses the **GDSCv2** dataset. Data preprocessing includes:

- **Gene expression features**: Top 2048 highly expressed genes encoded by Geneformer v1 → `(2048, 256)`
- **Atom-level drug features**: SMILES encoded by FG-BERT → `(n_atoms, 256)`
- **Global fingerprints**: RDKit 2048-bit Morgan fingerprints (radius 2)

## Quick Start

### Installation

```bash
git clone https://github.com/duolichate/BCA-HFP.git
cd BCA-HFP/DrugResponse_BCAHFP
conda env create -f environment.yml
conda activate bca-hfp
```

### Train

```bash
# Holdout training (default: Geneformer v1)
python -m bca_hfp.training.train_holdout --dataset GDSC --gene_version V1 --model_type attention

# Cross-validation
python -m bca_hfp.training.train_cv --dataset GDSC --gene_version V1 --model_type attention
```

### Evaluation

```bash
# Evaluate on the full test set
python -m bca_hfp.analysis.predict_all --dataset GDSC --gene_version V1 --model_type attention
```

## Citation

If you use BCA-HFP in your research, please cite:

```
Ye, Q.; Xie, X.; Song, Y.; Yu, H.; Lu, L. Bidirectional Cross-Attention with High-Resolution Fingerprints Enables Zero-Shot Anticancer Drug Response Prediction.
```

## License

MIT License
