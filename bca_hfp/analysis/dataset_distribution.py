# bca_hfp/analysis/dataset_distribution.py
"""Visualize label and chemical space distributions across data splits."""

import sys
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from bca_hfp.config.base_config import get_base_config, PROJECT_ROOT, root_dir
from bca_hfp.data.loader import (
    load_drug_atom_features, load_gene_embeddings,
    load_response, load_morgan_fingerprints
)
from bca_hfp.training.train_holdout import split_data_by_drug, split_data_by_cell, split_data_random


def visualize_data_distribution(train_df, val_df, test_df, label_col,
                                drug_global_dict, drug_id_col, output_path):
    """Visualize label distribution and feature space distribution."""
    plt.figure(figsize=(14, 6))

    # Subplot 1: Label density distribution
    plt.subplot(1, 2, 1)
    sns.kdeplot(train_df[label_col], label='Train', fill=True, alpha=0.4)
    sns.kdeplot(val_df[label_col], label='Val', fill=True, alpha=0.4)
    sns.kdeplot(test_df[label_col], label='Test_test', fill=True, alpha=0.4)
    plt.title('Target (Label) Distribution Density', fontsize=12)
    plt.xlabel('LN_IC50', fontsize=10)
    plt.ylabel('Density', fontsize=10)
    plt.legend(fontsize=9)

    # Subplot 2: PCA chemical space distribution
    plt.subplot(1, 2, 2)

    def get_valid_drugs(df):
        return [d for d in df[drug_id_col].unique() if str(d) in drug_global_dict]

    train_drugs = get_valid_drugs(train_df)
    val_drugs = get_valid_drugs(val_df)
    test_drugs = get_valid_drugs(test_df)

    print(f"PCA valid drugs: Train={len(train_drugs)}, Val={len(val_drugs)}, Test_test={len(test_drugs)}")

    if len(train_drugs) == 0 or len(val_drugs) == 0 or len(test_drugs) == 0:
        print("Warning: No valid drug fingerprints in some splits, skipping PCA plot")
        plt.text(0.5, 0.5, "No valid drugs for PCA", ha='center', va='center',
                 transform=plt.gca().transAxes)
    else:
        X_train = np.array([drug_global_dict[str(d)] for d in train_drugs])
        X_val = np.array([drug_global_dict[str(d)] for d in val_drugs])
        X_test = np.array([drug_global_dict[str(d)] for d in test_drugs])

        X_all = np.vstack([X_train, X_val, X_test])
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_all)

        len_train, len_val = len(X_train), len(X_val)
        pca_train = X_pca[:len_train]
        pca_val = X_pca[len_train: len_train + len_val]
        pca_test = X_pca[len_train + len_val:]

        plt.scatter(pca_train[:, 0], pca_train[:, 1], label='Train Drugs',
                   alpha=0.6, s=30, c='steelblue')
        plt.scatter(pca_val[:, 0], pca_val[:, 1], label='Val Drugs',
                   alpha=0.8, s=50, marker='^', c='orange')
        plt.scatter(pca_test[:, 0], pca_test[:, 1], label='Test_test Drugs',
                   alpha=0.4, s=50, marker='s', c='green')

        plt.title('Drug Chemical Space (PCA of Morgan Fingerprints)', fontsize=12)
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=10)
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=10)
        plt.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Distribution visualization saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate data distribution visualization')
    parser.add_argument('--dataset', type=str, default='GDSC', choices=['GDSC'])
    parser.add_argument('--split_method', type=str, default='drug',
                        choices=['random', 'cell', 'drug'])
    parser.add_argument('--output_path', type=str,
                        default=os.path.join(root_dir("Picture"), "distribution_check.png"))
    parser.add_argument('--gene_version', type=str, default='V1', choices=['V1', 'V2'])
    parser.add_argument('--fp_bits', type=int, default=2048)
    parser.add_argument('--random_state', type=int, default=42)
    args = parser.parse_args()

    config = get_base_config(args.dataset)
    config['data']['random_state'] = args.random_state

    base_data_path = root_dir("Data")
    if args.gene_version == 'V1':
        config['data']['gene_feat_dir'] = f"{base_data_path}/GDSC_EXP_GF_V1"
        config['data']['gene_input_dim'] = 256
    else:
        config['data']['gene_feat_dir'] = f"{base_data_path}/GDSC_EXP_GF_V2"
        config['data']['gene_input_dim'] = 768

    drug_id_col = config['data']['drug_id_col']
    cell_id_col = config['data']['cell_id_col']
    label_col = config['data']['label_col']

    # Load drug atom features (for filtering)
    print("Loading drug atom features...")
    drug_atom_dict = load_drug_atom_features(config['data']['drug_feat_path'])

    # Load gene embeddings (for filtering)
    print("Loading gene embeddings...")
    gene_emb_dict, _ = load_gene_embeddings(
        config['data']['gene_feat_dir'],
        cell_id_col
    )

    # Load Morgan fingerprints (for PCA visualization)
    print("Loading drug Morgan fingerprints...")
    drug_global_dict = load_morgan_fingerprints(
        config['data']['drug_feat_path'],
        fp_bits=args.fp_bits,
        fp_radius=2
    )
    config['data']['drug_global_dim'] = args.fp_bits

    # Load response data
    print("Loading response data...")
    response_df = load_response(
        config['data']['response_path'],
        cell_id_col,
        drug_id_col
    )

    # Filter valid samples (same logic as training script)
    print("Filtering valid samples...")
    valid_drug_mask = (
        response_df[drug_id_col].isin(drug_atom_dict) &
        response_df[drug_id_col].astype(str).isin(drug_global_dict)
    )
    valid_cell_mask = response_df[cell_id_col].isin(gene_emb_dict)
    valid_mask = valid_drug_mask & valid_cell_mask

    filtered_df = response_df[valid_mask].copy()
    print(f"Original samples: {len(response_df)}, Valid samples after filtering: {len(filtered_df)}")

    if len(filtered_df) == 0:
        raise ValueError("No valid samples after filtering, please check data")

    # Split data (same as training script)
    print(f"Splitting data by {args.split_method}...")
    if args.split_method == 'random':
        train_df, val_df, test_df = split_data_random(
            filtered_df, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1,
            random_state=args.random_state
        )
    elif args.split_method == 'cell':
        train_df, val_df, test_df = split_data_by_cell(
            filtered_df, cell_id_col, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1,
            random_state=args.random_state
        )
    else:
        train_df, val_df, test_df = split_data_by_drug(
            filtered_df, drug_id_col, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1,
            random_state=args.random_state
        )

    print(f"Train: {len(train_df)} samples, {train_df[drug_id_col].nunique()} drugs")
    print(f"Val: {len(val_df)} samples, {val_df[drug_id_col].nunique()} drugs")
    print(f"Test_test: {len(test_df)} samples, {test_df[drug_id_col].nunique()} drugs")

    # Generate visualization
    visualize_data_distribution(
        train_df, val_df, test_df,
        label_col, drug_global_dict, drug_id_col,
        args.output_path
    )


if __name__ == '__main__':
    main()
