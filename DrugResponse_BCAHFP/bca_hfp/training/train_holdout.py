# bca_hfp/training/train_holdout.py
"""Hold-out validation training."""

import sys
import os
import argparse
import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from bca_hfp.config.base_config import get_base_config, PROJECT_ROOT
from bca_hfp.data.loader import load_drug_atom_features, load_gene_embeddings, load_response, load_morgan_fingerprints
from bca_hfp.data.dataset import DrugGeneDataset, BaselineDataset, collate_fn
from bca_hfp.models import create_model
from bca_hfp.training.trainer import VariableLengthTrainer, FixedLengthTrainer
from bca_hfp.utils.reproducibility import set_seed


def split_data_random(df, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_state=42):
    """Random split without considering drug/cell grouping."""
    train_df, temp_df = train_test_split(df, test_size=(1 - train_ratio), random_state=random_state)
    val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(temp_df, test_size=(1 - val_ratio_adjusted), random_state=random_state)
    return train_df, val_df, test_df


def split_data_by_cell(df, cell_col, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_state=42):
    """Split by cell line to ensure no cell leakage between sets."""
    unique_cells = df[cell_col].unique()
    np.random.seed(random_state)
    np.random.shuffle(unique_cells)

    n_total = len(unique_cells)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_cells = unique_cells[:n_train]
    val_cells = unique_cells[n_train:n_train + n_val]
    test_cells = unique_cells[n_train + n_val:]

    train_df = df[df[cell_col].isin(train_cells)].copy()
    val_df = df[df[cell_col].isin(val_cells)].copy()
    test_df = df[df[cell_col].isin(test_cells)].copy()
    return train_df, val_df, test_df


def split_data_by_drug(df, drug_col, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_state=42):
    """Split by drug to ensure no drug leakage between sets."""
    unique_drugs = df[drug_col].unique()
    np.random.seed(random_state)
    np.random.shuffle(unique_drugs)

    n_total = len(unique_drugs)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_drugs = unique_drugs[:n_train]
    val_drugs = unique_drugs[n_train:n_train + n_val]
    test_drugs = unique_drugs[n_train + n_val:]

    train_df = df[df[drug_col].isin(train_drugs)].copy()
    val_df = df[df[drug_col].isin(val_drugs)].copy()
    test_df = df[df[drug_col].isin(test_drugs)].copy()
    return train_df, val_df, test_df


def run_holdout(args):
    set_seed(args.seed)
    config = get_base_config(args.dataset)

    base_data_path = os.path.join(PROJECT_ROOT, "Data")
    if args.gene_version == 'V1':
        config['data']['gene_feat_dir'] = f"{base_data_path}/GDSC_EXP_GF_V1"
        config['data']['gene_input_dim'] = 256
    else:
        config['data']['gene_feat_dir'] = f"{base_data_path}/GDSC_EXP_GF_V2"
        config['data']['gene_input_dim'] = 768

    if args.layer == 1:
        hidden_dims = [64]
    elif args.layer == 2:
        hidden_dims = [256, 128]
    else:
        hidden_dims = [512, 256, 128]
    config['model']['hidden_dims'] = hidden_dims
    config['model']['feature_dim'] = 256
    config['model']['num_heads'] = 8
    config['model']['dropout'] = 0.3
    config['model']['num_fusion_layers'] = 1

    fp_dim = 2048
    config['data']['drug_global_dim'] = fp_dim

    print("Generating drug global features...")
    drug_global_dict = load_morgan_fingerprints(
        config['data']['drug_feat_path'],
        fp_bits=fp_dim,
        fp_radius=2
    )

    print("=" * 60)
    print("Loading data...")
    print("=" * 60)

    drug_atom_dict = load_drug_atom_features(config['data']['drug_feat_path'])
    gene_emb_dict, gene_names = load_gene_embeddings(config['data']['gene_feat_dir'], config['data']['cell_id_col'])
    response_df = load_response(config['data']['response_path'],
                                config['data']['cell_id_col'],
                                config['data']['drug_id_col'])

    drug_col = config['data']['drug_id_col']
    cell_col = config['data']['cell_id_col']
    label_col = config['data']['label_col']

    valid_drug_mask = response_df[drug_col].isin(drug_atom_dict)
    valid_cell_mask = response_df[cell_col].isin(gene_emb_dict)
    valid_mask = valid_drug_mask & valid_cell_mask
    filtered_df = response_df[valid_mask].copy()
    print(f"Valid samples: {len(filtered_df)}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if args.split_method == 'random':
        train_df, val_df, test_df = split_data_random(
            filtered_df, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1,
            random_state=args.seed
        )
    elif args.split_method == 'cell':
        train_df, val_df, test_df = split_data_by_cell(
            filtered_df, cell_col, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1,
            random_state=args.seed
        )
    else:
        train_df, val_df, test_df = split_data_by_drug(
            filtered_df, drug_col, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1,
            random_state=args.seed
        )

    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    if args.model_type == 'attention':
        train_dataset = DrugGeneDataset(
            train_df[drug_col].tolist(), train_df[cell_col].tolist(),
            train_df[label_col].values, drug_atom_dict, gene_emb_dict, drug_global_dict
        )
        val_dataset = DrugGeneDataset(
            val_df[drug_col].tolist(), val_df[cell_col].tolist(),
            val_df[label_col].values, drug_atom_dict, gene_emb_dict, drug_global_dict
        )
        test_dataset = DrugGeneDataset(
            test_df[drug_col].tolist(), test_df[cell_col].tolist(),
            test_df[label_col].values, drug_atom_dict, gene_emb_dict, drug_global_dict
        )

        from functools import partial
        collate = partial(collate_fn, atom_mask_prob=config['training'].get('atom_mask_prob', 0.0))
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=config['training']['batch_size'],
            shuffle=True, collate_fn=collate, num_workers=config['training']['num_workers']
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=config['training']['batch_size'],
            shuffle=False, collate_fn=collate, num_workers=config['training']['num_workers']
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=config['training']['batch_size'],
            shuffle=False, collate_fn=collate, num_workers=config['training']['num_workers']
        )

        model = create_model('attention', config).to(device)
        trainer = VariableLengthTrainer(model, config, device)
    else:
        from bca_hfp.data.loader import pool_drug_atom_features, pool_gene_embeddings
        drug_pooled = pool_drug_atom_features(drug_atom_dict)
        gene_pooled = pool_gene_embeddings(gene_emb_dict)

        train_dataset = BaselineDataset(
            train_df[drug_col].tolist(), train_df[cell_col].tolist(),
            train_df[label_col].values, drug_pooled, gene_pooled
        )
        val_dataset = BaselineDataset(
            val_df[drug_col].tolist(), val_df[cell_col].tolist(),
            val_df[label_col].values, drug_pooled, gene_pooled
        )
        test_dataset = BaselineDataset(
            test_df[drug_col].tolist(), test_df[cell_col].tolist(),
            test_df[label_col].values, drug_pooled, gene_pooled
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=config['training']['batch_size'],
            shuffle=True, num_workers=config['training']['num_workers']
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=config['training']['batch_size'],
            shuffle=False, num_workers=config['training']['num_workers']
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=config['training']['batch_size'],
            shuffle=False, num_workers=config['training']['num_workers']
        )

        model = create_model('baseline', config).to(device)
        trainer = FixedLengthTrainer(model, config, device)

    save_dir = config['output']['model_save_dir']
    name = f"{args.model_type}_{args.split_method}"
    history, best_metrics, best_val_path, best_balance_path = trainer.train(
        train_loader, val_loader, config['training']['epochs'],
        save_dir, fold_idx=None, name=name
    )

    results = []
    if best_val_path and os.path.exists(best_val_path):
        checkpoint = torch.load(best_val_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        _, test_metrics = trainer.validate(test_loader)
        print(f"\nTest metrics (best val model): R²={test_metrics['r2']:.4f}, RMSE={test_metrics['rmse']:.4f}, Pearson={test_metrics['pearson']:.4f}")
        test_metrics['model_type'] = 'best_val'
        results.append(test_metrics)

    if best_balance_path and os.path.exists(best_balance_path):
        checkpoint = torch.load(best_balance_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        _, test_metrics = trainer.validate(test_loader)
        print(f"Test metrics (best balance model): R²={test_metrics['r2']:.4f}, RMSE={test_metrics['rmse']:.4f}, Pearson={test_metrics['pearson']:.4f}")
        test_metrics['model_type'] = 'best_balance'
        results.append(test_metrics)

    if results:
        metrics_df = pd.DataFrame(results)
        result_dir = config['output']['result_save_dir']
        os.makedirs(result_dir, exist_ok=True)
        metrics_path = os.path.join(result_dir, f"{args.model_type}_{args.split_method}_holdout_results.csv")
        metrics_df.to_csv(metrics_path, index=False)
        print(f"\nHold-out results saved to {metrics_path}")


def main():
    parser = argparse.ArgumentParser(description='Hold-out validation training')
    parser.add_argument('--dataset', type=str, default='GDSC', choices=['GDSC'])
    parser.add_argument('--model_type', type=str, default='attention', choices=['attention', 'baseline'])
    parser.add_argument('--split_method', type=str, default='drug', choices=['random', 'cell', 'drug'])
    parser.add_argument('--gene_version', type=str, default='V1', choices=['V1', 'V2'])
    parser.add_argument('--layer', type=int, default=3, choices=[1, 2, 3])
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    run_holdout(args)


if __name__ == '__main__':
    main()
