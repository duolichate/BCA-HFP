# bca_hfp/training/train_cv.py
"""5-fold cross-validation training."""

import sys
import os
import argparse
import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import KFold, GroupKFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from bca_hfp.config.base_config import get_base_config, PROJECT_ROOT
from bca_hfp.data.loader import load_drug_atom_features, load_gene_embeddings, load_response, load_morgan_fingerprints
from bca_hfp.data.dataset import DrugGeneDataset, BaselineDataset, collate_fn
from bca_hfp.models import create_model
from bca_hfp.training.trainer import VariableLengthTrainer, FixedLengthTrainer
from bca_hfp.utils.reproducibility import set_seed


def run_cross_validation(args):
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

    all_fold_metrics = []

    if args.split_method == 'random':
        kf = KFold(n_splits=5, shuffle=True, random_state=args.seed)
        splits = list(kf.split(filtered_df))
    elif args.split_method == 'cell':
        gkf = GroupKFold(n_splits=5)
        groups = filtered_df[cell_col].values
        splits = list(gkf.split(filtered_df, groups=groups))
    else:
        gkf = GroupKFold(n_splits=5)
        groups = filtered_df[drug_col].values
        splits = list(gkf.split(filtered_df, groups=groups))

    for fold_idx, (train_val_idx, test_idx) in enumerate(splits):
        print(f"\n{'=' * 60}")
        print(f"Fold {fold_idx + 1}/5")
        print(f"{'=' * 60}")

        train_val_df = filtered_df.iloc[train_val_idx].copy()
        test_df = filtered_df.iloc[test_idx].copy()

        if args.split_method == 'random':
            from sklearn.model_selection import train_test_split
            train_idx_inner, val_idx_inner = train_test_split(
                range(len(train_val_df)), test_size=0.125, random_state=args.seed
            )
            train_df = train_val_df.iloc[train_idx_inner].copy()
            val_df = train_val_df.iloc[val_idx_inner].copy()
        elif args.split_method == 'cell':
            unique_cells = train_val_df[cell_col].unique()
            np.random.seed(args.seed)
            val_cells = np.random.choice(unique_cells, size=int(len(unique_cells) * 0.125), replace=False)
            val_mask = train_val_df[cell_col].isin(val_cells)
            train_df = train_val_df[~val_mask].copy()
            val_df = train_val_df[val_mask].copy()
        else:
            unique_drugs = train_val_df[drug_col].unique()
            np.random.seed(args.seed)
            val_drugs = np.random.choice(unique_drugs, size=int(len(unique_drugs) * 0.125), replace=False)
            val_mask = train_val_df[drug_col].isin(val_drugs)
            train_df = train_val_df[~val_mask].copy()
            val_df = train_val_df[val_mask].copy()

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
            save_dir, fold_idx, name
        )

        if best_val_path and os.path.exists(best_val_path):
            checkpoint = torch.load(best_val_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            _, test_metrics = trainer.validate(test_loader)
            print(f"\nTest metrics (best val model): R²={test_metrics['r2']:.4f}, RMSE={test_metrics['rmse']:.4f}, Pearson={test_metrics['pearson']:.4f}")
            test_metrics['model_type'] = 'best_val'
            test_metrics['fold'] = fold_idx + 1
            all_fold_metrics.append(test_metrics)

        if best_balance_path and os.path.exists(best_balance_path):
            checkpoint = torch.load(best_balance_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            _, test_metrics = trainer.validate(test_loader)
            print(f"Test metrics (best balance model): R²={test_metrics['r2']:.4f}, RMSE={test_metrics['rmse']:.4f}, Pearson={test_metrics['pearson']:.4f}")
            test_metrics['model_type'] = 'best_balance'
            test_metrics['fold'] = fold_idx + 1
            all_fold_metrics.append(test_metrics)

    if all_fold_metrics:
        metrics_df = pd.DataFrame(all_fold_metrics)
        result_dir = config['output']['result_save_dir']
        os.makedirs(result_dir, exist_ok=True)
        metrics_path = os.path.join(result_dir, f"{args.model_type}_{args.split_method}_cv_results.csv")
        metrics_df.to_csv(metrics_path, index=False)
        print(f"\nCross-validation results saved to {metrics_path}")

        for mt in ['best_val', 'best_balance']:
            subset = metrics_df[metrics_df['model_type'] == mt]
            if len(subset) > 0:
                print(f"\n{mt} model summary:")
                print(f"  Mean R²: {subset['r2'].mean():.4f} ± {subset['r2'].std():.4f}")
                print(f"  Mean RMSE: {subset['rmse'].mean():.4f} ± {subset['rmse'].std():.4f}")
                print(f"  Mean Pearson: {subset['pearson'].mean():.4f} ± {subset['pearson'].std():.4f}")


def main():
    parser = argparse.ArgumentParser(description='5-fold cross-validation training')
    parser.add_argument('--dataset', type=str, default='GDSC', choices=['GDSC'])
    parser.add_argument('--model_type', type=str, default='attention', choices=['attention', 'baseline'])
    parser.add_argument('--split_method', type=str, default='drug', choices=['random', 'cell', 'drug'])
    parser.add_argument('--gene_version', type=str, default='V1', choices=['V1', 'V2'])
    parser.add_argument('--layer', type=int, default=3, choices=[1, 2, 3])
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    run_cross_validation(args)


if __name__ == '__main__':
    main()
