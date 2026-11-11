# bca_hfp/analysis/predict_all.py
"""Evaluate model performance on test set and rank predictions by error."""

import sys
import os
import argparse
import numpy as np
import torch
import pandas as pd
from functools import partial
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from bca_hfp.config.base_config import get_base_config, PROJECT_ROOT, root_dir
from bca_hfp.data.loader import load_drug_atom_features, load_gene_embeddings, load_response, load_morgan_fingerprints
from bca_hfp.data.dataset import DrugGeneDataset, collate_fn
from bca_hfp.models import create_model
from torch.utils.data import DataLoader
from bca_hfp.utils.reproducibility import set_seed

# Reuse data splitting function from training script
from bca_hfp.training.train_holdout import split_data_by_drug


def main():
    parser = argparse.ArgumentParser(description='Evaluate model and find most accurately predicted samples')
    parser.add_argument('--model_path', type=str, required=True, help='Model weights path')
    parser.add_argument('--dataset', type=str, default='GDSC', choices=['GDSC'])
    parser.add_argument('--gene_version', type=str, default='V1', choices=['V1', 'V2'])
    parser.add_argument('--layer', type=int, default=3, choices=[1,2,3], help='Number of MLP layers')
    parser.add_argument('--reg_norm', type=str, default='layernorm',
                        choices=['batchnorm', 'layernorm'],
                        help='Regression head norm layer (B6): must match the trained checkpoint')
    parser.add_argument('--fp_bits', type=int, default=2048, help='Morgan fingerprint bits')
    parser.add_argument('--fp_radius', type=int, default=2, help='Morgan fingerprint radius')
    parser.add_argument('--top_k', type=int, default=10, help='Show top k most accurately predicted samples')
    args = parser.parse_args()

    set_seed(42)
    config = get_base_config(args.dataset)

    # Gene feature path
    base_data_path = root_dir("Data")
    if args.gene_version == 'V1':
        config['data']['gene_feat_dir'] = f"{base_data_path}/GDSC_EXP_GF_V1"
        config['data']['gene_input_dim'] = 256
    else:
        config['data']['gene_feat_dir'] = f"{base_data_path}/GDSC_EXP_GF_V2"
        config['data']['gene_input_dim'] = 768

    # Model hyperparameters (consistent with training)
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
    config['model']['reg_norm'] = args.reg_norm  # B6: must match the training setting of the loaded checkpoint

    # Morgan fingerprints
    print("Generating drug global features (Morgan Fingerprints)...")
    drug_global_dict = load_morgan_fingerprints(
        config['data']['drug_feat_path'],
        fp_bits=args.fp_bits,
        fp_radius=args.fp_radius
    )
    config['data']['drug_global_dim'] = args.fp_bits

    # Load data
    print("Loading drug atom features...")
    drug_atom_dict = load_drug_atom_features(config['data']['drug_feat_path'])
    print("Loading gene embeddings...")
    gene_emb_dict, _ = load_gene_embeddings(config['data']['gene_feat_dir'], config['data']['cell_id_col'])
    print("Loading response data...")
    response_df = load_response(config['data']['response_path'],
                                config['data']['cell_id_col'],
                                config['data']['drug_id_col'])

    # Filter valid samples
    drug_col = config['data']['drug_id_col']
    cell_col = config['data']['cell_id_col']
    valid_mask = response_df[drug_col].isin(drug_atom_dict) & response_df[cell_col].isin(gene_emb_dict)
    filtered_df = response_df[valid_mask].copy()
    print(f"Total valid samples: {len(filtered_df)}")

    # Build drug name mapping (DRUG_ID -> DRUG_NAME)
    drug_name_map = filtered_df.drop_duplicates(drug_col)[[drug_col, 'DRUG_NAME']].set_index(drug_col)['DRUG_NAME'].to_dict()
    # Build cell line standard name mapping (GDSC_ID -> CELL_LINES_STD)
    cell_std_map = filtered_df.drop_duplicates(cell_col)[[cell_col, 'CELL_LINES_STD']].set_index(cell_col)['CELL_LINES_STD'].to_dict()

    # Split dataset by drug (identical to training)
    train_df, val_df, test_df = split_data_by_drug(
        filtered_df, drug_col,
        train_ratio=0.8, val_ratio=0.1, test_ratio=0.1,
        random_state=config['data']['random_state']
    )
    print(f"Test_test set samples: {len(test_df)}")
    print("Drugs in test set:", test_df[drug_col].unique())

    # Create test set DataLoader
    test_dataset = DrugGeneDataset(
        test_df[drug_col].tolist(),
        test_df[cell_col].tolist(),
        test_df[config['data']['label_col']].values,
        drug_atom_dict,
        gene_emb_dict,
        drug_global_dict
    )

    collate = partial(collate_fn, atom_mask_prob=0.0)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=collate)

    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model('attention', config).to(device)
    print(f"Loading model: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Prediction
    all_preds = []
    all_trues = []
    samples_info = []  # Store metadata for each sample

    drug_ids_list = test_df[drug_col].tolist()
    cell_ids_list = test_df[cell_col].tolist()
    true_labels = test_df[config['data']['label_col']].values

    with torch.no_grad():
        for batch_idx, (atom, atom_mask, gene, gene_mask, labels, drug_global) in enumerate(test_loader):
            atom = atom.to(device); atom_mask = atom_mask.to(device)
            gene = gene.to(device); gene_mask = gene_mask.to(device)
            drug_global = drug_global.to(device)
            preds = model(gene, atom, atom_mask, gene_mask, drug_global=drug_global).cpu().numpy()
            all_preds.extend(preds)
            all_trues.extend(labels.numpy())
            start_idx = batch_idx * test_loader.batch_size
            end_idx = start_idx + len(preds)
            for i in range(start_idx, end_idx):
                drug_id = drug_ids_list[i]
                cell_id = cell_ids_list[i]
                samples_info.append({
                    'drug_id': drug_id,
                    'drug_name': drug_name_map.get(drug_id, 'Unknown'),
                    'cell_line_id': cell_id,
                    'cell_line_name': cell_std_map.get(cell_id, 'Unknown'),
                    'true_ln_ic50': true_labels[i],
                    'pred_ln_ic50': preds[i - start_idx]
                })

    all_preds = np.array(all_preds)
    all_trues = np.array(all_trues)

    # Compute overall metrics
    r2 = r2_score(all_trues, all_preds)
    pearson, _ = pearsonr(all_trues, all_preds)
    mae = mean_absolute_error(all_trues, all_preds)
    rmse = np.sqrt(mean_squared_error(all_trues, all_preds))

    print("\n========== Test_test Set Overall Performance ==========")
    print(f"R2 = {r2:.4f}")
    print(f"Pearson = {pearson:.4f}")
    print(f"MAE = {mae:.4f}")
    print(f"RMSE = {rmse:.4f}")

    # Compute absolute error
    for sample in samples_info:
        sample['abs_error'] = abs(sample['true_ln_ic50'] - sample['pred_ln_ic50'])

    samples_sorted = sorted(samples_info, key=lambda x: x['abs_error'])
    top_k = min(args.top_k, len(samples_sorted))

    print(f"\n========== Top {top_k} Most Accurate Predictions ==========")
    for i, s in enumerate(samples_sorted[:top_k]):
        print(f"{i+1}. Drug ID: {s['drug_id']} ({s['drug_name']}), Cell Line: {s['cell_line_id']} ({s['cell_line_name']}), "
              f"True LN_IC50: {s['true_ln_ic50']:.4f}, Predicted: {s['pred_ln_ic50']:.4f}, "
              f"Abs Error: {s['abs_error']:.4f}")

    worst_k = 5
    print(f"\n========== Bottom {worst_k} Least Accurate Predictions ==========")
    for i, s in enumerate(samples_sorted[-worst_k:][::-1]):
        print(f"{i+1}. Drug ID: {s['drug_id']} ({s['drug_name']}), Cell Line: {s['cell_line_id']} ({s['cell_line_name']}), "
              f"True LN_IC50: {s['true_ln_ic50']:.4f}, Predicted: {s['pred_ln_ic50']:.4f}, "
              f"Abs Error: {s['abs_error']:.4f}")

    # Save CSV
    df_results = pd.DataFrame(samples_info)[['drug_id', 'drug_name', 'cell_line_id', 'cell_line_name', 'true_ln_ic50', 'pred_ln_ic50', 'abs_error']]
    output_csv = os.path.join(os.path.dirname(args.model_path), '..', 'results', 'test_predictions.csv')
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_results.to_csv(output_csv, index=False)
    print(f"\nAll test set predictions saved to: {output_csv}")


if __name__ == '__main__':
    main()
