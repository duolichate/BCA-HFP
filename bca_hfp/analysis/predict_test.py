# bca_hfp/analysis/predict_test.py
"""Predict drug response for specified drug-cell pairs."""

import sys
import os
import argparse
import numpy as np
import torch
import pandas as pd
from functools import partial

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from bca_hfp.config.base_config import get_base_config, PROJECT_ROOT, root_dir
from bca_hfp.data.loader import load_drug_atom_features, load_gene_embeddings, load_response, load_morgan_fingerprints
from bca_hfp.data.dataset import DrugGeneDataset, collate_fn
from bca_hfp.models import create_model
from torch.utils.data import DataLoader
from bca_hfp.utils.reproducibility import set_seed
from tqdm import tqdm

# Reuse data splitting functions from training script
from bca_hfp.training.train_holdout import split_data_by_drug, split_data_by_cell, split_data_random


def main():
    parser = argparse.ArgumentParser(description='Predict response for specified drug and cell line, or batch prediction')

    # Basic parameters
    parser.add_argument('--model_path', type=str, required=True, help='Model weights path (.pth)')
    parser.add_argument('--dataset', type=str, default='GDSC', choices=['GDSC'])
    parser.add_argument('--gene_version', type=str, default='V1', choices=['V1', 'V2'])
    parser.add_argument('--layer', type=int, default=3, help='Number of MLP layers (1,2,3)')
    parser.add_argument('--fp_bits', type=int, default=2048, help='Morgan fingerprint bits')

    parser.add_argument('--mode', type=str, default='single',
                        choices=['single', 'by_drug', 'by_cell'],
                        help='Prediction mode: single (single sample), by_drug (by drug), by_cell (by cell line)')

    parser.add_argument('--drug_id', type=str, default=None, help='Drug ID (e.g. 1175)')
    parser.add_argument('--cell_line', type=str, default=None, help='Cell line standard name or GDSC_ID (e.g. NCIH1155 or 908467)')

    parser.add_argument('--target_id', type=str, default=None,
                        help='by_drug mode: drug ID; by_cell mode: cell line standard name or GDSC_ID')
    parser.add_argument('--top_k', type=int, default=20, help='Show top K most accurate samples')
    parser.add_argument('--show_all', action='store_true', default=False,
                        help='Whether to show all samples (default shows only top_k)')
    parser.add_argument('--save_csv', type=str, default=None,
                        help='Save prediction results to CSV file path')

    args = parser.parse_args()

    set_seed(42)
    config = get_base_config(args.dataset)

    base_data_path = root_dir("Data")
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

    print("Generating drug global features (Morgan Fingerprints)...")
    drug_global_dict = load_morgan_fingerprints(
        config['data']['drug_feat_path'],
        fp_bits=args.fp_bits,
        fp_radius=2
    )
    config['data']['drug_global_dim'] = args.fp_bits

    print("Loading drug atom features...")
    drug_atom_dict = load_drug_atom_features(config['data']['drug_feat_path'])
    print("Loading gene embeddings...")
    gene_emb_dict, _ = load_gene_embeddings(config['data']['gene_feat_dir'], config['data']['cell_id_col'])
    print("Loading response data...")
    response_df = load_response(config['data']['response_path'],
                                config['data']['cell_id_col'],
                                config['data']['drug_id_col'])

    drug_col = config['data']['drug_id_col']
    cell_col = config['data']['cell_id_col']
    valid_mask = response_df[drug_col].isin(drug_atom_dict) & response_df[cell_col].isin(gene_emb_dict)
    filtered_df = response_df[valid_mask].copy()
    print(f"Total valid samples: {len(filtered_df)}")

    filtered_df = filtered_df.sort_values(drug_col).reset_index(drop=True)
    train_df, val_df, test_df = split_data_by_drug(
        filtered_df, drug_col,
        train_ratio=0.8, val_ratio=0.1, test_ratio=0.1,
        random_state=config['data']['random_state']
    )
    print(f"Train set: {len(train_df)}, Validation set: {len(val_df)}, Test set: {len(test_df)}")

    if args.mode == 'single':
        if args.drug_id is None or args.cell_line is None:
            raise ValueError("single mode requires --drug_id and --cell_line")

        # Filter target
        target_df = test_df[test_df[drug_col].astype(str) == str(args.drug_id)].copy()
        if target_df.empty:
            print(f"Drug {args.drug_id} not found in test set")
            return

        # Match cell line
        cell_std_col = 'CELL_LINES_STD'
        sample_row = None
        if cell_std_col in target_df.columns:
            std_mask = target_df[cell_std_col] == args.cell_line
            if std_mask.any():
                sample_row = target_df[std_mask].iloc[0]
            else:
                try:
                    cell_id = int(args.cell_line)
                    id_mask = target_df[cell_col] == cell_id
                    if id_mask.any():
                        sample_row = target_df[id_mask].iloc[0]
                    else:
                        raise ValueError(f"Cell line {args.cell_line} not found")
                except ValueError:
                    raise ValueError(f"Cell line {args.cell_line} not found")
        else:
            try:
                cell_id = int(args.cell_line)
                id_mask = target_df[cell_col] == cell_id
                if id_mask.any():
                    sample_row = target_df[id_mask].iloc[0]
                else:
                    raise ValueError(f"Cell line {args.cell_line} not found")
            except ValueError:
                raise ValueError(f"Invalid cell line identifier: {args.cell_line}")

        cell_id = sample_row[cell_col]
        true_label = sample_row[config['data']['label_col']]
        drug_name = sample_row.get('DRUG_NAME', 'Unknown')
        cell_std_name = sample_row.get(cell_std_col, 'Unknown')

        print(f"\nDrug: {args.drug_id} ({drug_name})")
        print(f"Cell line: {cell_id} ({cell_std_name})")
        print(f"True LN_IC50: {true_label:.4f} (IC50 = {np.exp(true_label):.2f} uM)")

        # Single sample prediction
        dataset = DrugGeneDataset(
            [str(args.drug_id)],
            [cell_id],
            [true_label],
            drug_atom_dict,
            gene_emb_dict,
            drug_global_dict
        )

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = create_model('attention', config).to(device)
        print(f"Loading model: {args.model_path}")
        checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        collate = partial(collate_fn, atom_mask_prob=0.0)
        loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate)
        with torch.no_grad():
            for atom, atom_mask, gene, gene_mask, labels, drug_global in loader:
                atom = atom.to(device)
                atom_mask = atom_mask.to(device)
                gene = gene.to(device)
                gene_mask = gene_mask.to(device)
                drug_global = drug_global.to(device)
                pred = model(gene, atom, atom_mask, gene_mask, drug_global=drug_global).item()
                print(f"\nPredicted LN_IC50: {pred:.4f} (IC50 = {np.exp(pred):.2f} uM)")
                print(f"Absolute error: {abs(pred - true_label):.4f}")

    elif args.mode == 'by_drug':
        if args.target_id is None:
            raise ValueError("by_drug mode requires --target_id (drug ID)")

        target_df = test_df[test_df[drug_col].astype(str) == str(args.target_id)].copy()
        if target_df.empty:
            print(f"Drug {args.target_id} not found in test set")
            return

        drug_name = target_df.iloc[0].get('DRUG_NAME', f"Drug {args.target_id}")
        print(f"\nBatch prediction by drug: {args.target_id} ({drug_name})")
        print(f"  Total {len(target_df)} cell line samples")

        # Create dataset
        dataset = DrugGeneDataset(
            target_df[drug_col].tolist(),
            target_df[cell_col].tolist(),
            target_df[config['data']['label_col']].values,
            drug_atom_dict,
            gene_emb_dict,
            drug_global_dict
        )

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = create_model('attention', config).to(device)
        print(f"Loading model: {args.model_path}")
        checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        collate = partial(collate_fn, atom_mask_prob=0.0)
        loader = DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=collate, num_workers=4)

        # Batch prediction
        all_preds = []
        all_trues = []
        all_cell_ids = []
        all_cell_std_names = []

        with torch.no_grad():
            for atom, atom_mask, gene, gene_mask, labels, drug_global in tqdm(loader, desc="Predicting"):
                atom = atom.to(device)
                atom_mask = atom_mask.to(device)
                gene = gene.to(device)
                gene_mask = gene_mask.to(device)
                drug_global = drug_global.to(device)
                preds = model(gene, atom, atom_mask, gene_mask, drug_global=drug_global)
                all_preds.extend(preds.cpu().numpy())
                all_trues.extend(labels.numpy())
                all_cell_ids.extend(target_df.iloc[len(all_trues) - len(preds):len(all_trues)][cell_col].tolist())

        # Build result DataFrame
        results = pd.DataFrame({
            'drug_id': [args.target_id] * len(all_trues),
            'drug_name': [drug_name] * len(all_trues),
            'cell_id': all_cell_ids,
            'cell_std_name': target_df['CELL_LINES_STD'].tolist() if 'CELL_LINES_STD' in target_df.columns else [
                                                                                                                    'Unknown'] * len(
                all_trues),
            'true_label': all_trues,
            'pred_label': all_preds,
            'abs_error': np.abs(np.array(all_trues) - np.array(all_preds))
        })

        # Sort by absolute error
        results_sorted = results.sort_values('abs_error').reset_index(drop=True)

        # Display
        top_k = min(args.top_k, len(results_sorted))
        print(f"\n========== Top {top_k} Most Accurate Predictions (Drug {args.target_id}) ==========")
        for i, row in results_sorted.head(top_k).iterrows():
            print(f"{i + 1}. Cell line: {row['cell_std_name']} (ID: {row['cell_id']}) | "
                  f"True: {row['true_label']:.4f} | Predicted: {row['pred_label']:.4f} | "
                  f"Error: {row['abs_error']:.4f}")

        if args.show_all:
            print(f"\n========== All {len(results_sorted)} Samples ==========")
            for i, row in results_sorted.iterrows():
                print(f"{i + 1}. Cell line: {row['cell_std_name']} | "
                      f"True: {row['true_label']:.4f} | Predicted: {row['pred_label']:.4f} | "
                      f"Error: {row['abs_error']:.4f}")

        # Save CSV
        if args.save_csv:
            results_sorted.to_csv(args.save_csv, index=False)
            print(f"\nResults saved to: {args.save_csv}")

    else:  # by_cell
        if args.target_id is None:
            raise ValueError("by_cell mode requires --target_id (cell line standard name or GDSC_ID)")

        # Find cell line
        cell_std_col = 'CELL_LINES_STD'
        if cell_std_col in test_df.columns:
            mask = test_df[cell_std_col] == args.target_id
            if mask.any():
                target_df = test_df[mask].copy()
                cell_std_name = args.target_id
            else:
                try:
                    cell_id = int(args.target_id)
                    mask = test_df[cell_col] == cell_id
                    if mask.any():
                        target_df = test_df[mask].copy()
                        cell_std_name = target_df.iloc[0].get(cell_std_col, f"Cell {cell_id}")
                    else:
                        raise ValueError(f"Cell line {args.target_id} not found")
                except ValueError:
                    raise ValueError(f"Cell line {args.target_id} not found")
        else:
            try:
                cell_id = int(args.target_id)
                mask = test_df[cell_col] == cell_id
                if mask.any():
                    target_df = test_df[mask].copy()
                    cell_std_name = f"Cell {cell_id}"
                else:
                    raise ValueError(f"Cell line {args.target_id} not found")
            except ValueError:
                raise ValueError(f"Invalid cell line identifier: {args.target_id}")

        print(f"\nBatch prediction by cell line: {cell_std_name}")
        print(f"  Total {len(target_df)} drug samples")

        # Create dataset
        dataset = DrugGeneDataset(
            target_df[drug_col].tolist(),
            target_df[cell_col].tolist(),
            target_df[config['data']['label_col']].values,
            drug_atom_dict,
            gene_emb_dict,
            drug_global_dict
        )

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = create_model('attention', config).to(device)
        print(f"Loading model: {args.model_path}")
        checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        collate = partial(collate_fn, atom_mask_prob=0.0)
        loader = DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=collate, num_workers=4)

        # Batch prediction
        all_preds = []
        all_trues = []
        all_drug_ids = []
        all_drug_names = []

        with torch.no_grad():
            for atom, atom_mask, gene, gene_mask, labels, drug_global in tqdm(loader, desc="Predicting"):
                atom = atom.to(device)
                atom_mask = atom_mask.to(device)
                gene = gene.to(device)
                gene_mask = gene_mask.to(device)
                drug_global = drug_global.to(device)
                preds = model(gene, atom, atom_mask, gene_mask, drug_global=drug_global)
                all_preds.extend(preds.cpu().numpy())
                all_trues.extend(labels.numpy())

        # Build result DataFrame
        results = pd.DataFrame({
            'drug_id': target_df[drug_col].tolist(),
            'drug_name': target_df['DRUG_NAME'].tolist() if 'DRUG_NAME' in target_df.columns else ['Unknown'] * len(
                all_trues),
            'cell_id': [target_df.iloc[0][cell_col]] * len(all_trues),
            'cell_std_name': [cell_std_name] * len(all_trues),
            'true_label': all_trues,
            'pred_label': all_preds,
            'abs_error': np.abs(np.array(all_trues) - np.array(all_preds))
        })

        # Sort by absolute error
        results_sorted = results.sort_values('abs_error').reset_index(drop=True)

        # Display
        top_k = min(args.top_k, len(results_sorted))
        print(f"\n========== Top {top_k} Most Accurate Predictions (Cell Line {cell_std_name}) ==========")
        for i, row in results_sorted.head(top_k).iterrows():
            print(f"{i + 1}. Drug: {row['drug_name']} (ID: {row['drug_id']}) | "
                  f"True: {row['true_label']:.4f} | Predicted: {row['pred_label']:.4f} | "
                  f"Error: {row['abs_error']:.4f}")

        if args.show_all:
            print(f"\n========== All {len(results_sorted)} Samples ==========")
            for i, row in results_sorted.iterrows():
                print(f"{i + 1}. Drug: {row['drug_name']} | "
                      f"True: {row['true_label']:.4f} | Predicted: {row['pred_label']:.4f} | "
                      f"Error: {row['abs_error']:.4f}")

        # Save CSV
        if args.save_csv:
            results_sorted.to_csv(args.save_csv, index=False)
            print(f"\nResults saved to: {args.save_csv}")


if __name__ == '__main__':
    main()
