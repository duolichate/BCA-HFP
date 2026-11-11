# bca_hfp/training/extract_attention.py
"""Extract gate weights and cross-attention weights from trained models."""

import sys
import os
import argparse
import numpy as np
import torch
import pandas as pd
from functools import partial
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from bca_hfp.config.base_config import get_base_config, PROJECT_ROOT
from bca_hfp.data.loader import load_drug_atom_features, load_gene_embeddings, load_response, load_morgan_fingerprints
from bca_hfp.data.dataset import DrugGeneDataset, collate_fn
from bca_hfp.models import create_model
from bca_hfp.utils.reproducibility import set_seed

from bca_hfp.training.train_holdout import split_data_by_drug, split_data_by_cell, split_data_random


def main():
    parser = argparse.ArgumentParser(description='Extract gate weights and cross-attention weights')
    parser.add_argument('--model_path', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--dataset', type=str, default='GDSC', choices=['GDSC'])
    parser.add_argument('--split_method', type=str, default='drug', choices=['random', 'cell', 'drug'])
    parser.add_argument('--split_set', type=str, default='test',
                        choices=['train', 'val', 'test', 'all'],
                        help='Which split to extract: train/val/test/all')
    parser.add_argument('--gene_version', type=str, default='V1', choices=['V1', 'V2'])
    parser.add_argument('--layer', type=int, default=3, choices=[1, 2, 3])
    parser.add_argument('--output_dir', type=str, default=os.path.join(PROJECT_ROOT, 'Result', 'Interpretability'))

    parser.add_argument('--mode', type=str, default='gate_only',
                        choices=['gate_only', 'full'],
                        help='Extraction mode: gate_only (gate weights CSV), full (gate+attention CSV+npy)')

    parser.add_argument('--drug_id', type=str, default=None, help='Filter by drug ID')
    parser.add_argument('--cell_std_name', type=str, default=None, help='Filter by cell line standard name')
    parser.add_argument('--max_samples', type=int, default=None, help='Max samples to extract')
    args = parser.parse_args()

    set_seed(42)
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

    if args.split_method == 'random':
        train_df, val_df, test_df = split_data_random(
            filtered_df, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1,
            random_state=config['data']['random_state']
        )
    elif args.split_method == 'cell':
        train_df, val_df, test_df = split_data_by_cell(
            filtered_df, cell_col, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1,
            random_state=config['data']['random_state']
        )
    else:
        train_df, val_df, test_df = split_data_by_drug(
            filtered_df, drug_col, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1,
            random_state=config['data']['random_state']
        )

    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    if args.split_set == 'train':
        sample_df = train_df.copy()
    elif args.split_set == 'val':
        sample_df = val_df.copy()
    elif args.split_set == 'test':
        sample_df = test_df.copy()
    else:
        sample_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    if args.drug_id is not None:
        sample_df = sample_df[sample_df[drug_col].astype(str) == str(args.drug_id)]
        print(f"Filtered by drug_id={args.drug_id}: {len(sample_df)} samples")
    if args.cell_std_name is not None:
        if 'CELL_LINES_STD' in sample_df.columns:
            sample_df = sample_df[sample_df['CELL_LINES_STD'] == args.cell_std_name]
            print(f"Filtered by cell_std_name={args.cell_std_name}: {len(sample_df)} samples")
        else:
            print(f"Warning: CELL_LINES_STD column not found, skipping this filter")
    if args.max_samples is not None and len(sample_df) > args.max_samples:
        sample_df = sample_df.sample(n=args.max_samples, random_state=42)
        print(f"Limited to {len(sample_df)} samples")

    if len(sample_df) == 0:
        print("No matching samples found")
        return

    print(f"Final sample count: {len(sample_df)}")

    dataset = DrugGeneDataset(
        sample_df[drug_col].tolist(),
        sample_df[cell_col].tolist(),
        sample_df[label_col].values,
        drug_atom_dict,
        gene_emb_dict,
        drug_global_dict
    )

    collate = partial(collate_fn, atom_mask_prob=0.0)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate,
        num_workers=4
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model('attention', config).to(device)

    print(f"\nLoading model: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    print("Model loaded successfully")

    dir_name = f"weights_{args.split_set}_{args.split_method}_{args.dataset}"
    if args.drug_id:
        dir_name = f"weights_drug{args.drug_id}_{args.split_set}"
    if args.cell_std_name:
        dir_name += f"_{args.cell_std_name}"
    if args.mode == 'gate_only':
        dir_name += "_gate_only"
    output_dir = os.path.join(args.output_dir, dir_name)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nExtraction mode: {args.mode}")
    print(f"Output directory: {output_dir}")

    all_gate_weights = []
    all_labels = []
    all_drug_ids = []
    all_drug_names = []
    all_cell_ids = []
    all_cell_std_names = []
    all_smiles = []
    all_real_atoms = []

    print(f"\nStarting extraction (mode: {args.mode})")
    print("=" * 60)

    if args.mode == 'gate_only' and len(sample_df) > 100:
        loader_fast = torch.utils.data.DataLoader(
            dataset,
            batch_size=64,
            shuffle=False,
            collate_fn=collate,
            num_workers=8
        )
        print("Using batch mode for faster gate weight extraction...")

        for batch in tqdm(loader_fast, desc="Extracting gate weights"):
            atom, atom_mask, gene, gene_mask, labels, drug_global = batch
            atom = atom.to(device)
            atom_mask = atom_mask.to(device)
            gene = gene.to(device)
            gene_mask = gene_mask.to(device)
            drug_global = drug_global.to(device)

            with torch.no_grad():
                _ = model(gene, atom, atom_mask, gene_mask, drug_global=drug_global, return_attn=False)
                gate_weights_batch = model.fusion_layers[0].last_gate_weights.cpu().numpy()

                for i in range(len(gate_weights_batch)):
                    all_gate_weights.append(gate_weights_batch[i])
                    idx = len(all_gate_weights) - 1
                    all_labels.append(sample_df.iloc[idx][label_col])
                    all_drug_ids.append(sample_df.iloc[idx][drug_col])
                    all_drug_names.append(sample_df.iloc[idx].get('DRUG_NAME', 'Unknown'))
                    all_cell_ids.append(sample_df.iloc[idx][cell_col])
                    all_cell_std_names.append(sample_df.iloc[idx].get('CELL_LINES_STD', ''))
                    all_smiles.append(sample_df.iloc[idx].get('DRUG_SMILE', ''))
                    all_real_atoms.append(0)

    else:
        for idx, batch in enumerate(tqdm(loader, desc="Extracting")):
            atom, atom_mask, gene, gene_mask, labels, drug_global = batch
            atom = atom.to(device)
            atom_mask = atom_mask.to(device)
            gene = gene.to(device)
            gene_mask = gene_mask.to(device)
            drug_global = drug_global.to(device)

            num_real_atoms = atom_mask.sum().item()
            num_genes = gene_mask.sum().item()

            with torch.no_grad():
                if args.mode == 'full':
                    _, drug2gene, gene2drug = model(
                        gene, atom, atom_mask, gene_mask,
                        drug_global=drug_global,
                        return_attn=True
                    )
                    drug2gene_np = drug2gene[0].cpu().numpy()
                    drug2gene_real = drug2gene_np[:, :num_real_atoms, :num_genes]
                    gene2drug_np = gene2drug[0].cpu().numpy()
                    gene2drug_real = gene2drug_np[:, :num_genes, :num_real_atoms]
                else:
                    _ = model(gene, atom, atom_mask, gene_mask, drug_global=drug_global, return_attn=False)

                gate_weights = model.fusion_layers[0].last_gate_weights.cpu().numpy().squeeze()

            all_gate_weights.append(gate_weights)
            all_labels.append(sample_df.iloc[idx][label_col])
            all_drug_ids.append(sample_df.iloc[idx][drug_col])
            all_drug_names.append(sample_df.iloc[idx].get('DRUG_NAME', 'Unknown'))
            all_cell_ids.append(sample_df.iloc[idx][cell_col])
            all_cell_std_names.append(sample_df.iloc[idx].get('CELL_LINES_STD', ''))
            all_real_atoms.append(num_real_atoms)
            all_smiles.append(sample_df.iloc[idx].get('DRUG_SMILE', ''))

            if args.mode == 'full':
                sample_name = f"sample_{idx:06d}"
                np.save(os.path.join(output_dir, f'{sample_name}_gate_weights.npy'), gate_weights)
                np.save(os.path.join(output_dir, f'{sample_name}_drug2gene_8heads.npy'), drug2gene_real)
                np.save(os.path.join(output_dir, f'{sample_name}_gene2drug_8heads.npy'), gene2drug_real)

    gate_df = pd.DataFrame(all_gate_weights, columns=['gene_pooled', 'global_gene', 'drug_global'])
    gate_df['label'] = all_labels
    gate_df['drug_id'] = all_drug_ids
    gate_df['drug_name'] = all_drug_names
    gate_df['cell_id'] = all_cell_ids
    gate_df['cell_std_name'] = all_cell_std_names
    gate_df['num_real_atoms'] = all_real_atoms
    gate_df['smiles'] = all_smiles
    gate_df.to_csv(os.path.join(output_dir, 'all_gate_weights.csv'), index=False)

    meta_df = pd.DataFrame({
        'sample_idx': range(len(all_labels)),
        'drug_id': all_drug_ids,
        'drug_name': all_drug_names,
        'cell_id': all_cell_ids,
        'cell_std_name': all_cell_std_names,
        'label': all_labels,
        'num_real_atoms': all_real_atoms,
        'smiles': all_smiles
    })
    meta_df.to_csv(os.path.join(output_dir, 'sample_metadata.csv'), index=False)

    print(f"\nExtraction complete!")
    print(f"  Output directory: {output_dir}")
    print(f"  Total samples extracted: {len(all_gate_weights)}")
    print(f"  Generated files: all_gate_weights.csv, sample_metadata.csv")
    if args.mode == 'full':
        print(f"  Plus per-sample .npy files")
    print("=" * 60)


if __name__ == '__main__':
    main()
