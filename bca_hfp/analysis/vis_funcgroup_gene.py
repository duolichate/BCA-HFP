# bca_hfp/analysis/vis_funcgroup_gene.py
"""Functional group-gene attention heatmap."""

import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import os
import argparse
import pickle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from bca_hfp.config.base_config import get_base_config, PROJECT_ROOT, root_dir
from bca_hfp.data.loader import load_drug_atom_features, load_gene_embeddings, load_response, load_morgan_fingerprints
from bca_hfp.data.dataset import DrugGeneDataset, collate_fn
from bca_hfp.models.main_model import Regressor
from torch.utils.data import DataLoader
from bca_hfp.utils.reproducibility import set_seed
from bca_hfp.analysis.funcgroup_pathway_mapping import get_fg_mapping, merge_fg_mapping, PATHWAY_SIMPLE_NAMES


# Local ID mapping functions
def load_ensembl_to_symbol_map():
    map_file = os.path.join(root_dir("Gene_Feature"), "geneformer", "gene_dictionaries_30m", "gene_name_id_dict_gc30M.pkl")
    with open(map_file, 'rb') as f:
        symbol_to_ensembl = pickle.load(f)
    ensembl_to_symbol = {}
    for symbol, ens_id in symbol_to_ensembl.items():
        ens_no_version = ens_id.split('.')[0]
        ensembl_to_symbol[ens_no_version] = symbol.upper()
    return ensembl_to_symbol


# Command line arguments
parser = argparse.ArgumentParser(description='Plot functional group-gene attention heatmap')
parser.add_argument('--drug_id', type=str, required=True, help='Drug ID (e.g. 1 or 1036)')
parser.add_argument('--cell_line', type=str, required=True, help='Cell line name or ID (e.g. NCIH1155, 908467, 687794)')
parser.add_argument('--model_path', type=str, required=True, help='Model weights path (.pth)')
parser.add_argument('--output_dir', type=str, default=root_dir("Picture"), help='Output directory')
parser.add_argument('--dataset', type=str, default='GDSC', choices=['GDSC'], help='Dataset name')
parser.add_argument('--gene_version', type=str, default='V1', choices=['V1', 'V2'], help='GeneFormer version')
parser.add_argument('--mode', type=str, default='weighted_mean',
                    choices=['weighted_mean', 'p90', 'dual'], help='Aggregation method')
parser.add_argument('--top_ratio', type=float, default=0.8, help='Ratio of high-ranking genes')
parser.add_argument('--total_genes', type=int, default=50, help='Total number of genes to display')
parser.add_argument('--heads', action='store_true', help='Plot separate heatmap for each attention head')
parser.add_argument('--layer', type=int, default=3, choices=[1, 2, 3], help='Number of MLP layers')
args = parser.parse_args()

set_seed(42)
config = get_base_config(args.dataset)

# Set gene feature directory and input dimension
base_data_path = root_dir("Data")
if args.gene_version == 'V1':
    config['data']['gene_feat_dir'] = f"{base_data_path}/GDSC_EXP_GF_V1"
    config['data']['gene_input_dim'] = 256
else:
    config['data']['gene_feat_dir'] = f"{base_data_path}/GDSC_EXP_GF_V2"
    config['data']['gene_input_dim'] = 768

# Load data
print("Loading drug atom features...")
drug_atom_dict = load_drug_atom_features(config['data']['drug_feat_path'])

print("Loading drug global features (Morgan Fingerprints)...")
fp_dim = 2048
drug_global_dict = load_morgan_fingerprints(config['data']['drug_feat_path'], fp_bits=fp_dim, fp_radius=2)
config['data']['drug_global_dim'] = fp_dim

print("Loading gene embeddings...")
gene_emb_dict, _ = load_gene_embeddings(config['data']['gene_feat_dir'], config['data']['cell_id_col'])
print(f"Gene embedding dictionary contains {len(gene_emb_dict)} cell lines")
print("Gene embedding dictionary key examples (first 5):", list(gene_emb_dict.keys())[:5])

print("Loading response data...")
response_df = load_response(config['data']['response_path'],
                            config['data']['cell_id_col'],
                            config['data']['drug_id_col'])


# Smart cell line lookup function
def find_cell_id(df, cell_input, dataset_type='GDSC'):
    """
    Return cell line ID that exists in gene_emb_dict based on user input.
    Supports: CELL_LINES_STD standard name, GDSC_ID.
    """
    id_col = config['data']['cell_id_col']
    std_name_col = 'CELL_LINES_STD'

    # 1. Try matching standard name directly
    if std_name_col in df.columns:
        mask = df[std_name_col] == cell_input
        if mask.any():
            matched = df[mask].iloc[0]
            matched_id = str(matched[id_col])
            print(f"Matched {id_col} = {matched_id} by standard name '{cell_input}'")
            return matched_id

    # 2. Match ID column directly
    mask = df[id_col].astype(str) == str(cell_input)
    if mask.any():
        matched_id = str(df[mask].iloc[0][id_col])
        print(f"Matched directly by {id_col} '{cell_input}', ID = {matched_id}")
        return matched_id

    # 3. If all methods fail, print available identifier examples
    print("\nError: Cell line identifier not found. Available examples:")
    if std_name_col in df.columns:
        print(f"Standard name example: {df[std_name_col].dropna().iloc[0] if len(df) > 0 else 'None'}")
    print(f"{id_col} example: {df[id_col].astype(str).iloc[0] if len(df) > 0 else 'None'}")
    raise ValueError(f"Cannot map '{cell_input}' to a valid cell line ID")

drug_col = config['data']['drug_id_col']
cell_col = config['data']['cell_id_col']
label_col = config['data']['label_col']


# Filter target drug
target_df = response_df[response_df[drug_col].astype(str) == str(args.drug_id)].copy()
if target_df.empty:
    raise ValueError(f"Drug {args.drug_id} not found")



# Find cell line ID
cell_id = find_cell_id(target_df, args.cell_line, args.dataset)
print(f"Final cell line ID used: {cell_id}")

# Check if cell line exists in gene embedding dictionary
if cell_id not in gene_emb_dict:
    print(f"Error: Cell line {cell_id} not found in gene embedding dictionary.")
    print("Available cell line ID examples:", list(gene_emb_dict.keys())[:10])
    raise KeyError(f"Cell line {cell_id} missing gene embedding")

# Get sample row
cell_mask = target_df[config['data']['cell_id_col']] == cell_id
if not cell_mask.any():
    raise ValueError(f"Cell line {cell_id} not found in samples for drug {args.drug_id}")
sample_row = target_df[cell_mask].iloc[0]
true_label = sample_row[config['data']['label_col']]
sample_idx = target_df.index.get_loc(sample_row.name)

# Create single-sample dataset (only containing selected cell line)
single_drug_ids = [str(sample_row[drug_col])]
single_cell_ids = [str(sample_row[cell_col])]
single_labels = [sample_row[label_col]]

dataset = DrugGeneDataset(
    single_drug_ids,
    single_cell_ids,
    single_labels,
    drug_atom_dict,
    gene_emb_dict,
    drug_global_dict
)
sample_atom, sample_gene, sample_label, sample_global = dataset[0]

# Load model
if args.layer == 1:
    hidden_dims = [64]
elif args.layer == 2:
    hidden_dims = [256, 128]
else:
    hidden_dims = [512, 256, 128]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config['model']['hidden_dims'] = hidden_dims
config['model']['feature_dim'] = 256
config['model']['num_heads'] = 8
config['model']['dropout'] = 0.3
config['model']['num_fusion_layers'] = 1
config['data']['gene_input_dim'] = 256

model = Regressor(
    feature_dim=config['model']['feature_dim'],
    num_heads=config['model']['num_heads'],
    dropout=config['model']['dropout'],
    hidden_dims=config['model']['hidden_dims'],
    gene_input_dim=config['data']['gene_input_dim'],
    drug_global_dim=config['data']['drug_global_dim'],
    num_fusion_layers=config['model']['num_fusion_layers']
).to(device)
print(f"Loading model: {args.model_path}")
checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Extract attention weights
single_loader = DataLoader([(sample_atom, sample_gene, sample_label, sample_global)], batch_size=1,
                           collate_fn=collate_fn)
with torch.no_grad():
    for atom, atom_mask, gene, gene_mask, labels, drug_global in single_loader:
        atom = atom.to(device);
        atom_mask = atom_mask.to(device)
        gene = gene.to(device);
        gene_mask = gene_mask.to(device)
        drug_global = drug_global.to(device)
        _, attn_weights = model(gene, atom, atom_mask, gene_mask, drug_global=drug_global, return_attn=True)
        if attn_weights.dim() == 4:
            has_heads = True
            num_heads = attn_weights.shape[1]
            print(f"Detected multi-head attention with {num_heads} heads")
        else:
            has_heads = False
            num_heads = 1
            print("Multi-head attention not detected, using averaged heads")
        num_atoms_original = sample_atom.shape[0]
        attn_weights = attn_weights[:, :, :num_atoms_original, :]  # (B, H, atoms, genes)

# Build global gene list and symbol mapping
print("Building gene name mapping...")
cell_to_genes = {}
for fname in os.listdir(config['data']['gene_feat_dir']):
    if not fname.endswith('.pt'):
        continue
    data = torch.load(os.path.join(config['data']['gene_feat_dir'], fname))
    meta = data['metadata']
    cid = str(meta.get(config['data']['cell_id_col']))
    if cid and cid != 'None':
        cell_to_genes[cid] = data['gene_names']
global_genes_set = set()
for genes in cell_to_genes.values():
    global_genes_set.update(genes)
global_genes = sorted(global_genes_set)
gene_to_global_idx = {g: i for i, g in enumerate(global_genes)}
num_global_genes = len(global_genes)

ensembl_to_symbol = load_ensembl_to_symbol_map()
global_symbols = [ensembl_to_symbol.get(g, g) for g in global_genes]

# Gene order for this sample
sample_cell_id = str(cell_id)
gene_names_cell = cell_to_genes.get(sample_cell_id, [])
if not gene_names_cell:
    raise ValueError(f"Gene list not found for cell line {sample_cell_id}")

# Functional group mapping
smiles = sample_row['DRUG_SMILE']
fg_mapping = get_fg_mapping(smiles)
if not fg_mapping:
    raise ValueError("No functional groups identified")
fg_mapping = merge_fg_mapping(fg_mapping)


# Align attention matrix
def align_attention(attn_np):
    # attn_np: (atoms, genes_cell)
    aligned = np.zeros((attn_np.shape[0], num_global_genes), dtype=np.float32)
    for j, g in enumerate(gene_names_cell):
        if g in gene_to_global_idx:
            aligned[:, gene_to_global_idx[g]] = attn_np[:, j]
    return aligned


# Compute gene importance
def compute_gene_importance(attn_matrix, mode):
    if mode == 'weighted_mean':
        return attn_matrix.mean(axis=0)
    elif mode == 'p90':
        return np.percentile(attn_matrix, 90, axis=0)
    elif mode == 'dual':
        return attn_matrix.mean(axis=0) + attn_matrix.max(axis=0)
    else:
        return attn_matrix.mean(axis=0)


def plot_fg_gene_heatmap(fg_weights, fg_names, global_symbols, gene_indices, mode, suffix, output_dir):
    """Plot functional group-gene heatmap"""
    # Build matrix
    fg_gene_matrix = np.vstack([fg_weights[fg][gene_indices] for fg in fg_names])
    # Sort functional groups (by total attention)
    fg_total = np.array([fg_weights[fg].sum() for fg in fg_names])
    fg_order = np.argsort(fg_total)[::-1]
    fg_names_sorted = [fg_names[i] for i in fg_order]
    fg_gene_matrix = fg_gene_matrix[fg_order]
    selected_symbols = [global_symbols[i] for i in gene_indices]
    # Color scale
    pos_vals = fg_gene_matrix[fg_gene_matrix > 0]
    if len(pos_vals) > 0:
        vmin, vmax = np.percentile(pos_vals, (0, 100))
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
    else:
        norm = plt.Normalize(vmin=0, vmax=1)
    plt.figure(figsize=(20, 8))
    ax = sns.heatmap(fg_gene_matrix,
                     xticklabels=selected_symbols,
                     yticklabels=fg_names_sorted,
                     norm=norm,
                     cmap='viridis',
                     annot=False,
                     cbar_kws={'label': 'Attention Weight'})
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=9)
    drug_name = sample_row.get("DRUG_NAME", args.drug_id)
    cell_std_name = sample_row.get("CELL_LINES_STD", args.cell_line)
    plt.title(f'{drug_name} in {cell_std_name} (IC50={np.exp(sample_label):.2f} uM)\n{mode} mode {suffix}')
    plt.tight_layout()
    out_path = os.path.join(output_dir, f'fg_gene_{args.drug_id}_{args.cell_line}_{mode}{suffix}.png')
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Heatmap saved: {out_path}")


# Process each head or averaged heads
modes = [args.mode]
for mode in modes:
    print(f"\nProcessing {mode} mode...")
    if has_heads and args.heads:
        for head_idx in range(num_heads):
            head_attn = attn_weights[0, head_idx].cpu().numpy()
            aligned = align_attention(head_attn)
            gene_imp = compute_gene_importance(aligned, mode)
            # Select top genes
            total_genes = args.total_genes
            top_n = int(total_genes * args.top_ratio)
            bottom_m = total_genes - top_n
            sorted_idx_desc = np.argsort(gene_imp)[::-1]
            top_indices = sorted_idx_desc[:top_n]
            sorted_idx_asc = np.argsort(gene_imp)
            bottom_indices = [idx for idx in sorted_idx_asc if idx not in top_indices][:bottom_m]
            selected_indices = np.concatenate([top_indices, bottom_indices])
            # Atom -> functional group aggregation
            fg_weights = {}
            for fg, indices in fg_mapping.items():
                valid = [i for i in indices if i < aligned.shape[0]]
                if valid:
                    fg_weights[fg] = np.mean(aligned[valid], axis=0)
                else:
                    fg_weights[fg] = np.zeros(aligned.shape[1])
            fg_names = list(fg_weights.keys())
            plot_fg_gene_heatmap(fg_weights, fg_names, global_symbols, selected_indices, mode, f'_head{head_idx + 1}',
                                 args.output_dir)
    else:
        if has_heads:
            avg_attn = attn_weights.mean(dim=1)[0].cpu().numpy()
        else:
            avg_attn = attn_weights[0].cpu().numpy()
        aligned = align_attention(avg_attn)
        gene_imp = compute_gene_importance(aligned, mode)
        total_genes = args.total_genes
        top_n = int(total_genes * args.top_ratio)
        bottom_m = total_genes - top_n
        sorted_idx_desc = np.argsort(gene_imp)[::-1]
        top_indices = sorted_idx_desc[:top_n]
        sorted_idx_asc = np.argsort(gene_imp)
        bottom_indices = [idx for idx in sorted_idx_asc if idx not in top_indices][:bottom_m]
        selected_indices = np.concatenate([top_indices, bottom_indices])
        fg_weights = {}
        for fg, indices in fg_mapping.items():
            valid = [i for i in indices if i < aligned.shape[0]]
            if valid:
                fg_weights[fg] = np.mean(aligned[valid], axis=0)
            else:
                fg_weights[fg] = np.zeros(aligned.shape[1])
        fg_names = list(fg_weights.keys())
        plot_fg_gene_heatmap(fg_weights, fg_names, global_symbols, selected_indices, mode, '_avg', args.output_dir)

print("\nDone!")
