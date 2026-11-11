# bca_hfp/analysis/vis_atom_gene.py
"""Atom-gene attention heatmap."""

import sys
import os
import argparse
import numpy as np
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from functools import partial
import pickle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from bca_hfp.config.base_config import get_base_config, PROJECT_ROOT, root_dir
from bca_hfp.data.loader import load_drug_atom_features, load_gene_embeddings, load_response, load_morgan_fingerprints
from bca_hfp.data.dataset import DrugGeneDataset, collate_fn
from bca_hfp.models.main_model import Regressor
from torch.utils.data import DataLoader
from bca_hfp.utils.reproducibility import set_seed

def load_ensembl_to_symbol_map():
    map_file = os.path.join(root_dir("Gene_Feature"), "geneformer", "gene_dictionaries_30m", "gene_name_id_dict_gc30M.pkl")
    with open(map_file, 'rb') as f:
        symbol_to_ensembl = pickle.load(f)
    ensembl_to_symbol = {}
    for symbol, ens_id in symbol_to_ensembl.items():
        ens_no_version = ens_id.split('.')[0]
        ensembl_to_symbol[ens_no_version] = symbol.upper()
    return ensembl_to_symbol

parser = argparse.ArgumentParser(description='Plot atom-gene attention heatmap')
parser.add_argument('--drug_id', type=str, default='1175', help='Drug ID')
parser.add_argument('--cell_line', type=str, default='NCIH1155', help='Cell line standard name or GDSC_ID')
parser.add_argument('--model_path', type=str, required=True, help='Model weights path (.pth)')
parser.add_argument('--output_dir', type=str, default=root_dir("Picture"), help='Output directory')
parser.add_argument('--dataset', type=str, default='GDSC', choices=['GDSC'])
parser.add_argument('--gene_version', type=str, default='V1', choices=['V1', 'V2'])
parser.add_argument('--mode', type=str, default='weighted_mean',
                    choices=['weighted_mean', 'p90', 'dual'], help='Aggregation method')
parser.add_argument('--total_genes', type=int, default=50, help='Total number of genes to display')
parser.add_argument('--heads', action='store_true', help='Plot separate heatmap for each attention head')
parser.add_argument('--layer', type=int, default=3, choices=[1,2,3], help='Number of MLP layers')
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

# Load data
print("Loading drug atom features...")
drug_atom_dict = load_drug_atom_features(config['data']['drug_feat_path'])
print("Loading gene embeddings...")
gene_emb_dict, _ = load_gene_embeddings(config['data']['gene_feat_dir'], config['data']['cell_id_col'])
print(f"Gene embedding dictionary contains {len(gene_emb_dict)} cell lines")
print("Loading response data...")
response_df = load_response(config['data']['response_path'],
                            config['data']['cell_id_col'],
                            config['data']['drug_id_col'])

# Generate Morgan fingerprints
fp_dim = 2048
drug_global_dict = load_morgan_fingerprints(config['data']['drug_feat_path'], fp_bits=fp_dim, fp_radius=2)
config['data']['drug_global_dim'] = fp_dim

# Filter target drug
drug_col = config['data']['drug_id_col']
target_df = response_df[response_df[drug_col].astype(str) == str(args.drug_id)].copy()
if target_df.empty:
    raise ValueError(f"Drug {args.drug_id} not found")

# Filter target cell line
cell_col = config['data']['cell_id_col']
cell_std_col = 'CELL_LINES_STD'
sample_row = None
if cell_std_col in target_df.columns:
    # Match by standard name
    mask = target_df[cell_std_col] == args.cell_line
    if mask.any():
        sample_row = target_df[mask].iloc[0]
        print(f"Matched cell line by standard name '{args.cell_line}'")
    else:
        # Try matching by GDSC_ID
        try:
            cell_id_int = int(args.cell_line)
            mask = target_df[cell_col] == cell_id_int
            if mask.any():
                sample_row = target_df[mask].iloc[0]
                print(f"Matched cell line by GDSC_ID '{cell_id_int}'")
            else:
                raise ValueError(f"Cell line {args.cell_line} not found")
        except ValueError:
            raise ValueError(f"Cell line {args.cell_line} not found")
else:
    # No standard name column, use GDSC_ID directly
    try:
        cell_id_int = int(args.cell_line)
        mask = target_df[cell_col] == cell_id_int
        if mask.any():
            sample_row = target_df[mask].iloc[0]
            print(f"Matched cell line by GDSC_ID '{cell_id_int}'")
        else:
            raise ValueError(f"Cell line {args.cell_line} not found")
    except ValueError:
        raise ValueError(f"Invalid cell line identifier: {args.cell_line}")

cell_id = str(sample_row[cell_col])   # Convert to string, consistent with gene_emb_dict keys
true_label = sample_row[config['data']['label_col']]
drug_name = sample_row.get('DRUG_NAME', 'Unknown')
cell_std_name = sample_row.get(cell_std_col, 'Unknown')
print(f"Drug: {args.drug_id} ({drug_name})")
print(f"Cell line: {cell_id} ({cell_std_name})")
print(f"True LN_IC50: {true_label:.4f} (IC50 = {np.exp(true_label):.2f} uM)")

# Check if cell line exists in gene embedding dictionary
if cell_id not in gene_emb_dict:
    raise KeyError(f"Cell line {cell_id} not found in gene embedding dictionary. Available cell line examples: {list(gene_emb_dict.keys())[:5]}")

# Create dataset
dataset = DrugGeneDataset(
    [str(args.drug_id)],
    [cell_id],
    [true_label],
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
single_loader = DataLoader([(sample_atom, sample_gene, sample_label, sample_global)], batch_size=1, collate_fn=collate_fn)
with torch.no_grad():
    for atom, atom_mask, gene, gene_mask, labels, drug_global in single_loader:
        atom = atom.to(device); atom_mask = atom_mask.to(device)
        gene = gene.to(device); gene_mask = gene_mask.to(device)
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
        # Trim to valid atom count
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
    cell_id_tmp = str(meta.get(config['data']['cell_id_col']))
    if cell_id_tmp and cell_id_tmp != 'None':
        cell_to_genes[cell_id_tmp] = data['gene_names']
global_genes_set = set()
for genes in cell_to_genes.values():
    global_genes_set.update(genes)
global_genes = sorted(global_genes_set)
gene_to_global_idx = {g: i for i, g in enumerate(global_genes)}
num_global_genes = len(global_genes)

ensembl_to_symbol = load_ensembl_to_symbol_map()
global_symbols = [ensembl_to_symbol.get(g, g) for g in global_genes]
symbol_to_idx = {s: i for i, s in enumerate(global_symbols)}

# Gene order for this sample
sample_cell_id = cell_id
gene_names_cell = cell_to_genes.get(sample_cell_id, [])
if not gene_names_cell:
    raise ValueError(f"Gene list not found for cell line {sample_cell_id}")

# Align attention matrix to global gene indices
def align_attention(attn_np):
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

# Plotting function
def plot_atomic_heatmap(attn_matrix, global_symbols, gene_indices, mode, suffix, output_dir, vmax_percentile=95):
    selected_symbols = [global_symbols[i] for i in gene_indices]
    atom_importance = attn_matrix.sum(axis=1)
    atom_order = np.argsort(atom_importance)[::-1]
    attn_sorted = attn_matrix[atom_order]
    pos_vals = attn_sorted[attn_sorted > 0]
    if len(pos_vals) > 0:
        vmax = np.percentile(pos_vals, vmax_percentile)
        vmin = np.percentile(pos_vals, 1)
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
    else:
        norm = plt.Normalize(vmin=0, vmax=1)
    plt.figure(figsize=(20, 12))
    ax = sns.heatmap(attn_sorted,
                     xticklabels=selected_symbols,
                     yticklabels=[f"Atom {i}" for i in atom_order],
                     norm=norm,
                     cmap='viridis',
                     cbar_kws={'label': 'Attention Weight'})
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=6)
    plt.title(f"Atom-Gene Attention ({mode} mode){suffix}\nDrug {args.drug_id}, Cell Line {args.cell_line}")
    plt.tight_layout()
    out_path = os.path.join(output_dir, f"atom_gene_{args.drug_id}_{args.cell_line}_{mode}{suffix}.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Atom-level heatmap saved: {out_path}")

# Process each head or averaged heads
for mode in [args.mode]:
    print(f"\nProcessing {mode} mode...")
    if has_heads and args.heads:
        for head_idx in range(num_heads):
            head_attn = attn_weights[0, head_idx].cpu().numpy()
            aligned = align_attention(head_attn)
            gene_imp = compute_gene_importance(aligned, mode)
            top_gene_idx = np.argsort(gene_imp)[-args.total_genes:][::-1]
            sub_matrix = aligned[:, top_gene_idx]
            plot_atomic_heatmap(sub_matrix, global_symbols, top_gene_idx, mode, f"_head{head_idx+1}", args.output_dir)
    else:
        if has_heads:
            avg_attn = attn_weights.mean(dim=1)[0].cpu().numpy()
        else:
            avg_attn = attn_weights[0].cpu().numpy()
        aligned = align_attention(avg_attn)
        gene_imp = compute_gene_importance(aligned, mode)
        top_gene_idx = np.argsort(gene_imp)[-args.total_genes:][::-1]
        sub_matrix = aligned[:, top_gene_idx]
        plot_atomic_heatmap(sub_matrix, global_symbols, top_gene_idx, mode, "_avg", args.output_dir)

print("Done!")
