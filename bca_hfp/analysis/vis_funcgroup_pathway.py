# bca_hfp/analysis/vis_funcgroup_pathway.py
"""Functional group-pathway attention heatmap."""

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
from bca_hfp.analysis.funcgroup_pathway_mapping import (
    get_fg_mapping, merge_fg_mapping, PATHWAY_SIMPLE_NAMES
)

def load_ensembl_to_symbol_map():
    map_file = os.path.join(root_dir("Gene_Feature"), "geneformer", "gene_dictionaries_30m", "gene_name_id_dict_gc30M.pkl")
    with open(map_file, 'rb') as f:
        symbol_to_ensembl = pickle.load(f)
    ensembl_to_symbol = {}
    for symbol, ens_id in symbol_to_ensembl.items():
        ens_no_version = ens_id.split('.')[0]
        ensembl_to_symbol[ens_no_version] = symbol.upper()
    return ensembl_to_symbol

parser = argparse.ArgumentParser(description='Plot functional group-pathway attention heatmap')
parser.add_argument('--drug_id', type=str, required=True, help='Drug ID')
parser.add_argument('--cell_line', type=str, required=True, help='Cell line standard name (e.g. NCIH1155) or GDSC_ID')
parser.add_argument('--model_path', type=str, required=True, help='Model weights path (.pth)')
parser.add_argument('--output_dir', type=str, default=root_dir("Picture"), help='Output directory')
parser.add_argument('--dataset', type=str, default='GDSC', choices=['GDSC'])
parser.add_argument('--gene_version', type=str, default='V1', choices=['V1', 'V2'])
parser.add_argument('--mode', type=str, default='weighted_mean',
                    choices=['weighted_mean', 'p90', 'dual'], help='Aggregation method')
parser.add_argument('--pathway_gmt', type=str, default=os.path.join(root_dir("Data"), "c2.cp.kegg.v7.5.1.symbols.gmt"),
                    help='KEGG pathway file path')
parser.add_argument('--heads', action='store_true', help='Plot separate heatmap for each attention head')
parser.add_argument('--layer', type=int, default=3, choices=[1,2,3], help='Number of MLP layers')
parser.add_argument('--fp_bits', type=int, default=2048, help='Morgan fingerprint bits')
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

# Model hyperparameters
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

# Morgan fingerprints
print("Generating drug global features (Morgan Fingerprints)...")
drug_global_dict = load_morgan_fingerprints(
    config['data']['drug_feat_path'],
    fp_bits=args.fp_bits,
    fp_radius=2
)
config['data']['drug_global_dim'] = args.fp_bits

# Load data
print("Loading drug atom features...")
drug_atom_dict = load_drug_atom_features(config['data']['drug_feat_path'])
print("Loading gene embeddings...")
gene_emb_dict, _ = load_gene_embeddings(config['data']['gene_feat_dir'], config['data']['cell_id_col'])
print("Loading response data...")
response_df = load_response(config['data']['response_path'], config['data']['cell_id_col'], config['data']['drug_id_col'])

# Filter target drug and cell line
drug_col = config['data']['drug_id_col']
cell_col = config['data']['cell_id_col']
target_df = response_df[response_df[drug_col].astype(str) == str(args.drug_id)].copy()
if target_df.empty:
    raise ValueError(f"Drug {args.drug_id} not found")

# Match cell line
cell_std_col = 'CELL_LINES_STD'
sample_row = None
if cell_std_col in target_df.columns:
    mask = target_df[cell_std_col] == args.cell_line
    if mask.any():
        sample_row = target_df[mask].iloc[0]
    else:
        # Try matching input as GDSC_ID
        try:
            cell_id_int = int(args.cell_line)
            mask = target_df[cell_col] == cell_id_int
            if mask.any():
                sample_row = target_df[mask].iloc[0]
            else:
                raise ValueError(f"Cell line {args.cell_line} not found")
        except:
            raise ValueError(f"Cell line {args.cell_line} not found")
else:
    try:
        cell_id_int = int(args.cell_line)
        mask = target_df[cell_col] == cell_id_int
        if mask.any():
            sample_row = target_df[mask].iloc[0]
        else:
            raise ValueError(f"Cell line {args.cell_line} not found")
    except:
        raise ValueError(f"Cell line {args.cell_line} not found")

cell_id = str(sample_row[cell_col])
true_label = sample_row[config['data']['label_col']]
sample_idx = target_df.index.get_loc(sample_row.name)

# Create dataset
single_drug_ids = [str(sample_row[config['data']['drug_id_col']])]
single_cell_ids = [str(sample_row[config['data']['cell_id_col']])]
single_labels = [sample_row[config['data']['label_col']]]

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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config['data']['gene_input_dim'] = 256   # Fixed for V1
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

# Extract attention weights (ensure multi-head weights are returned)
collate = collate_fn
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
        # Trim to valid atoms
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
    cid = str(meta.get(cell_col))
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
symbol_to_idx = {s: i for i, s in enumerate(global_symbols)}

# Gene order for this sample
sample_cell_id = cell_id
gene_names_cell = cell_to_genes.get(sample_cell_id, [])
if not gene_names_cell:
    raise ValueError(f"Gene list not found for cell line {sample_cell_id}")

# Functional group mapping
smiles = sample_row['DRUG_SMILE']
fg_mapping = get_fg_mapping(smiles)
fg_mapping = merge_fg_mapping(fg_mapping)
if not fg_mapping:
    raise ValueError("No functional groups identified")

# Load pathway file
def load_pathway_gmt(gmt_file):
    pathway_dict = {}
    with open(gmt_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue
            pathway = parts[0]
            genes = parts[2:]
            pathway_dict[pathway] = genes
    return pathway_dict

pathway_dict = load_pathway_gmt(args.pathway_gmt)
gene_set = set(global_symbols)
filtered_pathways = {}
for pathway, genes in pathway_dict.items():
    present = [g for g in genes if g in gene_set]
    if present:
        filtered_pathways[pathway] = present
path_names = list(filtered_pathways.keys())
print(f"Loaded {len(path_names)} pathways")

# Define pathway aggregation function
def compute_pathway_matrix(mode, fg_weights, path_names, symbol_to_idx, filtered_pathways):
    fg_list = [fg_weights[fg] for fg in fg_weights]
    n_fg = len(fg_list)
    n_path = len(path_names)
    matrix = np.zeros((n_fg, n_path))
    for i, fg_vec in enumerate(fg_list):
        for j, path in enumerate(path_names):
            gene_idx = [symbol_to_idx[g] for g in filtered_pathways[path] if g in symbol_to_idx]
            if gene_idx:
                if mode == 'weighted_mean':
                    val = np.mean(fg_vec[gene_idx])
                elif mode == 'p90':
                    val = np.percentile(fg_vec[gene_idx], 90)
                elif mode == 'dual':
                    val = np.mean(fg_vec[gene_idx]) + np.max(fg_vec[gene_idx])
                else:
                    val = np.mean(fg_vec[gene_idx])
                matrix[i, j] = val
    return matrix

# Helper function to align attention matrix
def align_attention(attn_np):
    # attn_np: (atoms, genes_cell)
    aligned = np.zeros((attn_np.shape[0], num_global_genes), dtype=np.float32)
    for j, g in enumerate(gene_names_cell):
        if g in gene_to_global_idx:
            aligned[:, gene_to_global_idx[g]] = attn_np[:, j]
    return aligned

# Process each head or averaged heads
modes = [args.mode]
for mode in modes:
    print(f"\nProcessing {mode} mode...")
    if has_heads and args.heads:
        for head_idx in range(num_heads):
            head_attn = attn_weights[0, head_idx].cpu().numpy()  # (atoms, genes_cell)
            aligned = align_attention(head_attn)
            # Atom -> functional group (mean aggregation)
            fg_weights = {}
            for fg, indices in fg_mapping.items():
                valid = [i for i in indices if i < aligned.shape[0]]
                if valid:
                    fg_weights[fg] = np.mean(aligned[valid], axis=0)
                else:
                    fg_weights[fg] = np.zeros(aligned.shape[1])
            if not fg_weights:
                continue
            fg_pathway_matrix = compute_pathway_matrix(mode, fg_weights, path_names, symbol_to_idx, filtered_pathways)
            # Pathway sorting
            threshold = np.percentile(fg_pathway_matrix[fg_pathway_matrix > 0], 50) if np.any(fg_pathway_matrix > 0) else 0
            pathway_max = fg_pathway_matrix.max(axis=0)
            pathway_freq = (fg_pathway_matrix > threshold).sum(axis=0)
            score = pathway_max + pathway_freq
            top_idx = np.argsort(score)[-20:][::-1]
            # Functional group sorting
            fg_total = np.array([fg_weights[fg].sum() for fg in fg_weights])
            fg_order = np.argsort(fg_total)[::-1]
            # Sub-matrix
            fg_pathway_matrix_top = fg_pathway_matrix[np.ix_(fg_order, top_idx)]
            path_names_top = [path_names[i] for i in top_idx]
            simplified_names = [PATHWAY_SIMPLE_NAMES.get(p, p.replace("KEGG_", "").replace("_", " ").title()) for p in path_names_top]
            fg_names_sorted = [list(fg_weights.keys())[i] for i in fg_order]
            # Color scale
            positive_vals = fg_pathway_matrix_top[fg_pathway_matrix_top > 0]
            if len(positive_vals) > 0:
                vmin = np.percentile(positive_vals, 5)
                vmax = np.percentile(positive_vals, 95)
                norm = plt.Normalize(vmin=vmin, vmax=vmax)
            else:
                norm = plt.Normalize(vmin=0, vmax=1)
            plt.figure(figsize=(16, 12))
            ax = sns.heatmap(fg_pathway_matrix_top,
                             xticklabels=simplified_names,
                             yticklabels=fg_names_sorted,
                             norm=norm,
                             cmap='viridis',
                             annot=False,
                             cbar_kws={'label': 'Attention Weight'})
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
            ax.set_yticklabels(ax.get_yticklabels(), fontsize=9)
            drug_name = sample_row.get("DRUG_NAME", args.drug_id)
            cell_std = sample_row.get(cell_std_col, cell_id)
            plt.title(f'{drug_name} in {cell_std} (IC50={np.exp(true_label):.2f} uM)\n{mode} mode Head {head_idx+1}')
            plt.tight_layout()
            output_path = os.path.join(args.output_dir, f'fg_pathway_{args.drug_id}_{args.cell_line}_{mode}_head{head_idx+1}.png')
            plt.savefig(output_path, dpi=300)
            plt.close()
            print(f"  Saved: {output_path}")
    else:
        # Averaged heads
        if has_heads:
            avg_attn = attn_weights.mean(dim=1)[0].cpu().numpy()
        else:
            avg_attn = attn_weights[0].cpu().numpy()
        aligned = align_attention(avg_attn)
        fg_weights = {}
        for fg, indices in fg_mapping.items():
            valid = [i for i in indices if i < aligned.shape[0]]
            if valid:
                fg_weights[fg] = np.mean(aligned[valid], axis=0)
            else:
                fg_weights[fg] = np.zeros(aligned.shape[1])
        if not fg_weights:
            print("Warning: No valid functional groups")
            continue
        fg_pathway_matrix = compute_pathway_matrix(mode, fg_weights, path_names, symbol_to_idx, filtered_pathways)
        threshold = np.percentile(fg_pathway_matrix[fg_pathway_matrix > 0], 50) if np.any(fg_pathway_matrix > 0) else 0
        pathway_max = fg_pathway_matrix.max(axis=0)
        pathway_freq = (fg_pathway_matrix > threshold).sum(axis=0)
        score = pathway_max + pathway_freq
        top_idx = np.argsort(score)[-20:][::-1]
        fg_total = np.array([fg_weights[fg].sum() for fg in fg_weights])
        fg_order = np.argsort(fg_total)[::-1]
        fg_pathway_matrix_top = fg_pathway_matrix[np.ix_(fg_order, top_idx)]
        path_names_top = [path_names[i] for i in top_idx]
        simplified_names = [PATHWAY_SIMPLE_NAMES.get(p, p.replace("KEGG_", "").replace("_", " ").title()) for p in path_names_top]
        fg_names_sorted = [list(fg_weights.keys())[i] for i in fg_order]
        positive_vals = fg_pathway_matrix_top[fg_pathway_matrix_top > 0]
        if len(positive_vals) > 0:
            vmin = np.percentile(positive_vals, 5)
            vmax = np.percentile(positive_vals, 95)
            norm = plt.Normalize(vmin=vmin, vmax=vmax)
        else:
            norm = plt.Normalize(vmin=0, vmax=1)
        plt.figure(figsize=(16, 12))
        ax = sns.heatmap(fg_pathway_matrix_top,
                         xticklabels=simplified_names,
                         yticklabels=fg_names_sorted,
                         norm=norm,
                         cmap='viridis',
                         annot=False,
                         cbar_kws={'label': 'Attention Weight'})
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=9)
        drug_name = sample_row.get("DRUG_NAME", args.drug_id)
        cell_std = sample_row.get(cell_std_col, cell_id)
        plt.title(f'{drug_name} in {cell_std} (IC50={np.exp(true_label):.2f} uM)\n{mode} mode (averaged heads)')
        plt.tight_layout()
        output_path = os.path.join(args.output_dir, f'fg_pathway_{args.drug_id}_{args.cell_line}_{mode}_avg.png')
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"  Saved: {output_path}")

print("\nDone!")
