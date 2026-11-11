# bca_hfp/analysis/vis_sensitive_resistant_diff.py
"""Compare functional group-pathway attention between sensitive and resistant cell lines."""

import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import os
import argparse
import pickle
from collections import defaultdict
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import fdrcorrection

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

parser = argparse.ArgumentParser(description='Compare functional group-pathway attention between sensitive vs resistant cell lines')
parser.add_argument('--drug_id', type=str, required=True, help='Drug ID')
parser.add_argument('--model_path', type=str, required=True, help='Model weights path (.pth)')
parser.add_argument('--output_dir', type=str, default=root_dir('Picture'), help='Output directory')
parser.add_argument('--dataset', type=str, default='GDSC', choices=['GDSC'])
parser.add_argument('--gene_version', type=str, default='V1', choices=['V1', 'V2'])
parser.add_argument('--mode', type=str, default='weighted_mean',
                    choices=['weighted_mean', 'p90', 'dual'], help='Aggregation method')
parser.add_argument('--sens_ratio', type=float, default=0.25, help='Sensitive group ratio')
parser.add_argument('--res_ratio', type=float, default=0.25, help='Resistant group ratio')
parser.add_argument('--pathway_gmt', type=str, default=os.path.join(root_dir('Data'), 'c2.cp.kegg.v7.5.1.symbols.gmt'),
                    help='KEGG pathway file path')
parser.add_argument('--heads', action='store_true', help='Whether to plot separate comparison for each attention head')
parser.add_argument('--layer', type=int, default=3, choices=[1,2,3], help='Number of MLP layers')
parser.add_argument('--fp_bits', type=int, default=2048, help='Morgan fingerprint bits')
args = parser.parse_args()

set_seed(42)
config = get_base_config(args.dataset)


drug_col = config['data']['drug_id_col']
cell_col = config['data']['cell_id_col']
label_col = config['data']['label_col']

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

# Filter target drug
drug_col = config['data']['drug_id_col']
cell_col = config['data']['cell_id_col']
target_df = response_df[response_df[drug_col].astype(str) == str(args.drug_id)].copy()
print(f"Drug {args.drug_id} original sample count: {len(target_df)}")

# Filter out cell lines with missing gene embeddings
valid_cell_mask = target_df[cell_col].astype(str).isin(gene_emb_dict)
target_df = target_df[valid_cell_mask].copy()
print(f"Sample count after filtering missing genes: {len(target_df)}")

if len(target_df) == 0:
    raise ValueError(f"Drug {args.drug_id} has no valid cell line samples")

# Create dataset (requires drug_global_dict)
dataset = DrugGeneDataset(
    target_df[drug_col].tolist(),
    target_df[cell_col].tolist(),
    target_df[label_col].values,
    drug_atom_dict,
    gene_emb_dict,
    drug_global_dict
)

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

# Build global gene mapping
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

# Load pathways
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

# Grouping
labels = target_df[config['data']['label_col']].values
ic50_sorted = np.argsort(labels)
n_samples = len(labels)
n_sens = int(args.sens_ratio * n_samples)
n_res = int(args.res_ratio * n_samples)
sens_indices = set(ic50_sorted[:n_sens])
res_indices = set(ic50_sorted[-n_res:])
print(f"Sensitive group samples: {len(sens_indices)}, Resistant group samples: {len(res_indices)}")

def accumulate_for_mode_and_head(mode, head_idx=None):
    pathway_values_sens = {path: [] for path in path_names}
    pathway_values_res = {path: [] for path in path_names}
    fg_pathway_accum_sens = defaultdict(float)
    fg_pathway_accum_res = defaultdict(float)
    fg_sample_count_sens = defaultdict(int)
    fg_sample_count_res = defaultdict(int)
    all_fg_names_set = set()

    loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate_fn, num_workers=4)
    cell_id_list = target_df[cell_col].tolist()

    print(f"Accumulating {mode} mode {'Head' + str(head_idx) if head_idx is not None else 'AvgHead'} functional group-pathway attention...")
    with torch.no_grad():
        for batch_idx, (atom, atom_mask, gene, gene_mask, labels, drug_global) in enumerate(loader):
            atom = atom.to(device); atom_mask = atom_mask.to(device)
            gene = gene.to(device); gene_mask = gene_mask.to(device)
            drug_global = drug_global.to(device)
            _, attn_weights = model(gene, atom, atom_mask, gene_mask, drug_global=drug_global, return_attn=True)
            if attn_weights.dim() == 4:
                if head_idx is not None:
                    attn_weights = attn_weights[:, head_idx, :, :]
                else:
                    attn_weights = attn_weights.mean(dim=1)
            for i in range(atom.size(0)):
                global_idx = batch_idx * atom.size(0) + i
                if global_idx >= len(cell_id_list):
                    continue
                cell_id = str(cell_id_list[global_idx])
                gene_names_cell = cell_to_genes.get(cell_id, [])
                if not gene_names_cell:
                    continue
                weight_i = attn_weights[i]
                if weight_i.dim() == 3:
                    attn = weight_i.mean(dim=0).cpu().numpy()
                else:
                    attn = weight_i.cpu().numpy()
                if attn.ndim != 2:
                    continue
                # Align to global genes
                aligned_attn = np.zeros((attn.shape[0], num_global_genes), dtype=np.float32)
                for j, g in enumerate(gene_names_cell):
                    if g in gene_to_global_idx:
                        aligned_attn[:, gene_to_global_idx[g]] = attn[:, j]
                smiles = target_df.iloc[global_idx]['DRUG_SMILE']
                fg_mapping = get_fg_mapping(smiles)
                if not fg_mapping:
                    continue
                fg_mapping = merge_fg_mapping(fg_mapping)
                num_atoms = aligned_attn.shape[0]
                fg_weights = {}
                for fg, indices in fg_mapping.items():
                    valid = [i for i in indices if i < num_atoms]
                    if valid:
                        fg_weights[fg] = np.mean(aligned_attn[valid], axis=0)
                    else:
                        fg_weights[fg] = np.zeros(aligned_attn.shape[1])
                all_fg_names_set.update(fg_weights.keys())
                is_sens = global_idx in sens_indices
                is_res = global_idx in res_indices
                if not (is_sens or is_res):
                    continue
                target_accum = fg_pathway_accum_sens if is_sens else fg_pathway_accum_res
                target_count = fg_sample_count_sens if is_sens else fg_sample_count_res
                for fg, vec in fg_weights.items():
                    target_count[fg] += 1
                    for j, path in enumerate(path_names):
                        gene_idx = [symbol_to_idx[g] for g in filtered_pathways[path] if g in symbol_to_idx]
                        if gene_idx:
                            if mode == 'weighted_mean':
                                val = np.mean(vec[gene_idx])
                            elif mode == 'p90':
                                val = np.percentile(vec[gene_idx], 90)
                            elif mode == 'dual':
                                val = np.mean(vec[gene_idx]) + np.max(vec[gene_idx])
                            else:
                                val = np.mean(vec[gene_idx])
                            target_accum[(fg, j)] += val
                            if is_sens:
                                pathway_values_sens[path].append(val)
                            else:
                                pathway_values_res[path].append(val)

    fg_name_list = sorted(all_fg_names_set)
    fg_to_idx = {fg: i for i, fg in enumerate(fg_name_list)}
    aggregated_sens = np.zeros((len(fg_name_list), len(path_names)))
    aggregated_res = np.zeros((len(fg_name_list), len(path_names)))
    for (fg, j), val in fg_pathway_accum_sens.items():
        aggregated_sens[fg_to_idx[fg], j] = val / fg_sample_count_sens[fg]
    for (fg, j), val in fg_pathway_accum_res.items():
        aggregated_res[fg_to_idx[fg], j] = val / fg_sample_count_res[fg]

    return aggregated_sens, aggregated_res, fg_name_list, pathway_values_sens, pathway_values_res

def plot_comparison(aggregated_sens, aggregated_res, fg_name_list, path_names, mode, suffix, output_dir,
                    pathway_values_sens, pathway_values_res):
    # Pathway sorting (based on sensitive group max + frequency)
    if np.any(aggregated_sens > 0):
        threshold = np.percentile(aggregated_sens[aggregated_sens > 0], 50)
    else:
        threshold = 0
    pathway_max = aggregated_sens.max(axis=0)
    pathway_freq = (aggregated_sens > threshold).sum(axis=0)
    score = pathway_max + pathway_freq
    top_idx = np.argsort(score)[-20:][::-1]
    # Functional group sorting
    fg_importance = aggregated_sens.sum(axis=1)
    fg_order = np.argsort(fg_importance)[::-1]

    aggregated_sens_top = aggregated_sens[np.ix_(fg_order, top_idx)]
    aggregated_res_top = aggregated_res[np.ix_(fg_order, top_idx)]
    path_names_top = [path_names[i] for i in top_idx]
    simplified_names = [PATHWAY_SIMPLE_NAMES.get(p, p.replace("KEGG_", "").replace("_", " ").title()) for p in path_names_top]
    fg_names_sorted = [fg_name_list[i] for i in fg_order]

    # Unified color scale
    all_vals = np.concatenate([aggregated_sens_top.flatten(), aggregated_res_top.flatten()])
    positive_vals = all_vals[all_vals > 0]
    if len(positive_vals) > 0:
        vmin = positive_vals.min()
        vmax = np.percentile(positive_vals, 95)
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
    else:
        vmin, vmax = 0, 1
        norm = plt.Normalize(vmin, vmax)

    # Side-by-side heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(24, 12), sharey=True)
    sns.heatmap(aggregated_sens_top, xticklabels=simplified_names, yticklabels=fg_names_sorted,
                ax=axes[0], cmap='viridis', norm=norm, annot=False, cbar=False)
    axes[0].set_title('Sensitive Cell Lines (Low IC50)', fontsize=12, fontweight='bold')
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha='right', fontsize=8)
    axes[0].set_yticklabels(axes[0].get_yticklabels(), fontsize=8)

    sns.heatmap(aggregated_res_top, xticklabels=simplified_names, yticklabels=fg_names_sorted,
                ax=axes[1], cmap='viridis', norm=norm, annot=False,
                cbar=True, cbar_kws={'label': 'Attention Weight', 'shrink': 0.8})
    axes[1].set_title('Resistant Cell Lines (High IC50)', fontsize=12, fontweight='bold')
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha='right', fontsize=8)
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'comparison_{mode}{suffix}.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"  Side-by-side heatmap saved: {output_path}")

    # Difference heatmap
    diff = aggregated_sens_top - aggregated_res_top
    max_abs = np.abs(diff).max()
    plt.figure(figsize=(16, 12))
    ax = sns.heatmap(diff, xticklabels=simplified_names, yticklabels=fg_names_sorted,
                     cmap='RdBu_r', center=0, vmin=-max_abs, vmax=max_abs,
                     annot=False, cbar_kws={'label': 'Attention Difference (Sensitive - Resistant)'})
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, fontsize=8, fontstyle='italic', ha='center')
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=9, rotation=0, ha='right', va='center')
    cbar = ax.collections[0].colorbar
    cbar.ax.set_ylabel('Attention Difference', rotation=90, labelpad=20, fontsize=10)
    plt.tight_layout()
    diff_path = os.path.join(output_dir, f'difference_{mode}{suffix}.png')
    plt.savefig(diff_path, dpi=300)
    plt.close()
    print(f"  Difference heatmap saved: {diff_path}")

    # Statistical tests
    print("\n  === Statistical Tests (Sensitive vs Resistant, attention values per pathway) ===")
    pvals = []
    for path in path_names:
        sens_vals = pathway_values_sens.get(path, [])
        res_vals = pathway_values_res.get(path, [])
        if len(sens_vals) > 1 and len(res_vals) > 1:
            _, p = ttest_ind(sens_vals, res_vals, equal_var=False)
        else:
            p = 1.0
        pvals.append(p)
    reject, qvals = fdrcorrection(pvals)
    print(f"{'Pathway':<50} {'p-value':<12} {'q-value':<12} {'Significant'}")
    for i, path in enumerate(path_names):
        print(f"{path:<50} {pvals[i]:<12.4e} {qvals[i]:<12.4e} {'Yes' if qvals[i] < 0.05 else 'No'}")

if args.heads:
    # Get number of attention heads
    dummy_loader = DataLoader([dataset[0]], batch_size=1, collate_fn=collate_fn)
    with torch.no_grad():
        for dummy_atom, dummy_mask, dummy_gene, dummy_gene_mask, _, _ in dummy_loader:
            _, dummy_attn = model(dummy_gene.to(device), dummy_atom.to(device), dummy_mask.to(device),
                                  dummy_gene_mask.to(device), drug_global=None, return_attn=True)
            if dummy_attn.dim() == 4:
                num_heads = dummy_attn.shape[1]
            else:
                num_heads = 1
    print(f"Detected {num_heads} attention heads")
    for head_idx in range(num_heads):
        aggregated_sens, aggregated_res, fg_name_list, sens_vals, res_vals = accumulate_for_mode_and_head(args.mode, head_idx=head_idx)
        suffix = f'_head{head_idx+1}'
        plot_comparison(aggregated_sens, aggregated_res, fg_name_list, path_names, args.mode, suffix, args.output_dir, sens_vals, res_vals)
else:
    aggregated_sens, aggregated_res, fg_name_list, sens_vals, res_vals = accumulate_for_mode_and_head(args.mode, head_idx=None)
    suffix = '_avg'
    plot_comparison(aggregated_sens, aggregated_res, fg_name_list, path_names, args.mode, suffix, args.output_dir, sens_vals, res_vals)

print("\nDone!")
