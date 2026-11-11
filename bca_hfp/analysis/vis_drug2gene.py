# bca_hfp/analysis/vis_drug2gene.py
"""Atom-to-gene attention heatmap."""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import torch
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from bca_hfp.config.base_config import PROJECT_ROOT, root_dir

GENE_FEAT_DIR = os.path.join(root_dir("Data"), "GDSC_EXP_GF_V1")


def load_ensembl_to_symbol_map():
    """Load ENSG -> Gene Symbol mapping"""
    map_file = os.path.join(root_dir("Gene_Feature"), "geneformer", "gene_dictionaries_30m", "gene_name_id_dict_gc30M.pkl")
    with open(map_file, 'rb') as f:
        symbol_to_ensembl = pickle.load(f)
    ensembl_to_symbol = {}
    for symbol, ens_id in symbol_to_ensembl.items():
        ens_no_version = ens_id.split('.')[0]
        ensembl_to_symbol[ens_no_version] = symbol.upper()
    return ensembl_to_symbol


def load_gene_names_from_pt(cell_id, gene_dir=GENE_FEAT_DIR):
    """Read gene name list from .pt file for specified cell line"""
    special_tokens = {'<cls>', '<eos>', '<pad>'}
    for fname in os.listdir(gene_dir):
        if not fname.endswith('.pt'):
            continue
        data = torch.load(os.path.join(gene_dir, fname))
        meta = data.get('metadata', {})
        if str(meta.get('GDSC_ID')) == str(cell_id):
            gene_names = data.get('gene_names', [])
            filtered = [name for name in gene_names if name not in special_tokens]
            return filtered
    raise ValueError(f"Gene embedding file not found for cell line {cell_id}")


def get_cell_id_from_metadata(weight_dir, sample_idx):
    """Read cell_id from sample_metadata.csv"""
    meta_path = os.path.join(weight_dir, 'sample_metadata.csv')
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"sample_metadata.csv not found: {meta_path}")
    df = pd.read_csv(meta_path)
    row = df[df['sample_idx'] == sample_idx]
    if row.empty:
        raise ValueError(f"Metadata not found for sample_idx={sample_idx}")
    return str(row.iloc[0]['cell_id'])


def main():
    parser = argparse.ArgumentParser(description='Plot Drug Queries Gene 8-head heatmap')
    parser.add_argument('--weight_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--sample_idx', type=int, required=True)

    parser.add_argument('--target_gene', type=str, nargs='+', default=['CAPNS1'],
                        help='Target genes, e.g. --target_gene CAPNS1')
    parser.add_argument('--top_genes', type=int, default=20)
    parser.add_argument('--head_idx', type=int, default=None, help='Only plot specified head (0-7)')
    parser.add_argument('--cell_width', type=float, default=0.6,
                        help='Width per cell in inches, default 0.6')
    parser.add_argument('--cell_height', type=float, default=0.08,
                        help='Height per cell in inches, default 0.08')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Read cell_id from metadata
    cell_id = get_cell_id_from_metadata(args.weight_dir, args.sample_idx)
    print(f"Cell line ID: {cell_id}")

    # 2. Read gene names from .pt file
    gene_names = load_gene_names_from_pt(cell_id)

    # 3. Load ENSG -> Symbol mapping
    ensembl_to_symbol = load_ensembl_to_symbol_map()
    gene_symbols = [ensembl_to_symbol.get(ensg, ensg) for ensg in gene_names]
    symbol_to_ensg = {v: k for k, v in ensembl_to_symbol.items()}
    print(f"Successfully loaded {len(gene_names)} gene names")

    # 4. Load weights
    weight_path = os.path.join(args.weight_dir, f'sample_{args.sample_idx:06d}_drug2gene_8heads.npy')
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"Weight file not found: {weight_path}")
    weights = np.load(weight_path)
    num_heads = weights.shape[0]
    num_atoms = weights.shape[1]
    num_genes = weights.shape[2]

    print(f"Loaded weights: {weight_path}")
    print(f"  Shape: {weights.shape}")
    print(f"  Atoms: {num_atoms}, Genes: {num_genes}, Heads: {num_heads}")

    # 5. Compute gene importance (sum across all heads)
    gene_attn = weights.sum(axis=1).sum(axis=0)
    print(f"Gene attention total score range: {gene_attn.min():.6f} ~ {gene_attn.max():.6f}")

    # Select top_genes genes
    top_indices = np.argsort(gene_attn)[-args.top_genes:][::-1]

    # 6. Force-include target genes (placed at end)
    target_indices = []
    if args.target_gene is not None:
        print("\nForce-including target genes:")
        for gene in args.target_gene:
            ensg = symbol_to_ensg.get(gene, gene)
            if ensg in gene_names:
                idx = gene_names.index(ensg)
                if idx not in top_indices:
                    target_indices.append(idx)
                    print(f"  {gene} (ENSG: {ensg}) force-included")
                else:
                    print(f"  {gene} (ENSG: {ensg}) already in top list")
            else:
                print(f"  {gene} not found in this cell line's gene list")

    if target_indices:
        remaining_top = [idx for idx in top_indices if idx not in target_indices]
        need = args.top_genes - len(target_indices)
        if need > 0:
            remaining_top = remaining_top[:need]
        else:
            remaining_top = []
        top_indices = remaining_top + target_indices
        print(f"\nFinal display: {len(top_indices)} genes")

    # 7. Build target gene position mapping (for highlighting)
    target_positions = {}
    for gene in args.target_gene:
        ensg = symbol_to_ensg.get(gene, gene)
        if ensg in gene_names:
            idx = gene_names.index(ensg)
            if idx in top_indices:
                target_positions[gene] = top_indices.index(idx)
                print(f"  {gene} at position {target_positions[gene]}")

    # 8. Print complete gene list
    print(f"\nGene list (sorted by attention total score):")
    for i, idx in enumerate(top_indices):
        sym = gene_symbols[idx]
        marker = " <-- TARGET!" if sym in args.target_gene else ""
        print(f"  {i+1:2d}. {sym}: {gene_attn[idx]:.6f}{marker}")

    # 9. Generate x-axis labels (all gene names)
    x_labels = [gene_symbols[i] for i in top_indices]

    # 10. Plot each head independently
    if args.head_idx is not None:
        head_list = [args.head_idx]
    else:
        head_list = range(num_heads)

    for h in head_list:
        print(f"\nPlotting Head {h + 1}...")

        head_data = weights[h, :, top_indices].T

        if head_data.max() == 0:
            print(f"  Head {h + 1} data is all zeros, skipping")
            continue

        n_cols = len(top_indices)
        n_rows = num_atoms
        fig_width = max(10, n_cols * args.cell_width)
        fig_height = max(5, n_rows * args.cell_height)

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        sns.heatmap(
            head_data,
            ax=ax,
            cmap='Reds',
            xticklabels=x_labels,
            yticklabels=False,
            cbar=True,
            cbar_kws={'label': 'Attention Weight', 'shrink': 0.8},
            square=False,
            annot=False,
            linewidths=0,
            vmin=0,
            vmax=None,
            rasterized=True
        )

        # Set x-axis labels: CAPNS1 in red bold
        for label in ax.get_xticklabels():
            text = label.get_text()
            if text == "CAPNS1":
                label.set_color('red')
                label.set_fontweight('bold')
                label.set_fontsize(8)
            else:
                label.set_color('black')
                label.set_fontsize(8)

        ax.set_xlabel("Gene", fontsize=12, fontweight='bold')
        ax.set_ylabel("Atom Index", fontsize=12, fontweight='bold')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='center')

        ax.set_title(f'Drug Queries Gene - Head {h + 1}', fontsize=14, fontweight='bold')

        plt.tight_layout()
        output_path = os.path.join(args.output_dir,
                                   f'micro_drug2gene_head{h + 1}_sample{args.sample_idx:06d}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Head {h + 1} saved: {os.path.basename(output_path)}")

    print(f"\nDone! Generated {len(head_list)} files")


if __name__ == '__main__':
    main()
