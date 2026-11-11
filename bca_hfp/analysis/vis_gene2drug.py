# bca_hfp/analysis/vis_gene2drug.py
"""Gene-to-atom attention heatmap."""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pickle
import torch
import pandas as pd
import math
from matplotlib.colors import PowerNorm

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
    """
    Read gene name list from .pt file for specified cell line.
    Args:
        cell_id: Cell line ID (e.g. '906836')
        gene_dir: Geneformer embedding directory
    Returns:
        list: Gene name list (special tokens filtered)
    """
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


def load_gene_order_from_file(filepath):
    """Read gene order list from file (one gene symbol per line)"""
    with open(filepath, 'r') as f:
        genes = [line.strip() for line in f if line.strip()]
    print(f"Read gene order from {filepath}, total {len(genes)} genes")
    return genes


def sci_notation(x, precision=2):
    """Scientific notation formatting (for color bar)"""
    if x == 0:
        return '0'
    sign = '' if x > 0 else '-'
    x_abs = abs(x)
    exp = math.floor(math.log10(x_abs))
    mant = x_abs / (10 ** exp)
    if abs(mant - 1.0) < 1e-6:
        return f'${sign}10^{{{exp}}}$'
    elif abs(mant - round(mant, precision)) < 1e-6:
        return f'${sign}{mant:.0f}\\times10^{{{exp}}}$'
    else:
        return f'${sign}{mant:.{precision}f}\\times10^{{{exp}}}$'


def main():
    parser = argparse.ArgumentParser(description='Plot Gene Queries Drug single-head heatmap (optimized)')
    parser.add_argument('--weight_dir', type=str, required=True,
                        help='Weight directory (containing sample_*_gene2drug_8heads.npy and sample_metadata.csv)')
    parser.add_argument('--output_dir', type=str, required=True, help='Output image directory')
    parser.add_argument('--sample_idx', type=int, required=True, help='Sample index')

    parser.add_argument('--gene_order_file', type=str, default=None,
                        help='Fixed gene order file (typically from Drug->Gene output). If provided, uses this order; otherwise sorts by own attention')
    parser.add_argument('--target_gene', type=str, nargs='+', default=None,
                        help='Target gene list to highlight in heatmap (yellow horizontal line + left arrow, does not change order)')
    parser.add_argument('--red_gene', type=str, nargs='+', default=None,
                        help='Specify gene labels to mark in red (Y-axis text becomes red bold)')
    parser.add_argument('--top_genes', type=int, default=20,
                        help='Number of genes to display (if order file provided, takes first top_genes)')
    parser.add_argument('--head_idx', type=int, default=None,
                        help='Only plot specified head (0-7). If not specified, generates all heads')

    # Color mapping parameters (aligned with paper_gene2drug)
    parser.add_argument('--lower_percentile', type=float, default=1.0,
                        help='Lower truncation percentile (default 1.0)')
    parser.add_argument('--upper_percentile', type=float, default=95.0,
                        help='Upper truncation percentile (default 95.0)')
    parser.add_argument('--gamma', type=float, default=0.35,
                        help='PowerNorm gamma parameter (default 0.35)')
    parser.add_argument('--dpi', type=int, default=300,
                        help='Output image DPI (default 300)')

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
    weight_path = os.path.join(args.weight_dir, f'sample_{args.sample_idx:06d}_gene2drug_8heads.npy')
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"Weight file not found: {weight_path}")
    weights = np.load(weight_path)
    num_heads = weights.shape[0]
    num_genes = weights.shape[1]
    num_atoms = weights.shape[2]

    print(f"Loaded weights: {weight_path}")
    print(f"  Shape: {weights.shape}")
    print(f"  Genes: {num_genes}, Atoms: {num_atoms}, Heads: {num_heads}")

    # 5. Determine gene order
    if args.gene_order_file and os.path.exists(args.gene_order_file):
        # Use fixed order file
        fixed_symbols = load_gene_order_from_file(args.gene_order_file)
        # Build mapping symbol -> index (position in weight matrix)
        symbol_to_idx = {gene_symbols[i]: i for i in range(len(gene_symbols))}
        selected_indices = []
        missing = []
        for sym in fixed_symbols[:args.top_genes]:  # Take first top_genes
            if sym in symbol_to_idx:
                selected_indices.append(symbol_to_idx[sym])
            else:
                missing.append(sym)
        if missing:
            print(f"The following genes do not exist in weight matrix and will be skipped: {missing}")
        # If fewer than top_genes, fill from remaining genes by max attention
        if len(selected_indices) < args.top_genes:
            all_indices = set(range(len(gene_symbols)))
            remaining = list(all_indices - set(selected_indices))
            # Sort by max attention (consistent with paper)
            gene_attn = weights.max(axis=(0, 2))
            remaining_sorted = sorted(remaining, key=lambda i: gene_attn[i], reverse=True)
            need = args.top_genes - len(selected_indices)
            selected_indices.extend(remaining_sorted[:need])
        # Truncate to top_genes
        selected_indices = selected_indices[:args.top_genes]
        # Get final gene symbol list (in fixed order)
        gene_symbols_sorted = [gene_symbols[i] for i in selected_indices]
        print(f"Using fixed gene order (first {len(gene_symbols_sorted)})")
    else:
        # Fallback: sort by own max attention (consistent with paper fallback)
        print("No gene order file provided, sorting by own max attention")
        gene_attn = weights.max(axis=(0, 2))  # (genes,)
        sorted_indices = np.argsort(gene_attn)[::-1]
        selected_indices = sorted_indices[:args.top_genes]
        gene_symbols_sorted = [gene_symbols[i] for i in selected_indices]
        print(f"Sorted by max attention, taking top {len(gene_symbols_sorted)}")

    # 6. Process target genes (highlighting)
    target_positions = []  # Positions in selected_indices
    if args.target_gene is not None:
        print("\nTarget gene highlighting:")
        for gene in args.target_gene:
            ensg = symbol_to_ensg.get(gene, gene)
            if ensg in gene_names:
                idx = gene_names.index(ensg)
                if idx in selected_indices:
                    pos = selected_indices.index(idx)
                    target_positions.append(pos)
                    print(f"  {gene} (position {pos}) will be highlighted")
                else:
                    print(f"  {gene} not in displayed gene list")
            else:
                print(f"  {gene} not in this cell line's gene list")
        if not target_positions:
            print("  No target genes highlighted")

    # 7. Process red-labeled genes
    red_gene_set = set()
    if args.red_gene is not None:
        print("\nRed-labeled genes:")
        for gene in args.red_gene:
            ensg = symbol_to_ensg.get(gene, gene)
            if ensg in gene_names:
                idx = gene_names.index(ensg)
                if idx in selected_indices:
                    red_gene_set.add(gene_symbols_sorted[selected_indices.index(idx)])
                    print(f"  {gene} will be marked in red")
                else:
                    print(f"  {gene} not in displayed gene list")
            else:
                print(f"  {gene} not in this cell line's gene list")
        if not red_gene_set:
            print("  No genes marked in red")

    # Print final gene list
    print(f"\nFinal displayed gene list (total {len(gene_symbols_sorted)}):")
    for i, sym in enumerate(gene_symbols_sorted):
        marker = " <-- TARGET" if i in target_positions else ""
        red_marker = " (RED)" if sym in red_gene_set else ""
        print(f"  {i+1:2d}. {sym}{marker}{red_marker}")

    # 8. Compute global statistics for truncation
    all_weights = weights.flatten()
    vmin = np.percentile(all_weights, args.lower_percentile)
    vmax = np.percentile(all_weights, args.upper_percentile)
    print(f"\nPercentile truncation: vmin={vmin:.6f} (P{args.lower_percentile}), vmax={vmax:.6f} (P{args.upper_percentile})")
    masked = all_weights[(all_weights >= vmin) & (all_weights <= vmax)]
    median_val = np.median(masked) if len(masked) > 0 else 0
    print(f"Median within truncation range: {median_val:.6f}")

    # 9. Extract data for corresponding heads
    if args.head_idx is not None:
        head_list = [args.head_idx]
    else:
        head_list = range(num_heads)

    # Prepare plotting data (heads, top_genes, atoms)
    mats = weights[:, selected_indices, :]  # (heads, top_genes, atoms)

    # 10. Plot each head
    for h in head_list:
        print(f"\nPlotting Head {h+1}...")
        data = mats[h]  # (top_genes, atoms)

        # Skip if data is all zeros
        if data.max() == 0:
            print(f"  Head {h+1} data is all zeros, skipping")
            continue

        # Create figure
        fig_width = max(10, num_atoms * 0.1)
        fig_height = max(5, args.top_genes * 0.35)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), facecolor='white')

        # Use PowerNorm (consistent with paper)
        im = ax.imshow(data, cmap='Blues',
                       norm=PowerNorm(gamma=args.gamma, vmin=vmin, vmax=vmax),
                       aspect="auto", interpolation='nearest')

        # Color bar (using scientific notation, annotated with percentiles)
        cbar = fig.colorbar(im, ax=ax, shrink=0.8, extend='max')
        cbar.set_ticks([vmin, median_val, vmax])
        tick_labels = [sci_notation(vmin, 2), sci_notation(median_val, 2), sci_notation(vmax, 2)]
        if vmin == 0:
            tick_labels[0] = '0 (P1)'
        tick_labels[1] = tick_labels[1] + '\n(Median)'
        tick_labels[2] = tick_labels[2] + f'\n(P{args.upper_percentile:.0f})'
        cbar.set_ticklabels(tick_labels)
        cbar.set_label('Attention Weight', fontsize=10, fontweight='bold')
        cbar.ax.tick_params(labelsize=8)

        # X-axis (atoms)
        x_ticks = [0, 15, 30, 45, 60, num_atoms - 1]
        x_ticks = sorted(set(x_ticks))
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([str(t) for t in x_ticks], fontsize=8)
        ax.tick_params(axis='x', length=4, direction='out', pad=1)
        ax.set_xlabel('Atom Index', fontsize=10, fontweight='bold')

        # Y-axis (genes)
        ax.set_yticks(range(len(gene_symbols_sorted)))
        ax.set_yticklabels(gene_symbols_sorted, fontsize=7, rotation=0, ha='right')
        ax.tick_params(axis='y', length=4, direction='out', pad=3)
        ax.set_ylabel('')  # No label

        # ----- Mark specified gene Y-axis labels in red -----
        y_labels = ax.get_yticklabels()
        for label in y_labels:
            text = label.get_text()
            if text in red_gene_set:
                label.set_color('red')
                label.set_fontweight('bold')
                label.set_fontsize(8)
            else:
                label.set_color('black')
                label.set_fontsize(8)

        # Highlight target gene rows (yellow horizontal line + left arrow)
        for pos in target_positions:
            ax.axhline(y=pos, color='yellow', linewidth=2, alpha=0.8, linestyle='-')
            ax.text(-0.5, pos, '>', ha='right', va='center', color='yellow', fontsize=14, fontweight='bold')

        # Title
        ax.set_title(f'Gene -> Drug Atoms - Head {h+1}', fontsize=12, fontweight='bold')

        # Adjust layout
        plt.tight_layout()

        # Save
        output_filename = f'micro_gene2drug_head{h+1}_sample{args.sample_idx:06d}.png'
        output_path = os.path.join(args.output_dir, output_filename)
        plt.savefig(output_path, dpi=args.dpi, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  Head {h+1} saved: {output_path}")

    print(f"\nDone! Generated {len(head_list)} images")


if __name__ == '__main__':
    main()
