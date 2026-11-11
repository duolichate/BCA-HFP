# bca_hfp/analysis/fig_drug2gene.py
"""Drug-to-gene multi-head attention heatmap."""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import pickle
import math
from matplotlib.gridspec import GridSpec
from matplotlib.colors import PowerNorm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from bca_hfp.config.base_config import PROJECT_ROOT, root_dir

GENE_FEAT_DIR = os.path.join(root_dir("Data"), "GDSC_EXP_GF_V1")


def load_ensembl_to_symbol_map():
    map_file = os.path.join(root_dir("Gene_Feature"), "geneformer", "gene_dictionaries_30m", "gene_name_id_dict_gc30M.pkl")
    with open(map_file, 'rb') as f:
        symbol_to_ensembl = pickle.load(f)
    ensembl_to_symbol = {}
    for symbol, ens_id in symbol_to_ensembl.items():
        ens_no_version = ens_id.split('.')[0]
        ensembl_to_symbol[ens_no_version] = symbol.upper()
    return ensembl_to_symbol


def load_gene_names_from_pt(cell_id, gene_dir=GENE_FEAT_DIR):
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
    meta_path = os.path.join(weight_dir, 'sample_metadata.csv')
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"sample_metadata.csv not found: {meta_path}")
    df = pd.read_csv(meta_path)
    row = df[df['sample_idx'] == sample_idx]
    if row.empty:
        raise ValueError(f"Metadata not found for sample_idx={sample_idx}")
    return str(row.iloc[0]['cell_id'])


def sci_notation(x, precision=2):
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
    parser = argparse.ArgumentParser(description='Highlight highest-weight gene in red')
    parser.add_argument('--weight_dir', type=str, required=True)
    parser.add_argument('--sample_idx', type=int, required=True)
    parser.add_argument('--output_dir', type=str, default='./figures')
    parser.add_argument('--top_genes', type=int, default=20)
    parser.add_argument('--target_genes', type=str, nargs='+', default=[],
                        help='Target gene to force at the end (uses the first one)')
    parser.add_argument('--dpi', type=int, default=600)
    parser.add_argument('--lower_percentile', type=float, default=1.0)
    parser.add_argument('--upper_percentile', type=float, default=95.0)
    parser.add_argument('--gamma', type=float, default=0.35)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load weights
    weight_path = os.path.join(args.weight_dir, f'sample_{args.sample_idx:06d}_drug2gene_8heads.npy')
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"Weight file not found: {weight_path}")
    weights = np.load(weight_path)
    num_heads, num_atoms, num_genes = weights.shape
    print(f"Loaded weights: {weight_path}")
    print(f"  Shape: (heads={num_heads}, atoms={num_atoms}, genes={num_genes})")

    # Load gene names
    cell_id = get_cell_id_from_metadata(args.weight_dir, args.sample_idx)
    print(f"Cell line ID: {cell_id}")
    gene_names = load_gene_names_from_pt(cell_id)
    ensembl_to_symbol = load_ensembl_to_symbol_map()
    gene_symbols = [ensembl_to_symbol.get(ensg, ensg) for ensg in gene_names]
    symbol_to_ensg = {v: k for k, v in ensembl_to_symbol.items()}
    print(f"Successfully loaded {len(gene_symbols)} gene names")


    # Compute total gene attention weights
    gene_attn = weights.sum(axis=0).sum(axis=0)

    # Print top 20 gene_attn values before sorting
    print("\n========== paper_drug2gene.py: gene_attn Top20 ==========")
    sorted_indices = np.argsort(gene_attn)[::-1]
    for i, idx in enumerate(sorted_indices[:20]):
        print(f"  {i + 1:2d}. {gene_symbols[idx]}: {gene_attn[idx]:.6f} (idx={idx})")
    print("=" * 50)

    # Force target gene to the end
    sorted_indices = np.argsort(gene_attn)[::-1]
    target_gene_specified = args.target_genes[0] if args.target_genes else None
    target_idx = None
    if target_gene_specified:
        ensg = symbol_to_ensg.get(target_gene_specified, target_gene_specified)
        if ensg in gene_names:
            target_idx = gene_names.index(ensg)
            if target_idx in sorted_indices:
                sorted_indices = sorted_indices[sorted_indices != target_idx]
                print(f"  Target gene {target_gene_specified} was in Top list, removed and will be appended")
            else:
                print(f"  Target gene {target_gene_specified} not in Top list, will be appended to end")
        else:
            print(f"  Warning: {target_gene_specified} not in this cell line's gene list, ignoring")
            target_idx = None

    if target_idx is not None:
        top_indices = sorted_indices[:args.top_genes - 1].tolist()
        top_indices.append(target_idx)
    else:
        top_indices = sorted_indices[:args.top_genes].tolist()

    # Print gene list
    print(f"\nFinal displayed gene list (target gene at end):")
    for i, idx in enumerate(top_indices):
        sym = gene_symbols[idx]
        marker = " <-- TARGET (end)" if target_idx is not None and idx == target_idx else ""
        print(f"  {i+1:2d}. {sym}: {gene_attn[idx]:.6f}{marker}")

    gene_symbols_sorted = [gene_symbols[i] for i in top_indices]
    mats_sorted = weights[:, :, top_indices]

    # Statistics
    all_weights = weights.flatten()
    print("\n========== Weight Statistics ==========")
    print(f"  Total weights: {len(all_weights)}")
    print(f"  Min:   {all_weights.min():.8f}")
    print(f"  Max:   {all_weights.max():.8f}")
    print(f"  Mean:  {all_weights.mean():.8f}")
    print(f"  Std:   {all_weights.std():.8f}")
    percentiles = [1, 5, 25, 50, 75, 95, 99]
    print("  Percentiles:")
    for p in percentiles:
        val = np.percentile(all_weights, p)
        print(f"    P{p:2d}: {val:.8f}")
    print("  Average weight per head:")
    for h in range(num_heads):
        print(f"    Head {h+1}: {weights[h].mean():.8f}")
    print("==================================\n")

    # Truncated color scale
    vmin = np.percentile(all_weights, args.lower_percentile)
    vmax = np.percentile(all_weights, args.upper_percentile)
    print(f"Percentile truncation thresholds: vmin={vmin:.6f} (P{args.lower_percentile}), vmax={vmax:.6f} (P{args.upper_percentile})")
    masked_weights = all_weights[(all_weights >= vmin) & (all_weights <= vmax)]
    median_val = np.median(masked_weights)
    print(f"Median within truncated range: {median_val:.6f}")

    # Multi-head classification
    head_types = ['Broad'] * num_heads
    head_types[4] = 'Focused'

    # Create figure
    fig = plt.figure(figsize=(24, 12), facecolor='white')
    gs = GridSpec(2, 5, width_ratios=[1, 1, 1, 1, 0.06],
                  wspace=0.3, hspace=0.35, top=0.92, bottom=0.12)
    axes = [fig.add_subplot(gs[i // 4, i % 4]) for i in range(8)]
    cbar_ax = fig.add_subplot(gs[:, 4])

    # Draw subplots
    for i in range(8):
        ax = axes[i]
        data = mats_sorted[i]

        im = ax.imshow(data, cmap='Reds',
                       norm=PowerNorm(gamma=args.gamma, vmin=vmin, vmax=vmax),
                       aspect='auto', interpolation='nearest')

        head_label = head_types[i]
        title_text = f"({chr(97 + i)}) Head {i+1}"
        ax.set_title(title_text, loc='left', y=1.02, fontsize=11, fontweight='bold')

        # Y-axis (only first column)
        if i % 4 == 0:
            y_ticks = [0, num_atoms // 2, num_atoms - 1]
            y_ticks = sorted(set(y_ticks))
            ax.set_yticks(y_ticks)
            ax.set_yticklabels([str(t) for t in y_ticks], fontsize=8)
            ax.tick_params(axis='y', length=4, direction='out')
            ax.set_ylabel('Atom Index', fontsize=10, fontweight='bold', labelpad=10)
        else:
            ax.set_yticks([])
            ax.set_ylabel('')

        # X-axis shows all gene names with tick marks
        show_idx = list(range(args.top_genes))
        ax.set_xticks(show_idx)
        labels = gene_symbols_sorted
        fontsize = 8 if args.top_genes <= 20 else max(5, 8 - (args.top_genes - 20) // 5)
        ax.set_xticklabels(labels, rotation=90, ha='center', fontsize=fontsize)
        ax.tick_params(axis='x', length=4, direction='out')
        # ax.set_xlabel('Gene', fontsize=11, fontweight='bold', labelpad=10)

        # ---- Highlight first column (highest-weight gene) in red ----
        tick_labels = ax.get_xticklabels()
        if tick_labels:
            tick_labels[0].set_color('red')
            tick_labels[0].set_fontweight('bold')

    # Color bar
    cbar = fig.colorbar(im, cax=cbar_ax, extend='max')
    cbar.set_ticks([vmin, median_val, vmax])
    tick_labels_cb = [sci_notation(vmin, 2), sci_notation(median_val, 2), sci_notation(vmax, 2)]
    if vmin == 0:
        tick_labels_cb[0] = '0 (P1)'
    tick_labels_cb[1] = tick_labels_cb[1] + '\n  (Median)'
    tick_labels_cb[2] = tick_labels_cb[2] + '\n     (P99)'
    cbar.set_ticklabels(tick_labels_cb)
    cbar.set_label('Attention Weight\nTruncated P1-P99', fontsize=12, fontweight='bold', labelpad=1)
    cbar.ax.tick_params(labelsize=9)


    fig.suptitle('Multi-head Cross-Attention between Drug Atoms and Target Genes',
                 fontsize=18, fontweight='bold', y=0.98)

    plt.subplots_adjust(top=0.94, bottom=0.3, left=0.10, right=0.94)

    # Save
    output_path = os.path.join(args.output_dir, f'Figure5A_multihead_2x4_sample{args.sample_idx:06d}.png')
    plt.savefig(output_path, dpi=args.dpi, facecolor='white')
    print(f"Optimized figure saved: {output_path}")

    pdf_path = output_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, facecolor='white')
    print(f"PDF vector version: {pdf_path}")

    plt.close()


if __name__ == '__main__':
    main()
