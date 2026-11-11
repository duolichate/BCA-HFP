# bca_hfp/analysis/fig_controls.py
"""R3-9 control figure: attention vs expression / cross-drug attention agreement / input removal.

all three panels are built from CSVs written by the control scripts, plotted locally (english labels).
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


def panel_attn_vs_expr(ax, csv_path):
    df = pd.read_csv(csv_path)
    key = ['RAB21', 'SURF6', 'PSMB2', 'CD2BP2', 'FAM216A']
    rho_v, p_v = spearmanr(df['expr_value'], df['attention'])
    rho_r, _ = spearmanr(df['attention'], df['expr_rank'])
    ax.scatter(df['expr_value'], df['attention'], s=6, c='#9a9a9a', alpha=0.5,
               edgecolors='none', label='all genes')
    cmap = {'RAB21': '#d62728', 'SURF6': '#d62728', 'PSMB2': '#d62728',
            'CD2BP2': '#ff7f0e', 'FAM216A': '#ff7f0e'}
    for s in key:
        sub = df[df['symbol'] == s]
        if not len(sub):
            continue
        ax.scatter(sub['expr_value'], sub['attention'], s=48,
                   c=cmap[s], edgecolors='k', linewidths=0.5, zorder=5, label=s)
    ax.set_xlabel('log-expression value')
    ax.set_ylabel('per-gene attention')
    ax.set_title('A. Attention vs expression\n'
                 f'Spearman(attn, expr)={rho_v:+.3f} (p={p_v:.2f})\n'
                 f'Spearman(attn, expr_rank)={rho_r:+.3f}')
    ax.legend(loc='upper right', fontsize=6, frameon=False)
    ax.spines[['top', 'right']].set_visible(False)


def panel_drug_stability(ax, csv_path, ref_col='1862'):
    df = pd.read_csv(csv_path)
    drug_cols = [c for c in df.columns if c not in ('gene_ensg', 'symbol')]
    ref = df[ref_col].values
    rhos = []
    for c in drug_cols:
        if c == ref_col:
            continue
        rhos.append(spearmanr(ref, df[c].values)[0])
    rhos = np.array(rhos)
    ax.hist(rhos, bins=30, color='#4c78a8', edgecolor='white')
    ax.axvline(rhos.mean(), color='#d62728', lw=1.5,
               label=f'mean {rhos.mean():.3f}')
    ax.set_xlabel('Spearman vs MG-132 (per-gene attention ranking)')
    ax.set_ylabel('number of drugs')
    ax.set_title(f'B. Cross-drug stability\n'
                 f'mean={rhos.mean():.3f}, std={rhos.std():.3f} (n={len(rhos)})')
    ax.legend(loc='upper left', fontsize=7, frameon=False)
    ax.spines[['top', 'right']].set_visible(False)


def panel_input_removal(ax, csv_path, topn=14):
    df = pd.read_csv(csv_path)
    gene = df[df['removed_type'] == 'gene'].copy()
    gene['abs_delta'] = gene['delta'].abs()
    gene = gene.sort_values('abs_delta', ascending=False)
    ctrl = df[df['removed_type'] == 'gene_ctrl']['delta'].abs()
    show = gene.head(topn).iloc[::-1]
    names = show['removed_name'].tolist()
    vals = show['abs_delta'].tolist()
    colors = ['#d62728' if n in ('RAB21', 'SURF6', 'PSMB2') else '#4c78a8'
              for n in names]
    ax.barh(range(len(vals)), vals, color=colors)
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel('|delta pred| after single-gene removal')
    ax.set_title(f'C. Input removal (genes)\n'
                 f'random ctrl mean={ctrl.mean():.4f}, max={ctrl.max():.4f}')
    ax.axvline(ctrl.mean(), color='#4c78a8', ls='--', lw=1.2,
               label='random mean')
    ax.axvline(ctrl.max(), color='#888', ls=':', lw=1.2, label='random max')
    ax.legend(loc='lower right', fontsize=7, frameon=False)
    ax.spines[['top', 'right']].set_visible(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--control_dir', default='Result/controls')
    ap.add_argument('--out', default=None)
    ap.add_argument('--dpi', type=int, default=200)
    args = ap.parse_args()

    out = args.out or os.path.join(args.control_dir, 'controls.png')
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    panel_attn_vs_expr(axes[0], os.path.join(args.control_dir, 'attn_vs_expr_1862_910919.csv'))
    panel_drug_stability(axes[1], os.path.join(args.control_dir, 'drugs_attn_910919.csv'))
    panel_input_removal(axes[2], os.path.join(args.control_dir, 'input_removal_1862_910919.csv'))
    fig.tight_layout()
    fig.savefig(out, dpi=args.dpi, bbox_inches='tight')
    print(f'saved: {out}')


if __name__ == '__main__':
    main()