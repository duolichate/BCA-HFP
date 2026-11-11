# bca_hfp/analysis/fig_gate_ablation.py
"""R3-3: gate routing figure + modality ablation figure.

input is the output dir of eval_modality_ablation.py (one subdir per fold, holding
gate_weights_test_fold{k}.csv and modality_ablation_test_fold{k}.csv).
files are searched recursively and deduplicated by fold, so a fold1 kept in a separate dir is still picked up.

output:
  gate_routing.{pdf,png}        stacked bars of mean gate weight per fold + argmax share
  modality_ablation.{pdf,png}   dR2 of each modality removal (fold mean +- sd + per-fold points)
  gate_routing_summary.csv / modality_ablation_summary.csv

usage (from the repo root):
python bca_hfp/analysis/fig_gate_ablation.py \
  --diag_dir ../5_Result/3_diagnostics/1_model_level/gate_routing_v1gc_c5 \
  --out_dir  ../6_Picture/3_verification/gate_v1gc
"""

import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_ID_COLS = {'DRUG_ID', 'GDSC_ID', 'true_label', 'predicted', 'fold',
            'sample_id', 'drug_id', 'cell_id', 'cell_std_name'}


def _find(diag_dir, pattern):
    """recursive file search, skips frozen subdirs, deduplicates by fold (first in sorted order wins)."""
    files = [f for f in glob.glob(os.path.join(diag_dir, '**', pattern), recursive=True)
             if 'frozen' not in f.lower()]
    picked = {}
    for f in sorted(files):
        m = re.search(r'fold(\d+)', os.path.basename(f))
        if not m:
            continue
        k = int(m.group(1))
        picked.setdefault(k, f)
    return picked


def load_gate(diag_dir):
    out = {}
    for k, f in sorted(_find(diag_dir, 'gate_weights_test_fold*.csv').items()):
        d = pd.read_csv(f)
        cols = [c for c in d.columns
                if c not in _ID_COLS and pd.api.types.is_numeric_dtype(d[c])]
        out[k] = d[cols].astype(float)
        print(f'[gate] fold{k} <- {f}  ({len(d)} samples, modalities {cols})')
    if not out:
        raise SystemExit(f'no gate_weights_test_fold*.csv under {diag_dir}')
    return out


def load_ablation(diag_dir):
    out = {}
    for k, f in sorted(_find(diag_dir, 'modality_ablation_test_fold*.csv').items()):
        d = pd.read_csv(f)
        if 'condition' not in d.columns or 'dR2' not in d.columns:
            continue
        d = d[d.condition.astype(str).str.startswith('no_')]
        if len(d) == 0:
            continue
        out[k] = d[['condition', 'dR2']].copy()
        print(f'[abl] fold{k} <- {f}  ({len(d)} removal conditions)')
    return out


def plot_gate(gate, out_dir, formats, dpi=300):
    folds = sorted(gate)
    mods = list(gate[folds[0]].columns)
    mean_w = np.array([[gate[k][m].mean() for m in mods] for k in folds])
    argmax = [gate[k][mods].values.argmax(axis=1) for k in folds]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    x = np.arange(len(folds))
    bottom = np.zeros(len(folds))
    colors = plt.cm.Set2(np.linspace(0, 1, len(mods)))
    for j, m in enumerate(mods):
        axes[0].bar(x, mean_w[:, j], bottom=bottom, label=m, color=colors[j],
                    edgecolor='white', linewidth=0.6)
        for i, v in enumerate(mean_w[:, j]):
            if v > 0.06:
                axes[0].text(i, bottom[i] + v / 2, f'{v:.2f}', ha='center',
                             va='center', fontsize=8)
        bottom += mean_w[:, j]
    axes[0].axhline(1.0 / len(mods), color='gray', ls='--', lw=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f'fold{k}' for k in folds])
    axes[0].set_ylim(0, 1.02)
    axes[0].set_ylabel('mean gate weight')
    axes[0].set_title('Routing weight per fold')
    axes[0].legend(frameon=False, fontsize=8, ncol=len(mods))

    width = 0.8 / len(mods)
    for j, m in enumerate(mods):
        share = [(argmax[i] == j).mean() for i in range(len(folds))]
        axes[1].bar(x + (j - (len(mods) - 1) / 2) * width, share, width=width,
                    label=m, color=colors[j], edgecolor='white', linewidth=0.6)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f'fold{k}' for k in folds])
    axes[1].set_ylim(0, 1.02)
    axes[1].set_ylabel('argmax share')
    axes[1].set_title('Which modality wins per sample')
    axes[1].legend(frameon=False, fontsize=8, ncol=len(mods))

    fig.tight_layout()
    for ext in formats:
        fig.savefig(os.path.join(out_dir, f'gate_routing.{ext}'), dpi=dpi)
    plt.close(fig)

    rows = []
    for i, k in enumerate(folds):
        r = {'fold': k}
        r.update({f'w_{m}': mean_w[i, j] for j, m in enumerate(mods)})
        r.update({f'argmax_{m}': (argmax[i] == j).mean() for j, m in enumerate(mods)})
        rows.append(r)
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, 'gate_routing_summary.csv'),
                              index=False)


def plot_ablation(abl, out_dir, formats, dpi=300):
    conds = sorted({c for d in abl.values() for c in d.condition})
    val = {c: [] for c in conds}
    for k, d in abl.items():
        for _, r in d.iterrows():
            val[r.condition].append(float(r.dR2))
    conds = [c for c in conds if len(val[c]) > 0]

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    colors = plt.cm.Set2(np.linspace(0, 1, len(conds)))
    for i, c in enumerate(conds):
        v = np.array(val[c])
        ax.bar(i, v.mean(), width=0.55, color=colors[i], alpha=0.75,
               edgecolor='black', linewidth=0.6)
        ax.errorbar(i, v.mean(), yerr=v.std(ddof=1) if len(v) > 1 else 0,
                    color='black', capsize=4, lw=1)
        ax.scatter(np.full(len(v), i) + np.linspace(-0.12, 0.12, len(v)), v,
                   s=16, color='black', zorder=3)
        ax.text(i, v.mean(), f'{v.mean():+.3f}\n±{v.std(ddof=1) if len(v) > 1 else 0:.3f}',
                ha='center', va='bottom' if v.mean() >= 0 else 'top', fontsize=8)
    ax.axhline(0, color='gray', lw=0.8)
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels([c.replace('no_', 'w/o ') for c in conds])
    ax.set_ylabel(r'$\Delta R^2$ (remove - full)')
    ax.set_title('Modality removal (per fold points, mean$\\pm$sd)')
    fig.tight_layout()
    for ext in formats:
        fig.savefig(os.path.join(out_dir, f'modality_ablation.{ext}'), dpi=dpi)
    plt.close(fig)

    rows = []
    for c in conds:
        v = np.array(val[c])
        r = {'condition': c, 'mean_dR2': v.mean(),
             'sd_dR2': v.std(ddof=1) if len(v) > 1 else np.nan, 'n_folds': len(v)}
        r.update({f'fold{k}': x for k, x in zip(sorted(abl), v)})
        rows.append(r)
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, 'modality_ablation_summary.csv'),
                              index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--diag_dir', required=True,
                    help='output root of eval_modality_ablation.py (searched recursively)')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--formats', default='pdf,png')
    ap.add_argument('--dpi', type=int, default=300)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    formats = [x.strip() for x in args.formats.split(',') if x.strip()]

    gate = load_gate(args.diag_dir)
    plot_gate(gate, args.out_dir, formats, args.dpi)
    print(f'[ok] gate_routing <- {args.out_dir}')

    abl = load_ablation(args.diag_dir)
    if abl:
        plot_ablation(abl, args.out_dir, formats, args.dpi)
        print(f'[ok] modality_ablation <- {args.out_dir}')
    else:
        print('[warn] no modality_ablation_test_fold*.csv found, skip the ablation figure')


if __name__ == '__main__':
    main()
