# bca_hfp/analysis/fig_cv_training_curves.py
"""plot the 5-fold CV training history as line charts.

input is attention_drug_training_history.csv written by train_cv (fold / epoch / metrics).
output is a 2x2 figure: (a) fold mean +- 1sd of train/val R2  (b) val R2 per fold
            (c) val Pearson per fold                  (d) val RMSE per fold

usage (from the repo root):
python bca_hfp/analysis/fig_cv_training_curves.py \
  --history "../Result/My_model/CV/<run>/results/attention_drug_training_history.csv" \
  --out_prefix figure_cv_training_seed42
a run dir can also be passed, results/..._training_history.csv is appended automatically
"""

import argparse
import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from bca_hfp.config.base_config import PROJECT_ROOT, root_dir

plt.rcParams['font.family'] = 'serif'
# Liberation Serif is a metric-compatible substitute for Times New Roman; Nimbus Roman ships
# only .t1 (Type1) Regular on some distros, which matplotlib can't load, so it is not a fallback
plt.rcParams['font.serif'] = ['Times New Roman', 'Liberation Serif', 'DejaVu Serif']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['lines.linewidth'] = 1.5
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.alpha'] = 0.2
plt.rcParams['grid.linestyle'] = '-'

FOLD_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
               '#8c564b', '#e377c2', '#17becf']


def resolve_history(path):
    """accepts a run dir, a results dir or the csv itself."""
    if os.path.isfile(path):
        return path
    for pat in ('results/*training_history.csv', '*training_history.csv'):
        hits = sorted(glob.glob(os.path.join(path, pat)))
        if hits:
            return hits[0]
    raise SystemExit(f'no *training_history.csv under {path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--history', required=True)
    ap.add_argument('--out_prefix', default='figure_cv_training_curves')
    ap.add_argument('--title', default='5-fold Cross-Validation Training Curves')
    ap.add_argument('--out_dir', default=root_dir('Picture'))
    ap.add_argument('--skip_first', action='store_true',
                    help='skip epoch 1 (warm-up) so the y-axis is not squeezed by -1.5; say so in the caption')
    args = ap.parse_args()

    csv = resolve_history(args.history)
    df = pd.read_csv(csv)
    if 'fold' not in df.columns:
        df['fold'] = 0
    if args.skip_first:
        df = df[df.epoch > df.epoch.min()].copy()
    df = df.sort_values(['fold', 'epoch']).reset_index(drop=True)
    folds = sorted(df['fold'].unique())
    print(f'[load] {csv}')
    print(f'[load] folds={folds}  epochs={df.epoch.min()}..{df.epoch.max()}  rows={len(df)}')
    if len(folds) < 5:
        print(f'[warn] only {len(folds)} folds read. train_cv overwrites the history CSV every fold, '
              f'so only the last fold survives; rebuild all folds from the log with parse_cv_log.py first.')

    os.makedirs(args.out_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.6), dpi=300)
    fig.suptitle(args.title, fontsize=13, y=0.98)

    # ---------- (a) fold mean ± 1sd: train / val R2 ----------
    ax = axes[0, 0]
    m_tr = df.groupby('epoch')['train_r2'].mean()
    m_va = df.groupby('epoch')['val_r2'].mean()
    s_va = df.groupby('epoch')['val_r2'].std(ddof=1)
    ax.plot(m_tr.index, m_tr.values, color='#0033ff', label='Train R² (mean)')
    ax.plot(m_va.index, m_va.values, color='#ff2222', label='Validation R² (mean)')
    ax.fill_between(m_va.index, m_va - s_va, m_va + s_va,
                    color='#ff2222', alpha=0.15, linewidth=0, label='±1 s.d.')
    ax.axhline(0, color='0.6', lw=0.8, ls='--')
    ax.set_title('R² on Training and Validation Sets')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('R²')
    ax.grid(True)
    ax.legend(frameon=False, loc='lower right')

    # ---------- (b) val R2 per fold ----------
    ax = axes[0, 1]
    for i, f in enumerate(folds):
        s = df[df.fold == f]
        ax.plot(s.epoch, s.val_r2, color=FOLD_COLORS[i % len(FOLD_COLORS)],
                lw=1.0, alpha=0.8, label=f'Fold {int(f)}')
    ax.plot(m_va.index, m_va.values, color='black', lw=2.0, label='Mean')
    best_ep = int(m_va.idxmax())
    best_v = float(m_va.max())
    ax.scatter([best_ep], [best_v], color='red', s=35, zorder=6,
               edgecolor='white', linewidth=0.5)
    ax.annotate(f'Best mean: {best_v:.4f} @ epoch {best_ep}',
                xy=(best_ep, best_v), xytext=(0.28, 0.06),
                textcoords='axes fraction', fontsize=8,
                arrowprops=dict(arrowstyle='-|>', color='black', lw=0.7, alpha=0.8))
    ax.axhline(0, color='0.6', lw=0.8, ls='--')
    ax.set_title('Validation R² by Fold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation R²')
    ax.grid(True)
    ax.legend(frameon=False, loc='lower right', ncol=2)

    # ---------- (c) val Pearson per fold ----------
    ax = axes[1, 0]
    for i, f in enumerate(folds):
        s = df[df.fold == f]
        ax.plot(s.epoch, s.val_pearson, color=FOLD_COLORS[i % len(FOLD_COLORS)],
                lw=1.0, alpha=0.8, label=f'Fold {int(f)}')
    m_pe = df.groupby('epoch')['val_pearson'].mean()
    ax.plot(m_pe.index, m_pe.values, color='black', lw=2.0, label='Mean')
    ax.set_title('Validation Pearson Correlation')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Pearson r')
    ax.grid(True)
    ax.legend(frameon=False, loc='lower right')

    # ---------- (d) val RMSE per fold ----------
    ax = axes[1, 1]
    for i, f in enumerate(folds):
        s = df[df.fold == f]
        ax.plot(s.epoch, s.val_rmse, color=FOLD_COLORS[i % len(FOLD_COLORS)],
                lw=1.0, alpha=0.8, label=f'Fold {int(f)}')
    m_rm = df.groupby('epoch')['val_rmse'].mean()
    ax.plot(m_rm.index, m_rm.values, color='black', lw=2.0, label='Mean')
    ax.set_title('Validation RMSE')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('RMSE')
    ax.grid(True)
    ax.legend(frameon=False, loc='upper right')

    plt.tight_layout(rect=[0, 0, 1, 0.955])
    for ax in axes.ravel():
        ax.set_xticks(range(2, int(df.epoch.max()) + 1, 2))
    pdf = os.path.join(args.out_dir, f'{args.out_prefix}.pdf')
    png = os.path.join(args.out_dir, f'{args.out_prefix}.png')
    plt.savefig(pdf, bbox_inches='tight')
    plt.savefig(png, dpi=600, bbox_inches='tight')
    print(f'[save] {pdf}')
    print(f'[save] {png}')

    # ---------- text summary ----------
    print('\n[summary] best val R2 per fold:')
    for f in folds:
        s = df[df.fold == f]
        i = s.val_r2.idxmax()
        print(f'  fold{int(f)}: R²={s.loc[i, "val_r2"]:+.4f} @ epoch {int(s.loc[i, "epoch"])}'
              f'  | last epoch R²={s.val_r2.iloc[-1]:+.4f}'
              f'  Pearson={s.val_pearson.iloc[-1]:.4f}'
              f'  RMSE={s.val_rmse.iloc[-1]:.4f}')
    print(f'[summary] fold mean: best {best_v:+.4f} @ epoch {best_ep}; '
          f'last epoch R²={m_va.iloc[-1]:+.4f}  Pearson={m_pe.iloc[-1]:.4f}  RMSE={m_rm.iloc[-1]:.4f}')


if __name__ == '__main__':
    main()
