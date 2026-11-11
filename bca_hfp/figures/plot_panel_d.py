# -*- coding: utf-8 -*-
"""Two replacement panels for architecture figure D (small size, for direct PPT use).

panelD_model_performance : G1 predicted vs observed scatter (R2=0.4134)
panelD_gating_sens_res   : mean modality gating weights for the sensitive and
                           resistant groups (replaces Key functional atoms)
"""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from bca_hfp.config.base_config import root_dir

RESULT = root_dir('Result')
ROOT = os.path.dirname(RESULT)
PICTURE = os.path.join(ROOT, '6_Picture', '1_manuscript')

OOF = os.path.join(RESULT, '3_diagnostics', '1_model_level',
                   'r36_v1gc_s42_5090_20261108', 'oof_swa.csv')
GATE = os.path.join(RESULT, '3_diagnostics', '1_model_level',
                    'gate_routing_v1gc_merged', 'gate_weights_all.csv')

plt.rcParams.update({
    'font.size': 6.5,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'figure.dpi': 200,
    # page is exported at 600 dpi inside the architecture PPT, so small panels
    # need pixel headroom
    'savefig.dpi': 1200,
    'savefig.bbox': 'tight',
})

COLORS = ['#C0392B', '#2980B9', '#7F8C8D']


def _save(fig, name):
    os.makedirs(PICTURE, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(PICTURE, f'{name}.{ext}'))
    plt.close(fig)
    print(f'  -> {os.path.join(PICTURE, name)}.png/.pdf')


def panel_model_performance(use_raw=False):
    if use_raw:
        src = os.path.join(os.path.dirname(OOF), 'raw_predictions', 'oof_raw_predictions.csv')
        d = pd.read_csv(src)
        y = d['y'].values.astype(float)
        p = d['p_raw'].values.astype(float)
        name = 'panelD_model_performance_raw'
    else:
        d = pd.read_csv(OOF)
        y = d['true_label'].values.astype(float)
        p = d['predicted'].values.astype(float)
        name = 'panelD_model_performance'
    r2 = 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    pear = float(np.corrcoef(y, p)[0, 1])

    fig, ax = plt.subplots(figsize=(1.9, 1.7))
    ax.hexbin(y, p, gridsize=36, cmap='Blues', mincnt=1, linewidths=0, bins='log')
    lim = [-8, 9]
    ax.plot(lim, lim, '--', color='#E24A33', lw=0.9)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel('Observed ln(IC50)', fontsize=6.5)
    ax.set_ylabel('Predicted ln(IC50)', fontsize=6.5)
    ax.tick_params(labelsize=6, length=2)
    ax.text(0.04, 0.96, f'R² = {r2:.3f}\nPearson = {pear:.3f}',
            transform=ax.transAxes, va='top', ha='left', fontsize=6)
    _save(fig, name)


def panel_gating_sens_res():
    g = pd.read_csv(GATE)
    cols = ['gene_pooled', 'global_gene', 'drug_global']
    g['pct'] = g.groupby('drug_id')['true_label'].rank(pct=True)
    sen = g.loc[g['pct'] < 0.2, cols].mean().values
    res = g.loc[g['pct'] > 0.8, cols].mean().values
    M = np.vstack([sen, res])

    fig, ax = plt.subplots(figsize=(1.9, 1.7))
    x = np.arange(2)
    bottom = np.zeros(2)
    for j, c in enumerate(COLORS):
        ax.bar(x, M[:, j], bottom=bottom, color=c, width=0.55,
               edgecolor='black', linewidth=0.6)
        for i, v in enumerate(M[:, j]):
            ax.text(x[i], bottom[i] + v / 2, f'{v * 100:.0f}%', ha='center',
                    va='center', color='white', fontsize=6, fontweight='bold')
        bottom += M[:, j]
    ax.set_xticks(x)
    ax.set_xticklabels(['Sensitive', 'Resistant'], fontsize=6.5)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Gating weight', fontsize=6.5)
    ax.tick_params(labelsize=6, length=2)
    ax.legend(['Gene Pooled', 'Global Gene', 'Drug Global'], fontsize=5.2,
              frameon=False, loc='upper center', bbox_to_anchor=(0.5, 1.22), ncol=3,
              handlelength=1.0, columnspacing=0.8)
    _save(fig, 'panelD_gating_sens_res')
    print(f'    sensitive: {np.round(sen, 3)}  resistant: {np.round(res, 3)}')


if __name__ == '__main__':
    panel_model_performance()
    panel_model_performance(use_raw=True)
    panel_gating_sens_res()
