# -*- coding: utf-8 -*-
"""Manuscript figure script (G1 main model).

Four figures:
  fig2_tanimoto   nearest-neighbour Tanimoto of each test drug to training
                  (replaces the old PCA panel)
  fig4_baselines  main model vs SOTA vs trivial baselines: raw -> clamp dumbbell
  fig5_fp_bits    fingerprint bit-width ablation (three-fold and five-fold
                  labelled separately)
  fig6_slack      concentration-window slack sensitivity

Usage (cwd under 4_DrugResponse_BCAHFP):
  python -m bca_hfp.figures.plot_manuscript_figs            # all
  python -m bca_hfp.figures.plot_manuscript_figs --which fig2,fig6
"""
import argparse
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

R36 = os.path.join(RESULT, '3_diagnostics', '1_model_level',
                   'r36_v1gc_s42_5090_20261108')
NN_CSV = os.path.join(R36, 'nn_tanimoto.csv')
SLACK_CSV = os.path.join(R36, 'raw_predictions', 'slack_sweep_pooled.csv')

plt.rcParams.update({
    'font.size': 8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.8,
    'figure.dpi': 130,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
})

C_OURS = '#C44E52'
C_SOTA = '#4C72B0'
C_BASE = '#8C8C8C'


# Journal minimum is 600 dpi. dpi is back-solved from a 7 in full-page width so
# PNG width stays >= 7*600=4200; tight bbox crops vary by figure, so remeasure
# after saving and top up.
MIN_PRINT_WIDTH_IN = 7.0
MIN_DPI = 600
MIN_WIDTH_PX = int(MIN_PRINT_WIDTH_IN * MIN_DPI)


def _save(fig, name):
    os.makedirs(PICTURE, exist_ok=True)
    w_in = fig.get_size_inches()[0]
    dpi = max(MIN_DPI, MIN_WIDTH_PX / (w_in * 0.90))
    png = os.path.join(PICTURE, f'{name}.png')
    fig.savefig(png, dpi=dpi)
    from PIL import Image
    with Image.open(png) as im:
        actual_w = im.size[0]
    if actual_w < MIN_WIDTH_PX:
        dpi *= MIN_WIDTH_PX * 1.02 / actual_w
        fig.savefig(png, dpi=dpi)
    fig.savefig(os.path.join(PICTURE, f'{name}.pdf'))
    plt.close(fig)
    print(f'  -> {os.path.join(PICTURE, name)}.png/.pdf  (dpi={dpi:.0f})')


def fig4_tanimoto():
    """Nearest-neighbour Tanimoto: train-max similarity per test drug when held out (242 drugs)."""
    if not os.path.exists(NN_CSV):
        raise SystemExit(f'missing {NN_CSV}')
    d = pd.read_csv(NN_CSV)
    s = d['nn_sim'].values

    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    bins = np.arange(0, 1.05, 0.05)
    ax.hist(s, bins=bins, color=C_SOTA, edgecolor='white', alpha=0.85)
    med = float(np.median(s))
    ax.axvline(med, color=C_OURS, ls='--', lw=1.2)
    ymax = ax.get_ylim()[1]
    ax.text(med + 0.02, ymax * 0.92, f'median = {med:.3f}', color=C_OURS, fontsize=7)
    ax.text(0.98, ymax * 0.92,
            f'>0.4: {100*(s > 0.4).mean():.1f}%\n>0.8: {100*(s > 0.8).mean():.1f}%',
            ha='right', va='top', fontsize=7)
    ax.set_xlim(0, 1)
    ax.set_xlabel('Max Tanimoto similarity to training drugs')
    ax.set_ylabel('Number of test drugs')
    _save(fig, 'fig4')


GATE_CSV = os.path.join(RESULT, '3_diagnostics', '1_model_level',
                        'gate_routing_v1gc_s42_swa_20261109', 'gate_weights_all.csv')
DRUG_RES = os.path.join(os.path.dirname(RESULT), '1_Data', 'Drug', 'GDSC_RES.csv')


def fig5_gate():
    """Gating weights: 3 sensitive/resistant pairs (same drug / broad-spectrum / targeted)."""
    picks = [
        ('909256', '1862', 'OCUB-M-MG-132', 'Sensitive'),
        ('907295', '1862', 'DMS-53-MG-132', 'Resistant'),
        ('1331048', '1190', 'VAL-GEMCITABINE', 'Sensitive'),
        ('910694', '1190', 'LN-405-GEMCITABINE', 'Resistant'),
        ('687980', '1011', 'COR-L311-NAVITOCLAX', 'Sensitive'),
        ('1240199', '1011', 'OVKATE-NAVITOCLAX', 'Resistant'),
    ]
    if not os.path.exists(GATE_CSV):
        raise SystemExit(f'missing {GATE_CSV}')
    d = pd.read_csv(GATE_CSV)
    d['drug_id'] = d['drug_id'].astype(str)
    d['cell_std_name'] = d['cell_std_name'].astype(str)

    rows, labels = [], []
    for cell, drug, name, status in picks:
        sub = d[(d['cell_std_name'] == cell) & (d['drug_id'] == drug)]
        if sub.empty:
            raise SystemExit(f'cell={cell} drug={drug} not found')
        r = sub.iloc[0]
        rows.append([r['gene_pooled'], r['global_gene'], r['drug_global']])
        labels.append(f'{name}\n({status})')
    M = np.array(rows, dtype=float)

    fig, ax = plt.subplots(figsize=(6.6, 3.5))
    x = np.arange(len(M))
    colors = ['#C0392B', '#2980B9', '#7F8C8D']
    names = ['Gene Pooled\n(Micro Target)', 'Global Gene\n(Macro Transcriptome)',
             'Drug Global\n(Macro Structure)']
    bottom = np.zeros(len(M))
    for j in range(3):
        ax.bar(x, M[:, j], bottom=bottom, color=colors[j], width=0.7,
               edgecolor='black', linewidth=0.8, label=names[j])
        for i, v in enumerate(M[:, j]):
            if v > 0.12:
                ax.text(x[i], bottom[i] + v / 2, f'{v * 100:.0f}%', ha='center',
                        va='center', color='white', fontsize=7, fontweight='bold')
        bottom += M[:, j]
    for xv in (1.5, 3.5):
        ax.axvline(xv, color='#888888', lw=0.8, ls=':')
    ax.set_xticks(x)
    ax.set_xticklabels([])
    for i, lbl in enumerate(labels):
        ax.text(x[i], -0.16, lbl, rotation=32, ha='center', va='top',
                rotation_mode='anchor', transform=ax.get_xaxis_transform(),
                fontsize=7, clip_on=False)
    ax.set_xlim(-0.6, 5.6)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Gating weight')
    ax.legend(ncol=3, fontsize=6.8, frameon=False, loc='upper center',
              bbox_to_anchor=(0.5, 1.20))
    _save(fig, 'fig5')


def fig2_baselines():
    """raw -> clamp dumbbell. Values from supplementary table A (same five folds, same labels)."""
    rows = [
        ('BCA-HFP', 0.2301, 0.4134, 'ours'),
        ('tCNNS', 0.2600, 0.3845, 'sota'),
        ('PaccMann', 0.1796, 0.3506, 'sota'),
        ('TransCDR', 0.1090, 0.3375, 'sota'),
        ('GraTransDRP', 0.0896, 0.2976, 'sota'),
        ('GraphDRP', 0.0738, 0.2067, 'sota'),
        ('GBM (raw expr 2048)', 0.1187, 0.2812, 'base'),
        ('GBM (GeneFormer 256)', 0.1549, 0.3097, 'base'),
        ('Cell-line mean', 0.0902, 0.3184, 'base'),
        ('Fingerprint ridge', 0.0395, 0.2274, 'base'),
        ('Nearest train drug', -0.2980, -0.0206, 'base'),
    ]
    rows.sort(key=lambda r: r[2])
    cmap = {'ours': C_OURS, 'sota': C_SOTA, 'base': C_BASE}

    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    for i, (name, raw, win, kind) in enumerate(rows):
        c = cmap[kind]
        lw = 2.4 if kind == 'ours' else 1.6
        ax.plot([raw, win], [i, i], color=c, lw=lw, alpha=0.85, zorder=1)
        ax.scatter([raw], [i], color=c, marker='o', s=22, zorder=2)
        ax.scatter([win], [i], color=c, marker='D', s=26, zorder=2)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=7)
    for lbl, c in zip(ax.get_yticklabels(), [cmap[r[3]] for r in rows]):
        if lbl.get_text().startswith('BCA-HFP'):
            lbl.set_color(c)
            lbl.set_fontweight('bold')
    ax.axvline(0, color='k', lw=0.6, ls=':')
    ax.set_xlabel('Pooled OOF R²')
    ax.set_title('circle = raw prediction, diamond = after concentration-window clamp',
                 fontsize=7)
    ax.grid(axis='x', ls=':', lw=0.5, alpha=0.6)
    _save(fig, 'fig2')


def fig3_fp_bits():
    """Fingerprint bit width (five-fold OOF, 256-4096)."""
    bits = [256, 512, 1024, 2048, 4096]
    r2 = [0.0386, 0.2680, 0.3871, 0.4134, 0.4015]

    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ax.plot(np.log2(bits), r2, 'o-', color=C_SOTA, lw=1.6, ms=4.5)
    offsets = [(6, 0, 'left', 'center'), (5, -10, 'left', 'top'), (0, -13, 'center', 'top'),
               (0, 9, 'center', 'bottom'), (5, 8, 'left', 'bottom')]
    for (b, v), (dx, dy, ha, va) in zip(zip(bits, r2), offsets):
        ax.annotate(f'{v:.4f}', (np.log2(b), v), textcoords='offset points',
                    xytext=(dx, dy), ha=ha, va=va, fontsize=6.2, color='#444444')
    ax.set_xticks(np.log2(bits))
    ax.set_xticklabels([str(b) for b in bits])
    ax.set_xlim(np.log2(256) - 0.5, np.log2(4096) + 0.5)
    ax.set_xlabel('Morgan fingerprint bits')
    ax.set_ylabel('Pooled OOF R²')
    ax.set_ylim(0, 0.47)
    _save(fig, 'fig3')


def fig6_slack():
    """Concentration-window slack sensitivity (none = no clamp; 2.0 = training setting)."""
    if not os.path.exists(SLACK_CSV):
        raise SystemExit(f'missing {SLACK_CSV}')
    d = pd.read_csv(SLACK_CSV)
    x = np.arange(len(d))
    labels = [str(v) for v in d['slack'].values]

    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    ax.plot(x, d['R2'].values, 'o-', color=C_SOTA, lw=1.6, ms=3.2)
    hit = np.where(d['slack'].astype(str).values == '2.0')[0]
    if len(hit):
        i = int(hit[0])
        ax.scatter([x[i]], [d['R2'].values[i]], color=C_OURS, s=10, zorder=4)
        ax.annotate(f'slack = 2.0 (training)\nR² = {d["R2"].values[i]:.4f}',
                    xy=(x[i], d['R2'].values[i]),
                    xytext=(0.98, 0.97), textcoords='axes fraction',
                    ha='right', va='top', fontsize=6.5, color=C_OURS,
                    arrowprops=dict(arrowstyle='-|>', color=C_OURS, lw=0.8,
                                    shrinkA=2, shrinkB=4))
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, fontsize=6.5)
    ax.set_xlabel('Slack beyond measured concentration window (ln units)')
    ax.set_ylabel('Pooled R²')
    ax.grid(axis='y', ls=':', lw=0.5, alpha=0.6)
    _save(fig, 'fig6')


FIGS = {
    'fig2': fig2_baselines,
    'fig3': fig3_fp_bits,
    'fig4': fig4_tanimoto,
    'fig5': fig5_gate,
    'fig6': fig6_slack,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--which', default='all',
                    help='all or a comma separated subset: fig2,fig4,fig5,fig6')
    args = ap.parse_args()

    want = list(FIGS) if args.which == 'all' else [s.strip() for s in args.which.split(',')]
    print(f'output dir: {PICTURE}')
    for k in want:
        if k not in FIGS:
            raise SystemExit(f'unknown figure name: {k} (choices {list(FIGS)})')
        print(f'[{k}]')
        FIGS[k]()


if __name__ == '__main__':
    main()
