# bca_hfp/analysis/master_table.py
"""collect review numbers scattered across files into one table with provenance.

reads the artifacts under Result/oof, computes and writes master_table.csv:
    section | item | metric | value | source

every row names its source file, so numbers from different rounds don't get mixed; the paper
should quote this table only.

usage:
python bca_hfp/analysis/master_table.py --dir Result/oof \
    --out Result/oof/master_table.csv
"""

import argparse
import os

import numpy as np
import pandas as pd


def r2(y, p):
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def rmse(y, p):
    return float(np.sqrt(((y - p) ** 2).mean()))


def pearson(y, p):
    if y.std() < 1e-12 or p.std() < 1e-12:
        return np.nan
    return float(np.corrcoef(y, p)[0, 1])


def spearman(y, p):
    yr = pd.Series(y).rank().values
    pr = pd.Series(p).rank().values
    if yr.std() < 1e-12 or pr.std() < 1e-12:
        return np.nan
    return float(np.corrcoef(yr, pr)[0, 1])


def decomp(df, gcol, ycol='y', pcol='p'):
    """between-drug / within-drug variance decomposition + drug-level correlation."""
    g = df.groupby(gcol)
    n = g.size()
    my = g[ycol].mean()
    mp = g[pcol].mean()
    gm = df[ycol].mean()
    SL = float((n * (my - gm) ** 2).sum())
    SY = float(((df[ycol] - gm) ** 2).sum())
    SW = SY - SL
    R2L = float(1 - (n * (mp - my) ** 2).sum() / SL)
    yc = df[ycol] - g[ycol].transform('mean')
    pc = df[pcol] - g[pcol].transform('mean')
    R2W = float(1 - ((yc - pc) ** 2).sum() / (yc ** 2).sum())
    return dict(r2_level=R2L, r2_within=R2W,
                ss_level_share=SL / SY, ss_within_share=SW / SY,
                drug_level_pearson=pearson(my.values, mp.values),
                drug_level_spearman=spearman(my.values, mp.values),
                n_drug=int(len(my)))


def full_block(tag, df, gcol, pcol, source):
    y = df['y'].values
    p = df[pcol].values
    d = decomp(df, gcol, 'y', pcol)
    rows = [
        ('overall', f'R2', r2(y, p)),
        ('overall', f'RMSE', rmse(y, p)),
        ('overall', f'Pearson', pearson(y, p)),
        ('overall', f'Spearman', spearman(y, p)),
        ('decomp', f'R2_level', d['r2_level']),
        ('decomp', f'R2_within', d['r2_within']),
        ('decomp', f'SS_level_share', d['ss_level_share']),
        ('decomp', f'SS_within_share', d['ss_within_share']),
        ('decomp', f'drug_level_pearson', d['drug_level_pearson']),
        ('decomp', f'drug_level_spearman', d['drug_level_spearman']),
    ]
    return [dict(section=s, item=tag, metric=m, value=v, source=source,
                 n_rows=len(df), n_drug=d['n_drug']) for s, m, v in rows]


def read_if(path, label, warn):
    if not os.path.exists(path):
        warn.append(f'missing {label}: {path}')
        return None
    return pd.read_csv(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True, help='artifact directory')
    ap.add_argument('--out', default='', help='output csv (default <dir>/master_table.csv)')
    args = ap.parse_args()
    d = args.dir
    out = args.out or os.path.join(d, 'master_table.csv')
    warn = []
    rows = []

    # ---------- main model OOF ----------
    oof = read_if(os.path.join(d, 'oof_predictions.csv'), 'E7 OOF predictions', warn)
    gmap = None
    if oof is not None:
        oof['DRUG_ID'] = oof['DRUG_ID'].astype(str)
        gmap = oof.drop_duplicates('DRUG_ID').set_index('DRUG_ID')['GROUP']
        rows += full_block('BCA-HFP (E7, swa)', oof, 'GROUP', 'p', 'oof_predictions.csv')

    # ---------- two GBM versions ----------
    for tag, fn in [('GBM-Expr', 'gbm_predictions_expr.csv'),
                    ('GBM-GF', 'gbm_predictions.csv')]:
        g = read_if(os.path.join(d, fn), tag, warn)
        if g is None:
            continue
        g['DRUG_ID'] = g['DRUG_ID'].astype(str)
        if gmap is not None:
            g['GROUP'] = g['DRUG_ID'].map(gmap)
            g['GROUP'] = g['GROUP'].fillna(g['DRUG_ID'])
        else:
            g['GROUP'] = g['DRUG_ID']
        pcol = 'p_win' if 'p_win' in g.columns else 'p'
        rows += full_block(tag, g, 'GROUP', pcol, fn)

    # ---------- per fold ----------
    pf = read_if(os.path.join(d, 'per_fold.csv'), 'per-fold metrics', warn)
    if pf is not None:
        for _, r in pf.iterrows():
            for m in ('r2', 'rmse', 'pearson', 'spearman'):
                if m in pf.columns:
                    rows.append(dict(section='per_fold', item=f'fold{int(r["fold"])}',
                                     metric=m, value=float(r[m]),
                                     source='per_fold.csv', n_rows=int(r.get('n', 0)),
                                     n_drug=int(r.get('n_drug', 0))))
        rows.append(dict(section='per_fold', item='mean', metric='r2',
                         value=float(pf.r2.mean()), source='per_fold.csv',
                         n_rows=0, n_drug=0))
        rows.append(dict(section='per_fold', item='std', metric='r2',
                         value=float(pf.r2.std(ddof=1)), source='per_fold.csv',
                         n_rows=0, n_drug=0))

    # ---------- confidence intervals ----------
    ov = read_if(os.path.join(d, 'overall.csv'), 'overall metrics', warn)
    if ov is not None:
        for _, r in ov.iterrows():
            for c, lab in [('ci_lo_drug', 'CI_lo_drug'), ('ci_hi_drug', 'CI_hi_drug')]:
                if c in ov.columns and pd.notna(r[c]):
                    rows.append(dict(section='ci', item=str(r['metric']), metric=lab,
                                     value=float(r[c]), source='overall.csv',
                                     n_rows=0, n_drug=0))

    # ---------- calibration ----------
    cal = read_if(os.path.join(d, 'calibration.csv'), 'calibration', warn)
    if cal is not None:
        for _, r in cal.iterrows():
            rows.append(dict(section='calibration', item=f'fold{int(r["fold"])}',
                             metric='r2_crossfold', value=float(r['r2_cf']),
                             source='calibration.csv', n_rows=int(r.get('n', 0)),
                             n_drug=0))
            rows.append(dict(section='calibration', item=f'fold{int(r["fold"])}',
                             metric='r2_raw', value=float(r['r2_raw']),
                             source='calibration.csv', n_rows=int(r.get('n', 0)),
                             n_drug=0))

    # ---------- trivial baselines ----------
    bl = read_if(os.path.join(d, 'baselines.csv'), 'trivial baselines', warn)
    if bl is not None:
        for _, r in bl.iterrows():
            for m in ('r2', 'rmse', 'pearson', 'spearman'):
                if m in bl.columns:
                    rows.append(dict(section='trivial', item=str(r['baseline']),
                                     metric=m, value=float(r[m]),
                                     source='baselines.csv', n_rows=int(r.get('n', 0)),
                                     n_drug=0))

    # ---------- Tanimoto ----------
    tn = read_if(os.path.join(d, 'nn_tanimoto.csv'), 'Tanimoto', warn)
    if tn is not None and 'nn_sim' in tn.columns:
        for q in (5, 25, 50, 75, 95):
            rows.append(dict(section='tanimoto', item=f'p{q}', metric='nn_sim',
                             value=float(np.percentile(tn.nn_sim, q)),
                             source='nn_tanimoto.csv', n_rows=len(tn), n_drug=0))
        for thr in (0.4, 0.5, 0.6, 0.7):
            rows.append(dict(section='tanimoto', item=f'>={thr}', metric='fraction',
                             value=float((tn.nn_sim >= thr).mean()),
                             source='nn_tanimoto.csv', n_rows=len(tn), n_drug=0))

    t = pd.DataFrame(rows)[['section', 'item', 'metric', 'value',
                            'n_rows', 'n_drug', 'source']]
    t.to_csv(out, index=False)

    print(f'[ok] {len(t)} rows -> {out}\n')
    print('core comparison (section=overall / decomp):')
    core = t[t.section.isin(['overall', 'decomp'])].pivot_table(
        index='item', columns='metric', values='value', aggfunc='first')
    cols = [c for c in ['R2', 'RMSE', 'Pearson', 'Spearman',
                        'R2_level', 'R2_within', 'drug_level_pearson'] if c in core.columns]
    print(core[cols].round(4).to_string())
    if warn:
        print('\n[warn] missing files, those rows were skipped:')
        for w in warn:
            print('  -', w)


if __name__ == '__main__':
    main()
