# bca_hfp/analysis/seed_paired_diff.py
"""compare two seeds over 5 folds: per-fold paired difference + stability summary.

input is oof_predictions.csv from pool_oof_stats.py (one per seed directory);
metrics are recomputed here, so per_fold.csv column names are not relied on.

usage:
python bca_hfp/analysis/seed_paired_diff.py \
  --a Result/oof/seed42 --b Result/oof/seed43 \
  --label_a seed42 --label_b seed43 \
  --out Result/oof/seed_paired_diff.csv
"""

import argparse
import os

import numpy as np
import pandas as pd


def _pick(df, cands, what):
    for c in cands:
        if c in df.columns:
            return c
    raise KeyError(f'{what} column not found, have: {list(df.columns)}')


def load_oof(path):
    f = os.path.join(path, 'oof_predictions.csv')
    if not os.path.exists(f):
        raise FileNotFoundError(f'{f} not found')
    d = pd.read_csv(f)
    fc = _pick(d, ['fold', 'Fold'], 'fold')
    yc = _pick(d, ['y', 'true', 'true_label', 'label'], 'label')
    pc = _pick(d, ['p', 'pred', 'predicted', 'yhat'], 'prediction')
    out = pd.DataFrame({'fold': d[fc].astype(int), 'y': d[yc].astype(float),
                        'p': d[pc].astype(float)})
    return out


def metrics(y, p):
    ss_res = float(((y - p) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot
    rmse = float(np.sqrt(((y - p) ** 2).mean()))
    pear = float(np.corrcoef(y, p)[0, 1])
    spear = float(pd.Series(y).corr(pd.Series(p), method='spearman'))
    return r2, rmse, pear, spear


def per_fold(d):
    rows = []
    for k, g in d.groupby('fold'):
        r2, rmse, pear, spear = metrics(g['y'].values, g['p'].values)
        rows.append(dict(fold=int(k), n=len(g), r2=r2, rmse=rmse,
                         pearson=pear, spearman=spear))
    return pd.DataFrame(rows).sort_values('fold').reset_index(drop=True)


def pooled(d):
    r2, rmse, pear, spear = metrics(d['y'].values, d['p'].values)
    return dict(n=len(d), r2=r2, rmse=rmse, pearson=pear, spearman=spear)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--a', required=True, help='OOF dir of seed A')
    ap.add_argument('--b', required=True, help='OOF dir of seed B')
    ap.add_argument('--label_a', default='seedA')
    ap.add_argument('--label_b', default='seedB')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    da, db = load_oof(args.a), load_oof(args.b)
    fa, fb = per_fold(da), per_fold(db)

    if len(fa) != len(fb) or set(fa.fold) != set(fb.fold):
        print(f'[warn] fold sets differ: A={list(fa.fold)} B={list(fb.fold)}')

    m = fa.merge(fb, on='fold', suffixes=('_a', '_b'))

    print('=' * 96)
    print(f'per-fold:  {args.label_a}  vs  {args.label_b}')
    print('=' * 96)
    head = (f'{"fold":>4} {"n_a":>7} {"n_b":>7} | {"R2_a":>8} {"R2_b":>8} {"ΔR2":>8} | '
            f'{"r_a":>7} {"r_b":>7} {"Δr":>7} | {"ρ_a":>7} {"ρ_b":>7} {"Δρ":>7}')
    print(head)
    print('-' * 96)
    for _, r in m.iterrows():
        print(f'{int(r.fold):>4} {int(r.n_a):>7} {int(r.n_b):>7} | '
              f'{r.r2_a:>+8.4f} {r.r2_b:>+8.4f} {r.r2_b - r.r2_a:>+8.4f} | '
              f'{r.pearson_a:>7.4f} {r.pearson_b:>7.4f} {r.pearson_b - r.pearson_a:>+7.4f} | '
              f'{r.spearman_a:>7.4f} {r.spearman_b:>7.4f} {r.spearman_b - r.spearman_a:>+7.4f}')

    d_r2 = (m.r2_b - m.r2_a).values
    d_r = (m.pearson_b - m.pearson_a).values
    print('-' * 96)
    print(f'paired diff ΔR2 : mean={d_r2.mean():+.4f}  sd={d_r2.std(ddof=1):.4f}  '
          f'|Δ|mean={np.abs(d_r2).mean():.4f}  range=[{d_r2.min():+.4f}, {d_r2.max():+.4f}]')
    print(f'paired diff Δr  : mean={d_r.mean():+.4f}  sd={d_r.std(ddof=1):.4f}  '
          f'|Δ|mean={np.abs(d_r).mean():.4f}  range=[{d_r.min():+.4f}, {d_r.max():+.4f}]')

    pa, pb = pooled(da), pooled(db)
    print('-' * 96)
    print(f'pooled OOF {args.label_a}: R2={pa["r2"]:+.4f}  RMSE={pa["rmse"]:.4f}  '
          f'Pearson={pa["pearson"]:.4f}  Spearman={pa["spearman"]:.4f}  (n={pa["n"]})')
    print(f'pooled OOF {args.label_b}: R2={pb["r2"]:+.4f}  RMSE={pb["rmse"]:.4f}  '
          f'Pearson={pb["pearson"]:.4f}  Spearman={pb["spearman"]:.4f}  (n={pb["n"]})')

    # stability: seed difference vs fold-to-fold difference
    fold_spread = float(fa.r2.std(ddof=1))
    print('-' * 96)
    print(f'between-fold SD({args.label_a}) = {fold_spread:.4f}   '
          f'between-fold SD({args.label_b}) = {float(fb.r2.std(ddof=1)):.4f}   '
          f'paired seed SD = {d_r2.std(ddof=1):.4f}')
    print(f'ratio: paired seed SD / between-fold SD = '
          f'{d_r2.std(ddof=1) / max(fold_spread, 1e-9):.2f}  '
          f'(<0.5 means seed noise is far below between-fold noise)')

    m['dR2'] = m.r2_b - m.r2_a
    m['dPearson'] = m.pearson_b - m.pearson_a
    m['dSpearman'] = m.spearman_b - m.spearman_a
    m.to_csv(args.out, index=False, float_format='%.6f')
    print(f'\n[ok] -> {args.out}')


if __name__ == '__main__':
    main()
