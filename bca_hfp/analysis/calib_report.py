# bca_hfp/analysis/calib_report.py
"""R3-2: report raw and rescaled R2 (y ~ a*p + b), plus a cross-fold transfer version.

three variants:
  raw          use predictions as they are
  self         a,b fitted on the whole OOF (upper bound, self-fitted on the test set, reference only)
  crossfold    for each fold, a,b fitted on the other folds then applied to this fold (no leakage, paper-ready)

usage:
python bca_hfp/analysis/calib_report.py \
  --cv_results ~/my_model/Result/My_model/CV/<e7>/results \
  --out_dir Result/oof
"""

import argparse
import glob
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


def fit_ab(p, y):
    """least-squares fit y ~ a*p + b."""
    A = np.vstack([p, np.ones_like(p)]).T
    (a, b), *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(a), float(b)


def load_oof(cv_results):
    files = sorted(glob.glob(os.path.join(cv_results, '*_fold*_swa_test_predictions.csv')))
    if not files:
        raise SystemExit(f'no prediction file found: {cv_results}')
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df.rename(columns={'true_label': 'y', 'predicted': 'p'})
    df['DRUG_ID'] = df['DRUG_ID'].astype(str)
    df['GDSC_ID'] = df['GDSC_ID'].astype(str)
    df['fold'] = df['fold'].astype(int)
    print(f'[load] {len(files)} folds, {len(df)} rows')
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cv_results', required=True)
    ap.add_argument('--out_dir', required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df = load_oof(args.cv_results)
    folds = sorted(df.fold.unique())
    y_all = df.y.values
    p_all = df.p.values

    # ---------- 1. raw ----------
    print('\n===== raw =====')
    print(f'  R2={r2(y_all, p_all):+.4f}  RMSE={rmse(y_all, p_all):.4f}  '
          f'Pearson={pearson(y_all, p_all):.4f}')

    # ---------- 2. self (upper bound) ----------
    a_s, b_s = fit_ab(p_all, y_all)
    p_self = a_s * p_all + b_s
    print('\n===== self-scaled (a,b fitted on the same OOF, upper bound only) =====')
    print(f'  a={a_s:.4f}  b={b_s:+.4f}')
    print(f'  R2={r2(y_all, p_self):+.4f}  RMSE={rmse(y_all, p_self):.4f}')

    # ---------- 3. cross-fold (safe for the paper) ----------
    print('\n===== cross-fold (a,b fitted on the other folds, applied to this one) =====')
    rows = []
    p_cf = np.full_like(p_all, np.nan)
    for k in folds:
        te = (df.fold == k).values
        tr = ~te
        a, b = fit_ab(p_all[tr], y_all[tr])
        p_cf[te] = a * p_all[te] + b
        rows.append(dict(fold=k, a=a, b=b,
                         r2_raw=r2(y_all[te], p_all[te]),
                         r2_cf=r2(y_all[te], p_cf[te]),
                         rmse_raw=rmse(y_all[te], p_all[te]),
                         rmse_cf=rmse(y_all[te], p_cf[te]),
                         n=int(te.sum())))
        print(f'  fold{k}: a={a:.4f} b={b:+.4f}  R2 raw={rows[-1]["r2_raw"]:+.4f} '
              f'-> cf={rows[-1]["r2_cf"]:+.4f}')
    t = pd.DataFrame(rows)
    print(f'  fold mean: raw {t.r2_raw.mean():+.4f} -> cf {t.r2_cf.mean():+.4f}')
    print(f'  pooled   : raw {r2(y_all, p_all):+.4f} -> cf {r2(y_all, p_cf):+.4f}  '
          f'RMSE {rmse(y_all, p_cf):.4f}  Pearson={pearson(y_all, p_cf):.4f}')
    print(f'  affine rescaling keeps the ranking, Pearson stays {pearson(y_all, p_all):.4f}, '
          f'so the R2 ceiling for cf is Pearson^2={pearson(y_all, p_all) ** 2:.4f}')

    t.to_csv(os.path.join(args.out_dir, 'calibration.csv'), index=False)
    print(f'\n[ok] -> {os.path.join(args.out_dir, "calibration.csv")}')


if __name__ == '__main__':
    main()
