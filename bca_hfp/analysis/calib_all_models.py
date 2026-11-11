# -*- coding: utf-8 -*-
"""R3-2 summary: R2 before clamping / fit on validation fold / fit across folds

two calibration variants
------------
* valfit (deployable): fit (a, b) on the predictions of that fold's validation subset, then apply to its test predictions.
* crossfold (diagnostic upper bound): fit (a, b) on the test predictions of the other four folds, then apply to the current fold.
  under the leave-drug-out split the val and test drugs do not overlap, so valfit drifts with the distribution
  shift; crossfold borrows labels from the other test folds and is only a scale upper bound. both are reported.

per-sample sources
----------
- BCA-HFP seed 42/43 : 1_model_level/{r36_v1gc_s42_5090_20261108,s43_raw_20261109}
                       validation fold: 1_model_level/bca_val_from_ckpt.csv (re-inferred by predict_cv_val.py)
- tCNNS / PaccMann / TransCDR / GraphDRP / GraTransDRP :
                       4_baselines/<name>_oof_from_ckpt.csv and <name>_oof_from_ckpt_val.csv

usage: python 4_DrugResponse_BCAHFP/bca_hfp/analysis/calib_all_models.py
output: 5_Result/<tables dir>/R3_2_calibrated_metrics.csv
"""

import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DIAG = os.path.join(ROOT, '5_Result', '3_diagnostics')
OUT = os.path.join(ROOT, 'results', 'tables', 'R3_2_calibrated_metrics.csv')

# (display name, test-fold CSV, val-fold CSV or None)
SOURCES = [
    ('BCA-HFP (seed 42)',
     os.path.join(DIAG, '1_model_level', 'r36_v1gc_s42_5090_20261108',
                  'raw_predictions', 'oof_raw_predictions.csv'),
     os.path.join(DIAG, '1_model_level', 'bca_val_from_ckpt.csv')),
    ('BCA-HFP (seed 43)',
     os.path.join(DIAG, '1_model_level', 's43_raw_20261109', 'oof_raw_predictions.csv'),
     os.path.join(DIAG, '1_model_level', 'bca_s43_val_from_ckpt.csv')),
    ('tCNNS', os.path.join(DIAG, '4_baselines', 'tcnns_oof_from_ckpt.csv'),
     os.path.join(DIAG, '4_baselines', 'tcnns_oof_from_ckpt_val.csv')),
    ('PaccMann', os.path.join(DIAG, '4_baselines', 'paccmann_oof_from_ckpt.csv'),
     os.path.join(DIAG, '4_baselines', 'paccmann_oof_from_ckpt_val.csv')),
    ('TransCDR', os.path.join(DIAG, '4_baselines', 'transcdr_oof_from_ckpt.csv'),
     os.path.join(DIAG, '4_baselines', 'transcdr_oof_from_ckpt_val.csv')),
    ('GraphDRP', os.path.join(DIAG, '4_baselines', 'graphdrp_oof_from_ckpt.csv'),
     os.path.join(DIAG, '4_baselines', 'graphdrp_oof_from_ckpt_val.csv')),
    ('GraTransDRP', os.path.join(DIAG, '4_baselines', 'gratransdrp_oof_from_ckpt.csv'),
     os.path.join(DIAG, '4_baselines', 'gratransdrp_oof_from_ckpt_val.csv')),
]

Y, P, F = 'y', 'p_raw', 'fold'


def r2(y, p):
    return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def crossfold(y, p, fold):
    pc = np.full_like(p, np.nan)
    slopes = []
    for k in np.unique(fold):
        tr, te = fold != k, fold == k
        a, b = np.polyfit(p[tr], y[tr], 1)
        pc[te] = a * p[te] + b
        slopes.append(a)
    return pc, slopes


def valfit(y, p, fold, val):
    pc = np.full_like(p, np.nan)
    slopes = []
    for k in np.unique(fold):
        v = val[val[F] == k]
        m = fold == k
        if len(v) < 10:
            pc[m] = p[m]
            continue
        a, b = np.polyfit(v[P].values, v[Y].values, 1)
        pc[m] = a * p[m] + b
        slopes.append(a)
    return pc, slopes


def main():
    rows, missing = [], []
    for name, test_path, val_path in SOURCES:
        if not os.path.exists(test_path):
            missing.append((name, 'test fold', test_path))
            continue
        d = pd.read_csv(test_path, usecols=[Y, P, F])
        y, p, fold = d[Y].values.astype(float), d[P].values.astype(float), d[F].values

        pc_cf, s_cf = crossfold(y, p, fold)
        rec = dict(model=name, n=len(y),
                   r2_preclamp=round(r2(y, p), 4),
                   r2_crossfold=round(r2(y, pc_cf), 4),
                   slope_crossfold=round(float(np.mean(s_cf)), 3))

        if val_path and os.path.exists(val_path):
            val = pd.read_csv(val_path, usecols=[Y, P, F])
            pc_v, s_v = valfit(y, p, fold, val)
            rec['n_val'] = len(val)
            rec['r2_valfit'] = round(r2(y, pc_v), 4)
            rec['slope_valfit'] = round(float(np.mean(s_v)), 3)
        else:
            rec['n_val'] = np.nan
            rec['r2_valfit'] = np.nan
            rec['slope_valfit'] = np.nan
            if val_path:
                missing.append((name, 'val fold', val_path))

        rec['pearson'] = round(float(np.corrcoef(y, p)[0, 1]), 4)
        rec['spearman'] = round(float(pd.Series(y).corr(pd.Series(p), method='spearman')), 4)
        rows.append(rec)
        print(f"{name:20s} n={len(y):6d}  raw={rec['r2_preclamp']:.4f}  "
              f"valfit={rec['r2_valfit']}  crossfold={rec['r2_crossfold']:.4f}")

    out = pd.DataFrame(rows)[['model', 'n', 'n_val', 'r2_preclamp', 'r2_valfit',
                              'r2_crossfold', 'slope_valfit', 'slope_crossfold',
                              'pearson', 'spearman']]
    out.to_csv(OUT, index=False, encoding='utf-8-sig')
    print(f'\nsaved {OUT}')
    if missing:
        print('\nmissing:')
        for n, k, p in missing:
            print(f'  {n:14s} {k} -> {p}')


if __name__ == '__main__':
    main()
