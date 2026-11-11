# -*- coding: utf-8 -*-
"""paired-per-drug comparison of the modality ablation: ΔSSE instead of per-fold ΔR2.

why: R2 is a ratio, each fold has its own SST and ΔR2 does not add up, so one narrow fold
can drag the mean away (Drug Global ΔR2 = -1.146 on fold 3). under LDO each drug is a test item
in exactly one fold, so "full model vs drop one modality" gives one prediction for the same samples
in both -> ΔSSE can be paired per drug, then bootstrapped over drugs for the interval. the unit of
comparison moves from 5 folds to 233 drugs.

usage (cwd in 4_DrugResponse_BCAHFP):
  python -m bca_hfp.analysis.ablation_paired_sse
"""
import glob
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
SRC = os.path.join(ROOT, '5_Result', '3_diagnostics', '1_model_level',
                   'modality_ablation_sse_v1gc_s42_swa_20261110')
OLD = os.path.join(ROOT, '5_Result', '3_diagnostics', '1_model_level',
                   'modality_ablation_v1gc_s42_swa_20261109')
OUT = os.path.join(ROOT, 'results', 'tables', 'S14_ablation_paired_sse.csv')

B = 1000
SEED = 0
MODS = [('no_gene_pooled', 'Gene Pooled'),
        ('no_global_gene', 'Global Gene'),
        ('no_drug_global', 'Drug Global')]


def load():
    files = sorted(glob.glob(os.path.join(SRC, 'fold*', 'preds_all_conditions_fold*.csv')))
    if len(files) != 5:
        raise SystemExit('fewer than 5 folds of predictions, got %d' % len(files))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df['GROUP'] = df['GROUP'].astype(str).str.strip().str.upper()
    return df


def boot_ci(vals, n, groups, rng):
    """resample whole drugs with replacement; returns (mean, lo, hi) for sum(w*vals) / sum(w*n)."""
    idx = np.arange(len(groups))
    out = np.empty(B)
    for b in range(B):
        pick = rng.choice(idx, size=len(idx), replace=True)
        num = vals[pick].sum()
        den = n[pick].sum()
        out[b] = num / den if den > 0 else np.nan
    return (float(vals.sum() / n.sum()),
            float(np.nanpercentile(out, 2.5)),
            float(np.nanpercentile(out, 97.5)))


def main():
    df = load()
    print('pairs=%d  groups=%d  folds=%s' % (len(df), df['GROUP'].nunique(),
                                             sorted(df['fold'].unique())))
    y = df['true_label'].values.astype(np.float64)
    full = df['pred_full'].values.astype(np.float64)

    # aggregate per drug
    g = df.groupby('GROUP', sort=True)
    groups = np.array(sorted(df['GROUP'].unique()))
    n_g = g.size().reindex(groups).values.astype(np.float64)

    def per_group_sum(vals):
        return pd.Series(vals, index=df.index).groupby(df['GROUP']).sum().reindex(groups).values

    sse_full = per_group_sum((full - y) ** 2)
    rng = np.random.default_rng(SEED)
    rows = []

    for cname, label in MODS:
        p = df['pred_' + cname].values.astype(np.float64)
        sse_mod = per_group_sum((p - y) ** 2)
        d_sse = sse_mod - sse_full

        mse_full = sse_full.sum() / n_g.sum()
        mse_mod = sse_mod.sum() / n_g.sum()
        d_mse, lo, hi = boot_ci(d_sse, n_g, groups, rng)
        rmse_full, rmse_mod = np.sqrt(mse_full), np.sqrt(mse_mod)
        # ΔRMSE interval: same weighted resampling applied to the sqrt
        idx = np.arange(len(groups))
        d_rmse_draws = np.empty(B)
        for b in range(B):
            pick = rng.choice(idx, size=len(idx), replace=True)
            d_rmse_draws[b] = (np.sqrt(sse_mod[pick].sum() / n_g[pick].sum())
                               - np.sqrt(sse_full[pick].sum() / n_g[pick].sum()))
        d_rmse = float(rmse_mod - rmse_full)

        worse = float((d_sse > 0).mean())
        med = float(np.median(d_sse / n_g))
        rows.append(dict(modality=label,
                         delta_MSE=round(d_mse, 4),
                         delta_MSE_lo=round(lo, 4), delta_MSE_hi=round(hi, 4),
                         delta_RMSE=round(d_rmse, 4),
                         delta_RMSE_lo=round(float(np.percentile(d_rmse_draws, 2.5)), 4),
                         delta_RMSE_hi=round(float(np.percentile(d_rmse_draws, 97.5)), 4),
                         frac_drugs_worse=round(worse, 4),
                         median_per_drug_dMSE=round(med, 4),
                         n_groups=len(groups)))
        print('%-12s dMSE=%+.4f ln2 [%+.4f, %+.4f] | dRMSE=%+.4f ln [%+.4f, %+.4f] '
              '| drugs worse %.1f%% | median per-drug dMSE %+.4f'
              % (label, d_mse, lo, hi, d_rmse,
                 np.percentile(d_rmse_draws, 2.5), np.percentile(d_rmse_draws, 97.5),
                 100 * worse, med))

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False, encoding='utf-8-sig')
    print('\nwrote', OUT)

    # ---- side by side with the old definition ----
    fs = sorted(glob.glob(os.path.join(OLD, 'fold*', 'modality_ablation_test_summary.csv')))
    if fs:
        old = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
        print('\nold definition (per-fold ΔR2, mean ± sd over 5 folds):')
        for cname, label in MODS:
            c = 'dR2_' + cname
            print('  %-12s %+.4f ± %.4f   per-fold: %s'
                  % (label, old[c].mean(), old[c].std(ddof=1),
                     ' '.join('%+.3f' % v for v in old[c])))


if __name__ == '__main__':
    main()
