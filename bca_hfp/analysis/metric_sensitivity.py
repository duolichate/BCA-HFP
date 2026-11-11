# -*- coding: utf-8 -*-
"""per-drug R2 / Pearson sensitivity to the within-drug label spread.

purpose: show that low per-drug R2 under LDO is a **denominator effect** (small within-drug label
variance), not a lack of signal; the same predictions scored with Pearson, or with the per-drug
offset removed, look much better.
"""
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
S42 = os.path.join(ROOT, '5_Result', '3_diagnostics', '1_model_level',
                   'r36_v1gc_s42_5090_20261108', 'oof_predictions.csv')

df = pd.read_csv(S42, usecols=['GROUP', 'DRUG_ID', 'GDSC_ID', 'y', 'p'])
df['GROUP'] = df['GROUP'].astype(str)

# ---- pooled (all pairs) ----
y, p = df['y'].values, df['p'].values
r_pool = np.corrcoef(y, p)[0, 1]
sst = ((y - y.mean()) ** 2).sum()
r2_pool = 1 - ((p - y) ** 2).sum() / sst
r2_pool_bias0 = 1 - (((p - y) - (p.mean() - y.mean())) ** 2).sum() / sst
print('== pooled (%d pairs) ==' % len(df))
print('  Pearson        = %.4f' % r_pool)
print('  R2             = %.4f' % r2_pool)
print('  R2 (global offset removed) = %.4f' % r2_pool_bias0)
print('  global offset   = %+.4f' % (p.mean() - y.mean()))
print('  label variance decomposition: between-drug %.2f%%' % (
    100 * df.groupby('GROUP')['y'].transform('mean').var(ddof=0) / y.var(ddof=0)))

# ---- per drug ----
rows = []
for g, d in df.groupby('GROUP'):
    if len(d) < 10:
        continue
    yy, pp = d['y'].values, d['p'].values
    if yy.std() < 1e-9 or pp.std() < 1e-9:
        continue
    r = np.corrcoef(yy, pp)[0, 1]
    sst_g = ((yy - yy.mean()) ** 2).sum()
    sse_g = ((pp - yy) ** 2).sum()
    # remove the constant offset of that drug (keeps only the within-drug ranking)
    sse_g_c = ((pp - pp.mean()) - (yy - yy.mean())) ** 2
    rows.append(dict(group=g, n=len(d), sd_y=yy.std(),
                     rng_y=yy.max() - yy.min(), bias=pp.mean() - yy.mean(),
                     sse=sse_g, sse_c=sse_g_c.sum(),
                     r=r, r2=1 - sse_g / sst_g, r2_c=1 - sse_g_c.sum() / sst_g))

t = pd.DataFrame(rows).dropna()
print('\n== per drug (%d drugs, >=10 pairs each) ==' % len(t))


def show(name, v):
    print('  %-12s median %.4f  IQR %.4f ~ %.4f  mean %.4f  min %.3f  max %.3f'
          % (name, v.median(), v.quantile(.25), v.quantile(.75), v.mean(), v.min(), v.max()))


show('Pearson', t['r'])
show('R2', t['r2'])
show('R2(no offset)', t['r2_c'])
print('  Pearson>0.5 : %.1f%%   >0.3 : %.1f%%   <0 : %.1f%%'
      % (100 * (t['r'] > .5).mean(), 100 * (t['r'] > .3).mean(), 100 * (t['r'] < 0).mean()))
print('  R2   >0.5 : %.1f%%   >0.3 : %.1f%%   <0 : %.1f%%'
      % (100 * (t['r2'] > .5).mean(), 100 * (t['r2'] > .3).mean(), 100 * (t['r2'] < 0).mean()))
print('  SD of per-drug bias = %.4f (median |bias| = %.4f)'
      % (t['bias'].std(), t['bias'].abs().median()))

# ---- error decomposition: share of the squared error taken by the per-drug constant offset ----
sse_all = t['sse'].sum()
sse_off = (t['n'] * t['bias'] ** 2).sum()
print('\n== squared error decomposition (these %d drugs) ==' % len(t))
print('  total SSE       = %.0f' % sse_all)
print('  per-drug constant offset = %.0f  (%.1f%%)' % (sse_off, 100 * sse_off / sse_all))
print('  within-drug residual = %.0f  (%.1f%%)'
      % (sse_all - sse_off, 100 * (1 - sse_off / sse_all)))
print('  with every drug offset flattened: pooled R2 %.4f -> %.4f'
      % (1 - sse_all / sst, 1 - (sse_all - sse_off) / sst))

# ---- buckets by within-drug label spread ----
t['q'] = pd.qcut(t['sd_y'], 4, labels=['Q1 narrow', 'Q2', 'Q3', 'Q4 wide'])
print('\n== quartiles of within-drug label SD ==')
agg = t.groupby('q').agg(n_drugs=('group', 'size'), sd_y_median=('sd_y', 'median'),
                         pearson_median=('r', 'median'), r2_median=('r2', 'median'),
                         r2_no_offset_median=('r2_c', 'median'),
                         frac_r2_negative=('r2', lambda v: 100 * (v < 0).mean()))
print(agg.round(4).to_string())

# ---- correlation: spread vs R2 ----
from scipy.stats import spearmanr
print('\nSpearman(within-drug label SD, per-drug R2)  = %.3f  (p=%.2g)'
      % spearmanr(t['sd_y'], t['r2'])[:2])
print('Spearman(within-drug label SD, per-drug Pearson) = %.3f  (p=%.2g)'
      % spearmanr(t['sd_y'], t['r'])[:2])

# ---- ranking metrics (what drug-priority ordering actually cares about) ----
top_prec = []
for g, d in df.groupby('GROUP'):
    if len(d) < 10:
        continue
    n_top = max(1, int(round(0.1 * len(d))))
    hit = set(d.nlargest(n_top, 'y')['GDSC_ID']) & set(d.nlargest(n_top, 'p')['GDSC_ID'])
    top_prec.append(len(hit) / n_top)
print('\nprecision@10%% per-drug median = %.3f  (mean %.3f)'
      % (np.median(top_prec), np.mean(top_prec)))

out = os.path.join(ROOT, 'results', 'tables', 'S12_metric_sensitivity_per_drug.csv')
t.to_csv(out, index=False)
print('\nwrote', out)
