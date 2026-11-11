# -*- coding: utf-8 -*-
"""cross-seed (42 vs 43) prediction agreement, per drug and pooled."""
import numpy as np
import pandas as pd

S42 = r'5_Result\3_diagnostics\1_model_level\r36_v1gc_s42_5090_20261108\oof_predictions.csv'
S43 = r'5_Result\3_diagnostics\1_model_level\s43_raw_20261109\oof_raw_predictions.csv'

a = pd.read_csv(S42, usecols=['GROUP', 'GDSC_ID', 'DRUG_ID', 'y', 'p'])
b = pd.read_csv(S43, usecols=['DRUG_ID', 'GDSC_ID', 'p_win']).rename(columns={'p_win': 'p43'})
for df in (a, b):
    df['DRUG_ID'] = df['DRUG_ID'].astype(str)
    df['GDSC_ID'] = df['GDSC_ID'].astype(str)
m = a.merge(b, on=['DRUG_ID', 'GDSC_ID'], how='inner')
print('merged pairs:', len(m), 'vs s42', len(a))

pooled = np.corrcoef(m['p'], m['p43'])[0, 1]
print('pooled corr(s42 pred, s43 pred) = %.4f' % pooled)

rows = []
for g, d in m.groupby('GROUP'):
    if len(d) < 10:
        continue
    r = np.corrcoef(d['p'], d['p43'])[0, 1]
    rows.append((g, len(d), r))
t = pd.DataFrame(rows, columns=['group', 'n', 'r'])
t = t.dropna()
print('drug groups:', len(t))
print('median per-drug corr = %.4f  (IQR %.4f-%.4f)' % (
    t['r'].median(), t['r'].quantile(.25), t['r'].quantile(.75)))
print('fraction with r > 0.8: %.1f%%' % (100 * (t['r'] > 0.8).mean()))

# rank agreement: per-drug Spearman on predictions, approximated by Pearson of ranks
sp = []
for g, d in m.groupby('GROUP'):
    if len(d) < 10:
        continue
    ra = d['p'].rank().values
    rb = d['p43'].rank().values
    sp.append(np.corrcoef(ra, rb)[0, 1])
sp = np.array(sp)
print('median per-drug rank agreement (Spearman) = %.4f' % np.nanmedian(sp))
