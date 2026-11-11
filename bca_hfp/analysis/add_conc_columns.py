# bca_hfp/analysis/add_conc_columns.py
"""merge the MIN_CONC / MAX_CONC columns of the official xlsx into Data/GDSC_RES.csv.

why: these two concentration window ends decide the censoring labels (right-censored = LN_IC50 >= ln(MAX_CONC));
the training side used to align them by row index through an external Result/diag/censoring_map.csv. with them
in the main table, regenerating the h5 gives the window straight from original_data, no external file needed.

columns are only appended, no existing column is touched; alignment is by row order (LN_IC50 checked row by row).

usage:
python bca_hfp/analysis/add_conc_columns.py \
  --csv ../Data/GDSC_RES.csv \
  --xlsx ../Data/GDSC2_fitted_dose_response_27Oct23.xlsx \
  --backup ../Data/GDSC_RES_backup_before_conccols.csv
"""
import argparse
import os
import shutil

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--xlsx', required=True)
    ap.add_argument('--backup', required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.csv, low_memory=False)
    x = pd.read_excel(args.xlsx, sheet_name=0)
    if len(df) != len(x):
        raise SystemExit(f'row count differs: csv={len(df)} xlsx={len(x)}')

    # row-order check: LN_IC50 must match row by row, otherwise the columns cannot be pasted by row
    a = pd.to_numeric(df['LN_IC50'], errors='coerce').values
    b = pd.to_numeric(x['LN_IC50'], errors='coerce').values
    ok = np.isclose(a, b, atol=1e-6, equal_nan=True)
    print(f'row-order check LN_IC50: {ok.mean() * 100:.4f}% identical')
    if ok.mean() < 0.9999:
        raise SystemExit('row order mismatch, aborting')

    before = df.shape
    df['MIN_CONC'] = x['MIN_CONC'].values
    df['MAX_CONC'] = x['MAX_CONC'].values
    # keep the order: ... DRUG_NAME, MIN_CONC, MAX_CONC, LN_IC50, ...
    order = [c for c in df.columns if c not in ('MIN_CONC', 'MAX_CONC')]
    i = order.index('LN_IC50')
    order = order[:i] + ['MIN_CONC', 'MAX_CONC'] + order[i:]
    df = df[order]

    if not os.path.exists(args.backup):
        shutil.copy2(args.csv, args.backup)
        print(f'backed up -> {args.backup}')
    df.to_csv(args.csv, index=False)
    print(f'wrote {args.csv}: {before} -> {df.shape}')
    print('column order:', list(df.columns))

    # ---- self-check ----
    chk = pd.read_csv(args.csv, low_memory=False)
    print('\nself-check:')
    print(f'  rows {len(chk)}')
    for col in ['MIN_CONC', 'MAX_CONC']:
        same = np.isclose(pd.to_numeric(chk[col], errors='coerce').values,
                          pd.to_numeric(x[col], errors='coerce').values,
                          atol=0, equal_nan=True)
        print(f'  {col:<9s}: {same.mean() * 100:.4f}% match xlsx')
    same = (chk['DRUG_ID'].astype(str).values == x['DRUG_ID'].astype(str).values)
    print(f'  {"DRUG_ID":<9s}: {same.mean() * 100:.4f}%')
    for col in ['LN_IC50', 'AUC', 'RMSE', 'Z_SCORE']:
        same = np.isclose(pd.to_numeric(chk[col], errors='coerce').values,
                          pd.to_numeric(x[col], errors='coerce').values,
                          atol=1e-6, equal_nan=True)
        print(f'  {col:<9s}: {same.mean() * 100:.4f}%')
    print(f'  missing: MIN_CONC={int(chk["MIN_CONC"].isna().sum())} '
          f'MAX_CONC={int(chk["MAX_CONC"].isna().sum())}')

    # censoring distribution (same definition as Result/diag/censoring_map.csv)
    lo = np.log(chk['MIN_CONC'].astype(float).values)
    hi = np.log(chk['MAX_CONC'].astype(float).values)
    y = chk['LN_IC50'].astype(float).values
    right = int((y >= hi).sum())
    left = int((y <= lo).sum())
    print(f'  right-censored {right:,} ({right / len(chk) * 100:.2f}%)  '
          f'left-censored {left:,} ({left / len(chk) * 100:.2f}%)  '
          f'inside window {len(chk) - right - left:,}')


if __name__ == '__main__':
    main()
