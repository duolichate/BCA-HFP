# bca_hfp/data/rebuild_gdsc_exp.py
"""rebuild GDSC_EXP.csv from two official inputs and compare it value by value with the existing file.

Inputs (both from GDSC / Sanger)
  Cell_line_RMA_proc_basalExp.xlsx   RMA-normalised expression (gene x cell line, columns DATA.<COSMIC>)
  Cell_Lines_Details.xlsx            cell line metadata (COSMIC identifier / Sample Name / Tissue descriptor)

Steps (lines 330-448 of the preprocessing script)
  1. drop the GENE_title column
  2. filter empty gene names     -> 17,419 genes
  3. transpose                   -> cell line x gene
  4. strip the DATA. prefix, convert to COSMIC ID
  5. intersect RMA with Details  -> 968 cell lines
  6. join CELL_LINES / ORGAN
  7. write the CSV

Usage
  python bca_hfp/data/rebuild_gdsc_exp.py \
    --rma ../Data/Cell_line_RMA_proc_basalExp.xlsx \
    --details ../Data/Cell_Lines_Details.xlsx \
    --reference ../Data/GDSC_EXP.csv \
    --out ../Data/GDSC_EXP_v2.csv
"""
import argparse
import datetime
import os
import time

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rma', required=True)
    ap.add_argument('--details', required=True)
    ap.add_argument('--reference', default='', help='existing GDSC_EXP.csv used for the value-by-value comparison')
    ap.add_argument('--out', required=True)
    ap.add_argument('--n_check', type=int, default=200, help='number of random (cell line, gene) pairs to compare')
    args = ap.parse_args()

    t0 = time.time()
    print('reading RMA ...')
    raw = pd.read_excel(args.rma, sheet_name=0)
    print(f'  {raw.shape}  {time.time() - t0:.0f}s')

    gcol = raw.columns[0]
    title_cols = [c for c in raw.columns if 'GENE_title' in str(c)]
    if title_cols:
        raw = raw.drop(columns=title_cols)

    def _norm_gene(v):
        if isinstance(v, (pd.Timestamp, datetime.datetime, datetime.date)):
            return str((pd.Timestamp(v) - pd.Timestamp('1899-12-30')).days)
        return str(v).strip()

    raw[gcol] = raw[gcol].map(_norm_gene)
    n0 = len(raw)
    raw = raw[raw[gcol].notna() & (raw[gcol] != '') & (raw[gcol] != 'nan')]
    print(f'  {n0} rows after dropping title columns; {len(raw)} genes after filtering empty names')

    dup = raw[gcol].duplicated().sum()
    if dup:
        print(f'  [warn] {dup} duplicated gene names, keeping the first')
        raw = raw.drop_duplicates(subset=[gcol], keep='first')

    print('transposing ...')
    mat = raw.set_index(gcol).T
    mat.index = pd.Index([str(i).replace('DATA.', '') for i in mat.index], name='GDSC_ID')
    mat.index = pd.to_numeric(mat.index, errors='coerce')
    keep = ~pd.isna(mat.index)
    mat = mat.loc[keep]
    mat.index = mat.index.astype('int64')
    print(f'  {mat.shape} (cell lines x genes)')

    print('reading cell line metadata ...')
    cd = pd.read_excel(args.details, sheet_name=0)
    cd['GDSC_ID'] = pd.to_numeric(cd['COSMIC identifier'], errors='coerce')
    cd = cd.dropna(subset=['GDSC_ID']).drop_duplicates('GDSC_ID')
    cd['GDSC_ID'] = cd['GDSC_ID'].astype('int64')
    organ_col = next((c for c in cd.columns if 'Tissue descriptor 1' in str(c)), None)
    print(f'  {len(cd)} cell lines; ORGAN column = {organ_col!r}')

    ids = sorted(set(mat.index) & set(cd['GDSC_ID']))
    print(f'  RMA & Details = {len(ids)} cell lines')

    meta = (cd.set_index('GDSC_ID')
            .loc[ids, ['Sample Name'] + ([organ_col] if organ_col else [])]
            .rename(columns={'Sample Name': 'CELL_LINES', organ_col: 'ORGAN'}))
    out = meta.reset_index()
    gene_mat = mat.loc[ids].reset_index()
    out = out.merge(gene_mat, on='GDSC_ID', how='left')
    if out['GDSC_ID'].duplicated().any():
        n_dup = int(out['GDSC_ID'].duplicated().sum())
        print(f'  [warn] {n_dup} duplicated GDSC_ID after merge, keeping the first row')
        out = out.drop_duplicates('GDSC_ID', keep='first')
    print(f'output {out.shape}')

    out.to_csv(args.out, index=False)
    print(f'wrote {args.out}')

    # ---------------- compare with the existing file ----------------
    if not args.reference or not os.path.exists(args.reference):
        return
    print('\n' + '=' * 70)
    print('comparing with the existing GDSC_EXP.csv')
    print('=' * 70)
    ref_head = pd.read_csv(args.reference, nrows=1, low_memory=False)
    ref_meta = [c for c in ['GDSC_ID', 'CCLE_ID', 'CELL_LINES', 'CELL_LINES_STD', 'ORGAN']
                if c in ref_head.columns]
    ref_genes = [c for c in ref_head.columns if c not in ref_meta]

    print(f'  existing: {len(ref_head.columns)} cols = {len(ref_meta)} metadata + {len(ref_genes)} genes')
    _n_gene_new = len(out.columns.intersection(ref_genes))
    print(f'  rebuilt: {out.shape[1]} cols = {out.shape[1] - _n_gene_new} metadata + {_n_gene_new} genes')

    id_ref = set(pd.read_csv(args.reference, usecols=['GDSC_ID'], low_memory=False)['GDSC_ID'].astype(str))
    id_new = set(out['GDSC_ID'].astype(str))
    print(f'  cell lines: existing {len(id_ref)}, rebuilt {len(id_new)}, '
          f'only in existing {len(id_ref - id_new)}, only in rebuilt {len(id_new - id_ref)}')

    common_genes = [g for g in ref_genes if g in out.columns]
    print(f'  shared gene columns {len(common_genes)} / existing {len(ref_genes)}')

    # sampled value-by-value comparison
    rng = np.random.RandomState(42)
    ref = pd.read_csv(args.reference, low_memory=False,
                      usecols=['GDSC_ID'] + common_genes)
    ref = ref[ref['GDSC_ID'].astype(str).isin(id_new)].copy()
    ref['GDSC_ID'] = ref['GDSC_ID'].astype(str)
    new = out[['GDSC_ID'] + common_genes].copy()
    new['GDSC_ID'] = new['GDSC_ID'].astype(str)
    m = ref.merge(new, on='GDSC_ID', suffixes=('_ref', '_new'))
    print(f'  pairable cell lines {len(m)}')

    picks = rng.choice(len(common_genes), size=min(args.n_check, len(common_genes)), replace=False)
    max_diff, n_bad = 0.0, 0
    for gi in picks:
        g = common_genes[gi]
        a = pd.to_numeric(m[f'{g}_ref'], errors='coerce').values
        b = pd.to_numeric(m[f'{g}_new'], errors='coerce').values
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() == 0:
            continue
        d = np.abs(a[ok] - b[ok])
        max_diff = max(max_diff, float(d.max()))
        n_bad += int((d > 1e-5).sum())
    print(f'  random {len(picks)} genes x {len(m)} cell lines: '
          f'max|diff|={max_diff:.3e}, {n_bad} values over 1e-5')

    # metadata comparison
    if 'CELL_LINES' in out.columns:
        rr = pd.read_csv(args.reference, low_memory=False, usecols=['GDSC_ID', 'CELL_LINES'])
        rr['GDSC_ID'] = rr['GDSC_ID'].astype(str)
        mm = rr.merge(out[['GDSC_ID', 'CELL_LINES']].assign(GDSC_ID=out['GDSC_ID'].astype(str)),
                      on='GDSC_ID', suffixes=('_ref', '_new'))
        same = (mm['CELL_LINES_ref'].astype(str).str.strip().str.upper()
                == mm['CELL_LINES_new'].astype(str).str.strip().str.upper())
        print(f'  CELL_LINES match: {int(same.sum())}/{len(mm)}')
    print(f'\ntotal time {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
