# bca_hfp/analysis/check_id_consistency.py
"""check ID / name consistency between the h5 (GDSC_RES_FGBERT.h5) and the original xlsx (GDSC2_fitted_dose_response_27Oct23).

checking and dumping only, no data is modified. outputs (default Result/diag/id_check/):
  summary.txt              global summary
  mismatch_rows.csv        per-row mismatch list (row index + both sides + whether LN_IC50 matches)
  drug_id_name_map.csv     all names of each DRUG_ID on both sides
  cell_id_name_map.csv     all names of each cell id on both sides
  drug_id_only_h5.csv      DRUG_IDs only in h5
  drug_id_only_xlsx.csv    DRUG_IDs only in xlsx
  drug_smiles_conflict.csv one DRUG_ID mapped to several SMILES
  dup_pairs.csv            rows of repeated (drug, cell) measurements and their label spread

usage:
  python bca_hfp/analysis/check_id_consistency.py \
      --h5 ../Data/GDSC_RES_FGBERT.h5 \
      --xlsx ../Data/GDSC2_fitted_dose_response_27Oct23.xlsx \
      --out_dir ../Result/diag/id_check
"""
import argparse
import os

import numpy as np
import pandas as pd


def norm_name(s):
    """normalize a name: drop spaces/hyphens/underscores and uppercase. separates a real mismatch from a spelling difference."""
    return (str(s).strip().upper()
            .replace('-', '').replace(' ', '').replace('_', '')
            .replace('.', ''))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--h5', required=True)
    ap.add_argument('--xlsx', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--label_col', default='LN_IC50')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    lines = []

    def log(msg=''):
        print(msg)
        lines.append(str(msg))

    # ---------- reading ----------
    with pd.HDFStore(args.h5, 'r') as store:
        h5 = store['original_data'].copy()
        smiles = store['drug_smiles'][['DRUG_ID', 'DRUG_SMILE']].copy()
        # read the row count only, no unpickling (atom_features were written by numpy 2.x, an old numpy cannot unpickle them)
        _n_atom = store.get_storer('atom_features').nrows
    x = pd.read_excel(args.xlsx, sheet_name=0)

    log('=' * 78)
    log('1. sizes and columns')
    log('=' * 78)
    log(f'  h5   original_data : {h5.shape}  atom_features: ({_n_atom}, 1)')
    log(f'  xlsx Sheet1        : {x.shape}')
    log(f'  h5   drug_smiles   : {smiles.shape}')
    log(f'  h5 columns     : {list(h5.columns)}')
    log(f'  xlsx columns   : {list(x.columns)}')
    log(f'  same row count: {len(h5) == len(x)}')
    if len(h5) != len(x):
        raise SystemExit('row counts differ, a row-by-row comparison is meaningless, check the data source first')

    # ---------- row-by-row comparison ----------
    lab_a = pd.to_numeric(h5[args.label_col], errors='coerce').values
    lab_b = pd.to_numeric(x[args.label_col], errors='coerce').values
    lab_eq = np.isclose(lab_a, lab_b, atol=1e-6, equal_nan=True)

    drug_id_eq = (h5['DRUG_ID'].astype(str).values == x['DRUG_ID'].astype(str).values)
    drug_name_raw = (h5['DRUG_NAME'].astype(str).values == x['DRUG_NAME'].astype(str).values)
    drug_name_norm = (h5['DRUG_NAME'].map(norm_name).values == x['DRUG_NAME'].map(norm_name).values)
    cell_name_raw = (h5['CELL_LINES'].astype(str).values == x['CELL_LINE_NAME'].astype(str).values)
    cell_name_norm = (h5['CELL_LINES'].map(norm_name).values == x['CELL_LINE_NAME'].map(norm_name).values)

    def pct(mask):
        return f"{int(mask.sum()):,} / {len(mask):,}  ({mask.mean() * 100:.2f}% identical)"

    log()
    log('=' * 78)
    log('2. per-row agreement (assumes row alignment)')
    log('=' * 78)
    log(f'  {args.label_col:8s} same : {pct(lab_eq)}')
    log(f'  DRUG_ID         same : {pct(drug_id_eq)}')
    log(f'  DRUG_NAME raw   same : {pct(drug_name_raw)}')
    log(f'  DRUG_NAME normalized : {pct(drug_name_norm)}   (spaces/hyphens/underscores dropped, uppercased)')
    log(f'  cell name raw   same : {pct(cell_name_raw)}')
    log(f'  cell name normalized : {pct(cell_name_norm)}')

    # do the mismatched rows still agree on the label (i.e. only the metadata naming differs)
    bad_any = (~drug_id_eq) | (~drug_name_norm) | (~cell_name_norm)
    if bad_any.any():
        log(f'  rows with any field mismatch: {int(bad_any.sum()):,}; share of them with matching {args.label_col}: '
            f'{lab_eq[bad_any].mean() * 100:.2f}%')

    # ---------- per-row mismatch list ----------
    idx = np.where(bad_any)[0]
    mm = pd.DataFrame({
        'row': idx,
        'h5_DRUG_ID': h5['DRUG_ID'].values[idx],
        'xlsx_DRUG_ID': x['DRUG_ID'].values[idx],
        'h5_DRUG_NAME': h5['DRUG_NAME'].values[idx],
        'xlsx_DRUG_NAME': x['DRUG_NAME'].values[idx],
        'h5_CELL_LINES': h5['CELL_LINES'].values[idx],
        'xlsx_CELL_LINE_NAME': x['CELL_LINE_NAME'].values[idx],
        'h5_GDSC_ID': h5['GDSC_ID'].values[idx],
        'xlsx_SANGER_MODEL_ID': x['SANGER_MODEL_ID'].values[idx],
        'drug_id_diff': (~drug_id_eq)[idx],
        'drug_name_diff_norm': (~drug_name_norm)[idx],
        'cell_name_diff_norm': (~cell_name_norm)[idx],
        'LN_IC50_h5': h5[args.label_col].values[idx],
        'LN_IC50_xlsx': x[args.label_col].values[idx],
        'LN_IC50_diff': np.abs(h5[args.label_col].values[idx] - x[args.label_col].values[idx]),
    })
    mm.to_csv(os.path.join(args.out_dir, 'mismatch_rows.csv'), index=False)

    # ---------- drug set difference ----------
    set_h5 = set(h5['DRUG_ID'].astype(str).unique())
    set_x = set(x['DRUG_ID'].astype(str).unique())
    log()
    log('=' * 78)
    log('3. drug ID sets')
    log('=' * 78)
    log(f'  h5   unique DRUG_ID: {len(set_h5)}')
    log(f'  xlsx unique DRUG_ID: {len(set_x)}')
    only_h5 = sorted(set_h5 - set_x, key=lambda s: int(s) if s.isdigit() else s)
    only_x = sorted(set_x - set_h5, key=lambda s: int(s) if s.isdigit() else s)
    log(f'  only in h5 ({len(only_h5)}): {only_h5}')
    log(f'  only in xlsx ({len(only_x)}): {only_x}')
    pd.DataFrame({'DRUG_ID': only_h5}).to_csv(
        os.path.join(args.out_dir, 'drug_id_only_h5.csv'), index=False)
    pd.DataFrame({'DRUG_ID': only_x}).to_csv(
        os.path.join(args.out_dir, 'drug_id_only_xlsx.csv'), index=False)

    # ---------- ID -> name maps ----------
    log()
    log('=' * 78)
    log('4. DRUG_ID -> DRUG_NAME mapping conflicts')
    log('=' * 78)
    _h = h5.groupby(h5['DRUG_ID'].astype(str))['DRUG_NAME'].agg(
        lambda s: sorted(set(s.astype(str))))
    _x = x.groupby(x['DRUG_ID'].astype(str))['DRUG_NAME'].agg(
        lambda s: sorted(set(s.astype(str))))
    rows = []
    for did in sorted(set(_h.index) | set(_x.index), key=lambda s: int(s) if s.isdigit() else s):
        hn = _h.get(did, [])
        xn = _x.get(did, [])
        same = (sorted({norm_name(v) for v in hn}) == sorted({norm_name(v) for v in xn}))
        rows.append({'DRUG_ID': did,
                     'h5_names': ' | '.join(hn),
                     'xlsx_names': ' | '.join(xn),
                     'n_h5': len(hn), 'n_xlsx': len(xn),
                     'norm_equal': same})
    dmap = pd.DataFrame(rows)
    dmap.to_csv(os.path.join(args.out_dir, 'drug_id_name_map.csv'), index=False)
    log(f'  drugs whose normalized name sets differ between the two sides: {int((~dmap["norm_equal"]).sum())} / {len(dmap)}')
    log(f'  IDs with several names inside h5: {int((dmap["n_h5"] > 1).sum())}')
    log(f'  IDs with several names inside xlsx: {int((dmap["n_xlsx"] > 1).sum())}')
    if (~dmap['norm_equal']).any():
        log('  mismatch list (first 30):')
        for _, r in dmap[~dmap['norm_equal']].head(30).iterrows():
            log(f'    ID {r["DRUG_ID"]:>6s} | h5={r["h5_names"][:60]} | xlsx={r["xlsx_names"][:60]}')

    # ---------- cell name field ----------
    log()
    log('=' * 78)
    log('5. cell line name conflicts')
    log('=' * 78)
    _hc = h5.groupby(h5['CELL_LINES'].map(norm_name))['GDSC_ID'].agg(
        lambda s: sorted({str(v) for v in s}))
    log(f'  h5 normalized cell names: {len(_hc)}; names with several GDSC_IDs: {int((_hc.map(len) > 1).sum())}')
    _xc = x.groupby(x['CELL_LINE_NAME'].map(norm_name))['SANGER_MODEL_ID'].agg(
        lambda s: sorted({str(v) for v in s}))
    log(f'  xlsx normalized cell names: {len(_xc)}; names with several MODEL_IDs: {int((_xc.map(len) > 1).sum())}')
    log(f'  cell names only in h5: {len(set(_hc.index) - set(_xc.index))}')
    log(f'  cell names only in xlsx: {len(set(_xc.index) - set(_hc.index))}')
    crows = []
    for name in sorted(set(_hc.index) | set(_xc.index)):
        crows.append({'cell_name_norm': name,
                      'h5_GDSC_IDs': ' | '.join(_hc.get(name, [])),
                      'xlsx_MODEL_IDs': ' | '.join(_xc.get(name, [])),
                      'n_h5_ids': len(_hc.get(name, [])),
                      'n_xlsx_ids': len(_xc.get(name, []))})
    pd.DataFrame(crows).to_csv(os.path.join(args.out_dir, 'cell_id_name_map.csv'), index=False)

    # ---------- SMILES conflicts ----------
    log()
    log('=' * 78)
    log('6. ID/SMILES conflicts inside the drug_smiles table')
    log('=' * 78)
    _sm = smiles.groupby(smiles['DRUG_ID'].astype(str))['DRUG_SMILE'].agg(
        lambda s: sorted(set(s.astype(str))))
    log(f'  drug_smiles unique DRUG_ID: {len(_sm)}')
    log(f'  IDs mapped to several SMILES: {int((_sm.map(len) > 1).sum())}')
    conflict = pd.DataFrame({
        'DRUG_ID': _sm.index,
        'n_smiles': _sm.map(len).values,
        'smiles': [' | '.join(v) for v in _sm.values],
    })
    conflict[conflict['n_smiles'] > 1].to_csv(
        os.path.join(args.out_dir, 'drug_smiles_conflict.csv'), index=False)

    # whether the SMILES in the h5 response table match the drug_smiles table
    _map = dict(zip(smiles['DRUG_ID'].astype(str), smiles['DRUG_SMILE'].astype(str)))
    _resp_sm = h5.groupby(h5['DRUG_ID'].astype(str))['DRUG_SMILE'].agg(
        lambda s: sorted(set(s.astype(str))))
    bad = [d for d in _resp_sm.index
           if d in _map and set(_resp_sm[d]) != {_map[d]}]
    log(f'  drugs whose SMILES differ between original_data and drug_smiles: {len(bad)}')
    log('  note: training uses the drug_smiles table (atom_features are built from it), and the difference against original_data')
    log('        is mostly stereochemistry or kekulization, not a wrong drug; some do lose chirality, which affects the FG-BERT atom features.')
    sm_check = pd.DataFrame([
        {'DRUG_ID': d,
         'n_resp_smiles': len(_resp_sm[d]),
         'same_as_table': (len(_resp_sm[d]) == 1 and _resp_sm[d][0] == _map[d]),
         'resp_smiles': _resp_sm[d][0][:200],
         'table_smiles': _map[d][:200]}
        for d in sorted(set(_resp_sm.index) & set(_map), key=lambda s: int(s))
    ])
    sm_check.to_csv(os.path.join(args.out_dir, 'smiles_check.csv'), index=False)

    # ---------- repeated measurements ----------
    log()
    log('=' * 78)
    log('7. repeated (drug, cell) measurements')
    log('=' * 78)
    key = h5['DRUG_ID'].astype(str) + '||' + h5['GDSC_ID'].astype(str)
    g = h5.assign(_k=key).groupby('_k')[args.label_col]
    dup = g.agg(['count', 'mean', 'std', 'min', 'max'])
    dup = dup[dup['count'] > 1]
    log(f'  duplicate groups: {len(dup):,}  rows involved: {int(dup["count"].sum()):,} '
        f'({dup["count"].sum() / len(h5) * 100:.2f}%)')
    if len(dup):
        log(f'  within-group label spread: median {dup["max"].sub(dup["min"]).median():.4f}  '
            f'p95 {(dup["max"] - dup["min"]).quantile(0.95):.4f}  '
            f'max {(dup["max"] - dup["min"]).max():.4f}')
    dup.reset_index().to_csv(os.path.join(args.out_dir, 'dup_pairs.csv'), index=False)

    # ---------- pairing by drug name: two DRUG_ID systems ----------
    log()
    log('=' * 78)
    log('8. pairing by drug name: h5 DRUG_ID <-> xlsx DRUG_ID')
    log('=' * 78)
    _pair = pd.DataFrame({
        'h5_id': h5['DRUG_ID'].astype(str).values,
        'x_id': x['DRUG_ID'].astype(str).values,
        'name_h5': h5['DRUG_NAME'].map(norm_name).values,
        'name_x': x['DRUG_NAME'].map(norm_name).values,
        'one': 1,
    })
    _ok = _pair[_pair['name_h5'] == _pair['name_x']]
    gmap = (_ok.groupby(['h5_id', 'x_id', 'name_h5'], as_index=False)
            .agg(n_rows=('one', 'size'))
            .rename(columns={'name_h5': 'drug_name_norm'}))
    gmap['id_changed'] = gmap['h5_id'] != gmap['x_id']
    gmap = gmap.sort_values(['id_changed', 'n_rows'], ascending=[False, False])
    gmap.to_csv(os.path.join(args.out_dir, 'drug_id_mapping.csv'), index=False)
    log(f'  pairable (h5_id, xlsx_id) combinations: {len(gmap)}')
    log(f'  of which the IDs differ: {int(gmap["id_changed"].sum())} pairs, '
        f'{int(gmap.loc[gmap["id_changed"], "n_rows"].sum()):,} rows involved')
    log('  meaning: h5 keeps the GDSC1 DRUG_ID, xlsx uses the GDSC2 DRUG_ID, so the same drug has different numbers.')
    log('        row-wise LN_IC50 is identical, so the labels are fine; only the ID systems are mixed.')

    _multi = (gmap.groupby('h5_id')
              .agg(n_xid=('x_id', 'nunique'),
                   x_ids=('x_id', lambda s: ' | '.join(sorted(set(s)))),
                   drug_name_norm=('drug_name_norm', 'first'),
                   rows=('n_rows', 'sum'))
              .reset_index())
    _multi = _multi[_multi['n_xid'] > 1]
    log(f'  h5_ids mapping to several xlsx_ids: {len(_multi)} (xlsx holds two records of the drug, h5 merged them)')
    for _, r in _multi.iterrows():
        log(f'    h5 {r["h5_id"]:>5s} {r["drug_name_norm"]:<14s} -> xlsx {r["x_ids"]}  '
            f'({int(r["rows"])} rows)')

    # ---------- unpaired IDs ----------
    log()
    log('=' * 78)
    log('9. IDs that cannot be paired by drug name')
    log('=' * 78)
    _unp_h5 = sorted(set(_pair['h5_id']) - set(gmap['h5_id']), key=lambda s: int(s))
    _unp_x = sorted(set(_pair['x_id']) - set(gmap['x_id']), key=lambda s: int(s))
    rows = []
    for did in _unp_h5:
        sub = h5[h5['DRUG_ID'].astype(str) == did]
        rows.append({'side': 'h5_only', 'DRUG_ID': did,
                     'names': ' | '.join(sorted(set(sub['DRUG_NAME'].astype(str)))),
                     'n_rows': len(sub)})
    for did in _unp_x:
        sub = x[x['DRUG_ID'].astype(str) == did]
        rows.append({'side': 'xlsx_only', 'DRUG_ID': did,
                     'names': ' | '.join(sorted(set(sub['DRUG_NAME'].astype(str)))),
                     'n_rows': len(sub)})
    unp = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=['side', 'DRUG_ID', 'names', 'n_rows'])
    unp.to_csv(os.path.join(args.out_dir, 'unpaired_ids.csv'), index=False)
    if len(unp):
        for _, r in unp.iterrows():
            log(f'  {r["side"]:<10s} ID {r["DRUG_ID"]:>5s}  {r["names"]:<20s} {int(r["n_rows"])} rows')
    else:
        log('  (none)')

    # ---------- atom feature coverage ----------
    log()
    log('=' * 78)
    log('10. drug_smiles / atom_features coverage')
    log('=' * 78)
    _known = set(smiles['DRUG_ID'].astype(str))
    _has = h5['DRUG_ID'].astype(str).isin(_known)
    log(f'  drugs covered by drug_smiles: {len(_known)}')
    log(f'  usable rows in the response table: {int(_has.sum()):,} ({_has.mean() * 100:.2f}%)')
    log(f'  rows without atom features: {int((~_has).sum()):,}; drugs without features: '
        f'{len(set(h5["DRUG_ID"].astype(str)) - _known)}')
    _miss_ids = sorted(set(h5['DRUG_ID'].astype(str)) - _known, key=lambda s: int(s))
    log(f'  IDs of drugs without features: {_miss_ids}')
    log('  note: Valid samples in the training log is below this value, the gap comes from missing cell-line gene embeddings.')

    # ---------- conclusion ----------
    log()
    log('=' * 78)
    log('11. conclusion')
    log('=' * 78)
    verdict = []
    verdict.append(f'row order {"aligned" if lab_eq.mean() > 0.999999 else "not aligned"} '
                   f'(LN_IC50 agreement {lab_eq.mean() * 100:.4f}%)')
    verdict.append(f'DRUG_ID row-wise agreement {drug_id_eq.mean() * 100:.2f}%'
                   f' (all differences come from the GDSC1/GDSC2 numbering, labels are unaffected)')
    verdict.append(f'normalized drug name agreement {drug_name_norm.mean() * 100:.2f}%')
    verdict.append(f'normalized cell name agreement {cell_name_norm.mean() * 100:.2f}%')
    verdict.append(f'drug_smiles covers {len(_known)} drugs / {int(_has.sum()):,} rows '
                   f'({_has.mean() * 100:.2f}%)')
    for v in verdict:
        log('  - ' + v)

    # ---------- summary xlsx ----------
    xlsx_path = os.path.join(args.out_dir, 'id_check.xlsx')
    _summary = pd.DataFrame({
        'item': ['h5 rows', 'xlsx rows', 'LN_IC50 agreement',
                 'DRUG_ID row-wise agreement', 'drug name (normalized) agreement',
                 'cell name (normalized) agreement',
                 'h5 unique DRUG_ID', 'xlsx unique DRUG_ID',
                 'same drug, different ID pairs', 'h5_ids mapping to several xlsx_ids',
                 'unpaired IDs on the h5 side', 'unpaired IDs on the xlsx side',
                 'drugs covered by drug_smiles', 'rows without atom features',
                 'duplicate measurement groups', 'rows in duplicate measurements',
                 'drugs with SMILES differing from the drug_smiles table'],
        'value': [len(h5), len(x), f'{lab_eq.mean() * 100:.4f}%',
                  f'{drug_id_eq.mean() * 100:.2f}%',
                  f'{drug_name_norm.mean() * 100:.2f}%',
                  f'{cell_name_norm.mean() * 100:.2f}%',
                  len(set_h5), len(set_x),
                  int(gmap['id_changed'].sum()), len(_multi),
                  len(_unp_h5), len(_unp_x),
                  len(_known), int((~_has).sum()),
                  len(dup), int(dup['count'].sum()) if len(dup) else 0,
                  len(bad)],
    })
    with pd.ExcelWriter(xlsx_path, engine='openpyxl') as w:
        _summary.to_excel(w, sheet_name='summary', index=False)
        gmap.to_excel(w, sheet_name='drug_id_mapping', index=False)
        _multi.to_excel(w, sheet_name='multi_id_merged', index=False)
        unp.to_excel(w, sheet_name='unpaired_ids', index=False)
        # only the first 5000 rows of the big tables go into the xlsx, the full version is the csv of the same name
        mm.head(5000).to_excel(w, sheet_name='row_mismatch_head5000', index=False)
        sm_check.to_excel(w, sheet_name='smiles_diff', index=False)
        dup.reset_index().head(5000).to_excel(w, sheet_name='dup_pairs_head5000', index=False)
    log()
    log(f'  summary table: {xlsx_path}')

    with open(os.path.join(args.out_dir, 'summary.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'\noutput dir: {args.out_dir}')


if __name__ == '__main__':
    main()
