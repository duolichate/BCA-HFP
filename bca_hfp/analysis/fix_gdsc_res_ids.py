# bca_hfp/analysis/fix_gdsc_res_ids.py
"""fix the DRUG_ID / DRUG_NAME / DRUG_SMILE columns of Data/GDSC_RES.csv against the official
GDSC2_fitted_dose_response_27Oct23.xlsx.

why
----
GDSC_RES.csv was stitched from the GDSC1 + GDSC2 sheets and then left-joined with the compound structure
table on DRUG_ID, which left:
  1) the same drug under its GDSC1 id (e.g. Erlotinib h5=1, GDSC2=1168) -- 27 pairs, 21,200 rows
  2) 6 groups of rows whose structure/name belong to drug A while LN_IC50 comes from drug B -- 4,466 rows
     (h5 1580 GSK2801 <-> GDSC2 1940 WIKI4 and so on)

only these three columns are touched, LN_IC50 / AUC / RMSE / Z_SCORE / cell line columns stay as they are.
for the 6 misplaced drugs the standard SMILES comes from PubChem by official INCHI_KEY; if not found the
row SMILES is left empty (better one trainable drug less than training on a wrong structure).

usage
----
python bca_hfp/analysis/fix_gdsc_res_ids.py \
  --csv ../Data/GDSC_RES.csv \
  --xlsx ../Data/GDSC2_fitted_dose_response_27Oct23.xlsx \
  --inchi ../Data/screened_compounds_rel_8.5_inchi.csv \
  --backup ../Data/GDSC_RES_backup_before_id_fix.csv \
  --report ../Result/diag/id_check/fix_report.txt
"""
import argparse
import json
import os
import shutil
import time
import urllib.request

import numpy as np
import pandas as pd

_PUBCHEM = ('https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/'
            '{key}/property/SMILES,MolecularFormula/JSON')


def norm_name(s):
    return (str(s).strip().upper().replace('-', '').replace(' ', '')
            .replace('_', '').replace('.', ''))


def pubchem_smiles(key, cache):
    if not isinstance(key, str) or len(key) < 14:
        return None, None
    if key in cache:
        return cache[key].get('smiles'), cache[key].get('formula')
    url = _PUBCHEM.format(key=key)
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'gdsc-id-fix/1.0'})
            with urllib.request.urlopen(req, timeout=25) as resp:
                data = json.loads(resp.read().decode('utf-8'))
            p = data['PropertyTable']['Properties'][0]
            smi, formula = p.get('SMILES'), p.get('MolecularFormula')
            cache[key] = {'smiles': smi, 'formula': formula}
            return smi, formula
        except Exception:
            time.sleep(1.5 * (attempt + 1))
    cache[key] = {'smiles': None, 'formula': None}
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--xlsx', required=True)
    ap.add_argument('--inchi', required=True)
    ap.add_argument('--backup', required=True)
    ap.add_argument('--report', required=True)
    ap.add_argument('--cache', default='')
    args = ap.parse_args()

    lines = []

    def log(msg=''):
        print(msg)
        lines.append(str(msg))

    df = pd.read_csv(args.csv, low_memory=False)
    x = pd.read_excel(args.xlsx, sheet_name=0)
    inchi = pd.read_csv(args.inchi, low_memory=False)
    log(f'GDSC_RES.csv : {df.shape}')
    log(f'xlsx         : {x.shape}')
    if len(df) != len(x):
        raise SystemExit('row count differs, cannot align by row')

    new_id_num = x['DRUG_ID'].values          # keep numeric, used when writing back
    new_id = x['DRUG_ID'].astype(str).values  # only for lookups / comparison
    new_name = x['DRUG_NAME'].astype(str).str.upper().values
    old_id = df['DRUG_ID'].astype(str).values
    old_name = df['DRUG_NAME'].astype(str).values

    same = np.array([norm_name(a) == norm_name(b) for a, b in zip(old_name, new_name)])
    log(f'\nrows with matching drug name: {int(same.sum()):,} ({same.mean() * 100:.2f}%)')
    log(f'rows with mismatched drug name: {int((~same).sum()):,}')

    # ---- new ids that need a new structure ----
    inchi['DRUG_ID'] = inchi['DRUG_ID'].astype(str)
    key_map = dict(zip(inchi['DRUG_ID'], inchi['INCHI_KEY']))
    need_ids = sorted(set(new_id[~same]), key=lambda s: int(s))
    log(f'drugs that need a structure (GDSC2 ID): {need_ids}')

    cache = {}
    if args.cache and os.path.exists(args.cache):
        with open(args.cache, 'r', encoding='utf-8') as f:
            cache = json.load(f)
    fix = {}
    log('\nfetch standard SMILES from PubChem by official INCHI_KEY:')
    for did in need_ids:
        row = x[x['DRUG_ID'].astype(str) == did]
        nm = row['DRUG_NAME'].iloc[0] if len(row) else '?'
        key = key_map.get(did)
        smi, formula = pubchem_smiles(key, cache)
        fix[did] = smi
        log(f'  {did:>5s} {str(nm):<18s} key={key}  formula={formula}')
        log(f'        SMILES={smi}')
    if args.cache:
        with open(args.cache, 'w', encoding='utf-8') as f:
            json.dump(cache, f, ensure_ascii=False, indent=1)

    # ---- write back ----
    out = df.copy()
    out['DRUG_ID'] = new_id_num
    out['DRUG_NAME'] = new_name
    smi_col = out['DRUG_SMILE'].astype(object).copy()
    n_blank = 0
    for i in np.where(~same)[0]:
        v = fix.get(new_id[i])
        smi_col.iat[i] = v
        if v is None:
            n_blank += 1
    out['DRUG_SMILE'] = smi_col
    log(f'\nrows with SMILES cleared: {n_blank:,}')

    # ---- backup + overwrite ----
    if not os.path.exists(args.backup):
        shutil.copy2(args.csv, args.backup)
        log(f'original backed up -> {args.backup}')
    else:
        log(f'backup already exists, not overwritten -> {args.backup}')
    out.to_csv(args.csv, index=False)
    log(f'wrote corrected {args.csv}  ({out.shape})')

    # ---- self-check ----
    chk = pd.read_csv(args.csv, low_memory=False)
    log('\n' + '=' * 70)
    log('self-check after the fix (row by row against the xlsx)')
    log('=' * 70)
    log(f'  rows            : {len(chk)} vs {len(x)}  {"OK" if len(chk) == len(x) else "FAIL"}')
    id_eq = (chk['DRUG_ID'].astype(str).values == x['DRUG_ID'].astype(str).values)
    nm_eq = np.array([norm_name(a) == norm_name(b)
                      for a, b in zip(chk['DRUG_NAME'], x['DRUG_NAME'])])
    for col in ['LN_IC50', 'AUC', 'RMSE', 'Z_SCORE']:
        a = pd.to_numeric(chk[col], errors='coerce').values
        b = pd.to_numeric(x[col], errors='coerce').values
        ok = np.isclose(a, b, atol=1e-6, equal_nan=True)
        log(f'  {col:<15s}: {ok.mean() * 100:.4f}% identical')
    log(f'  DRUG_ID         : {id_eq.mean() * 100:.4f}% identical')
    log(f'  DRUG_NAME (norm): {nm_eq.mean() * 100:.4f}% identical')
    log(f'  unique DRUG_ID  : {chk["DRUG_ID"].nunique()}')
    log(f'  rows with SMILES: {int(chk["DRUG_SMILE"].notna().sum()):,} '
        f'({chk["DRUG_SMILE"].notna().mean() * 100:.2f}%)')
    log(f'  drugs with structure: {chk.loc[chk["DRUG_SMILE"].notna(), "DRUG_ID"].nunique()}')
    _sm = (chk.dropna(subset=['DRUG_SMILE'])
           .groupby(chk['DRUG_ID'].astype(str))['DRUG_SMILE'].nunique())
    log(f'  drugs with several SMILES: {int((_sm > 1).sum())}')

    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'\nreport: {args.report}')


if __name__ == '__main__':
    main()
