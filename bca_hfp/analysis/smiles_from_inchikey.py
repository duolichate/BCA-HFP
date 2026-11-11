# bca_hfp/analysis/smiles_from_inchikey.py
"""make DRUG_SMILE in GDSC_RES.csv come from a single chain: official compound table INCHI_KEY -> PubChem.

why
---
the old GDSC_RES.csv SMILES came from an R script: drug name/alias -> fuzzy match in the CTRP library.
that path is not reproducible (alias table version and match order both change the result), and the six
mis-assigned drug names came from it.

replaced by:
    screened_compounds_rel_8.5_inchi.csv  (official GDSC, with INCHI_KEY)
      -> PubChem REST (by InChIKey)
      -> SMILES
      -> DRUG_SMILE column of GDSC_RES.csv

InChIKey is a unique hash of the molecule, so a lookup by it cannot land on another drug.
drugs not found (no INCHI_KEY in the compound table) are set empty and filtered downstream, no guessing.

output
----
  Data/GDSC_RES.csv                          DRUG_SMILE column overwritten
  Result/diag/smiles_from_inchikey.csv       per-drug provenance table (for reviewers)
  Data/GDSC_RES_backup_before_smiles_inchikey.csv   backup

usage
----
python bca_hfp/analysis/smiles_from_inchikey.py \
  --csv ../Data/GDSC_RES.csv \
  --inchi ../Data/screened_compounds_rel_8.5_inchi.csv \
  --cache ../Data/smiles_by_inchikey_cache.json \
  --backup ../Data/GDSC_RES_backup_before_smiles_inchikey.csv \
  --report ../Result/diag/smiles_from_inchikey.csv
"""
import argparse
import json
import os
import shutil
import time
import urllib.error
import urllib.request

import numpy as np
import pandas as pd

_PUBCHEM = ('https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/'
            '{key}/property/SMILES/JSON')


def fetch_pubchem(key, cache, retries=3):
    if not isinstance(key, str) or len(key) < 14:
        return None
    if key in cache:
        return cache[key]
    url = _PUBCHEM.format(key=key)
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'gdsc-smiles/1.0'})
            with urllib.request.urlopen(req, timeout=20) as r:
                data = json.loads(r.read().decode())
            smi = data['PropertyTable']['Properties'][0].get('SMILES')
            cache[key] = smi
            return smi
        except urllib.error.HTTPError as e:
            if e.code == 404:
                cache[key] = None
                return None
            time.sleep(1.0 * (i + 1))
        except Exception:
            time.sleep(1.0 * (i + 1))
    cache[key] = None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--inchi', required=True)
    ap.add_argument('--cache', required=True)
    ap.add_argument('--backup', required=True)
    ap.add_argument('--report', required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.csv, low_memory=False)
    inc = pd.read_csv(args.inchi, low_memory=False)
    inc['DRUG_ID'] = inc['DRUG_ID'].astype(str)
    df['_id'] = df['DRUG_ID'].astype(str)

    key_map = dict(zip(inc['DRUG_ID'], inc['INCHI_KEY']))
    name_map = (df.drop_duplicates('_id').set_index('_id')['DRUG_NAME'].to_dict())
    old_smi = (df.dropna(subset=['DRUG_SMILE'])
               .groupby('_id')['DRUG_SMILE'].first().to_dict())

    cache = {}
    if os.path.exists(args.cache):
        with open(args.cache, 'r', encoding='utf-8') as f:
            cache = json.load(f)

    ids = sorted(df['_id'].unique(), key=lambda s: int(s))
    print(f'drugs {len(ids)}; with InChIKey {sum(1 for d in ids if isinstance(key_map.get(d), str) and len(key_map[d]) > 10)}')

    smi_of, src_of = {}, {}
    for i, d in enumerate(ids):
        k = key_map.get(d)
        smi = fetch_pubchem(k, cache)
        smi_of[d] = smi
        src_of[d] = ('PubChem:InChIKey' if smi else
                     ('no_InChIKey' if not isinstance(k, str) or len(k) < 14
                      else 'not_in_PubChem'))
        if (i + 1) % 50 == 0:
            print(f'  {i + 1}/{len(ids)}')
            with open(args.cache, 'w', encoding='utf-8') as f:
                json.dump(cache, f)
    with open(args.cache, 'w', encoding='utf-8') as f:
        json.dump(cache, f)

    # write back DRUG_SMILE
    new_smi = df['_id'].map(smi_of)
    n_old = int(df['DRUG_SMILE'].notna().sum())
    df['DRUG_SMILE'] = new_smi.values
    n_new = int(df['DRUG_SMILE'].notna().sum())

    if not os.path.exists(args.backup):
        shutil.copy2(args.csv, args.backup)
        print(f'backed up -> {args.backup}')
    df.drop(columns=['_id']).to_csv(args.csv, index=False)

    # provenance table
    rep = pd.DataFrame({
        'DRUG_ID': ids,
        'DRUG_NAME': [name_map.get(d, '') for d in ids],
        'INCHI_KEY': [key_map.get(d, '') for d in ids],
        'SMILES': [smi_of.get(d) for d in ids],
        'source': [src_of.get(d) for d in ids],
        'n_rows': [int((df['_id'] == d).sum()) for d in ids],
        'old_SMILES': [old_smi.get(d) for d in ids],
    })
    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    rep.to_csv(args.report, index=False)

    # stats
    has_old = set(old_smi)
    has_new = {d for d in ids if smi_of.get(d)}
    print('\n' + '=' * 66)
    print(f'  drugs with old SMILES: {len(has_old)}   new: {len(has_new)}')
    print(f'  added: {sorted(has_new - has_old, key=lambda s: int(s))}')
    print(f'  lost: {sorted(has_old - has_new, key=lambda s: int(s))}')
    print(f'  rows with old SMILES: {n_old:,}   new: {n_new:,}')
    same = sum(1 for d in has_old & has_new if old_smi.get(d) == smi_of.get(d))
    print(f'  SMILES string identical: {same} / {len(has_old & has_new)} '
          f'(a different spelling does not mean a different molecule, compare after RDKit canonicalization)')
    print(f'  drugs with structure: {len(has_new)}')
    print('=' * 66)
    print(f'provenance table: {args.report}')


if __name__ == '__main__':
    main()
