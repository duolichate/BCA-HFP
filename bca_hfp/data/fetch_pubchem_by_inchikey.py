# bca_hfp/data/fetch_pubchem_by_inchikey.py
"""exact PubChem lookup by InChIKey, to identify what the remaining MISMATCH drugs actually are.

Two queries per DRUG_ID:
  1) official INCHI_KEY -> PubChem Title / CanonicalSMILES (what the official structure is)
  2) key computed from the current SMILES -> PubChem Title (what drug we actually hold)

Prints a side-by-side table, enough to tell a salt/metal representation
difference from a real mismatch. Read-only over the network; no data file is
modified.
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

import pandas as pd
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

_SMI_COLS = ['DRUG_SMILE', 'SMILES', 'smiles', 'drug_smiles']
_LFC = rdMolStandardize.LargestFragmentChooser()
_PUBCHEM = ('https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/'
            '{key}/property/SMILES,CanonicalSMILES,IsomericSMILES,Title/JSON')


def pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f'no column found, candidates={candidates}, actual={list(df.columns)}')


def key_of(smi):
    if not isinstance(smi, str) or not smi.strip() or smi.lower() == 'nan':
        return None
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    try:
        mol = _LFC.choose(mol)
    except Exception:
        pass
    try:
        return Chem.MolToInchiKey(mol)
    except Exception:
        return None


def query(key):
    if not isinstance(key, str) or len(key) < 14:
        return None, None, None, 'no_key'
    url = _PUBCHEM.format(key=key)
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'smiles-audit/1.0'})
        with urllib.request.urlopen(req, timeout=25) as resp:
            data = json.loads(resp.read().decode('utf-8'))
        props = data['PropertyTable']['Properties'][0]
        # the SMILES field name differs across PubChem API versions; try each in turn
        smi = (props.get('CanonicalSMILES') or props.get('SMILES')
               or props.get('ConnectivitySMILES') or props.get('IsomericSMILES') or '')
        return props.get('CID'), props.get('Title'), smi, 'ok'
    except urllib.error.HTTPError as e:
        return None, None, None, f'http_{e.code}'
    except Exception as e:
        return None, None, None, f'error:{type(e).__name__}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--official', required=True)
    ap.add_argument('--old_csv', required=True, help='GDSC_RES holding the current SMILES')
    ap.add_argument('--drug_ids', nargs='+', type=int, required=True)
    ap.add_argument('--out_csv', required=True)
    args = ap.parse_args()

    off = pd.read_csv(args.official)
    old = pd.read_csv(args.old_csv)
    smi_col = pick_col(old, _SMI_COLS)

    off_map = {}
    for r in off.itertuples():
        did = int(getattr(r, 'DRUG_ID'))
        off_map.setdefault(did, {
            'name': getattr(r, 'DRUG_NAME', ''),
            'target': getattr(r, 'TARGET', ''),
            'pathway': getattr(r, 'TARGET_PATHWAY', ''),
            'key': getattr(r, 'INCHI_KEY', None),
        })

    old_first = old.groupby('DRUG_ID')[smi_col].first()

    rows = []
    for did in args.drug_ids:
        info = off_map.get(did, {})
        off_key = info.get('key')
        old_smi = old_first.get(did)
        old_key = key_of(old_smi)

        cid1, title1, smi1, st1 = query(off_key)
        time.sleep(0.35)
        cid2, title2, smi2, st2 = query(old_key)
        time.sleep(0.35)

        rows.append({
            'DRUG_ID': did,
            'name_in_gdsc_res': info.get('name', ''),
            'target': info.get('target', ''),
            'pathway': info.get('pathway', ''),
            'official_key': off_key or '',
            'official_pubchem_title': title1 or '',
            'official_pubchem_smiles': smi1 or '',
            'official_query': st1,
            'old_key': old_key or '',
            'old_pubchem_title': title2 or '',
            'old_query': st2,
            'verdict': ('same' if off_key and old_key and off_key == old_key
                        else 'differs'),
        })

    out = pd.DataFrame(rows)
    pd.set_option('display.width', 240)
    pd.set_option('display.max_colwidth', 70)
    print(out.to_string(index=False))
    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f'\nsaved -> {args.out_csv}')
    print('\nhow to read:')
    print('  official_query=ok and title matches DRUG_NAME -> official key trusted, rebuild from official_smiles')
    print('  official_query=http_404 -> official key not found in PubChem, keep the old SMILES and log it')
    print('  old_pubchem_title names another drug -> the old structure is a confirmed mismatch')


if __name__ == '__main__':
    main()
