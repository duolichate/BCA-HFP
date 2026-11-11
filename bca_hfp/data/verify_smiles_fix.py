# bca_hfp/data/verify_smiles_fix.py
"""verify the SMILES fix: per-drug InChIKey against the official GDSC annotation, and confirm labels/keys are unchanged.

Criterion (InChIKey):
  full 27 chars match   -> exact         same structure
  first 14 chars match  -> connectivity  same connectivity (salt/stereo/protonation differ)
  first 14 differ       -> MISMATCH      still wrong, must be handled

Read-only, no file is modified.
"""
import argparse
import os

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

_SMI_COLS = ['DRUG_SMILE', 'SMILES', 'smiles', 'drug_smiles']


def pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f'no column found, candidates={candidates}, actual={list(df.columns)}')


def make_keyer(desalt=True):
    lfc = rdMolStandardize.LargestFragmentChooser()

    def key(smi):
        if not isinstance(smi, str) or not smi.strip() or smi.lower() == 'nan':
            return None, 'empty'
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None, 'invalid_smiles'
        if desalt:
            try:
                mol = lfc.choose(mol)
            except Exception:
                pass
        try:
            return Chem.MolToInchiKey(mol), 'ok'
        except Exception:
            return None, 'inchi_failed'

    return key


def judge(new_key, official_key):
    if not new_key:
        return 'empty'
    if not isinstance(official_key, str) or len(official_key) < 14:
        return 'no_official'
    if new_key == official_key:
        return 'exact'
    if new_key[:14] == official_key[:14]:
        return 'connectivity'
    return 'MISMATCH'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--old_csv', required=True, help='GDSC_RES backup taken before the fix')
    ap.add_argument('--new_csv', required=True, help='GDSC_RES after the fix')
    ap.add_argument('--official', required=True, help='screened_compounds_rel_8.5_inchi.csv')
    ap.add_argument('--output_csv', default='smiles_fix_verification.csv')
    ap.add_argument('--keep_salt', action='store_true',
                    help='skip desalting before the InChIKey comparison (default: desalt first)')
    args = ap.parse_args()

    old = pd.read_csv(args.old_csv)
    new = pd.read_csv(args.new_csv)
    off = pd.read_csv(args.official)

    print('=' * 78)
    print('1) table shape / row count / label consistency')
    print('=' * 78)
    print(f'old rows={len(old)}  new rows={len(new)}')
    print(f'old cols={len(old.columns)}  new cols={len(new.columns)}')
    if len(old) != len(new):
        print('[FAIL] row counts differ, unusable for training')
    for c in ['GDSC_ID', 'DRUG_ID', 'LN_IC50']:
        if c not in old.columns or c not in new.columns:
            print(f'[WARN] missing column {c}')
            continue
        if c == 'LN_IC50':
            diff = ~np.isclose(old[c].values.astype(float), new[c].values.astype(float),
                               rtol=0, atol=1e-9)
        else:
            diff = old[c].astype(str).values != new[c].astype(str).values
        print(f'  {c}: {int(diff.sum())} mismatched rows'
              f'{"  [FAIL]" if diff.sum() else "  [OK]"}')

    print()
    print('=' * 78)
    print('2) one SMILES per DRUG_ID')
    print('=' * 78)
    old_smi_col = pick_col(old, _SMI_COLS)
    new_smi_col = pick_col(new, _SMI_COLS)
    g_old = old.groupby('DRUG_ID')[old_smi_col].nunique(dropna=True)
    g_new = new.groupby('DRUG_ID')[new_smi_col].nunique(dropna=True)
    print(f'  old: DRUG_IDs with several values {int((g_old > 1).sum())}')
    print(f'  new: DRUG_IDs with several values {int((g_new > 1).sum())}'
          f'{"  [FAIL]" if (g_new > 1).sum() else "  [OK]"}')

    print()
    print('=' * 78)
    print('3) per-drug InChIKey vs official')
    print('=' * 78)
    official = {int(r.DRUG_ID): r.INCHI_KEY for r in off.itertuples()
                if pd.notna(getattr(r, 'INCHI_KEY', None))}
    keyer = make_keyer(desalt=not args.keep_salt)

    old_first = old.groupby('DRUG_ID')[old_smi_col].first()
    new_first = new.groupby('DRUG_ID')[new_smi_col].first()

    rows = []
    for did in sorted(set(old_first.index) | set(new_first.index)):
        old_smi = old_first.get(did)
        new_smi = new_first.get(did)
        old_key, old_status = keyer(old_smi)
        new_key, new_status = keyer(new_smi)
        off_key = official.get(int(did))
        rows.append({
            'DRUG_ID': int(did),
            'old_smiles': old_smi if isinstance(old_smi, str) else '',
            'new_smiles': new_smi if isinstance(new_smi, str) else '',
            'old_key': old_key or '',
            'new_key': new_key or '',
            'official_key': off_key or '',
            'old_verdict': judge(old_key, off_key),
            'new_verdict': judge(new_key, off_key),
            'new_parse': new_status,
            'changed': bool(str(old_smi) != str(new_smi)),
        })
    res = pd.DataFrame(rows)

    print(f'  drugs: {len(res)}   with official INCHI_KEY: {len(official)}')
    for col in ['old_verdict', 'new_verdict']:
        vc = res[col].value_counts()
        print(f'\n  [{col}]')
        for k, v in vc.items():
            print(f'    {k:16s} {v}')

    bad = res[res.new_verdict == 'MISMATCH']
    print(f'\n  still MISMATCH after the fix: {len(bad)}'
          f'{"  [FAIL]" if len(bad) else "  [OK]"}')
    if len(bad):
        print(bad[['DRUG_ID', 'new_key', 'official_key']].to_string(index=False))

    print()
    print('=' * 78)
    print('4) data loss from blanked drugs')
    print('=' * 78)
    emptied = res[(res.new_verdict == 'empty') & (res.old_verdict != 'empty')]
    print(f'  had SMILES, blanked by the fix: {len(emptied)} drugs')
    if len(emptied):
        print('  ' + ', '.join(str(x) for x in emptied['DRUG_ID'].tolist()))
        print('  suggestion: check one by one whether the official INCHI_KEY really has no lookup; these drugs drop out of training')

    print()
    print('=' * 78)
    print('5) spot check of key drugs')
    print('=' * 78)
    for did in [1811, 1230, 1620, 1046, 1003, 1248]:
        r = res[res.DRUG_ID == did]
        if len(r):
            r = r.iloc[0]
            print(f'  {did}: old={r.old_verdict:12s} new={r.new_verdict:12s} '
                  f'changed={r.changed}')

    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    res.to_csv(args.output_csv, index=False)
    print(f'\ndetails saved -> {args.output_csv}')
    print('pass criterion: sections 1/2 all OK; section 3 has no new_verdict MISMATCH; section 4 loss acceptable.')


if __name__ == '__main__':
    main()
