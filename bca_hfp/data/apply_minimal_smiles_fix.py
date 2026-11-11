# bca_hfp/data/apply_minimal_smiles_fix.py
"""minimal SMILES fix: replace only drugs whose InChIKey disagrees with the official one, leave the rest untouched.

Why:
  overwriting everything with PubChem lookups turns many correct SMILES into a
  different representation (salt/stereo/protonation), which perturbs the Morgan
  fingerprints for no reason and makes old and new results incomparable.
  This script only touches drugs whose first 14 InChIKey chars differ from the
  official key.

Rules (per DRUG_ID):
  old exact / connectivity                    -> keep the old SMILES
  old MISMATCH + candidate exact/connectivity -> use the candidate (desalted main fragment)
  old MISMATCH + candidate empty              -> blank it (prefer missing features over a wrong match)
  old MISMATCH + candidate still MISMATCH     -> keep the old value, flag keep_old_still_bad (manual review)
  old empty + candidate has a valid structure -> fill it back
  otherwise                                   -> keep the old value

Only the SMILES column changes; row count, row order and label columns stay as they are.
"""
import argparse
import os

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

_SMI_COLS = ['DRUG_SMILE', 'SMILES', 'smiles', 'drug_smiles']
_LFC = rdMolStandardize.LargestFragmentChooser()


def pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f'no column found, candidates={candidates}, actual={list(df.columns)}')


def desalt_smiles(smi):
    """return the desalted main-fragment SMILES; None if parsing fails."""
    if not isinstance(smi, str) or not smi.strip() or smi.lower() == 'nan':
        return None
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    try:
        mol = _LFC.choose(mol)
    except Exception:
        pass
    return Chem.MolToSmiles(mol)


def inchi_key(smi):
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


def verdict(key, official_key):
    if not key:
        return 'empty'
    if not isinstance(official_key, str) or len(official_key) < 14:
        return 'no_official'
    if key == official_key:
        return 'exact'
    if key[:14] == official_key[:14]:
        return 'connectivity'
    return 'MISMATCH'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--old_csv', required=True, help='pre-fix backup (holds the originally correct SMILES)')
    ap.add_argument('--candidate_csv', required=True, help='corrected version from the full PubChem lookup')
    ap.add_argument('--official', required=True, help='screened_compounds_rel_8.5_inchi.csv')
    ap.add_argument('--out_csv', required=True)
    ap.add_argument('--report_csv', required=True)
    ap.add_argument('--drug_name_col', default='DRUG_NAME')
    args = ap.parse_args()

    old = pd.read_csv(args.old_csv)
    cand = pd.read_csv(args.candidate_csv)
    off = pd.read_csv(args.official)

    if len(old) != len(cand):
        raise ValueError(f'row count mismatch: old={len(old)} candidate={len(cand)}')

    smi_old = pick_col(old, _SMI_COLS)
    smi_cand = pick_col(cand, _SMI_COLS)

    official = {}
    for r in off.itertuples():
        did = int(getattr(r, 'DRUG_ID'))
        key = getattr(r, 'INCHI_KEY', None)
        if pd.notna(key) and isinstance(key, str) and len(key) >= 14:
            official.setdefault(did, key)

    old_first = old.groupby('DRUG_ID')[smi_old].first()
    cand_first = cand.groupby('DRUG_ID')[smi_cand].first()
    name_map = (old.groupby('DRUG_ID')[args.drug_name_col].first()
                if args.drug_name_col in old.columns else {})

    rows = []
    final_by_drug = {}
    for did in sorted(set(old_first.index) | set(cand_first.index)):
        did_int = int(did)
        o = old_first.get(did)
        c = cand_first.get(did)
        off_key = official.get(did_int)

        o_key = inchi_key(o)
        c_key = inchi_key(c)
        o_v = verdict(o_key, off_key)
        c_v = verdict(c_key, off_key)

        action = 'keep_old'
        final = o if isinstance(o, str) else ''

        if o_v in ('exact', 'connectivity'):
            action = 'keep_old_ok'
            final = o
        elif o_v == 'MISMATCH':
            if c_v in ('exact', 'connectivity'):
                action = 'replaced'
                final = desalt_smiles(c) or c
            elif c_v == 'empty':
                action = 'emptied'
                final = ''
            else:
                action = 'keep_old_still_bad'
                final = o
        elif o_v == 'empty':
            if c_v in ('exact', 'connectivity'):
                action = 'filled'
                final = desalt_smiles(c) or c
            else:
                action = 'keep_empty'
                final = ''
        # no_official -> keep_old

        if isinstance(final, str):
            final = final.strip()
            if final.lower() == 'nan':
                final = ''

        final_by_drug[did] = final
        rows.append({
            'DRUG_ID': did_int,
            'DRUG_NAME': name_map.get(did, '') if hasattr(name_map, 'get') else '',
            'old_smiles': o if isinstance(o, str) else '',
            'candidate_smiles': c if isinstance(c, str) else '',
            'final_smiles': final,
            'old_key': o_key or '',
            'candidate_key': c_key or '',
            'official_key': off_key or '',
            'old_verdict': o_v,
            'candidate_verdict': c_v,
            'action': action,
        })

    report = pd.DataFrame(rows)

    # write back: replace only the SMILES column, row order and other columns untouched
    out = old.copy()
    out[smi_old] = out['DRUG_ID'].map(final_by_drug).fillna('')
    # keep blanks as NaN, matching the original convention
    out.loc[out[smi_old] == '', smi_old] = np.nan

    print('=' * 78)
    print('action summary')
    print('=' * 78)
    print(report['action'].value_counts().to_string())

    still_bad = report[report['action'] == 'keep_old_still_bad']
    print(f'\nneeds manual review (old value itself wrong and candidate did not fix it): {len(still_bad)}')
    if len(still_bad):
        print(still_bad[['DRUG_ID', 'DRUG_NAME', 'old_key', 'official_key']].to_string(index=False))

    changed = int((report['old_smiles'].fillna('') != report['final_smiles'].fillna('')).sum())
    print(f'\nchanged drugs: {changed} / {len(report)}')
    print(f'output rows: {len(out)} (was {len(old)})')
    for c in ['GDSC_ID', 'DRUG_ID', 'LN_IC50']:
        if c in old.columns and c in out.columns:
            if c == 'LN_IC50':
                bad = int((~np.isclose(old[c].values.astype(float),
                                       out[c].values.astype(float), rtol=0, atol=1e-9)).sum())
            else:
                bad = int((old[c].astype(str).values != out[c].astype(str).values).sum())
            print(f'  {c} mismatched rows: {bad}')

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    report.to_csv(args.report_csv, index=False)
    print(f'\nwrote: {args.out_csv}')
    print(f'report: {args.report_csv}')


if __name__ == '__main__':
    main()
