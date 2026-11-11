# bca_hfp/analysis/compare_smiles_sets.py
"""compare old and new DRUG_SMILE: RDKit canonicalization + InChIKey to tell whether the molecule itself changed.

different strings do not mean a different molecule (aromatic form, kekulization, atom order all vary).
canonicalize first, then compare, to see whether old drugs keep the same atom_features after h5 regeneration.

usage (needs rdkit):
python bca_hfp/analysis/compare_smiles_sets.py \
  --old ../Data/GDSC_RES_backup_before_smiles_inchikey.csv \
  --new ../Data/GDSC_RES.csv \
  --out ../Result/diag/smiles_change_report.csv
"""
import argparse

import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog('rdApp.*')


def canon(smi):
    if not isinstance(smi, str) or not smi.strip():
        return None, None
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None, None
    try:
        can = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
        key = Chem.MolToInchiKey(mol)
    except Exception:
        return None, None
    return can, key


def per_drug(df):
    df = df.copy()
    df['_id'] = df['DRUG_ID'].astype(str)
    sub = df.dropna(subset=['DRUG_SMILE'])
    g = sub.groupby('_id')['DRUG_SMILE'].first()
    name = df.drop_duplicates('_id').set_index('_id')['DRUG_NAME'].to_dict()
    nrow = df.groupby('_id').size().to_dict()
    return {d: (s, name.get(d, ''), nrow.get(d, 0)) for d, s in g.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--old', required=True)
    ap.add_argument('--new', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    old = per_drug(pd.read_csv(args.old, low_memory=False))
    new = per_drug(pd.read_csv(args.new, low_memory=False))

    rows = []
    for d in sorted(set(old) | set(new), key=lambda s: int(s)):
        so = old.get(d, (None, '', 0))[0]
        sn = new.get(d, (None, '', 0))[0]
        nm = (new.get(d) or old.get(d))[1]
        n = (new.get(d) or old.get(d))[2]
        co, ko = canon(so)
        cn, kn = canon(sn)
        if ko is None and kn is None:
            status = 'both_missing'
        elif ko is None:
            status = 'added'
        elif kn is None:
            status = 'removed'
        elif ko == kn:
            status = 'same_molecule'
        else:
            status = 'MOLECULE_CHANGED'
        rows.append({'DRUG_ID': d, 'DRUG_NAME': nm, 'n_rows': n,
                     'status': status,
                     'old_canon': co, 'new_canon': cn,
                     'old_key': ko, 'new_key': kn})

    rep = pd.DataFrame(rows)
    rep.to_csv(args.out, index=False)

    print('=' * 70)
    print(rep['status'].value_counts().to_string())
    print('=' * 70)
    chg = rep[rep['status'] == 'MOLECULE_CHANGED']
    if len(chg):
        print(f'\ndrugs whose molecule actually changed ({len(chg)} drugs, {int(chg["n_rows"].sum())} rows):')
        for _, r in chg.iterrows():
            print(f'  {r["DRUG_ID"]:>5s} {str(r["DRUG_NAME"])[:22]:<22s} {int(r["n_rows"]):5d} rows')
            print(f'        old {str(r["old_canon"])[:80]}')
            print(f'        new {str(r["new_canon"])[:80]}')
    else:
        print('\nno molecule changed, old drugs should have the same atom_features after h5 regeneration.')
    print(f'\nreport: {args.out}')


if __name__ == '__main__':
    main()
