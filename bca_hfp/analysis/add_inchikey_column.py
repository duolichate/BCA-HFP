# bca_hfp/analysis/add_inchikey_column.py
"""write the official INCHI_KEY column into GDSC_RES.csv so DRUG_SMILE can be verified.

why
---
DRUG_SMILE comes from the official compound INCHI_KEY via PubChem. with INCHI_KEY stored alongside:
  - anyone can compute the InChIKey of DRUG_SMILE with rdkit and compare it row by row with this column
  - no dependency on how PubChem spells the string, reproducibility rests on the two official files only

column position: before DRUG_SMILE.

usage:
python bca_hfp/analysis/add_inchikey_column.py \
  --csv ../Data/GDSC_RES.csv \
  --inchi ../Data/screened_compounds_rel_8.5_inchi.csv \
  --backup ../Data/GDSC_RES_backup_before_inchikey_col.csv
"""
import argparse
import os
import shutil

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--inchi', required=True)
    ap.add_argument('--backup', required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.csv, low_memory=False)
    inc = pd.read_csv(args.inchi, low_memory=False)
    inc['DRUG_ID'] = inc['DRUG_ID'].astype(str)

    # a DRUG_ID should have exactly one record in the compound table
    dup = inc.groupby('DRUG_ID').size()
    dup = dup[dup > 1]
    if len(dup):
        print(f'[warn] {len(dup)} DRUG_IDs have several records in the compound table, taking the first: {list(dup.index)[:10]}')
    key_map = (inc.drop_duplicates('DRUG_ID')
               .set_index('DRUG_ID')['INCHI_KEY'].to_dict())

    before = list(df.columns)
    df['INCHI_KEY'] = df['DRUG_ID'].astype(str).map(key_map)
    cols = [c for c in df.columns if c != 'INCHI_KEY']
    i = cols.index('DRUG_SMILE')
    df = df[cols[:i] + ['INCHI_KEY'] + cols[i:]]

    if not os.path.exists(args.backup):
        shutil.copy2(args.csv, args.backup)
        print(f'backed up -> {args.backup}')
    df.to_csv(args.csv, index=False)

    # ---- self-check ----
    chk = pd.read_csv(args.csv, low_memory=False)
    has_smi = chk['DRUG_SMILE'].notna()
    has_key = chk['INCHI_KEY'].notna()
    print('\ncolumns:', list(chk.columns))
    print(f'rows {len(chk):,}')
    print(f'rows with SMILES {int(has_smi.sum()):,} ({has_smi.mean() * 100:.2f}%)')
    print(f'rows with INCHI_KEY {int(has_key.sum()):,} ({has_key.mean() * 100:.2f}%)')
    print(f'both {int((has_smi & has_key).sum()):,}')
    print(f'SMILES but no KEY {int((has_smi & ~has_key).sum()):,}')
    print(f'KEY but no SMILES {int((~has_smi & has_key).sum()):,}')
    print(f'drugs with structure {chk.loc[has_smi, "DRUG_ID"].nunique()}; '
          f'drugs with KEY {chk.loc[has_key, "DRUG_ID"].nunique()}')
    # one drug with several KEYs
    g = chk.dropna(subset=['INCHI_KEY']).groupby(chk['DRUG_ID'].astype(str))['INCHI_KEY'].nunique()
    print(f'drugs with several KEYs: {int((g > 1).sum())}')
    print('\nwhether DRUG_SMILE and INCHI_KEY are the same molecule still needs an rdkit check:')
    print('  Chem.MolToInchiKey(Chem.MolFromSmiles(row.DRUG_SMILE)) == row.INCHI_KEY')


if __name__ == '__main__':
    main()
