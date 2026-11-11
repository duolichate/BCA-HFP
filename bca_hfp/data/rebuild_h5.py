# bca_hfp/data/rebuild_h5.py
"""rebuild the h5 from the corrected Data/GDSC_RES.csv, reusing existing atom_features where possible.

Background
----------
DRUG_ID in GDSC_RES.csv now follows GDSC2 naming, while drug_smiles / atom_features
in the h5 still sit in the 228 GDSC1 IDs. Replacing only original_data would break
the ID join and filter out nearly all training samples.

This script does three things:
  1) original_data <- new CSV (row-wise, with MIN_CONC / MAX_CONC)
  2) drug_smiles   <- SMILES text of the old table (already RDKit canonical), only DRUG_ID renumbered
  3) atom_features <- realigned row-wise with drug_smiles; new drugs sharing a SMILES copy the old features

Reuse rules (no FG-BERT rerun needed):
  - old ID still present -> use as is
  - old ID renamed via the 27-pair mapping -> use the mapped new ID
  - second entry of the 9 duplicate-ID drugs -> same SMILES as the main ID, copy features
Drugs that cannot be reused (the 6 with newly added structures) are left out of
drug_smiles and get filtered at training time.

Usage (numpy>=2, run on the server):
python bca_hfp/data/rebuild_h5.py \
  --old_h5 ../Data/GDSC_RES_FGBERT.h5 \
  --csv    ../Data/GDSC_RES.csv \
  --mapping ../Result/diag/id_check/drug_id_mapping.csv \
  --out    ../Data/GDSC_RES_FGBERT_v2.h5 \
  --backup ../Data/GDSC_RES_FGBERT_backup_before_id_fix.h5
"""
import argparse
import os
import shutil

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--old_h5', required=True)
    ap.add_argument('--csv', required=True)
    ap.add_argument('--mapping', required=True, help='drug_id_mapping.csv: h5_id -> x_id')
    ap.add_argument('--out', required=True)
    ap.add_argument('--backup', default='', help='back up here if out overwrites the original file')
    ap.add_argument('--overwrite', action='store_true', help='allow out == old_h5')
    args = ap.parse_args()

    if os.path.abspath(args.out) == os.path.abspath(args.old_h5) and not args.overwrite:
        raise SystemExit('out equals old_h5; pass --overwrite to replace it')
    if args.overwrite and args.backup and not os.path.exists(args.backup):
        shutil.copy2(args.old_h5, args.backup)
        print(f'backed up -> {args.backup}')

    print('reading old h5 ...')
    with pd.HDFStore(args.old_h5, 'r') as s:
        old_ds = s['drug_smiles'].copy()
        old_af = s['atom_features']['atom_features'].tolist()
    old_ds['DRUG_ID'] = old_ds['DRUG_ID'].astype(str)
    assert len(old_ds) == len(old_af), 'drug_smiles and atom_features row counts differ'
    print(f'  old drug_smiles: {len(old_ds)} rows, unique SMILES: {old_ds["DRUG_SMILE"].nunique()}')

    print('reading new CSV ...')
    df = pd.read_csv(args.csv, low_memory=False)
    df['DRUG_ID'] = df['DRUG_ID'].astype(str)
    has_smi = df.dropna(subset=['DRUG_SMILE'])
    # drug -> raw SMILES (a drug may have several rows; keep the first)
    drug_smi_raw = has_smi.groupby('DRUG_ID')['DRUG_SMILE'].first().to_dict()
    print(f'  drugs with structure: {len(drug_smi_raw)}, rows with structure: {len(has_smi):,}')

    print('reading ID mapping ...')
    mp = pd.read_csv(args.mapping)
    old2new = {str(a): str(b) for a, b in zip(mp['h5_id'], mp['x_id'])}
    print(f'  mapping pairs: {len(old2new)}')

    # ---- pass 1: old ID -> new ID ----
    entries = {}          # new_id -> (smiles, atom_feat)
    src = {}              # new_id -> provenance note
    for i, r in old_ds.iterrows():
        old_id = r['DRUG_ID']
        new_id = old2new.get(old_id, old_id)
        if new_id not in drug_smi_raw:
            continue
        if new_id in entries:
            print(f'  [warn] new ID {new_id} claimed by several old IDs, keeping the first ({src[new_id]})')
            continue
        entries[new_id] = (r['DRUG_SMILE'], old_af[i])
        src[new_id] = f'old {old_id}'

    # ---- pass 2: duplicate-ID entries sharing a SMILES, copy features ----
    smi2new = {}
    for nid, (smi, _) in entries.items():
        smi2new.setdefault(smi, nid)
    n_copy = 0
    for nid, smi_raw in drug_smi_raw.items():
        if nid in entries:
            continue
        # find another ID of the same record via the raw SMILES (duplicate-ID drug)
        dup_ids = [k for k, v in drug_smi_raw.items() if v == smi_raw and k in entries]
        if dup_ids:
            base = dup_ids[0]
            entries[nid] = (entries[base][0], entries[base][1])
            src[nid] = f'copied from {base}'
            n_copy += 1
    print(f'  features copied for same SMILES: {n_copy} drugs')

    missing = sorted(set(drug_smi_raw) - set(entries), key=lambda s: int(s))
    print(f'  drugs without usable atom_features ({len(missing)}): {missing}')

    # ---- assemble the three tables ----
    order = [d for d in drug_smi_raw if d in entries]
    new_ds = pd.DataFrame({
        'DRUG_ID': [int(d) if d.isdigit() else d for d in order],
        'DRUG_SMILE': [entries[d][0] for d in order],
    })
    new_af = pd.DataFrame({'atom_features': [entries[d][1] for d in order]})

    keep_ids = set(order)
    n_covered = int(df['DRUG_ID'].isin(keep_ids).sum())
    # original_data keeps all 242,036 rows (including rows without atom features),
    # matching the old h5; load_response filters them by isin(drug_atom_dict) later.
    new_orig = df.copy()
    new_orig['DRUG_ID'] = pd.to_numeric(new_orig['DRUG_ID'], errors='coerce').astype('int64')
    print(f'\nnew tables: drug_smiles {len(new_ds)} rows; original_data {len(new_orig):,} rows '
          f'({n_covered:,} with features, {n_covered / len(new_orig) * 100:.2f}%)')

    print(f'writing -> {args.out}')
    with pd.HDFStore(args.out, 'w') as s:
        s['original_data'] = new_orig
        s['drug_smiles'] = new_ds
        s['atom_features'] = new_af

    # ---- self-check ----
    with pd.HDFStore(args.out, 'r') as s:
        c_orig = s['original_data']
        c_ds = s['drug_smiles']
        c_af = s['atom_features']
    print('\nself-check:')
    print(f'  original_data  {c_orig.shape}')
    print(f'  drug_smiles    {c_ds.shape}')
    print(f'  atom_features  {c_af.shape}')
    ids_o = set(c_orig['DRUG_ID'].astype(str))
    ids_d = set(c_ds['DRUG_ID'].astype(str))
    print(f'  drugs in response table {len(ids_o)}, in feature table {len(ids_d)}, '
          f'covered rows {int(c_orig["DRUG_ID"].astype(str).isin(ids_d).sum()):,}')
    print(f'  uncovered drugs: {sorted(ids_o - ids_d, key=lambda s: int(s))}')
    print(f'  drugs with empty SMILES: {int(c_ds["DRUG_SMILE"].isna().sum())}')
    print(f'  atom_features shape: {np.asarray(c_af["atom_features"].iloc[0]).shape}')


if __name__ == '__main__':
    main()
