# bca_hfp/analysis/build_fixed_drug_smiles.py
"""build a GDSC_RES.csv that never carries a mismatched SMILES.

principle: rather empty than wrong.
  1. drug with an INCHI_KEY in the official screened_compounds_rel_8.5_inchi.csv -> look the key up in
     PubChem and overwrite with the authoritative CanonicalSMILES (structure certainly belongs to that DRUG_ID).
  2. no official INCHI_KEY, but the old SMILES matches the drug name in the PubChem synonyms -> keep the old value.
  3. anything else (not found / name disagrees with official / unverifiable) -> SMILES set empty, the
     training script then filters the drug out.

the original file is backed up before it is overwritten.
"""
import argparse
import csv
import json
import os
import re
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request

BASE = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug'


def norm(s):
    return re.sub(r'[^A-Z0-9]', '', str(s).upper())


def get_json(url, retries=2, timeout=15):
    for i in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(2 + 2 * i)
        except Exception:
            time.sleep(2 + 2 * i)
    return None


def smiles_from_inchikey(ik):
    q = urllib.parse.quote(ik, safe='')
    d = get_json(f'{BASE}/compound/inchikey/{q}/property/CanonicalSMILES/JSON')
    if not d:
        return None
    props = d.get('PropertyTable', {}).get('Properties', [])
    return props[0].get('CanonicalSMILES') or props[0].get('ConnectivitySMILES') if props else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default='Data/GDSC_RES.csv')
    ap.add_argument('--official', default='Data/screened_compounds_rel_8.5.csv')
    ap.add_argument('--official_inchi', default='Data/screened_compounds_rel_8.5_inchi.csv')
    ap.add_argument('--pubchem_check', default='Result/smiles_name_check.csv')
    ap.add_argument('--backup', default='Data/GDSC_RES_backup_before_smiles_fix.csv')
    ap.add_argument('--report', default='Result/drug_smiles_fix_report.csv')
    ap.add_argument('--sleep', type=float, default=0.2)
    args = ap.parse_args()

    # ---- official tables ----
    official = {}
    with open(args.official, encoding='utf-8-sig', newline='') as f:
        for r in csv.DictReader(f):
            did = str(r['DRUG_ID']).strip()
            names = [r.get('DRUG_NAME', '')]
            names += [x.strip() for x in str(r.get('SYNONYMS', '')).split(',') if x.strip()]
            official.setdefault(did, {'name': r.get('DRUG_NAME', ''), 'names': []})
            official[did]['names'].extend(names)
    with open(args.official_inchi, encoding='utf-8-sig', newline='') as f:
        for r in csv.DictReader(f):
            did = str(r['DRUG_ID']).strip()
            ik = (r.get('INCHI_KEY') or '').strip()
            if did in official and ik:
                official[did]['inchikey'] = ik

    # ---- PubChem name-match result of the old SMILES ----
    name_ok = {}
    if os.path.exists(args.pubchem_check):
        with open(args.pubchem_check, encoding='utf-8-sig', newline='') as f:
            for r in csv.DictReader(f):
                name_ok[str(r['DRUG_ID'])] = (r.get('status') == 'OK_NAME_MATCH')

    # ---- read the original CSV ----
    with open(args.csv, encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
    smi_col = next(c for c in ('DRUG_SMILE', 'SMILES', 'smiles') if c in fieldnames)
    ours = []
    seen = set()
    for r in rows:
        did = str(r['DRUG_ID']).strip()
        if did in seen:
            continue
        seen.add(did)
        ours.append((did, r.get('DRUG_NAME', '').strip(), str(r.get(smi_col, '') or '').strip()))
    print(f'CSV rows {len(rows)}, unique drugs {len(ours)}')

    # ---- decide the new SMILES per drug ----
    new_smi, report = {}, []
    stats = {'official': 0, 'kept': 0, 'cleared': 0}
    for i, (did, name, old) in enumerate(ours, 1):
        off = official.get(did)
        off_ik = (off or {}).get('inchikey', '')
        off_names = (off or {}).get('names', [])
        new, source = '', ''

        # name agreement with the official entry (when it has a name)
        name_consistent = True
        if off_names:
            n = norm(name)
            name_consistent = any(n == norm(x) or (len(n) >= 5 and (n in norm(x) or norm(x) in n))
                                  for x in off_names if x)

        if not name_consistent:
            new, source = '', 'name_mismatch_cleared'
        elif off_ik:
            cand = smiles_from_inchikey(off_ik)
            time.sleep(args.sleep)
            if cand:
                new, source = cand, 'official_inchikey'
            else:
                new, source = '', 'inchikey_not_found_cleared'
        elif name_ok.get(did):
            new, source = old, 'name_match_kept'
        else:
            new, source = '', 'unverified_cleared'

        new_smi[did] = new
        stats['official' if source == 'official_inchikey' else
              ('kept' if source == 'name_match_kept' else 'cleared')] += 1
        report.append([did, name, old, new, source, bool(new and new != old)])
        if i % 20 == 0:
            print(f'[{i}/{len(ours)}] ...', flush=True)

    # ---- backup + overwrite ----
    shutil.copy2(args.csv, args.backup)
    print(f'original backed up -> {args.backup}')
    for r in rows:
        did = str(r['DRUG_ID']).strip()
        r[smi_col] = new_smi.get(did, '')
    with open(args.csv, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        w.writeheader()
        w.writerows(rows)
    print(f'written back {args.csv}')

    os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
    with open(args.report, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        w.writerow(['DRUG_ID', 'DRUG_NAME', 'OLD_SMILES', 'NEW_SMILES', 'SOURCE', 'CHANGED'])
        w.writerows(report)

    print('\n===== summary =====')
    print(f'  rebuilt from official InChIKey : {stats["official"]}')
    print(f'  name match, old value kept     : {stats["kept"]}')
    print(f'  cleared (unverifiable / mismatch): {stats["cleared"]}')
    print(f'  report -> {args.report}')


if __name__ == '__main__':
    main()
