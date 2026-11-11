# bca_hfp/analysis/audit_drug_smiles.py
"""check DRUG_NAME / DRUG_SMILE of GDSC_RES.csv against the official GDSC compound table.

sources:
  Data/screened_compounds_rel_8.5.csv        -> DRUG_ID, DRUG_NAME, SYNONYMS
  Data/screened_compounds_rel_8.5_inchi.csv  -> extra INCHI_KEY (structural key)

two checks:
  1) name: does our DRUG_NAME appear in the official DRUG_NAME / SYNONYMS
  2) structure: convert our SMILES to InChIKey via PubChem, compare the first 14 chars with the
     official INCHI_KEY (first 14 = connectivity block, a mismatch means a different structure.
     needs network, 100 CIDs per request)
"""
import argparse
import csv
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request

BASE = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug'


def norm(s):
    return re.sub(r'[^A-Z0-9]', '', str(s).upper())


def get_json(url, retries=2, timeout=20):
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


def load_official(path_csv, path_inchi):
    """DRUG_ID -> {name, names:[...], inchikey}"""
    info = {}
    with open(path_csv, encoding='utf-8-sig', newline='') as f:
        for r in csv.DictReader(f):
            did = str(r['DRUG_ID']).strip()
            names = [r.get('DRUG_NAME', '')]
            names += [x.strip() for x in str(r.get('SYNONYMS', '')).split(',') if x.strip()]
            info.setdefault(did, {'name': r.get('DRUG_NAME', ''), 'names': []})
            info[did]['names'].extend(names)
    with open(path_inchi, encoding='utf-8-sig', newline='') as f:
        for r in csv.DictReader(f):
            did = str(r['DRUG_ID']).strip()
            ik = (r.get('INCHI_KEY') or '').strip()
            if ik and did in info:
                info[did]['inchikey'] = ik
    return info


def load_ours(path):
    rows, seen = [], set()
    with open(path, encoding='utf-8-sig', newline='') as f:
        for r in csv.DictReader(f):
            did = str(r['DRUG_ID']).strip()
            if did in seen:
                continue
            seen.add(did)
            smi = next((r[k] for k in ('DRUG_SMILE', 'SMILES', 'smiles') if k in r), '')
            rows.append((did, r.get('DRUG_NAME', '').strip(), str(smi).strip()))
    return rows


def fetch_inchikeys(cids):
    """batch CID -> InChIKey (100 per request)."""
    out = {}
    for i in range(0, len(cids), 100):
        chunk = cids[i:i + 100]
        url = f"{BASE}/compound/cid/{','.join(map(str, chunk))}/property/InChIKey/JSON"
        d = get_json(url)
        if d:
            for p in d['PropertyTable']['Properties']:
                out[int(p['CID'])] = p.get('InChIKey', '')
        time.sleep(0.3)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default='Data/GDSC_RES.csv')
    ap.add_argument('--official', default='Data/screened_compounds_rel_8.5.csv')
    ap.add_argument('--official_inchi', default='Data/screened_compounds_rel_8.5_inchi.csv')
    ap.add_argument('--pubchem_check', default='Result/smiles_name_check.csv',
                    help='output of check_smiles_pubchem.py, provides SMILES->CID')
    ap.add_argument('--out', default='Result/drug_smiles_audit.csv')
    args = ap.parse_args()

    official = load_official(args.official, args.official_inchi)
    ours = load_ours(args.csv)
    print(f'official drugs: {len(official)}  ours: {len(ours)}')

    # SMILES -> CID (from the PubChem step above)
    cid_map, how_map = {}, {}
    if os.path.exists(args.pubchem_check):
        with open(args.pubchem_check, encoding='utf-8-sig', newline='') as f:
            for r in csv.DictReader(f):
                if r.get('CID'):
                    cid_map[str(r['DRUG_ID'])] = int(r['CID'])
                    how_map[str(r['DRUG_ID'])] = r.get('match_type', '')

    ik_map = fetch_inchikeys(sorted(set(cid_map.values())))
    print(f'InChIKey fetched: {len(ik_map)} / {len(set(cid_map.values()))}')

    rows, n_name_bad, n_struct_bad = [], 0, 0
    for did, name, smi in ours:
        off = official.get(did)
        off_name = off['name'] if off else ''
        off_ik = (off or {}).get('inchikey', '')

        # name
        name_ok = None
        if off and name:
            n = norm(name)
            name_ok = any(n == norm(x) or (len(n) >= 5 and (n in norm(x) or norm(x) in n))
                          for x in off['names'] if x)
            if not name_ok:
                n_name_bad += 1

        # structure
        cid = cid_map.get(did)
        our_ik = ik_map.get(cid, '') if cid else ''
        struct_ok = None
        if our_ik and off_ik:
            struct_ok = our_ik[:14] == off_ik[:14]
            if not struct_ok:
                n_struct_bad += 1

        notes = []
        if not off:
            notes.append('id missing in official table')
        if not cid:
            notes.append('no CID for SMILES')
        if our_ik and off_ik and not struct_ok:
            notes.append('structure mismatch')
        if name_ok is False:
            notes.append('name mismatch')
        rows.append([did, name, off_name, name_ok, cid or '', how_map.get(did, ''),
                     our_ik, off_ik, struct_ok, '; '.join(notes)])

    print('\n===== structure mismatch (first 14 chars of InChIKey differ) =====')
    bad = [r for r in rows if r[8] is False]
    for r in bad:
        print(f'  {r[0]:>6} {r[1]:<26} CID={r[4]:<10} ours={r[6][:14]}  official={r[7][:14]}')

    print('\n===== name mismatch =====')
    for r in rows:
        if r[3] is False:
            print(f'  {r[0]:>6} ours={r[1]:<26} official={r[2]}')

    print('\n===== summary =====')
    print(f'  structure comparable: {sum(1 for r in rows if r[8] is not None)}')
    print(f'  structure mismatch: {n_struct_bad}')
    print(f'  name mismatch: {n_name_bad}')
    print(f'  no structure check (no CID for SMILES): {sum(1 for r in rows if not r[4])}')

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        w.writerow(['DRUG_ID', 'OUR_NAME', 'OFFICIAL_NAME', 'NAME_OK', 'CID', 'match_type',
                    'OUR_INCHIKEY', 'OFFICIAL_INCHIKEY', 'STRUCT_OK', 'NOTE'])
        w.writerows(rows)
    print(f'\nresult -> {args.out}')


if __name__ == '__main__':
    main()
