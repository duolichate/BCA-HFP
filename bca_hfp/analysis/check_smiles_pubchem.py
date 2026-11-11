# bca_hfp/analysis/check_smiles_pubchem.py
"""check with PubChem PUG REST that DRUG_NAME and DRUG_SMILE in the CSV belong together.

steps:
  1. SMILES -> CID (exact match first, fastidentity as fallback)
  2. CID -> synonym list
  3. test whether DRUG_NAME appears in the synonyms (compared after normalization, handles salt forms / aliases)

resumable: results are appended to --out line by line, an already done DRUG_ID is skipped on rerun.
rate limit <= 5 requests/s (PubChem recommendation).
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
HEADER = ['DRUG_ID', 'DRUG_NAME', 'SMILES', 'CID', 'MolecularFormula',
          'IUPACName', 'match_type', 'synonym_hit', 'status']


def norm(s):
    """normalize a name: uppercase, letters and digits only (drops spaces, hyphens, brackets, dots)."""
    return re.sub(r'[^A-Z0-9]', '', str(s).upper())


def get_json(url, retries=2, timeout=12):
    for i in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if e.code == 429:            # rate limited: back off longer
                time.sleep(5 + 3 * i)
                continue
            time.sleep(1 + i)
        except Exception:
            time.sleep(1 + i)
    return None


def query_smiles(smi):
    q = urllib.parse.quote(smi, safe='')
    d = get_json(f'{BASE}/compound/smiles/{q}/property/IUPACName,MolecularFormula/JSON')
    if d:
        p = d['PropertyTable']['Properties'][0]
        return p['CID'], p.get('MolecularFormula'), p.get('IUPACName'), 'exact'
    d = get_json(f'{BASE}/compound/fastidentity/smiles/{q}/cids/JSON')
    cids = (d or {}).get('IdentifierList', {}).get('CID', [])
    if cids:
        cid = cids[0]
        d2 = get_json(f'{BASE}/compound/cid/{cid}/property/IUPACName,MolecularFormula/JSON')
        if d2:
            p = d2['PropertyTable']['Properties'][0]
            return cid, p.get('MolecularFormula'), p.get('IUPACName'), 'identity'
        return cid, None, None, 'identity'
    return None, None, None, 'not_found'


def query_synonyms(cid):
    d = get_json(f'{BASE}/compound/cid/{cid}/synonyms/JSON')
    if not d:
        return []
    info = d.get('InformationList', {}).get('Information', [{}])
    return info[0].get('Synonym', []) if info else []


def name_match(name, syns):
    n = norm(name)
    for s in syns:
        if norm(s) == n:
            return True, s
    if len(n) >= 6:
        for s in syns:
            sn = norm(s)
            if n in sn or sn in n:
                return True, s
    return False, ''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default='Data/GDSC_RES.csv')
    ap.add_argument('--out', default='Result/smiles_name_check.csv')
    ap.add_argument('--sleep', type=float, default=0.2, help='sleep seconds after each request')
    ap.add_argument('--limit', type=int, default=0, help='only check the first N drugs (debug)')
    args = ap.parse_args()

    rows = []
    with open(args.csv, encoding='utf-8-sig', newline='') as f:
        for r in csv.DictReader(f):
            rows.append(r)
    smi_col = next((c for c in ('DRUG_SMILE', 'SMILES', 'smiles') if c in rows[0]), None)
    assert smi_col, f'no SMILES column, have={list(rows[0])}'

    drugs, seen = [], set()
    for r in rows:
        did = str(r.get('DRUG_ID', '')).strip()
        if did in seen:
            continue
        seen.add(did)
        drugs.append((did, r.get('DRUG_NAME', '').strip(), str(r.get(smi_col, '')).strip()))
    if args.limit:
        drugs = drugs[:args.limit]

    done = set()
    if os.path.exists(args.out):
        with open(args.out, encoding='utf-8-sig', newline='') as f:
            for r in csv.DictReader(f):
                done.add(str(r.get('DRUG_ID', '')))
        print(f'resume: {len(done)} already done')
    else:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, 'w', encoding='utf-8-sig', newline='') as f:
            csv.writer(f).writerow(HEADER)

    todo = [d for d in drugs if d[0] not in done]
    print(f'drugs to check: {len(todo)} / {len(drugs)} (CSV rows {len(rows)})', flush=True)

    stats = {'match': 0, 'mismatch': 0, 'not_found': 0, 'error': 0}
    with open(args.out, 'a', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        for i, (did, name, smi) in enumerate(todo, 1):
            if not smi or smi.lower() == 'nan':
                w.writerow([did, name, smi, '', '', '', '', '', 'ERROR_EMPTY_SMILES'])
                f.flush()
                stats['error'] += 1
                continue
            cid, formula, iupac, how = query_smiles(smi)
            time.sleep(args.sleep)
            if cid is None:
                status, syn_hit = 'SMILES_NOT_FOUND', ''
                stats['not_found'] += 1
            else:
                syns = query_synonyms(cid)
                time.sleep(args.sleep)
                ok, syn_hit = name_match(name, syns)
                status = 'OK_NAME_MATCH' if ok else 'CID_FOUND_NAME_MISMATCH'
                stats['match' if ok else 'mismatch'] += 1
            w.writerow([did, name, smi, cid or '', formula or '', iupac or '',
                        how, syn_hit, status])
            f.flush()
            flag = '' if status == 'OK_NAME_MATCH' else '  <-- needs a manual look'
            print(f'[{i}/{len(todo)}] {did:>6} {name:<28} cid={cid} {how:<9} '
                  f'{status}{flag}', flush=True)

    print('\n===== summary of this run =====')
    print(f'  name match            : {stats["match"]}')
    print(f'  CID found, name differs: {stats["mismatch"]}')
    print(f'  SMILES not found      : {stats["not_found"]}')
    print(f'  other errors          : {stats["error"]}')
    print(f'  result -> {args.out}')


if __name__ == '__main__':
    main()
