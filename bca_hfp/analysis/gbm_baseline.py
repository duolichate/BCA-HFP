# bca_hfp/analysis/gbm_baseline.py
"""R3-6: gradient boosting baseline (classic learner on concatenated inputs).

inputs come from the same source as the main model:
  drug = Morgan(2048) + 8 physicochemical descriptors (same width as drug_global in training)
  cell = expression of the selected genes (by default the Ensembl ids in global_gene_names_GDSC.txt,
         read as columns from GDSC_EXP_ensembl.csv; --no_genes disables this)
labels and splits reuse the E7 OOF directly (same held-out drugs, same fold assignment), so they are comparable.

usage (from the repo root):
python bca_hfp/analysis/gbm_baseline.py \
  --cv_results ~/my_model/Result/My_model/CV/<e7>/results \
  --res_csv ../Data/GDSC_RES.csv \
  --exp_csv ../Data/GDSC_EXP_ensembl.csv \
  --gene_names ~/my_model/Result/Interpretability/global_gene_names_GDSC.txt \
  --out_dir Result/oof \
  --folds 1,2,3,4,5 --max_iter 150
"""

import argparse
import glob
import math
import os
import sys

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs, Descriptors
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from bca_hfp.config.base_config import root_dir

RDLogger.DisableLog('rdApp.*')


def r2(y, p):
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def rmse(y, p):
    return float(np.sqrt(((y - p) ** 2).mean()))


def pearson(y, p):
    if y.std() < 1e-12 or p.std() < 1e-12:
        return np.nan
    return float(np.corrcoef(y, p)[0, 1])


def load_oof(cv_results):
    files = sorted(glob.glob(os.path.join(cv_results, '*_fold*_swa_test_predictions.csv')))
    if not files:
        raise SystemExit(f'no prediction file found: {cv_results}')
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df.rename(columns={'true_label': 'y', 'predicted': 'p'})
    for c in ('DRUG_ID', 'GDSC_ID'):
        df[c] = df[c].astype(str)
    print(f'[load] {len(files)} folds, {len(df)} rows')
    return df


def drug_matrix(res_csv, n_bits=2048):
    res = pd.read_csv(res_csv, low_memory=False)
    res['DRUG_ID'] = res['DRUG_ID'].astype(str)
    res = res.drop_duplicates('DRUG_ID')
    out, bad = {}, []
    for d, s in zip(res.DRUG_ID, res.DRUG_SMILE):
        mol = Chem.MolFromSmiles(s) if isinstance(s, str) and s else None
        if mol is None:
            bad.append(d)
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits)
        a = np.zeros((n_bits,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, a)
        desc = np.array([
            Descriptors.MolWt(mol), Descriptors.MolLogP(mol), Descriptors.TPSA(mol),
            Descriptors.NumHDonors(mol), Descriptors.NumHAcceptors(mol),
            Descriptors.NumAromaticRings(mol), Descriptors.NumRotatableBonds(mol),
            Descriptors.FractionCSP3(mol),
        ], dtype=np.float32)
        out[d] = np.concatenate([a, desc])
    print(f'[drug] features for {len(out)} drugs, {len(bad)} failed')
    return out, n_bits + 8


def _top_by_mean(exp_csv, n_top):
    """without a gene list: take the top n_top genes by mean expression across cell lines, matching the
    "top-N highly expressed genes" rule of the main model."""
    head = pd.read_csv(exp_csv, nrows=0).columns.tolist()
    gcols = [c for c in head if c.upper().startswith('ENSG')]
    if not gcols:
        raise SystemExit(f'{exp_csv} has no column starting with ENSG, cannot pick genes automatically')
    print(f'[gene] expression table has {len(gcols)} gene columns, taking the top {n_top} by mean expression')
    dt = {c: np.float32 for c in gcols}
    dt['GDSC_ID'] = str
    exp = pd.read_csv(exp_csv, usecols=['GDSC_ID'] + gcols, dtype=dt)
    mu = exp[gcols].mean(axis=0)
    return mu.sort_values(ascending=False).head(n_top).index.tolist()


def gene_matrix(exp_csv, gene_names, cells, auto_genes=0):
    if gene_names and os.path.exists(gene_names):
        genes = [x.strip() for x in open(gene_names) if x.strip()]
        print(f'[gene] list file {gene_names}: {len(genes)} genes')
    elif auto_genes > 0:
        genes = _top_by_mean(exp_csv, auto_genes)
    else:
        raise SystemExit(
            f'gene list not found: {gene_names}\n'
            f'  look for it with find ~/my_model -name "*gene_names*", or use '
            f'--auto_genes 2048 to pick by mean expression.')
    head = pd.read_csv(exp_csv, nrows=0).columns.tolist()
    use = [g for g in genes if g in set(head)]
    print(f'[gene] {len(use)} found in the expression table')
    if not use:
        raise SystemExit('gene list and expression columns do not match (check that both use Ensembl ids)')
    dtypes = {c: np.float32 for c in use}
    dtypes['GDSC_ID'] = str
    exp = pd.read_csv(exp_csv, usecols=['GDSC_ID'] + use, dtype=dtypes)
    exp['GDSC_ID'] = exp['GDSC_ID'].astype(str).str.strip()
    exp = exp.drop_duplicates('GDSC_ID').set_index('GDSC_ID')
    miss = len(set(cells) - set(exp.index))
    print(f'[gene] expression table has {len(exp)} rows; {miss} cell lines missing (filled with the column mean)')
    exp = exp.reindex(sorted(set(cells)))
    return exp[use].values, use


def gf_matrix(pt_dir, cells, pool='mean', max_missing=0.02):
    """pool the per-cell GeneFormer embeddings (n_genes, 256) over the gene axis.

    same representation as the main model, only reduced for a non-sequence model.
    the cell key comes from metadata['GDSC_ID'] **inside the .pt**, not from the file name
    (in practice the names do not match the GDSC_IDs, so matching on file names mostly misses).
    pool='mean'    -> 256 dims
    pool='meanmax' -> 512 dims
    """
    import torch
    files = sorted(glob.glob(os.path.join(pt_dir, '*.pt')))
    if not files:
        raise SystemExit(f'no .pt file under {pt_dir}')
    out, shapes, dup, no_key = {}, set(), 0, 0
    for f in files:
        d = torch.load(f, map_location='cpu', weights_only=False)
        emb = None
        cid = ''
        if isinstance(d, dict):
            meta = d.get('metadata') or {}
            cid = str(meta.get('GDSC_ID', '') or '').strip()
            emb = d.get('embeddings', None)
            if emb is None:
                for k in ('emb', 'feat', 'features', 'x'):
                    if k in d:
                        emb = d[k]
                        break
        else:
            emb = d
        if emb is None:
            continue
        if (not cid) or cid in ('None', 'nan'):
            cid = os.path.splitext(os.path.basename(f))[0].replace('DATA.', '').strip()
            no_key += 1
        a = emb.float().numpy() if hasattr(emb, 'float') else np.asarray(emb, dtype=np.float32)
        a = np.asarray(a, dtype=np.float32)
        if a.ndim == 1:
            v = a
        else:
            shapes.add(a.shape)
            v = (np.concatenate([a.mean(axis=0), a.max(axis=0)])
                 if pool == 'meanmax' else a.mean(axis=0))
        if cid in out:
            dup += 1
            continue
        out[cid] = np.nan_to_num(v, nan=0.0)

    dims = {v.shape[0] for v in out.values()}
    print(f'[gf] {len(files)} files -> {len(out)} cell keys (duplicates {dup}, no metadata {no_key})')
    print(f'[gf] pooled dims {dims}, sample raw shapes {list(shapes)[:3]}')
    if len(dims) != 1:
        raise SystemExit(f'pooled dims are inconsistent: {dims}')

    want = set(cells)
    miss = want - set(out)
    print(f'[gf] {len(want)} cell lines needed, {len(miss)} missing '
          f'({100.0 * len(miss) / len(want):.1f}%)')
    if miss and len(miss) / len(want) > max_missing:
        ex = sorted(miss)[:5]
        raise SystemExit(
            f'missing rate too high, aborting (threshold {100 * max_missing:.0f}%). examples: {ex}\n'
            f'  check that metadata["GDSC_ID"] and the OOF GDSC_ID use the same convention '
            f'instead of carrying on with zero-filled values.')

    dim = dims.pop()
    order = sorted(want)
    mat = np.zeros((len(order), dim), dtype=np.float32)
    for i, c in enumerate(order):
        if c in out:
            mat[i] = out[c]
    return mat, dim


def trivial_baselines(df, fp_map, folds, n_bits=2048):
    """the three trivial baselines from R3-6, on the same OOF fold split as GBM, training folds only.

    cellmean : mean lnIC50 of that cell line over the training folds (unseen cell lines fall back to the global mean)
    nn_drug  : nearest training drug by Morgan-Tanimoto, take its mean lnIC50
    fp_const : Ridge regressing the drug-mean lnIC50 on the fingerprint only, one constant per test drug
    """
    from sklearn.linear_model import Ridge

    drug_ids = sorted(set(df.DRUG_ID))
    dim = next(iter(fp_map.values())).shape[0]
    fp_mat = np.zeros((len(drug_ids), dim), dtype=np.float32)
    for i, d in enumerate(drug_ids):
        if d in fp_map:
            fp_mat[i] = fp_map[d]
    idx = {d: i for i, d in enumerate(drug_ids)}
    B = (fp_mat[:, :n_bits] > 0).astype(np.float32)

    out = {k: np.full(len(df), np.nan, dtype=np.float64)
           for k in ('cellmean', 'nn_drug', 'fp_const')}
    for k in folds:
        te = (df.fold == k).values
        dtr, dte = df[~te], df[te]
        gmean = float(dtr.y.mean())
        # 1) cell-line mean
        cm = dtr.groupby('GDSC_ID').y.mean()
        out['cellmean'][te] = dte.GDSC_ID.map(cm).fillna(gmean).values
        # 2) nearest Tanimoto training drug
        train_drugs = sorted(set(dtr.DRUG_ID))
        test_drugs = sorted(set(dte.DRUG_ID))
        dm = dtr.groupby('DRUG_ID').y.mean()
        ti = [idx[d] for d in train_drugs]
        nn = {}
        for d in test_drugs:
            v = B[idx[d]]
            dot = B[ti] @ v
            denom = B[ti].sum(1) + v.sum() - dot
            sim = np.where(denom > 0, dot / np.maximum(denom, 1e-9), 0.0)
            nn[d] = train_drugs[int(np.argmax(sim))]
        out['nn_drug'][te] = dte.DRUG_ID.map(
            lambda d: float(dm.get(nn[d], gmean))).values
        # 3) fingerprint only -> drug-mean constant
        ytr = dm.reindex(train_drugs).values.astype(np.float64)
        ok = np.isfinite(ytr)
        rg = Ridge(alpha=1.0, solver='lsqr').fit(
            fp_mat[[idx[d] for d in np.array(train_drugs)[ok]]], ytr[ok])
        pm = pd.Series(rg.predict(fp_mat[[idx[d] for d in test_drugs]]),
                       index=test_drugs)
        out['fp_const'][te] = dte.DRUG_ID.map(pm).values
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cv_results', required=True)
    ap.add_argument('--res_csv', default='',
                    help='response CSV (with SMILES / concentration window); empty means 1_Data/Drug/GDSC_RES.csv')
    ap.add_argument('--exp_csv', default='')
    ap.add_argument('--gene_names', default='')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--folds', default='1,2,3,4,5')
    ap.add_argument('--max_iter', type=int, default=150)
    ap.add_argument('--max_leaf_nodes', type=int, default=31)
    ap.add_argument('--no_genes', action='store_true')
    ap.add_argument('--no_gbm', action='store_true',
                    help='only the three trivial baselines (skip GBM, finishes in seconds)')
    ap.add_argument('--gf_dir', default='',
                    help='GeneFormer embedding dir (e.g. ../Data/GDSC_EXP_GF_V1); when given, use the pooled '
                         'representation of all gene tokens per cell line, same source as the main model')
    ap.add_argument('--gf_pool', default='mean', choices=['mean', 'meanmax'],
                    help='pooling of the GeneFormer tokens: mean (256 dims) | meanmax (512 dims)')
    ap.add_argument('--gf_max_missing', type=float, default=0.02,
                    help='upper bound on the cell-line missing rate, abort above it (default 2%%)')
    ap.add_argument('--auto_genes', type=int, default=0,
                    help='[not recommended] take the top N raw genes by global mean expression. this does not '
                         'match the main model (per-cell top-2048 + rank encoding + housekeeping removal), reference only')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    if not args.res_csv:
        args.res_csv = os.path.join(root_dir('Data'), 'Drug', 'GDSC_RES.csv')
    print(f'[res] {args.res_csv}')

    df = load_oof(args.cv_results)
    dm, d_dim = drug_matrix(args.res_csv)

    X = None
    if not args.no_gbm:
        blocks = [np.array([dm[d] for d in df.DRUG_ID], dtype=np.float32)]
        names = [f'drug({d_dim})']

        if not args.no_genes:
            cells = df.GDSC_ID.values
            if args.gf_dir:
                gmat, gdim = gf_matrix(args.gf_dir, cells, args.gf_pool, args.gf_max_missing)
                names.append(f'gf-{args.gf_pool}({gdim})')
            else:
                if not args.exp_csv:
                    raise SystemExit('needs --exp_csv or --gf_dir, or add --no_genes')
                gmat, use = gene_matrix(args.exp_csv, args.gene_names, cells, args.auto_genes)
                names.append(f'gene({len(use)})')
            idx = {c: i for i, c in enumerate(sorted(set(cells)))}
            pos = np.array([idx[c] for c in cells])
            gmat = np.nan_to_num(gmat, nan=0.0)
            blocks.append(gmat[pos])

        X = np.concatenate(blocks, axis=1)
        del blocks
        print(f'[feat] X shape={X.shape}  ({X.nbytes / 2**30:.2f} GB)  source: {", ".join(names)}')

    y = df.y.values
    folds = [int(x) for x in args.folds.split(',') if x.strip()]

    # main model (OOF swa) + GBM + the three trivial baselines, all on the same fold split and the same window clamp
    preds = {'main_swa': df.p.values.astype(np.float64)}
    if not args.no_gbm:
        pred_all = np.full(len(df), np.nan)
        for k in folds:
            te = (df.fold == k).values
            tr = ~te
            mdl = HistGradientBoostingRegressor(
                max_iter=args.max_iter, learning_rate=0.1,
                max_leaf_nodes=args.max_leaf_nodes, early_stopping=False,
                random_state=0, verbose=0)
            mdl.fit(X[tr], y[tr])
            pred_all[te] = mdl.predict(X[te])
            print(f'  fold{k}: GBM trained n_train={int(tr.sum())} n_test={int(te.sum())}')
        preds['gbm'] = pred_all
        del X

    preds.update(trivial_baselines(df, dm, folds, n_bits=2048))

    # ---------- same definition as the main model: winsorize with the concentration window ----------
    res = pd.read_csv(args.res_csv, low_memory=False)
    for c in ('DRUG_ID', 'GDSC_ID'):
        res[c] = res[c].astype(str)
    b = res.drop_duplicates(['DRUG_ID', 'GDSC_ID'])[
        ['DRUG_ID', 'GDSC_ID', 'MIN_CONC', 'MAX_CONC']]
    base = df[['DRUG_ID', 'GDSC_ID', 'fold']].copy()
    base['y'] = y
    base = base.merge(b, on=['DRUG_ID', 'GDSC_ID'], how='left')
    print(f'\n[windows] {int(base.MIN_CONC.isna().sum())} rows lack a concentration window')
    lo = np.log(base.MIN_CONC.values)
    hi = np.log(base.MAX_CONC.values)

    rows = []
    for name, pv in preds.items():
        mm = base.copy()
        mm['p'] = pv
        mm['p_win'] = np.clip(np.clip(mm.p.values, lo - 2.0, hi + 2.0),
                              math.log(0.001), math.log(100.0))
        for k in folds:
            s = mm[(mm.fold == k) & np.isfinite(mm.p.values)]
            if len(s) == 0:
                continue
            rows.append(dict(
                model=name, fold=k, n=len(s),
                r2_raw=r2(s.y.values, s.p.values),
                r2_win=r2(s.y.values, s.p_win.values),
                pearson_raw=pearson(s.y.values, s.p.values),
                pearson_win=pearson(s.y.values, s.p_win.values),
                rmse_raw=rmse(s.y.values, s.p.values),
                rmse_win=rmse(s.y.values, s.p_win.values),
                clipped=float((np.abs(s.p_win.values - s.p.values) > 1e-6).mean())))
        done = np.isfinite(mm.p.values)
        rows.append(dict(
            model=name, fold=0, n=int(done.sum()),
            r2_raw=r2(mm.y.values[done], mm.p.values[done]),
            r2_win=r2(mm.y.values[done], mm.p_win.values[done]),
            pearson_raw=pearson(mm.y.values[done], mm.p.values[done]),
            pearson_win=pearson(mm.y.values[done], mm.p_win.values[done]),
            rmse_raw=rmse(mm.y.values[done], mm.p.values[done]),
            rmse_win=rmse(mm.y.values[done], mm.p_win.values[done]),
            clipped=float((np.abs(mm.p_win.values[done] - mm.p.values[done]) > 1e-6).mean())))
    t = pd.DataFrame(rows)

    print('\n  == pooled (fold=0) ==')
    print('  model        R2_raw   R2_win   Pearson   RMSE_win')
    for name in preds:
        r = t[(t.model == name) & (t.fold == 0)].iloc[0]
        print(f'  {name:<12s} {r.r2_raw:+.4f}  {r.r2_win:+.4f}  '
              f'{r.pearson_win:.4f}    {r.rmse_win:.4f}')

    t.to_csv(os.path.join(args.out_dir, 'r36_baselines.csv'), index=False)
    print(f'\n[ok] -> {args.out_dir}/r36_baselines.csv')


if __name__ == '__main__':
    main()
