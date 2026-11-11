# bca_hfp/analysis/table_a_rebuild.py
"""rebuild supplementary table A in one go (same five folds, same labels, same window clamp) and fill the columns that were missing.

why it is needed
------------------
table A used to mix two batches: the two GBM rows came from gbm_baseline.py (window clamped), the three
trivial baselines from pool_oof_stats.py (not clamped), and all of them lacked Spearman and
R2_level / R2_within. all 6 rows are recomputed here with the same OOF fold assignment, the same metrics
and the same window post-processing, and the per-sample predictions are written too so anyone can recompute.

what is computed (CPU only, the GPU stays free for training)
  1) ours      : the E7 OOF predictions (already clamped, only metrics are computed)
  2) gbm_expr  : Morgan(2048)+8 descriptors + 2048 raw expression genes
  3) gbm_gf    : Morgan(2048)+8 descriptors + GeneFormer pool 256
  4) cellmean  : label mean of that cell line over the training folds
  5) nn_drug   : observation of the nearest training drug on that cell line
  6) fp_ridge  : ridge from the Morgan fingerprint to the drug-level mean, spread out as a constant

usage (from the repo root; limiting the thread count is advised so a running training job keeps its CPU)
  OMP_NUM_THREADS=6 python bca_hfp/analysis/table_a_rebuild.py \
    --cv_results ~/my_model/Result/My_model/CV/GDSC_attention_3l_V1_drug_fp2048_batchnorm_md0_seed42_cv5_ne20_lrfix_swa5_desc_pc_gbn_mse_dml0.5_e7/results \
    --res_csv ../Data/GDSC_RES.csv \
    --exp_csv ../Data/GDSC_EXP_ensembl.csv \
    --gene_names ~/my_model/Result/Interpretability/global_gene_names_GDSC.txt \
    --gf_dir ../Data/GDSC_EXP_GF_V1 \
    --out_dir ~/my_model/Result/diag/tableA_v2 \
    --threads 6

--oof_csv can also take an already exported OOF (columns: DRUG_ID/GDSC_ID/fold/y/p).
trivial baselines only (no GBM, a few tens of seconds): add --skip_gbm.
"""

import os
import sys

# ---- the thread count must be fixed before sklearn is imported ----
_threads = '6'
for _i, _a in enumerate(sys.argv):
    if _a == '--threads' and _i + 1 < len(sys.argv):
        _threads = sys.argv[_i + 1]
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'NUMEXPR_NUM_THREADS'):
    os.environ[_k] = _threads

import argparse
import glob
import math

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


# ---------------------------------------------------------------- feature building
def load_oof_from_results(cv_results):
    """read the OOF from the *_swa_test_predictions.csv written by train_cv (same source as the main table)."""
    files = sorted(glob.glob(os.path.join(cv_results, '*_fold*_swa_test_predictions.csv')))
    if not files:
        raise SystemExit(f'no *_fold*_swa_test_predictions.csv under {cv_results}')
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df.rename(columns={'true_label': 'y', 'predicted': 'p'})
    for c in ('DRUG_ID', 'GDSC_ID'):
        df[c] = df[c].astype(str)
    dups = int(df.duplicated(['DRUG_ID', 'GDSC_ID']).sum())
    print(f'[oof] {len(files)} folds, {len(df)} rows, duplicate keys {dups}')
    if dups:
        raise SystemExit('the OOF has duplicate (DRUG_ID, GDSC_ID) pairs, the folds may overlap, check that first')
    return df


def drug_matrix(res_csv, n_bits=2048):
    """drug-side features: Morgan(n_bits) + 8 physicochemical descriptors (aligned with drug_global of the main model)."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem, DataStructs, Descriptors
    RDLogger.DisableLog('rdApp.*')
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
    """without a gene list: take the top n_top genes by mean expression over cell lines (reproducible, paper-ready)."""
    head = pd.read_csv(exp_csv, nrows=0).columns.tolist()
    gcols = [c for c in head if c.upper().startswith('ENSG')]
    if not gcols:
        raise SystemExit(f'{exp_csv} has no column starting with ENSG')
    print(f'[gene] expression table has {len(gcols)} gene columns, taking the top {n_top} by mean expression')
    dt = {c: np.float32 for c in gcols}
    dt['GDSC_ID'] = str
    exp = pd.read_csv(exp_csv, usecols=['GDSC_ID'] + gcols, dtype=dt)
    mu = exp[gcols].mean(axis=0)
    return mu.sort_values(ascending=False).head(n_top).index.tolist()


def gene_matrix(exp_csv, gene_names, cells, auto_genes=0, gene_stats='', min_frac=0.0):
    """cell-side features: raw expression (Ensembl ids).

    three ways to pick the columns (in priority order):
      1) --gene_names file (a fixed list, e.g. the 2048 genes chosen by frequency)
      2) --gene_stats + --min_frac (keep genes present in >= min_frac of the cell lines)
      3) --auto_genes N (top N by mean expression over cell lines)
    """
    if gene_names and os.path.exists(gene_names):
        genes = [x.strip() for x in open(gene_names) if x.strip()]
        print(f'[gene] list has {len(genes)} genes')
    elif gene_stats and os.path.exists(gene_stats):
        st = pd.read_csv(gene_stats)
        for c in ('gene_id', 'frac_cells'):
            if c not in st.columns:
                raise SystemExit(f'{gene_stats} lacks the column {c}; generate it with build_global_gene_list.py')
        st = st.sort_values(['n_cells', 'mean_rank'], ascending=[False, True]) \
            if 'mean_rank' in st.columns else st.sort_values('n_cells', ascending=False)
        genes = st.loc[st.frac_cells >= min_frac, 'gene_id'].tolist()
        print(f'[gene] {len(genes)} genes with frac_cells >= {min_frac:g} '
              f'from {os.path.basename(gene_stats)}')
    elif auto_genes > 0:
        genes = _top_by_mean(exp_csv, auto_genes)
    else:
        raise SystemExit(f'gene list not found: {gene_names}\n'
                         f'  use --gene_stats + --min_frac instead, or --auto_genes 2048')
    head = pd.read_csv(exp_csv, nrows=0).columns.tolist()
    use = [g for g in genes if g in set(head)]
    print(f'[gene] {len(genes)} in the list, {len(use)} found in the expression table')
    if not use:
        raise SystemExit('gene list and expression columns do not match (both should be Ensembl ids)')
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
    """cell-side features: GeneFormer per-cell (2048, 256) pooled over the gene axis."""
    import torch
    files = sorted(glob.glob(os.path.join(pt_dir, '*.pt')))
    if not files:
        raise SystemExit(f'no .pt file under {pt_dir}')
    out, dup, no_key = {}, 0, 0
    for f in files:
        d = torch.load(f, map_location='cpu', weights_only=False)
        meta = (d.get('metadata') or {}) if isinstance(d, dict) else {}
        cid = str(meta.get('GDSC_ID', '') or '').strip()
        emb = d.get('embeddings', None) if isinstance(d, dict) else d
        if emb is None:
            continue
        if (not cid) or cid in ('None', 'nan'):
            cid = os.path.splitext(os.path.basename(f))[0].replace('DATA.', '').strip()
            no_key += 1
        a = emb.float().numpy() if hasattr(emb, 'float') else np.asarray(emb, dtype=np.float32)
        a = np.asarray(a, dtype=np.float32)
        v = a if a.ndim == 1 else (np.concatenate([a.mean(axis=0), a.max(axis=0)])
                                   if pool == 'meanmax' else a.mean(axis=0))
        if cid in out:
            dup += 1
            continue
        out[cid] = np.nan_to_num(v, nan=0.0)

    dims = {v.shape[0] for v in out.values()}
    print(f'[gf] {len(files)} files -> {len(out)} cell keys (duplicates {dup}, no metadata {no_key})')
    print(f'[gf] pooled dims {dims}')
    if len(dims) != 1:
        raise SystemExit(f'pooled dims are inconsistent: {dims}')
    want = set(cells)
    miss = want - set(out)
    print(f'[gf] {len(want)} cell lines needed, {len(miss)} missing '
          f'({100.0 * len(miss) / len(want):.1f}%)')
    if miss and len(miss) / len(want) > max_missing:
        raise SystemExit(f'missing rate too high, aborting (threshold {100 * max_missing:.0f}%). '
                         f'examples: {sorted(miss)[:5]}')
    dim = dims.pop()
    order = sorted(want)
    mat = np.zeros((len(order), dim), dtype=np.float32)
    for i, c in enumerate(order):
        if c in out:
            mat[i] = out[c]
    return mat, dim


# ---------------------------------------------------------------- metrics
def r2(y, p):
    y = np.asarray(y, np.float64); p = np.asarray(p, np.float64)
    return float(1.0 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def rmse(y, p):
    y = np.asarray(y, np.float64); p = np.asarray(p, np.float64)
    return float(np.sqrt(((y - p) ** 2).mean()))


def pearson(y, p):
    y = np.asarray(y, np.float64); p = np.asarray(p, np.float64)
    if y.std() < 1e-12 or p.std() < 1e-12:
        return np.nan
    return float(np.corrcoef(y, p)[0, 1])


def spearman(y, p):
    yr = pd.Series(np.asarray(y, np.float64)).rank().values
    pr = pd.Series(np.asarray(p, np.float64)).rank().values
    if yr.std() < 1e-12 or pr.std() < 1e-12:
        return np.nan
    return float(np.corrcoef(yr, pr)[0, 1])


def decompose(df, y_col='y', p_col='p'):
    """between-drug / within-drug decomposition, same definition as pool_oof_stats.section_decomp (grouped by drug name)."""
    g = df.groupby('GROUP')
    n = g.size()
    my = g[y_col].mean()
    mp = g[p_col].mean()
    gm = df[y_col].mean()
    SL = float((n * (my - gm) ** 2).sum())
    SY = float(((df[y_col] - gm) ** 2).sum())
    r2_level = float(1 - (n * (mp - my) ** 2).sum() / SL)
    yc = df[y_col] - g[y_col].transform('mean')
    pc = df[p_col] - g[p_col].transform('mean')
    r2_within = float(1 - ((yc - pc) ** 2).sum() / (yc ** 2).sum())
    return r2_level, r2_within, SL / SY


def full_metrics(df, tag):
    y = df.y.values
    p = df.p.values
    lvl, win, share = decompose(df)
    return dict(model=tag, n=len(df),
                R2=r2(y, p), RMSE=rmse(y, p), Pearson=pearson(y, p),
                Spearman=spearman(y, p),
                R2_level=lvl, R2_within=win, ss_level_share=share)


# ---------------------------------------------------------------- data
def load_oof(args):
    if args.oof_csv:
        df = pd.read_csv(args.oof_csv)
        need = {'DRUG_ID', 'GDSC_ID', 'fold', 'y', 'p'}
        miss = need - set(df.columns)
        if miss:
            raise SystemExit(f'--oof_csv is missing columns: {sorted(miss)}')
        df['DRUG_ID'] = df['DRUG_ID'].astype(str)
        df['GDSC_ID'] = df['GDSC_ID'].astype(str)
        print(f'[oof] {args.oof_csv}: {len(df)} rows')
        return df
    if not args.cv_results:
        raise SystemExit('needs --cv_results or --oof_csv')
    return load_oof_from_results(args.cv_results)


def attach_group_and_window(df, res_csv):
    res = pd.read_csv(res_csv, low_memory=False)
    for c in ('DRUG_ID', 'GDSC_ID'):
        res[c] = res[c].astype(str)
    meta = res.drop_duplicates('DRUG_ID')[['DRUG_ID', 'DRUG_NAME', 'DRUG_SMILE']].copy()
    meta['GROUP'] = meta['DRUG_NAME'].astype(str).str.strip().str.upper()
    if 'GROUP' not in df.columns:
        df = df.merge(meta[['DRUG_ID', 'GROUP']], on='DRUG_ID', how='left')
        if df['GROUP'].isna().any():
            raise SystemExit('some rows have no DRUG_NAME, the DRUG_ID convention differs')
    chk = df.groupby('GROUP')['fold'].nunique()
    if (chk > 1).any():
        raise SystemExit('a drug group spans several folds, inconsistent with the leave-drug split, check that first')
    win = res.drop_duplicates(['DRUG_ID', 'GDSC_ID'])[
        ['DRUG_ID', 'GDSC_ID', 'MIN_CONC', 'MAX_CONC']]
    df = df.merge(win, on=['DRUG_ID', 'GDSC_ID'], how='left')
    miss = int(df['MIN_CONC'].isna().sum())
    print(f'[window] {miss} rows lack a concentration window')
    if miss:
        raise SystemExit('concentration window missing, cannot apply the same post-processing as the main model')
    return df, meta


def clamp(df):
    lo = np.log(df.MIN_CONC.values)
    hi = np.log(df.MAX_CONC.values)
    return np.clip(np.clip(df.p.values, lo - 2.0, hi + 2.0),
                   math.log(0.001), math.log(100.0))


# ---------------------------------------------------------------- baselines
def run_gbm(df, X, folds, tag, max_iter, max_leaf_nodes):
    y = df.y.values
    pred = np.full(len(df), np.nan)
    for k in folds:
        te = (df.fold == k).values
        if not te.any():
            print(f'  [{tag}] fold{k}: no samples, skipped')
            continue
        tr = ~te
        from sklearn.ensemble import HistGradientBoostingRegressor
        mdl = HistGradientBoostingRegressor(
            max_iter=max_iter, learning_rate=0.1, max_leaf_nodes=max_leaf_nodes,
            early_stopping=False, random_state=0, verbose=0)
        mdl.fit(X[tr], y[tr])
        pred[te] = mdl.predict(X[te])
        print(f'  [{tag}] fold{k}: n_train={int(tr.sum())} n_test={int(te.sum())}')
    return pred


def run_trivial(df, fps, folds, n_bits=2048):
    from sklearn.linear_model import Ridge

    out = {k: np.full(len(df), np.nan) for k in ('cellmean', 'nn_drug', 'fp_ridge')}

    def fp_array(bv, n_bits=n_bits):
        a = np.zeros((n_bits,), dtype=np.int8)
        from rdkit.Chem import DataStructs
        DataStructs.ConvertToNumpyArray(bv, a)
        return a

    for k in folds:
        te = (df.fold == k).values
        if not te.any():
            print(f'  [trivial] fold{k}: no samples, skipped')
            continue
        tr = df[~te]
        te_df = df[te]
        idx = np.where(te)[0]

        # 1) cell-line mean
        cl_mean = tr.groupby('GDSC_ID').y.mean()
        fb = float(tr.y.mean())
        p1 = te_df.GDSC_ID.map(cl_mean).fillna(fb).values

        # 2) observation of the nearest training drug on that cell line
        from rdkit.Chem import DataStructs
        tr_ids = [d for d in sorted(tr.DRUG_ID.unique()) if d in fps]
        nn_map = {}
        for d in sorted(te_df.DRUG_ID.unique()):
            if d not in fps or not tr_ids:
                nn_map[d] = None
                continue
            sims = [(DataStructs.TanimotoSimilarity(fps[d], fps[t]), t) for t in tr_ids]
            nn_map[d] = max(sims, key=lambda x: x[0])[1]
        lookup = tr.groupby(['DRUG_ID', 'GDSC_ID']).y.mean()
        keys = list(zip(te_df.DRUG_ID.map(nn_map), te_df.GDSC_ID))
        p2 = np.asarray(lookup.reindex(keys).values, dtype=float)
        p2 = np.where(np.isfinite(p2), p2, p1)

        # 3) ridge from the fingerprint to the drug mean
        dmean = tr.groupby('DRUG_ID').y.mean()
        tr_d = [d for d in dmean.index if d in fps]
        Xtr = np.stack([fp_array(fps[d]) for d in tr_d]).astype(np.float32)
        mdl = Ridge(alpha=1.0).fit(Xtr, dmean.loc[tr_d].values)
        te_d = [d for d in sorted(te_df.DRUG_ID.unique()) if d in fps]
        Xte = np.stack([fp_array(fps[d]) for d in te_d]).astype(np.float32)
        pred_d = dict(zip(te_d, mdl.predict(Xte)))
        p3 = te_df.DRUG_ID.map(pred_d).fillna(fb).values

        out['cellmean'][idx] = p1
        out['nn_drug'][idx] = p2
        out['fp_ridge'][idx] = p3
        print(f'  [trivial] fold{k}: {len(te_df)} rows')
    return out


def build_fps(meta, n_bits=2048):
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem
    RDLogger.DisableLog('rdApp.*')
    fp = {}
    for d, s in zip(meta.DRUG_ID, meta.DRUG_SMILE):
        mol = Chem.MolFromSmiles(str(s)) if isinstance(s, str) and s else None
        if mol is not None:
            fp[d] = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits)
    print(f'[fp] Morgan built for {len(fp)} / {len(meta)}')
    return fp


# ---------------------------------------------------------------- main flow
def main():
    ap = argparse.ArgumentParser(description='rebuild supplementary table A (all baselines recomputed the same way)')
    ap.add_argument('--cv_results', default='', help='E7 results dir (holding *_swa_test_predictions.csv)')
    ap.add_argument('--oof_csv', default='', help='or the exported OOF directly (DRUG_ID/GDSC_ID/fold/y/p)')
    ap.add_argument('--res_csv', required=True)
    ap.add_argument('--exp_csv', default='', help='needed by gbm_expr')
    ap.add_argument('--gene_names', default='', help='2048-gene list; if absent use --gene_stats or --auto_genes')
    ap.add_argument('--gene_stats', default='',
                    help='gene_stats_*.csv from build_global_gene_list.py; with --min_frac it selects columns by coverage')
    ap.add_argument('--min_frac', type=float, default=0.0,
                    help='with --gene_stats: keep only genes present in >= this fraction of cell lines (e.g. 0.5)')
    ap.add_argument('--auto_genes', type=int, default=0,
                    help='take the top N genes by mean expression over the expression table (2048 recommended, reproducible and paper-ready)')
    ap.add_argument('--gf_dir', default='', help='GeneFormer embedding dir (needed by gbm_gf)')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--folds', default='1,2,3,4,5')
    ap.add_argument('--max_iter', type=int, default=300, help='frozen setting: 300 iterations')
    ap.add_argument('--max_leaf_nodes', type=int, default=31, help='frozen setting: 31 leaves')
    ap.add_argument('--threads', type=int, default=6)
    ap.add_argument('--skip_gbm', action='store_true', help='only the three trivial baselines')
    ap.add_argument('--n_bits', type=int, default=2048,
                    help='Morgan fingerprint width (default 2048). used for the R1-8/R1-9 width sweep: '
                         '256/512/1024/2048/4096; it also changes the drug-side input of fp_ridge, nn_drug and gbm')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df = load_oof(args)
    df, meta = attach_group_and_window(df, args.res_csv)
    folds = [int(x) for x in args.folds.split(',') if x.strip()]
    print(f'[data] {len(df)} rows, {df.DRUG_ID.nunique()} drugs / {df.GROUP.nunique()} groups, '
          f'{df.GDSC_ID.nunique()} cell lines, folds {folds}')

    preds = {}
    if not args.skip_gbm:
        dm, d_dim = drug_matrix(args.res_csv, args.n_bits)
        base_drug = np.array([dm[d] for d in df.DRUG_ID], dtype=np.float32)

        # --- gbm_expr ---
        if not args.exp_csv:
            raise SystemExit('gbm_expr needs --exp_csv')
        gmat, use = gene_matrix(args.exp_csv, args.gene_names, df.GDSC_ID.values,
                                auto_genes=args.auto_genes,
                                gene_stats=args.gene_stats, min_frac=args.min_frac)
        with open(os.path.join(args.out_dir, 'gene_list_used.txt'), 'w') as fh:
            fh.write('\n'.join(use) + '\n')
        print(f'[gene] the {len(use)} genes actually used were written to {args.out_dir}/gene_list_used.txt')
        idx = {c: i for i, c in enumerate(sorted(set(df.GDSC_ID)))}
        pos = np.array([idx[c] for c in df.GDSC_ID.values])
        X = np.concatenate([base_drug, np.nan_to_num(gmat, nan=0.0)[pos]], axis=1)
        print(f'[feat] gbm_expr X={X.shape} ({X.nbytes / 2**30:.2f} GB)')
        preds['gbm_expr'] = run_gbm(df, X, folds, 'gbm_expr',
                                    args.max_iter, args.max_leaf_nodes)
        del X, gmat

        # --- gbm_gf ---
        if args.gf_dir:
            gmat2, gdim = gf_matrix(args.gf_dir, df.GDSC_ID.values, 'mean')
            idx2 = {c: i for i, c in enumerate(sorted(set(df.GDSC_ID)))}
            pos2 = np.array([idx2[c] for c in df.GDSC_ID.values])
            X2 = np.concatenate([base_drug, np.nan_to_num(gmat2, nan=0.0)[pos2]], axis=1)
            print(f'[feat] gbm_gf X={X2.shape} ({X2.nbytes / 2**30:.2f} GB)')
            preds['gbm_gf'] = run_gbm(df, X2, folds, 'gbm_gf',
                                      args.max_iter, args.max_leaf_nodes)
            del X2, gmat2
        else:
            print('[skip] no --gf_dir given, skipping gbm_gf')

    fps = build_fps(meta, args.n_bits)
    triv = run_trivial(df, fps, folds, args.n_bits)
    preds.update(triv)
    preds['ours'] = df.p.values.copy()   # the E7 OOF was already clamped during training

    # ---------- common post-processing + metrics ----------
    rows, keep = [], df[['DRUG_ID', 'GROUP', 'GDSC_ID', 'fold', 'y']].copy()
    keep['p_ours'] = preds['ours']
    lo_all = np.log(df.MIN_CONC.values)
    hi_all = np.log(df.MAX_CONC.values)
    for name, p in preds.items():
        ok = np.isfinite(p)
        if ok.sum() == 0:
            print(f'[skip] {name}: no valid prediction')
            continue
        if ok.sum() != len(df):
            print(f'  [warn] {name}: only {int(ok.sum())}/{len(df)} rows have a prediction '
                  f'(folds={folds}); normal for a smoke test on a few folds, '
                  f'but then this row covers only those rows and is not comparable with the others')
        d = df.loc[ok, ['GROUP', 'y']].copy()
        d['p'] = p[ok]
        m_raw = full_metrics(d, name + '_raw')
        p_win = np.clip(np.clip(p[ok], lo_all[ok] - 2.0, hi_all[ok] + 2.0),
                        math.log(0.001), math.log(100.0))
        d['p'] = p_win
        m_win = full_metrics(d, name)
        for k in ('R2', 'RMSE', 'Pearson', 'Spearman', 'R2_level', 'R2_within'):
            m_win[k + '_raw'] = m_raw[k]
        rows.append(m_win)
        keep['p_' + name] = p
        keep['p_' + name + '_win'] = np.nan
        keep.loc[ok, 'p_' + name + '_win'] = p_win

    t = pd.DataFrame(rows)
    order = ['ours', 'gbm_expr', 'gbm_gf', 'cellmean', 'nn_drug', 'fp_ridge']
    t['_o'] = t.model.map({k: i for i, k in enumerate(order)}).fillna(99)
    t = t.sort_values('_o').drop(columns='_o')

    cols = ['model', 'n', 'R2', 'R2_raw', 'RMSE', 'Pearson', 'Spearman',
            'R2_level', 'R2_within', 'ss_level_share']
    t = t[[c for c in cols if c in t.columns]]
    csv_path = os.path.join(args.out_dir, 'table_a_rebuild.csv')
    t.to_csv(csv_path, index=False)
    keep.to_csv(os.path.join(args.out_dir, 'table_a_predictions.csv'), index=False)

    print('\n' + '=' * 96)
    print('table A (all window post-processed; R2_raw is before clamping)')
    print('=' * 96)
    print(f'{"model":<10}{"n":>8}{"R2":>9}{"R2_raw":>9}{"Pearson":>9}{"Spearman":>10}'
          f'{"R2_level":>10}{"R2_within":>11}')
    for _, r in t.iterrows():
        print(f'{r.model:<10}{int(r.n):>8}{r.R2:>+9.4f}{r.R2_raw:>+9.4f}'
              f'{r.Pearson:>9.4f}{r.Spearman:>10.4f}{r.R2_level:>+10.4f}'
              f'{r.R2_within:>+11.4f}')
    print('=' * 96)
    print('markdown:')
    print('| model | OOF R2 | OOF Pearson | OOF Spearman | R2_level | R2_within |')
    print('|---|---|---|---|---|---|')
    name_cn = {'ours': 'BCA-HFP (seed 42)', 'gbm_expr': 'gradient boosting (raw expression 2048)',
               'gbm_gf': 'gradient boosting (GeneFormer pool 256)', 'cellmean': 'cell-line mean',
               'nn_drug': 'nearest training drug', 'fp_ridge': 'fingerprint ridge (drug-level constant)'}
    for _, r in t.iterrows():
        print(f'| {name_cn.get(r.model, r.model)} | {r.R2:.4f} | {r.Pearson:.4f} | '
              f'{r.Spearman:.4f} | {r.R2_level:.4f} | {r.R2_within:.4f} |')
    print(f'\n[ok] -> {csv_path}')
    print(f'[ok] -> {os.path.join(args.out_dir, "table_a_predictions.csv")}')


if __name__ == '__main__':
    main()
