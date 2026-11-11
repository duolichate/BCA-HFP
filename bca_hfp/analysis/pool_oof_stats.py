# bca_hfp/analysis/pool_oof_stats.py
"""Pool statistics computed once from 5-fold OOF predictions; no training involved.

Outputs per-drug correlations, between/within-drug variance decomposition, three
trivial baselines, label sd reconciliation (pooled vs per-fold), Spearman with
bootstrap confidence intervals, nearest Tanimoto similarity from each test drug to
the training drugs, and performance stratified by target pathway.

input columns:
  cv_results    directory of *_fold*_swa_test_predictions.csv, each with
                DRUG_ID, GDSC_ID, true_label, predicted, fold
  res_csv       compound table with DRUG_ID, DRUG_NAME, DRUG_SMILE
  compound_csv  optional table with a DRUG_ID column and a PATHWAY or TARGET column
  h5            optional, only checked for existence
  out_dir       directory receiving the csv tables
  n_boot        number of bootstrap resamples, default 500

usage (from the repository root):
python bca_hfp/analysis/pool_oof_stats.py \
  --cv_results "../Result/My_model/CV/<run_dir>/results" \
  --h5 ../Data/GDSC_RES_FGBERT_full.h5 \
  --res_csv ../Data/GDSC_RES.csv \
  --out_dir ../Result/diag/pool
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs

RDLogger.DisableLog('rdApp.*')

GROUP_NORM = lambda s: s.astype(str).str.strip().str.upper()   # same as _group in train_cv


# ----------------------------------------------------------------------------- metrics
def r2(y, p):
    y = np.asarray(y, np.float64); p = np.asarray(p, np.float64)
    return float(1.0 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def rmse(y, p):
    y = np.asarray(y, np.float64); p = np.asarray(p, np.float64)
    return float(np.sqrt(((y - p) ** 2).mean()))


def mae(y, p):
    y = np.asarray(y, np.float64); p = np.asarray(p, np.float64)
    return float(np.abs(y - p).mean())


def pearson(y, p):
    y = np.asarray(y, np.float64); p = np.asarray(p, np.float64)
    if y.std() < 1e-12 or p.std() < 1e-12:
        return np.nan
    return float(np.corrcoef(y, p)[0, 1])


def spearman(y, p):
    """rank correlation; pandas ranks (average method) avoid an extra scipy dependency."""
    yr = pd.Series(np.asarray(y, np.float64)).rank().values
    pr = pd.Series(np.asarray(p, np.float64)).rank().values
    if yr.std() < 1e-12 or pr.std() < 1e-12:
        return np.nan
    return float(np.corrcoef(yr, pr)[0, 1])


def all_metrics(y, p):
    return dict(n=len(y), r2=r2(y, p), rmse=rmse(y, p), mae=mae(y, p),
                pearson=pearson(y, p), spearman=spearman(y, p), sd_y=float(np.std(y)))


def bootstrap_ci(y, p, clusters=None, n_boot=500, seed=0):
    """bootstrap 95% CI; resample whole groups when clusters are given (drug as the unit), else rows."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y, np.float64); p = np.asarray(p, np.float64)
    if clusters is None:
        idx_groups = None
        n = len(y)
    else:
        cl = pd.Series(clusters)
        codes = cl.astype('category').cat.codes.values
        uniq = np.unique(codes)
        idx_by_group = [np.where(codes == u)[0] for u in uniq]
        n = len(y)

    keys = ['r2', 'rmse', 'pearson', 'spearman']
    acc = {k: [] for k in keys}
    for _ in range(n_boot):
        if clusters is None:
            sel = rng.integers(0, n, n)
        else:
            pick = rng.integers(0, len(idx_by_group), len(idx_by_group))
            sel = np.concatenate([idx_by_group[i] for i in pick])
        yy, pp = y[sel], p[sel]
        if yy.std() < 1e-12 or pp.std() < 1e-12:
            continue
        acc['r2'].append(r2(yy, pp))
        acc['rmse'].append(rmse(yy, pp))
        acc['pearson'].append(pearson(yy, pp))
        acc['spearman'].append(spearman(yy, pp))
    out = {}
    for k in keys:
        v = np.asarray(acc[k], np.float64)
        v = v[np.isfinite(v)]
        out[k] = (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if len(v) else (np.nan, np.nan)
    return out


# ----------------------------------------------------------------------------- data
def load_oof(cv_results):
    pats = [os.path.join(cv_results, '*_fold*_swa_test_predictions.csv')]
    files = sorted(sum([glob.glob(p) for p in pats], []))
    if not files:
        raise SystemExit(f'no *_fold*_swa_test_predictions.csv under {cv_results}')
    frames = [pd.read_csv(f) for f in files]
    df = pd.concat(frames, ignore_index=True)
    df['DRUG_ID'] = df['DRUG_ID'].astype(str)
    df['GDSC_ID'] = df['GDSC_ID'].astype(str)
    df = df.rename(columns={'true_label': 'y', 'predicted': 'p'})
    dups = df.duplicated(['DRUG_ID', 'GDSC_ID']).sum()
    print(f'[oof] read {len(files)} folds, {len(df)} rows, duplicate keys {dups}')
    if dups:
        raise SystemExit('duplicate (DRUG_ID, GDSC_ID) in OOF, folds overlap; resolve before running stats')
    return df, files


def attach_group(df, res_csv):
    res = pd.read_csv(res_csv, low_memory=False)
    res['DRUG_ID'] = res['DRUG_ID'].astype(str)
    meta = res.drop_duplicates('DRUG_ID')[['DRUG_ID', 'DRUG_NAME', 'DRUG_SMILE']].copy()
    meta['GROUP'] = GROUP_NORM(meta['DRUG_NAME'])
    m = df.merge(meta, on='DRUG_ID', how='left')
    miss = m['GROUP'].isna().sum()
    print(f'[group] rows without a matched drug name: {miss}')
    if miss:
        raise SystemExit('some rows match no drug name, DRUG_ID convention differs; later stats unusable')
    # each drug group must fall entirely inside one fold
    chk = m.groupby('GROUP')['fold'].nunique()
    assert (chk <= 1).all(), 'a drug group spans folds, grouping and split disagree'
    m['fold'] = m['fold'].astype(int)
    return m, meta


def build_fps(meta):
    fp, ok = {}, []
    for d, s in zip(meta['DRUG_ID'], meta['DRUG_SMILE']):
        mol = Chem.MolFromSmiles(str(s)) if isinstance(s, str) and s else None
        if mol is None:
            ok.append(d); continue
        fp[d] = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    print(f'[fp] morgan fingerprint built {len(fp)} / {len(meta)}; failed {len(ok)}')
    return fp


def tanimoto(a, b):
    return float(DataStructs.TanimotoSimilarity(a, b))


def fp_array(fp, n_bits=2048):
    a = np.zeros((n_bits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, a)
    return a


# ----------------------------------------------------------------------------- sections
def section_overall(df, n_boot):
    print('\n===== [1] overall metrics and bootstrap 95% CI =====')
    m = all_metrics(df.y.values, df.p.values)
    ci_row = bootstrap_ci(df.y.values, df.p.values, None, n_boot)
    ci_drug = bootstrap_ci(df.y.values, df.p.values, df.GROUP.values, n_boot)
    print(f'  N={m["n"]}  sd(y)={m["sd_y"]:.4f}')
    print(f'  R2={m["r2"]:+.4f}  RMSE={m["rmse"]:.4f}  MAE={m["mae"]:.4f}  '
          f'Pearson={m["pearson"]:.4f}  Spearman={m["spearman"]:.4f}')
    for k in ['r2', 'rmse', 'pearson', 'spearman']:
        print(f'  {k:9s} CI(row) [{ci_row[k][0]:+.4f}, {ci_row[k][1]:+.4f}]   '
              f'CI(drug cluster) [{ci_drug[k][0]:+.4f}, {ci_drug[k][1]:+.4f}]')
    return m, ci_row, ci_drug


def section_folds(df):
    print('\n===== [2] per-fold metrics and label sd reconciliation =====')
    rows = []
    for k in sorted(df.fold.unique()):
        s = df[df.fold == k]
        r = all_metrics(s.y.values, s.p.values)
        r['fold'] = k
        r['n_drug'] = s.GROUP.nunique()
        rows.append(r)
    t = pd.DataFrame(rows)[['fold', 'n', 'n_drug', 'sd_y', 'r2', 'rmse', 'pearson', 'spearman']]
    for _, r in t.iterrows():
        print(f'  fold{r["fold"]}  n={int(r["n"]):6d}  drugs={int(r["n_drug"]):3d}  '
              f'sd(y)={r["sd_y"]:.4f}  R2={r["r2"]:+.4f}  Pearson={r["pearson"]:.4f}  '
              f'Spearman={r["spearman"]:.4f}')
    pooled = all_metrics(df.y.values, df.p.values)
    print(f'  mean per-fold R2 = {t.r2.mean():.4f} ± {t.r2.std(ddof=1):.4f}')
    print(f'  pooled OOF  R2   = {pooled["r2"]:.4f}   sd(y)={pooled["sd_y"]:.4f}')
    print(f'  mean per-fold sd(y) = {t.sd_y.mean():.4f}  vs pooled sd(y) = {pooled["sd_y"]:.4f}')
    print('  pooling mixes drug means from different folds, which inflates sd(y), so pooled')
    print('  R2 sits above the per-fold mean; report both, keep mean±std in the main table.')
    return t, pooled


def section_decomp(df):
    print('\n===== [3] between/within-drug variance decomposition and per-drug correlation =====')
    g = df.groupby('GROUP')
    n = g.size()
    my = g.y.mean()
    mp = g.p.mean()
    gm = df.y.mean()
    SL = float((n * (my - gm) ** 2).sum())
    SY = float(((df.y - gm) ** 2).sum())
    SW = SY - SL
    R2L = float(1 - (n * (mp - my) ** 2).sum() / SL)
    yc = df.y - g.y.transform('mean')
    pc = df.p - g.p.transform('mean')
    R2W = float(1 - ((yc - pc) ** 2).sum() / (yc ** 2).sum())
    print(f'  SS_level/SS_y = {SL / SY:.4f}   R2_level  = {R2L:+.4f}')
    print(f'  SS_within/SS_y= {SW / SY:.4f}   R2_within = {R2W:+.4f}')
    print(f'  recombined = {(SL / SY) * R2L + (SW / SY) * R2W:+.4f}  (equals overall R2)')
    # per-drug correlation
    per = []
    for name, s in df.groupby('GROUP'):
        if len(s) < 10:
            continue
        per.append(dict(drug=name, n=len(s),
                        pearson=pearson(s.y.values, s.p.values),
                        spearman=spearman(s.y.values, s.p.values),
                        r2=r2(s.y.values, s.p.values)))
    per = pd.DataFrame(per)
    per = per[np.isfinite(per.pearson)]
    print(f'  per-drug Pearson: median {per.pearson.median():.4f}  '
          f'IQR [{per.pearson.quantile(.25):.4f}, {per.pearson.quantile(.75):.4f}]  '
          f'drugs={len(per)}')
    print(f'  per-drug Spearman: median {per.spearman.median():.4f}  '
          f'IQR [{per.spearman.quantile(.25):.4f}, {per.spearman.quantile(.75):.4f}]')
    print(f'  drug-level Pearson (rank of drug means) = {pearson(my.values, mp.values):+.4f}  '
          f'Spearman = {spearman(my.values, mp.values):+.4f}')
    return dict(ss_level_share=SL / SY, r2_level=R2L, ss_within_share=SW / SY,
                r2_within=R2W, drug_level_pearson=pearson(my.values, mp.values),
                drug_level_spearman=spearman(my.values, mp.values)), per


def section_tanimoto(df, meta, fps, out_dir):
    print('\n===== [4] nearest Tanimoto from test drug to training drugs =====')
    rows = []
    for k in sorted(df.fold.unique()):
        te = sorted(df[df.fold == k].DRUG_ID.unique())
        tr = sorted(df[df.fold != k].DRUG_ID.unique())
        te = [d for d in te if d in fps]
        tr = [d for d in tr if d in fps]
        for d in te:
            sims = [(tanimoto(fps[d], fps[t]), t) for t in tr]
            best = max(sims, key=lambda x: x[0])
            rows.append(dict(fold=k, drug=d, nn_sim=best[0], nn_drug=best[1]))
    t = pd.DataFrame(rows)
    print(f'  drug-fold observations {len(t)}')
    for q in [0, 5, 25, 50, 75, 95, 100]:
        print(f'    nn_sim p{q:<3d} = {np.percentile(t.nn_sim, q):.3f}')
    for thr in [0.4, 0.5, 0.6, 0.7]:
        print(f'    nn_sim >= {thr}: {100 * (t.nn_sim >= thr).mean():.1f}% of (drug, fold)')
    t.to_csv(os.path.join(out_dir, 'nn_tanimoto.csv'), index=False)
    return t


def section_trivial(df, meta, fps, out_dir):
    print('\n===== [5] three trivial baselines =====')
    from sklearn.linear_model import Ridge

    folds = sorted(df.fold.unique())
    res = {k: dict(y=[], p=[]) for k in ['cellmean', 'nn_drug', 'fp_ridge', 'ours']}
    for k in folds:
        te_mask = (df.fold == k).values
        tr = df[~te_mask]
        te = df[te_mask]
        # --- baseline 1: cell line mean over every row outside this fold ---
        cl_mean = tr.groupby('GDSC_ID').y.mean()
        fb = float(tr.y.mean())
        p1 = te.GDSC_ID.map(cl_mean).fillna(fb).values
        # --- baseline 2: observed value of the nearest training drug on that cell line ---
        tr_ids = [d for d in sorted(tr.DRUG_ID.unique()) if d in fps]
        nn_map = {}
        for d in sorted(te.DRUG_ID.unique()):
            if d not in fps or not tr_ids:
                nn_map[d] = None; continue
            sims = [(tanimoto(fps[d], fps[t]), t) for t in tr_ids]
            nn_map[d] = max(sims, key=lambda x: x[0])[1]
        lookup = tr.set_index(['DRUG_ID', 'GDSC_ID']).y
        keys = list(zip(te.DRUG_ID.map(nn_map), te.GDSC_ID))
        p2 = np.asarray(lookup.reindex(keys).values, dtype=float)
        p2 = np.where(np.isfinite(p2), p2, p1)
        # --- baseline 3: fingerprint only -> drug mean (ridge), spread as a constant per drug ---
        dmean = tr.groupby('DRUG_ID').y.mean()
        tr_d = [d for d in dmean.index if d in fps]
        Xtr = np.stack([fp_array(fps[d]) for d in tr_d]).astype(np.float32)
        mdl = Ridge(alpha=1.0).fit(Xtr, dmean.loc[tr_d].values)
        te_d = [d for d in sorted(te.DRUG_ID.unique()) if d in fps]
        Xte = np.stack([fp_array(fps[d]) for d in te_d]).astype(np.float32)
        pred_d = dict(zip(te_d, mdl.predict(Xte)))
        p3 = te.DRUG_ID.map(pred_d).fillna(fb).values

        res['cellmean']['y'].append(te.y.values); res['cellmean']['p'].append(p1)
        res['nn_drug']['y'].append(te.y.values); res['nn_drug']['p'].append(p2)
        res['fp_ridge']['y'].append(te.y.values); res['fp_ridge']['p'].append(p3)
        res['ours']['y'].append(te.y.values); res['ours']['p'].append(te.p.values)

    out = {}
    print('  all three baselines use every row outside the fold (val included), a lenient setting for them.')
    for name, d in res.items():
        y = np.concatenate(d['y']); p = np.concatenate(d['p'])
        m = all_metrics(y, p)
        out[name] = m
        print(f'  {name:9s}  R2={m["r2"]:+.4f}  RMSE={m["rmse"]:.4f}  '
              f'Pearson={m["pearson"]:.4f}  Spearman={m["spearman"]:.4f}')
    pd.DataFrame([dict(baseline=k, **v) for k, v in out.items()]).to_csv(
        os.path.join(out_dir, 'baselines.csv'), index=False)
    return out


def section_pathway(df, compound_csv, out_dir):
    print('\n===== [6] performance by target pathway =====')
    if not compound_csv or not os.path.exists(compound_csv):
        print('  [skip] no compound csv supplied')
        return None
    cp = pd.read_csv(compound_csv, low_memory=False)
    cand_id = [c for c in cp.columns if c.strip().upper() in ('DRUG_ID', 'DRUGID')]
    cand_pw = [c for c in cp.columns if 'PATHWAY' in c.upper() or c.strip().upper() in ('TARGET',)]
    if not cand_id or not cand_pw:
        print(f'  [skip] no drug id or pathway column; columns: {list(cp.columns)[:20]}')
        return None
    idc, pwc = cand_id[0], cand_pw[0]
    cp[idc] = cp[idc].astype(str)
    meta = cp.drop_duplicates(idc)[[idc, pwc]].rename(columns={idc: 'DRUG_ID', pwc: 'PATHWAY'})
    m = df.merge(meta, on='DRUG_ID', how='left')
    m['PATHWAY'] = m['PATHWAY'].fillna('UNKNOWN')
    rows = []
    for pw, s in m.groupby('PATHWAY'):
        if len(s) < 500:
            continue
        rows.append(dict(pathway=pw, n=len(s), n_drug=s.GROUP.nunique(),
                         r2=r2(s.y.values, s.p.values),
                         pearson=pearson(s.y.values, s.p.values),
                         spearman=spearman(s.y.values, s.p.values)))
    t = pd.DataFrame(rows).sort_values('n', ascending=False)
    print(f'  pathways with n >= 500: {len(t)}')
    for _, r in t.head(15).iterrows():
        print(f'   {str(r["pathway"])[:45]:45s} n={int(r["n"]):6d} drugs={int(r["n_drug"]):3d} '
              f'R2={r["r2"]:+.3f} r={r["pearson"]:+.3f}')
    t.to_csv(os.path.join(out_dir, 'by_pathway.csv'), index=False)
    return t


def section_export(df, meta, out_dir):
    print('\n===== [7] export fold membership and per-drug predictions =====')
    fold_of_drug = df.groupby('DRUG_ID').agg(GROUP=('GROUP', 'first'), fold=('fold', 'first'),
                                             n=('y', 'size')).reset_index()
    fold_of_drug = fold_of_drug.merge(meta[['DRUG_ID', 'DRUG_NAME']], on='DRUG_ID', how='left')
    fold_of_drug.to_csv(os.path.join(out_dir, 'fold_membership.csv'), index=False)
    df[['DRUG_ID', 'GROUP', 'fold', 'GDSC_ID', 'y', 'p']].to_csv(
        os.path.join(out_dir, 'oof_predictions.csv'), index=False)
    print(f'  fold_membership.csv: {len(fold_of_drug)} drugs / {fold_of_drug.GROUP.nunique()} groups')
    print('  drug groups per fold:', fold_of_drug.groupby('fold')['GROUP'].nunique().to_dict())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cv_results', required=True, help='results directory holding *_fold*_swa_test_predictions.csv')
    ap.add_argument('--h5', default='', help='optional, only checked for existence')
    ap.add_argument('--res_csv', required=True)
    ap.add_argument('--compound_csv', default='')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--n_boot', type=int, default=500)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df, files = load_oof(args.cv_results)
    df, meta = attach_group(df, args.res_csv)
    fps = build_fps(meta)

    overall, ci_row, ci_drug = section_overall(df, args.n_boot)
    ftab, pooled = section_folds(df)
    dec, per_drug = section_decomp(df)
    section_tanimoto(df, meta, fps, args.out_dir)
    section_trivial(df, meta, fps, args.out_dir)
    section_pathway(df, args.compound_csv, args.out_dir)
    section_export(df, meta, args.out_dir)

    ftab.to_csv(os.path.join(args.out_dir, 'per_fold.csv'), index=False)
    per_drug.to_csv(os.path.join(args.out_dir, 'per_drug_corr.csv'), index=False)
    _rows = []
    for k in ['r2', 'rmse', 'mae', 'pearson', 'spearman', 'sd_y']:
        lo_r, hi_r = ci_row.get(k, (np.nan, np.nan))
        lo_d, hi_d = ci_drug.get(k, (np.nan, np.nan))
        _rows.append(dict(metric=k, value=overall[k],
                          ci_lo_row=lo_r, ci_hi_row=hi_r,
                          ci_lo_drug=lo_d, ci_hi_drug=hi_d))
    pd.DataFrame(_rows).to_csv(os.path.join(args.out_dir, 'overall.csv'), index=False)
    print(f'\nall results written to {args.out_dir}')


if __name__ == '__main__':
    main()
