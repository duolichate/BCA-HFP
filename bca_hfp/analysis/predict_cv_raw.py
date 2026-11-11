# bca_hfp/analysis/predict_cv_raw.py
"""export the **pre-clamp** predictions of the CV folds (OOF), for the two-variant comparison of table A.

why
----
training sets `pred_clamp=True`, which clips predictions to the measured concentration window of that drug
±2 and then to [ln0.001, ln100], so the saved OOF is already clamped. for the "before vs after clamping"
comparison the main-model row has no raw values and is asymmetric against the other baselines, so this
script reruns a forward pass with the same checkpoints (no post-processing at all).

points
----
* data is loaded once and reused over the 5 folds (saves 5 h5 reads and 1635 .pt files).
* reproduces the training split exactly: GroupKFold + grouping by drug name (--group_by_name, on by default).
* writes per-fold and pooled metrics (raw and clamped, for cross-checking).
* self-check: against the saved `*_swa_test_predictions.csv`, rows that were not clipped must match value by value.

usage
----
python bca_hfp/analysis/predict_cv_raw.py \
  --models_dir ~/my_model/Result/My_model/CV/GDSC_attention_3l_V1_drug_fp2048_batchnorm_md0_seed42_cv5_ne20_lrfix_swa5_desc_pc_gbn_mse_dml0.5_e7/models \
  --name attention_drug --ckpt swa \
  --res_csv ../Data/GDSC_RES.csv \
  --folds 1,2,3,4,5 \
  --out_dir ~/my_model/Result/diag/ours_raw
"""

import argparse
import glob
import math
import os
import sys
from functools import partial

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from bca_hfp.data.dataset import DrugGeneDataset, collate_fn
from bca_hfp.data.loader import (load_drug_atom_features, load_gene_embeddings,
                                 load_morgan_fingerprints, load_response)
from bca_hfp.models import create_model
from bca_hfp.utils.reproducibility import set_seed


def metrics(y, p):
    from scipy.stats import pearsonr
    return dict(n=len(y),
                R2=float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()),
                RMSE=float(np.sqrt(((y - p) ** 2).mean())),
                Pearson=float(pearsonr(y, p)[0]))


def spearman(y, p):
    yr = pd.Series(np.asarray(y, np.float64)).rank().values
    pr = pd.Series(np.asarray(p, np.float64)).rank().values
    if yr.std() < 1e-12 or pr.std() < 1e-12:
        return float('nan')
    return float(np.corrcoef(yr, pr)[0, 1])


def main():
    ap = argparse.ArgumentParser(description='export the pre-clamp CV predictions')
    ap.add_argument('--models_dir', required=True)
    ap.add_argument('--name', default='attention_drug', help='checkpoint prefix (the name used by train_cv)')
    ap.add_argument('--ckpt', default='swa', choices=['swa', 'best_val', 'final'],
                    help='which checkpoint to use; the OOF main table uses swa')
    ap.add_argument('--res_csv', required=True)
    ap.add_argument('--h5', default='', help='override drug_feat_path from the config (optional)')
    ap.add_argument('--gene_feat_dir', default='', help='override gene_feat_dir from the config (optional)')
    ap.add_argument('--folds', default='1,2,3,4,5')
    ap.add_argument('--slacks', default='0,0.5,1,1.5,2,3,4,6,10',
                    help='comma separated slack list (ln units; 2 matches the training setting)')
    ap.add_argument('--n_splits', type=int, default=5)
    ap.add_argument('--no_group_by_name', action='store_true')
    ap.add_argument('--fold_csv', default='',
                    help='external fixed split CSV used in training (GROUP,DRUG_ID,fold); when given the '
                         'split follows it, exactly as in training (GroupKFold differs across sklearn versions)')
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--out_dir', required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(0)

    folds = [int(x) for x in args.folds.split(',') if x.strip()]
    slacks = sorted({float(x) for x in args.slacks.split(',') if x.strip()})
    print(f'[slack] sweep {slacks} (2 matches pred_clamp in training)')

    # ---------- checkpoint and config (taken from the first fold) ----------
    first = os.path.join(args.models_dir, f'{args.name}_fold{folds[0]}_{args.ckpt}.pth')
    if not os.path.exists(first):
        cand = sorted(glob.glob(os.path.join(args.models_dir, f'{args.name}_fold*_{args.ckpt}.pth')))
        raise SystemExit(f'{first} not found\n  available in the dir: {[os.path.basename(c) for c in cand][:6]}')
    ckpt0 = torch.load(first, map_location='cpu', weights_only=False)
    config = ckpt0.get('config') if isinstance(ckpt0, dict) else None
    if config is None:
        here = os.path.dirname(os.path.abspath(first))
        for fn in sorted(os.listdir(here)):
            if not fn.endswith('.pth') or fn == os.path.basename(first):
                continue
            other = torch.load(os.path.join(here, fn), map_location='cpu', weights_only=False)
            if isinstance(other, dict) and other.get('config'):
                config = other['config']
                print(f'  [info] {os.path.basename(first)} has no config (SWA stores weights only), reuse {fn} from the same dir')
                break
        if config is None:
            raise SystemExit(f'{first} has no config, and no .pth with a config in the same dir')
    del ckpt0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        free, total = torch.cuda.mem_get_info()
        print(f'GPU free {free / 2**30:.1f} / {total / 2**30:.1f} GiB')

    data_cfg = config['data']
    drug_col, cell_col, label_col = (data_cfg['drug_id_col'], data_cfg['cell_id_col'],
                                     data_cfg['label_col'])
    h5_path = args.h5 or data_cfg['drug_feat_path']
    use_desc = int(data_cfg['drug_global_dim']) > 2048
    fp_bits = 2048 if use_desc else int(data_cfg['drug_global_dim'])
    print(f'config: use_desc={use_desc} fp_bits={fp_bits} '
          f'reg_norm={config["model"].get("reg_norm")} '
          f'pred_clamp={config["training"].get("pred_clamp")} (no clamping here)')

    # ---------- data (loaded once) ----------
    print('Loading data...')
    drug_atom_dict = load_drug_atom_features(h5_path)
    gene_emb_dict, _ = load_gene_embeddings(args.gene_feat_dir or data_cfg['gene_feat_dir'], cell_col)
    response_df = load_response(h5_path, cell_col, drug_col)
    drug_global_dict = load_morgan_fingerprints(h5_path, fp_bits=fp_bits, fp_radius=2,
                                                use_desc=use_desc)
    valid = (response_df[drug_col].isin(drug_atom_dict)
             & response_df[cell_col].isin(gene_emb_dict))
    filtered_df = response_df[valid].copy()
    filtered_df['_group'] = (filtered_df[drug_col] if args.no_group_by_name else
                             filtered_df['DRUG_NAME'].astype(str).str.strip().str.upper())
    print(f'valid samples {len(filtered_df)}, '
          f'{filtered_df["_group"].nunique()} groups / {filtered_df[drug_col].nunique()} drugs / '
          f'{filtered_df[cell_col].nunique()} cell lines')

    res = pd.read_csv(args.res_csv, low_memory=False)
    for c in (drug_col, cell_col):
        res[c] = res[c].astype(str)
    res[cell_col] = res[cell_col].astype(str).str.strip()
    win = res.drop_duplicates([drug_col, cell_col])[[drug_col, cell_col, 'MIN_CONC', 'MAX_CONC']]

    if args.fold_csv:
        # external fixed split identical to train_cv.py: GROUP (drug name) -> fold
        fm = pd.read_csv(args.fold_csv)
        key = 'GROUP' if 'GROUP' in fm.columns else ('DRUG_NAME' if 'DRUG_NAME' in fm.columns
                                                    else drug_col)
        fm[key] = fm[key].astype(str).str.strip().str.upper()
        g2f = dict(zip(fm[key], fm['fold'].astype(int)))
        _keys = (filtered_df['_group'].astype(str).str.strip().str.upper()
                 if not args.no_group_by_name else filtered_df[drug_col].astype(str))
        _fold = _keys.map(g2f)
        if _fold.isna().any():
            raise SystemExit(f'--fold_csv does not cover {int(_fold.isna().sum())} rows '
                             f'(check that grouping key {key} matches --group_by_name)')
        splits = []
        for k in sorted(fm['fold'].unique()):
            te = (_fold == k).values
            splits.append((np.where(~te)[0], np.where(te)[0]))
        print(f'  [fold_csv] using the external split {args.fold_csv}: '
              f'{len(splits)} folds, drug groups per fold={[int((_fold == k).sum()) for k in sorted(fm["fold"].unique())]}')
    else:
        splits = list(GroupKFold(n_splits=args.n_splits).split(
            filtered_df, groups=filtered_df['_group'].values))

    rows, per_fold = [], []
    LN_MIN, LN_MAX = math.log(0.001), math.log(100.0)
    for k in folds:
        train_val_idx, test_idx = splits[k - 1]
        test_df = filtered_df.iloc[test_idx].copy()
        dataset = DrugGeneDataset(
            test_df[drug_col].tolist(), test_df[cell_col].tolist(),
            test_df[label_col].values, drug_atom_dict, gene_emb_dict, drug_global_dict)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False,
            collate_fn=partial(collate_fn), num_workers=args.num_workers)

        path = os.path.join(args.models_dir, f'{args.name}_fold{k}_{args.ckpt}.pth')
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        model = create_model('attention', config).to(device)
        missing, unexpected = model.load_state_dict(ckpt['model_state_dict'], strict=False)
        if unexpected or missing:
            raise SystemExit(f'fold{k} weight mismatch: missing={missing[:3]} unexpected={unexpected[:3]}')
        model.eval()

        preds = []
        with torch.no_grad():
            for batch in loader:
                atom, atom_mask, gene, gene_mask, labels, drug_global = batch
                out = model(gene.to(device), atom.to(device), atom_mask.to(device),
                            gene_mask.to(device), drug_global=drug_global.to(device))
                preds.extend(out.cpu().float().numpy())
        preds = np.asarray(preds, dtype=np.float32)
        assert len(preds) == len(test_df), (len(preds), len(test_df))

        sub = pd.DataFrame({
            'DRUG_ID': test_df[drug_col].values,
            'GDSC_ID': test_df[cell_col].values,
            'fold': k,
            'GROUP': test_df['_group'].values,
            'y': test_df[label_col].values,
            'p_raw': preds,
        })
        sub = sub.merge(win.rename(columns={drug_col: 'DRUG_ID', cell_col: 'GDSC_ID'}),
                        on=['DRUG_ID', 'GDSC_ID'], how='left')
        if sub.MIN_CONC.isna().any():
            raise SystemExit(f'fold{k}: {int(sub.MIN_CONC.isna().sum())} rows have no concentration window, cannot compute the clamped value')
        lo = np.log(sub.MIN_CONC.values)
        hi = np.log(sub.MAX_CONC.values)

        # default variant (slack=2), identical to pred_clamp in training
        sub['p_win'] = np.clip(np.clip(sub.p_raw.values, lo - 2.0, hi + 2.0), LN_MIN, LN_MAX)
        sub['clipped'] = np.abs(sub.p_win.values - sub.p_raw.values) > 1e-6

        # sensitivity over slack
        cols = []
        for s in slacks:
            col = f'p_s{s:g}'
            sub[col] = np.clip(np.clip(sub.p_raw.values, lo - s, hi + s), LN_MIN, LN_MAX)
            m = metrics(sub.y.values, sub[col].values)
            fr = float(np.mean(np.abs(sub[col].values - sub.p_raw.values) > 1e-6))
            per_fold.append(dict(fold=k, slack=s, n=len(sub), clipped=fr,
                                 R2=m['R2'], RMSE=m['RMSE'], Pearson=m['Pearson']))
            cols.append(col)

        m_raw = metrics(sub.y.values, sub.p_raw.values)
        print(f'fold{k}: n={len(sub)}  R2_raw={m_raw["R2"]:+.4f}  ' +
              '  '.join(f's={s:g}:{per_fold[-len(slacks) + i]["R2"]:+.4f}'
                        f'({100 * per_fold[-len(slacks) + i]["clipped"]:.0f}%)'
                        for i, s in enumerate(slacks)))
        rows.append(sub[['DRUG_ID', 'GDSC_ID', 'fold', 'GROUP', 'y',
                         'p_raw', 'p_win'] + cols])
        del model, ckpt

    oof = pd.concat(rows, ignore_index=True)
    out_csv = os.path.join(args.out_dir, 'oof_raw_predictions.csv')
    oof.to_csv(out_csv, index=False)
    pf = pd.DataFrame(per_fold)
    pf.to_csv(os.path.join(args.out_dir, 'slack_sweep_per_fold.csv'), index=False)

    # ---------- self-check: value by value against the saved (clamped) predictions ----------
    ref_files = sorted(glob.glob(os.path.join(
        os.path.dirname(args.models_dir), 'results',
        f'*_fold*_{args.ckpt}_test_predictions.csv')))
    if ref_files:
        ref = pd.concat([pd.read_csv(f) for f in ref_files], ignore_index=True)
        ref = ref.rename(columns={'true_label': 'y_ref', 'predicted': 'p_ref'})
        ref['DRUG_ID'] = ref['DRUG_ID'].astype(str)
        ref['GDSC_ID'] = ref['GDSC_ID'].astype(str)
        m = oof.merge(ref[['DRUG_ID', 'GDSC_ID', 'y_ref', 'p_ref']],
                      on=['DRUG_ID', 'GDSC_ID'], how='inner')
        unclipped = m[np.abs(m.p_raw.values - m.p_win.values) <= 1e-6]
        diff = float(np.abs(unclipped.p_raw.values - unclipped.p_ref.values).max()) if len(unclipped) else np.nan
        print(f'\n[check] matched {len(m)}/{len(oof)} rows against the saved predictions; '
              f'max value difference over the {len(unclipped)} unclipped rows = {diff:.2e}')
        if np.isfinite(diff) and diff > 1e-4:
            print('  [warn] large difference: the checkpoint and the saved OOF may not come from the same run')
    else:
        print(f'\n[check] no prediction file under {os.path.dirname(args.models_dir)}/results, skip the self-check')

    # ---------- pooled metrics + between-drug / within-drug decomposition ----------
    def pooled(col):
        y, p = oof.y.values, oof[col].values
        r = metrics(y, p)
        r['Spearman'] = spearman(y, p)
        g = oof.groupby('GROUP')
        n = g.size()
        my, mp = g.y.mean(), g[col].mean()
        gm = y.mean()
        SL = float((n * (my - gm) ** 2).sum())
        r['R2_level'] = float(1 - (n * (mp - my) ** 2).sum() / SL)
        yc = oof.y - g.y.transform('mean')
        pc = oof[col] - g[col].transform('mean')
        r['R2_within'] = float(1 - ((yc - pc) ** 2).sum() / (yc ** 2).sum())
        r['clipped'] = float(np.mean(np.abs(p - oof.p_raw.values) > 1e-6))
        return r

    sweep = []
    sweep.append(dict(slack='none(raw)', **pooled('p_raw')))
    for s in slacks:
        sweep.append(dict(slack=s, **pooled(f'p_s{s:g}')))
    sw = pd.DataFrame(sweep)
    sw.to_csv(os.path.join(args.out_dir, 'slack_sweep_pooled.csv'), index=False)

    sw2 = pd.DataFrame([dict(condition='raw', **pooled('p_raw')),
                        dict(condition='clamped', **pooled('p_win'))])
    sw2.to_csv(os.path.join(args.out_dir, 'oof_raw_summary.csv'), index=False)

    print('\n' + '=' * 84)
    print('pooled OOF (slack = allowed overshoot of the measured window in ln units; none = no clamping)')
    print(f'{"slack":>10}{"clip%":>8}{"R2":>9}{"Pearson":>9}{"Spearman":>10}'
          f'{"R2_level":>10}{"R2_within":>11}')
    for _, r in sw.iterrows():
        print(f'{str(r.slack):>10}{100 * r.clipped:>7.1f}%{r.R2:>+9.4f}{r.Pearson:>9.4f}'
              f'{r.Spearman:>10.4f}{r.R2_level:>+10.4f}{r.R2_within:>+11.4f}')
    print('=' * 84)
    print(f'reference: R2 at slack=2 = {[r.R2 for _, r in sw.iterrows() if str(r.slack) == "2.0"][0]:+.4f}'
          f' (should be close to the clamped OOF metric saved by training)')
    print(f'[ok] -> {out_csv}')
    print(f'[ok] -> {os.path.join(args.out_dir, "slack_sweep_pooled.csv")}')
    print(f'[ok] -> {os.path.join(args.out_dir, "slack_sweep_per_fold.csv")}')


if __name__ == '__main__':
    main()
