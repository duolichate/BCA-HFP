# bca_hfp/analysis/predict_cv_val.py
"""export the pre-clamp predictions of the five **validation subsets** of the main model (for the "fit on validation" calibration in R3-2).

why
----
training only saved test-fold predictions, the per-sample output of the validation fold (12.5% of each
train_val part, split by drug name) was never written. reviewer R3-2 asks to fit slope and intercept on
the validation predictions and then report R2 before and after calibration, so this file is needed.

implementation
----
* reproduces the val split of train_cv.py: `_group`=drug name, np.random.seed(split_seed),
  np.random.choice(unique_drugs, int(len*val_ratio), replace=False).
* everything else (fold split, model, weights, data) is identical to predict_cv_raw.py, forward only, no clamping.
* self-check: the per-fold val R2 printed here should be close to the best val R2 of that fold in the training log (same weights, same split).

usage
----
python bca_hfp/analysis/predict_cv_val.py \
  --models_dir 5_Result/1_main_models/RTX5090_v1_gc_c5_s42_20261106_1822/models \
  --res_csv 1_Data/Drug/GDSC_RES.csv \
  --fold_csv 1_Data/Membership/fold_var3_fromckpt.csv \
  --out_csv 5_Result/3_diagnostics/1_model_level/bca_val_from_ckpt.csv
"""

import argparse
import os
import sys
from functools import partial

# compat shim: atom_features in drug_features.h5 were written by numpy 2, and an older numpy that
# unpickles them looks for numpy._core.* (the numpy 2 module path). map numpy._core.* to numpy.core.*
# before pandas/tables read the h5.
try:
    import numpy.core as _ncore
    for _n in ('numeric', 'multiarray', '_multiarray_umath', 'umath'):
        _m = getattr(_ncore, _n, None)
        if _m is not None:
            sys.modules.setdefault(f'numpy._core.{_n}', _m)
    sys.modules.setdefault('numpy._core', _ncore)
except Exception:
    pass

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


def main():
    ap = argparse.ArgumentParser(description='export the validation-subset predictions of the five folds')
    ap.add_argument('--models_dir', required=True)
    ap.add_argument('--name', default='attention_drug')
    ap.add_argument('--ckpt', default='swa', choices=['swa', 'best_val', 'final'])
    ap.add_argument('--res_csv', required=True)
    ap.add_argument('--h5', default='')
    ap.add_argument('--gene_feat_dir', default='')
    ap.add_argument('--folds', default='1,2,3,4,5')
    ap.add_argument('--fold_csv', default='')
    ap.add_argument('--val_ratio', type=float, default=0.125)
    ap.add_argument('--split_seed', type=int, default=42)
    ap.add_argument('--n_splits', type=int, default=5)
    ap.add_argument('--batch_size', type=int, default=128)
    ap.add_argument('--num_workers', type=int, default=0)
    ap.add_argument('--out_csv', required=True)
    args = ap.parse_args()

    set_seed(0)
    folds = [int(x) for x in args.folds.split(',') if x.strip()]

    first = os.path.join(args.models_dir, f'{args.name}_fold{folds[0]}_{args.ckpt}.pth')
    if not os.path.exists(first):
        raise SystemExit(f'{first} not found')
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
                print(f'  [info] reuse the config from {fn} in the same dir')
                break
        if config is None:
            raise SystemExit(f'{first} has no config')
    del ckpt0

    data_cfg = config['data']
    drug_col, cell_col, label_col = (data_cfg['drug_id_col'], data_cfg['cell_id_col'],
                                     data_cfg['label_col'])
    h5_path = args.h5 or data_cfg['drug_feat_path']
    use_desc = int(data_cfg['drug_global_dim']) > 2048
    fp_bits = 2048 if use_desc else int(data_cfg['drug_global_dim'])

    print('Loading data ...')
    drug_atom_dict = load_drug_atom_features(h5_path)
    gene_emb_dict, _ = load_gene_embeddings(args.gene_feat_dir or data_cfg['gene_feat_dir'], cell_col)
    response_df = load_response(h5_path, cell_col, drug_col)
    drug_global_dict = load_morgan_fingerprints(h5_path, fp_bits=fp_bits, fp_radius=2,
                                                use_desc=use_desc)
    valid = (response_df[drug_col].isin(drug_atom_dict)
             & response_df[cell_col].isin(gene_emb_dict))
    filtered_df = response_df[valid].copy()
    filtered_df['_group'] = filtered_df['DRUG_NAME'].astype(str).str.strip().str.upper()
    print(f'valid samples {len(filtered_df)}')

    if args.fold_csv:
        fm = pd.read_csv(args.fold_csv)
        key = 'GROUP' if 'GROUP' in fm.columns else 'DRUG_NAME'
        fm[key] = fm[key].astype(str).str.strip().str.upper()
        g2f = dict(zip(fm[key], fm['fold'].astype(int)))
        _fold = filtered_df['_group'].map(g2f)
        if _fold.isna().any():
            raise SystemExit(f'--fold_csv does not cover {int(_fold.isna().sum())} rows')
        splits = []
        for k in sorted(fm['fold'].unique()):
            te = (_fold == k).values
            splits.append((np.where(~te)[0], np.where(te)[0]))
    else:
        splits = list(GroupKFold(n_splits=args.n_splits).split(
            filtered_df, groups=filtered_df['_group'].values))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    rows = []
    for k in folds:
        train_val_idx, _ = splits[k - 1]
        tv = filtered_df.iloc[train_val_idx].copy()

        unique_drugs = tv['_group'].unique()
        np.random.seed(args.split_seed)
        val_drugs = np.random.choice(unique_drugs,
                                     size=int(len(unique_drugs) * args.val_ratio),
                                     replace=False)
        val_df = tv[tv['_group'].isin(val_drugs)].copy()

        dataset = DrugGeneDataset(
            val_df[drug_col].tolist(), val_df[cell_col].tolist(),
            val_df[label_col].values, drug_atom_dict, gene_emb_dict, drug_global_dict)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False,
            collate_fn=partial(collate_fn), num_workers=args.num_workers)

        path = os.path.join(args.models_dir, f'{args.name}_fold{k}_{args.ckpt}.pth')
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        model = create_model('attention', config).to(device)
        missing, unexpected = model.load_state_dict(ckpt['model_state_dict'], strict=False)
        if unexpected or missing:
            raise SystemExit(f'fold{k} weight mismatch: {missing[:3]} / {unexpected[:3]}')
        model.eval()

        preds = []
        with torch.no_grad():
            for batch in loader:
                atom, atom_mask, gene, gene_mask, labels, drug_global = batch
                out = model(gene.to(device), atom.to(device), atom_mask.to(device),
                            gene_mask.to(device), drug_global=drug_global.to(device))
                preds.extend(out.cpu().float().numpy())
        preds = np.asarray(preds, dtype=np.float32)
        assert len(preds) == len(val_df), (len(preds), len(val_df))

        y = val_df[label_col].values.astype(np.float64)
        r2 = 1 - ((y - preds) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        rows.append(pd.DataFrame({'fold': k,
                                  'DRUG_ID': val_df[drug_col].values,
                                  'GDSC_ID': val_df[cell_col].values,
                                  'y': y,
                                  'p_raw': preds}))
        print(f'  fold {k}: n={len(preds)}  val R²={r2:.4f}  (drug groups {len(val_drugs)})')

    out = pd.concat(rows, ignore_index=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    out.to_csv(args.out_csv, index=False, encoding='utf-8-sig')
    print(f'saved {args.out_csv}  rows={len(out)}')


if __name__ == '__main__':
    main()
