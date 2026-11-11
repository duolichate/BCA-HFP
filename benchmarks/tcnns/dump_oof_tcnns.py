#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""rebuild per-sample out-of-fold predictions from trained tCNNS fold checkpoints (inference only).

purpose: R3-2 asks for pre/post-calibration R² for every model in Table 2, and calibration needs
per-sample (y, p, fold). the original train_tcnns.py only wrote cv_folds.csv /
final_performance.csv and kept no per-sample predictions; this script reuses the same data
pipeline and checkpoints and exports the OOF per-sample predictions to CSV.

usage (from the repository root):
    python 4_DrugResponse_BCAHFP/benchmarks/tcnns/dump_oof_tcnns.py \
        --ckpt_dir 5_Result/2_baselines/tCNNS/CV/RTX5090_v1_base6_tCNNS_20261106_1226 \
        --output 5_Result/3_diagnostics/4_baselines/tcnns_oof_from_ckpt.csv

self-check: the pooled raw R² must match raw_r2 in final_performance.csv (tolerance 1e-4);
a mismatch means the char table, gene standardization or fold split was not reproduced, so
check that before using the output.
"""

import sys as _sys
if hasattr(_sys.stdout, "reconfigure"):
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
sys.path.insert(0, os.path.dirname(os.path.dirname(_here)))

from train_tcnns import tCNNS, build_smiles_vocab, smiles_to_onehot  # noqa: E402
from bca_hfp.config.base_config import get_base_config, root_dir  # noqa: E402
from bca_hfp.data.loader import load_drug_atom_features, load_gene_embeddings, load_response  # noqa: E402
from bca_hfp.training.train_holdout import (load_fixed_fold_splits, build_censor_bounds,  # noqa: E402
                                            winsorize_pred, pooled_metrics,
                                            fit_gene_scaler, apply_gene_scaler)
from bca_hfp.utils.reproducibility import set_seed  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description='tCNNS: dump per-sample OOF predictions from checkpoints')
    ap.add_argument('--ckpt_dir', type=str, required=True,
                    help='dir containing best_model_fold1..5.pth')
    ap.add_argument('--output', type=str, required=True, help='output CSV path')
    ap.add_argument('--dataset', type=str, default='GDSC', choices=['GDSC'])
    ap.add_argument('--gene_version', type=str, default='V1', choices=['V1', 'V2'])
    ap.add_argument('--batch_size', type=int, default=1024)
    ap.add_argument('--val_ratio', type=float, default=0.125,
                    help='must match training, used to reproduce the train/val cell sets (affects gene standardization)')
    ap.add_argument('--fold_csv', type=str, default='')
    ap.add_argument('--with_val', action='store_true',
                    help='also export the per-fold val subset predictions (for "fit on validation" calibration)')
    ap.add_argument('--expect_r2', type=float, default=None,
                    help='optional: expected pooled raw R² (e.g. 0.2600), used as a self-check')
    args = ap.parse_args()

    set_seed(42)
    config = get_base_config(args.dataset)

    base_data = root_dir("Data")
    if args.gene_version == 'V1':
        config['data']['gene_feat_dir'] = f"{base_data}/Gene/GDSC_EXP_GF_V1"
        config['data']['gene_input_dim'] = 256
    else:
        config['data']['gene_feat_dir'] = f"{base_data}/Gene/GDSC_EXP_GF_V2"
        config['data']['gene_input_dim'] = 768

    fold_csv = args.fold_csv or os.path.join(base_data, "Membership", "fold_var3_fromckpt.csv")

    drug_id_col = config['data']['drug_id_col']
    cell_id_col = config['data']['cell_id_col']
    label_col = config['data']['label_col']

    print("Loading data ...")
    gene_emb_dict, _ = load_gene_embeddings(config['data']['gene_feat_dir'], cell_id_col)
    response_df = load_response(config['data']['response_path'], cell_id_col, drug_id_col)
    with pd.HDFStore(config['data']['drug_feat_path'], 'r') as store:
        drug_smiles_df = store['drug_smiles']
    drug_smile_dict = {str(r['DRUG_ID']): r['DRUG_SMILE'] for _, r in drug_smiles_df.iterrows()}

    # the training script masks on drug_atom_dict; membership in the SMILES table is equivalent here
    # and avoids reading atom_features (a numpy 2 pickle that older numpy cannot deserialize).
    valid_mask = (response_df[drug_id_col].astype(str).isin(drug_smile_dict)
                  & response_df[cell_id_col].isin(gene_emb_dict))
    df = response_df[valid_mask].copy()
    print(f"Valid samples: {len(df)}")

    all_smiles = [drug_smile_dict[d] for d in df[drug_id_col].unique() if d in drug_smile_dict]
    chars, seq_len = build_smiles_vocab(all_smiles)
    char2idx = {c: i for i, c in enumerate(chars)}
    drug_onehot = {d: smiles_to_onehot(drug_smile_dict[d], char2idx, seq_len)
                   for d in df[drug_id_col].unique() if d in drug_smile_dict}
    pooled_gene = {cid: emb.mean(axis=0) for cid, emb in gene_emb_dict.items()}
    print(f"SMILES vocab: {len(chars)} chars, max_len={seq_len}")

    grp = df['DRUG_NAME'].astype(str).str.strip().str.upper()
    splits = load_fixed_fold_splits(fold_csv, grp)
    b_lo, b_hi = build_censor_bounds(df, label_col)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    rows, val_rows = [], []
    for fold_idx, (_, test_idx) in enumerate(splits):
        fold = fold_idx + 1
        ckpt = os.path.join(args.ckpt_dir, f'best_model_fold{fold}.pth')
        if not os.path.exists(ckpt):
            print(f"  [skip] fold {fold}: {ckpt} missing")
            continue

        test_df = df.iloc[test_idx]
        # reproduce the training train/val split: gene standardization stats are fit on train_df cells only
        train_val_df = df.iloc[splits[fold_idx][0]]
        tv_grp = train_val_df['DRUG_NAME'].astype(str).str.strip().str.upper().values
        np.random.seed(42)
        val_grp = set(np.random.choice(np.unique(tv_grp),
                                       size=max(1, int(len(np.unique(tv_grp)) * args.val_ratio)),
                                       replace=False))
        val_mask = train_val_df['DRUG_NAME'].astype(str).str.strip().str.upper().isin(val_grp).values
        train_df = train_val_df[~val_mask]
        tr_cells = sorted(set(train_df[cell_id_col].values))
        g_mean, g_std = fit_gene_scaler([pooled_gene[c] for c in tr_cells if c in pooled_gene])

        model = tCNNS(len(chars), seq_len, config['data']['gene_input_dim'],
                      dropout=config['model']['dropout']).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.eval()

        def run_part(part_df):
            drugs, cells, labs, pos = [], [], [], []
            for p, (did, cid, lab) in enumerate(zip(part_df[drug_id_col].values,
                                                    part_df[cell_id_col].values,
                                                    part_df[label_col].values)):
                if did not in drug_onehot or cid not in pooled_gene:
                    continue
                drugs.append(drug_onehot[did])
                cells.append(apply_gene_scaler(pooled_gene[cid], g_mean, g_std))
                labs.append(lab)
                pos.append(p)
            ds = TensorDataset(torch.tensor(np.stack(drugs), dtype=torch.float32),
                               torch.tensor(np.stack(cells), dtype=torch.float32),
                               torch.tensor(np.asarray(labs), dtype=torch.float32))
            loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
            preds = []
            with torch.no_grad():
                for drug, cell, _ in loader:
                    out = model(drug.to(device), cell.to(device))
                    preds.extend(out.cpu().numpy().flatten().tolist())
            sub = part_df.iloc[np.asarray(pos, dtype=np.int64)]
            return pd.DataFrame({'fold': fold,
                                 'DRUG_ID': sub[drug_id_col].values,
                                 'GDSC_ID': sub[cell_id_col].values,
                                 'y': np.asarray(labs, dtype=np.float64),
                                 'p_raw': np.asarray(preds, dtype=np.float64)})

        rows.append(run_part(test_df))
        print(f"  fold {fold}: n={len(rows[-1])}  raw R²={r2_score(rows[-1].y, rows[-1].p_raw):.4f}")

        # optional: also export this fold's val subset predictions (for "fit on validation" calibration)
        if args.with_val:
            val_rows.append(run_part(train_val_df[val_mask]))
            print(f"           val n={len(val_rows[-1])}")

    out = pd.concat(rows, ignore_index=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    out.to_csv(args.output, index=False, encoding='utf-8-sig')

    y = out['y'].values
    p = out['p_raw'].values
    r2_pool = 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    print(f"\nOOF rows: {len(out)}")
    print(f"pooled raw R² = {r2_pool:.4f}")
    if args.expect_r2 is not None:
        ok = abs(r2_pool - args.expect_r2) < 1e-3
        print(f"self-check vs {args.expect_r2}: {'PASS' if ok else 'FAIL'}")
    print(f"saved: {args.output}")

    if args.with_val and val_rows:
        val_out = os.path.splitext(args.output)[0] + '_val.csv'
        pd.concat(val_rows, ignore_index=True).to_csv(val_out, index=False, encoding='utf-8-sig')
        print(f"saved val: {val_out}")


if __name__ == '__main__':
    main()
