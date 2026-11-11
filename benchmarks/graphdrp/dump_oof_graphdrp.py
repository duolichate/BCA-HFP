#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""rebuild per-sample out-of-fold predictions from trained GraphDRP / GraTransDRP fold checkpoints (inference only).

needs torch_geometric and rdkit (present in the gene5090 env on the 5090 box), must run on the 5090 box.

usage (from the project root):
    # GraphDRP (--model gcn)
    python 4_DrugResponse_BCAHFP/benchmarks/graphdrp/dump_oof_graphdrp.py \
        --model gcn \
        --ckpt_dir 5_Result/2_baselines/GraphDRP/CV/RTX5090_v1_base2_GraphDRP_20261106_0811 \
        --output 5_Result/3_diagnostics/4_baselines/graphdrp_oof_from_ckpt.csv \
        --expect_r2 0.0738

    # GraTransDRP (--model gat_gcn)
    python 4_DrugResponse_BCAHFP/benchmarks/graphdrp/dump_oof_graphdrp.py \
        --model gat_gcn \
        --ckpt_dir 5_Result/2_baselines/GraTransDRP/CV/RTX5090_v1_base3_GraTransDRP_20261106_0315 \
        --output 5_Result/3_diagnostics/4_baselines/gratransdrp_oof_from_ckpt.csv \
        --expect_r2 0.0896
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
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
sys.path.insert(0, os.path.dirname(os.path.dirname(_here)))

from train_graphdrp import (GraphDRPSimpleDataset, gra_collate_fn,  # noqa: E402
                            GCNNet, GATNet, GAT_GCN, GINConvNet)
from bca_hfp.config.base_config import get_base_config, root_dir  # noqa: E402
from bca_hfp.data.loader import load_gene_embeddings, load_response  # noqa: E402
from bca_hfp.training.train_holdout import load_fixed_fold_splits, fit_gene_scaler  # noqa: E402
from bca_hfp.utils.reproducibility import set_seed  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description='GraphDRP/GraTransDRP: dump per-sample OOF predictions')
    ap.add_argument('--ckpt_dir', type=str, required=True)
    ap.add_argument('--output', type=str, required=True)
    ap.add_argument('--model', type=str, default='gcn', choices=['gcn', 'gat', 'gat_gcn', 'ginconv'])
    ap.add_argument('--dataset', type=str, default='GDSC', choices=['GDSC'])
    ap.add_argument('--gene_version', type=str, default='V1', choices=['V1', 'V2'])
    ap.add_argument('--batch_size', type=int, default=1024)
    ap.add_argument('--val_ratio', type=float, default=0.125,
                    help='must match training, used to reproduce the train/val cell sets (affects gene standardization)')
    ap.add_argument('--fold_csv', type=str, default='')
    ap.add_argument('--with_val', action='store_true',
                    help='also export the per-fold val subset predictions (for "fit on validation" calibration)')
    ap.add_argument('--expect_r2', type=float, default=None)
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

    model_map = {'gcn': GCNNet, 'gat': GATNet, 'gat_gcn': GAT_GCN, 'ginconv': GINConvNet}
    model_cls = model_map[args.model]

    print("Loading data ...")
    gene_emb_dict, _ = load_gene_embeddings(config['data']['gene_feat_dir'], cell_id_col)
    response_df = load_response(config['data']['response_path'], cell_id_col, drug_id_col)
    with pd.HDFStore(config['data']['drug_feat_path'], 'r') as store:
        drug_smiles_df = store['drug_smiles']
    drug_smile_dict = {str(r['DRUG_ID']): r['DRUG_SMILE'] for _, r in drug_smiles_df.iterrows()}

    df = response_df[(response_df[drug_id_col].astype(str).isin(drug_smile_dict))
                     & (response_df[cell_id_col].isin(gene_emb_dict))].copy()
    print(f"Valid samples: {len(df)}")

    grp = df['DRUG_NAME'].astype(str).str.strip().str.upper()
    splits = load_fixed_fold_splits(fold_csv, grp)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    rows, val_rows = [], []
    for fold_idx, (train_val_idx, test_idx) in enumerate(splits):
        fold = fold_idx + 1
        ckpt = os.path.join(args.ckpt_dir, f'best_model_fold{fold}.pth')
        if not os.path.exists(ckpt):
            print(f"  [skip] fold {fold}: checkpoint missing")
            continue

        train_val_df = df.iloc[train_val_idx]
        tv_grp = train_val_df['DRUG_NAME'].astype(str).str.strip().str.upper().values
        np.random.seed(42)
        val_grp = set(np.random.choice(np.unique(tv_grp),
                                       size=max(1, int(len(np.unique(tv_grp)) * args.val_ratio)),
                                       replace=False))
        val_mask = train_val_df['DRUG_NAME'].astype(str).str.strip().str.upper().isin(val_grp).values
        train_df = train_val_df[~val_mask]
        test_df = df.iloc[test_idx]

        tr_cells = sorted(set(train_df[cell_id_col].values))
        scaler = fit_gene_scaler([gene_emb_dict[c].mean(axis=0)
                                  for c in tr_cells if c in gene_emb_dict])

        model = model_cls(num_features_xd=78, num_features_xt=config['data']['gene_input_dim'],
                          output_dim=128, dropout=0.2).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        model.eval()

        def run_part(part_df):
            ds = GraphDRPSimpleDataset(part_df[drug_id_col].tolist(), part_df[cell_id_col].tolist(),
                                       part_df[label_col].values, drug_smile_dict, gene_emb_dict,
                                       gene_scaler=scaler)
            loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                                collate_fn=gra_collate_fn)
            preds = []
            with torch.no_grad():
                for graphs, _ in loader:
                    graphs = graphs.to(device)
                    out, _ = model(graphs)
                    preds.extend(out.cpu().numpy().flatten().tolist())
            sub = part_df.iloc[np.asarray(ds.row_idx, dtype=np.int64)]
            return pd.DataFrame({'fold': fold,
                                 'DRUG_ID': sub[drug_id_col].values,
                                 'GDSC_ID': sub[cell_id_col].values,
                                 'y': sub[label_col].values.astype(np.float64),
                                 'p_raw': np.asarray(preds, dtype=np.float64)})

        rows.append(run_part(test_df))
        print(f"  fold {fold}: n={len(rows[-1])}  raw R²={r2_score(rows[-1].y, rows[-1].p_raw):.4f}")

        if args.with_val:
            val_rows.append(run_part(train_val_df[val_mask]))
            print(f"           val n={len(val_rows[-1])}")

    out = pd.concat(rows, ignore_index=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    out.to_csv(args.output, index=False, encoding='utf-8-sig')

    y, p = out['y'].values, out['p_raw'].values
    r2_pool = 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    print(f"\nOOF rows: {len(out)}\npooled raw R² = {r2_pool:.4f}")
    if args.expect_r2 is not None:
        print(f"self-check vs {args.expect_r2}: "
              f"{'PASS' if abs(r2_pool - args.expect_r2) < 1e-3 else 'FAIL'}")
    print(f"saved: {args.output}")

    if args.with_val and val_rows:
        val_out = os.path.splitext(args.output)[0] + '_val.csv'
        pd.concat(val_rows, ignore_index=True).to_csv(val_out, index=False, encoding='utf-8-sig')
        print(f"saved val: {val_out}")


if __name__ == '__main__':
    main()
