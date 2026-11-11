# bca_hfp/analysis/predict_cv_fold.py
"""predict val / test drugs of one CV fold and write a per-sample CSV.

why: train_cv.py only saves test predictions, but diagnosing a weak val fold needs val predictions.
model structure is restored from the config inside the checkpoint (no flags to repeat), the same
GroupKFold + val_ratio split as train_cv is reproduced, forward pass only, no training.

output CSV has DRUG_ID / GDSC_ID / true_label / predicted, can be fed straight to diag_per_drug.py.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from bca_hfp.data.loader import (load_drug_atom_features, load_gene_embeddings,
                                 load_response, load_morgan_fingerprints)
from bca_hfp.data.dataset import DrugGeneDataset, collate_fn
from bca_hfp.models import create_model
from bca_hfp.utils.reproducibility import set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_path', required=True, help='.pth of one fold (contains config)')
    ap.add_argument('--fold', type=int, required=True, help='1-based fold index')
    ap.add_argument('--which', type=str, default='val', choices=['val', 'test'])
    ap.add_argument('--n_splits', type=int, default=5)
    ap.add_argument('--val_ratio', type=float, default=0.125,
                    help='must match training (val fraction cut from the train+val part)')
    ap.add_argument('--split_seed', type=int, default=42)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--output_csv', required=True)
    args = ap.parse_args()

    set_seed(0)
    ckpt = torch.load(args.model_path, map_location='cpu', weights_only=False)
    config = ckpt['config']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_cfg = config['data']
    drug_col = data_cfg['drug_id_col']
    cell_col = data_cfg['cell_id_col']
    label_col = data_cfg['label_col']
    h5_path = data_cfg['drug_feat_path']
    use_desc = int(data_cfg['drug_global_dim']) > 2048

    print(f'config: query_source={config["model"].get("query_source")} '
          f'add_atom_modality={config["model"].get("add_atom_modality")} '
          f'drop_drug_global={config["model"].get("drop_drug_global_modality")} '
          f'decouple={config["model"].get("decouple_atom")} '
          f'use_desc={use_desc} reg_norm={config["model"].get("reg_norm")}')

    drug_atom_dict = load_drug_atom_features(h5_path)
    gene_emb_dict, _ = load_gene_embeddings(data_cfg['gene_feat_dir'], cell_col)
    response_df = load_response(h5_path, cell_col, drug_col)
    drug_global_dict = load_morgan_fingerprints(h5_path, fp_bits=2048, fp_radius=2,
                                                use_desc=use_desc)

    valid = (response_df[drug_col].isin(drug_atom_dict)
             & response_df[cell_col].isin(gene_emb_dict))
    filtered_df = response_df[valid].copy()

    # ---- reproduce the train_cv split ----
    gkf = GroupKFold(n_splits=args.n_splits)
    splits = list(gkf.split(filtered_df, groups=filtered_df[drug_col].values))
    train_val_idx, test_idx = splits[args.fold - 1]
    train_val_df = filtered_df.iloc[train_val_idx].copy()

    if args.which == 'test':
        eval_df = filtered_df.iloc[test_idx].copy()
    else:
        unique_drugs = train_val_df[drug_col].unique()
        np.random.seed(args.split_seed)
        val_drugs = np.random.choice(
            unique_drugs, size=int(len(unique_drugs) * args.val_ratio), replace=False)
        eval_df = train_val_df[train_val_df[drug_col].isin(val_drugs)].copy()

    print(f'fold{args.fold} {args.which}: {len(eval_df)} samples, '
          f'{eval_df[drug_col].nunique()} drugs')

    dataset = DrugGeneDataset(
        eval_df[drug_col].tolist(), eval_df[cell_col].tolist(),
        eval_df[label_col].values, drug_atom_dict, gene_emb_dict, drug_global_dict)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=args.num_workers)

    model = create_model('attention', config).to(device)
    model.load_state_dict(ckpt['model_state_dict'], strict=True)
    model.eval()

    preds = []
    with torch.no_grad():
        for batch in loader:
            atom_feat, atom_mask, gene_feat, gene_mask, labels, drug_global = batch
            out = model(gene_feat.to(device), atom_feat.to(device), atom_mask.to(device),
                        gene_mask.to(device), drug_global=drug_global.to(device))
            preds.extend(out.cpu().float().numpy())

    preds = np.asarray(preds, dtype=np.float32)
    assert len(preds) == len(eval_df), (len(preds), len(eval_df))
    df_out = pd.DataFrame({
        'DRUG_ID': eval_df[drug_col].values,
        'GDSC_ID': eval_df[cell_col].values,
        'true_label': eval_df[label_col].values,
        'predicted': preds,
    })
    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    df_out.to_csv(args.output_csv, index=False)

    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    from scipy.stats import pearsonr
    y = df_out['true_label'].values
    p = df_out['predicted'].values
    print(f'[overall] R2={r2_score(y, p):+.4f}  Pearson={pearsonr(y, p)[0]:.4f}  '
          f'MAE={mean_absolute_error(y, p):.4f}  RMSE={np.sqrt(mean_squared_error(y, p)):.4f}  '
          f'sd(true)={y.std():.3f}  sd(pred)={p.std():.3f}  shrink={p.std() / y.std():.3f}')
    print(f'saved -> {args.output_csv}')
    print('next: python bca_hfp/analysis/pool_oof_stats.py '
          f'--pred_csv {args.output_csv} --output_dir <diag output dir>')


if __name__ == '__main__':
    main()
