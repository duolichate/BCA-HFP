#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tCNNS training (Liu et al., BMC Bioinformatics 2019, 20:408) on the main model's fixed var3 5-fold split.

architecture (faithful to the paper):
- twin CNN: drug SMILES one-hot (charset x max_len) -> 3 x Conv1d(7)+ReLU+MaxPool(3) [40/80/60]
            cell gene expression vector -> 3 x Conv1d(7)+ReLU+MaxPool(3) [40/80/60]
- flatten both branches, concat -> FC(1024) -> FC(1024) -> FC(1)
deviations (required by the unified protocol; see Supplementary Material):
- cell input uses the mean GeneFormer expression embedding (paper uses a 735-dim mutation profile; this dataset has no mutations);
- output regresses raw LN_IC50 directly (paper uses sigmoid + normalized IC50, so numbers are not directly comparable to the main model);
- the split is read from fold_var3_fromckpt.csv, each fold's test predictions are winsorized by concentration window and pooled for metrics.

Output: Result/tCNNS/CV/{dataset}_{gene_version}/
"""

import sys as _sys
# windows console defaults to GBK, printing "R²" (U+00B2) raises UnicodeEncodeError and kills the run;
# switch stdout/stderr to utf-8 (unencodable chars replaced), with or without a console.
if hasattr(_sys.stdout, "reconfigure"):
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import sys
import os
import time
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from bca_hfp.config.base_config import get_base_config, root_dir, detect_env
from bca_hfp.data.loader import load_drug_atom_features, load_gene_embeddings, load_response
from bca_hfp.training.train_holdout import (load_fixed_fold_splits, build_censor_bounds,
                                            winsorize_pred, pooled_metrics,
                                            fit_gene_scaler, apply_gene_scaler)
from bca_hfp.utils.reproducibility import set_seed


# ---------------------------------------------------------------- SMILES encoding
def string2smiles_list(string):
    """same tokenization as data/preprocess.py in the paper: a lowercase char merges with the previous char (Cl / Br / nH in [nH] etc.)."""
    char_list = []
    i = 1
    while i < len(string):
        c = string[i]
        if c.islower():
            char_list.append(string[i - 1:i + 1])
            i += 1
        else:
            char_list.append(string[i - 1])
        i += 1
    if string and not string[-1].islower():
        char_list.append(string[-1])
    return char_list


def build_smiles_vocab(smiles_list):
    """build the char table from all SMILES in the data (sorted for reproducibility) plus the max length."""
    chars = set()
    max_len = 1
    for s in smiles_list:
        toks = string2smiles_list(s)
        chars.update(toks)
        max_len = max(max_len, len(toks))
    return sorted(chars), max_len


def smiles_to_onehot(smiles, char2idx, seq_len):
    """charset x seq_len one-hot matrix (same as onehot_encode in the paper, zero-padded on the right)."""
    toks = string2smiles_list(smiles)[:seq_len]
    mat = np.zeros((len(char2idx), seq_len), dtype=np.float32)
    for pos, tok in enumerate(toks):
        idx = char2idx.get(tok)
        if idx is not None:
            mat[idx, pos] = 1.0
    return mat


# ---------------------------------------------------------------- model
class tCNNS(nn.Module):
    """twin CNN: drug SMILES one-hot and cell expression vector each run through 3 Conv1d layers."""

    def __init__(self, n_chars, drug_len, cell_dim, dropout=0.5):
        super().__init__()
        # padding=3 + ceil_mode=True is equivalent to 'SAME' conv/pooling in the original TF code
        self.drug_conv = nn.Sequential(
            nn.Conv1d(n_chars, 40, 7, padding=3), nn.ReLU(), nn.MaxPool1d(3, ceil_mode=True),
            nn.Conv1d(40, 80, 7, padding=3), nn.ReLU(), nn.MaxPool1d(3, ceil_mode=True),
            nn.Conv1d(80, 60, 7, padding=3), nn.ReLU(), nn.MaxPool1d(3, ceil_mode=True),
        )
        self.cell_conv = nn.Sequential(
            nn.Conv1d(1, 40, 7, padding=3), nn.ReLU(), nn.MaxPool1d(3, ceil_mode=True),
            nn.Conv1d(40, 80, 7, padding=3), nn.ReLU(), nn.MaxPool1d(3, ceil_mode=True),
            nn.Conv1d(80, 60, 7, padding=3), nn.ReLU(), nn.MaxPool1d(3, ceil_mode=True),
        )
        d_flat = self._flat_dim(self.drug_conv, n_chars, drug_len)
        c_flat = self._flat_dim(self.cell_conv, 1, cell_dim)

        self.fc1 = nn.Linear(d_flat + c_flat, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 1)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()

    @staticmethod
    def _flat_dim(conv, in_ch, length):
        with torch.no_grad():
            out = conv(torch.zeros(1, in_ch, length))
        return int(out.numel())

    def forward(self, drug, cell):
        d = self.drug_conv(drug).flatten(1)
        c = self.cell_conv(cell.unsqueeze(1)).flatten(1)
        h = self.act(self.fc1(torch.cat([d, c], dim=1)))
        h = self.dropout(h)
        h = self.dropout(self.act(self.fc2(h)))
        return self.fc3(h).squeeze(-1)


# ---------------------------------------------------------------- train/eval
def _predict(model, loader, device):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for drug, cell, labels in loader:
            out = model(drug.to(device), cell.to(device))
            preds.extend(out.cpu().numpy().flatten().tolist())
            trues.extend(labels.numpy().flatten().tolist())
    return np.asarray(trues, dtype=np.float64), np.asarray(preds, dtype=np.float64)


def main():
    parser = argparse.ArgumentParser(description='tCNNS training (fair 5-fold)')
    parser.add_argument('--dataset', type=str, default='GDSC', choices=['GDSC', 'CCLE'])
    parser.add_argument('--gene_version', type=str, default='V1', choices=['V1', 'V2'])
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--fold_csv', type=str, default='',
                        help='external fixed split CSV; default Data/Membership/fold_var3_fromckpt.csv')
    parser.add_argument('--val_ratio', type=float, default=0.125,
                        help='val fraction of the non-test samples per fold (matches main model --val_ratio default)')
    parser.add_argument('--pred_clamp', action='store_true',
                        help='append _clamp to the output dir (both raw and clamped metrics are written)')
    parser.add_argument('--only_folds', type=str, default='',
                        help='run only these folds (comma separated, 1-based)')
    parser.add_argument('--ablate', type=str, default='', choices=['', 'drug', 'gene'],
                        help='diagnostic ablation: drug=zero drug input, gene=zero cell vector')
    parser.add_argument('--variant', type=str, default='',
                        help='variant id (e.g. v1_base6_tCNNS). when set the output dir is '
                             '<env>_<variant>_<timestamp> instead of bs/lr')
    parser.add_argument('--env', type=str, default='auto',
                        choices=['auto', 'RTX5090', 'TeslaT4', '7900XT', 'cpu', 'loc', 'srv'],
                        help='machine tag (default: detect from GPU name)')
    args = parser.parse_args()

    set_seed(42)
    config = get_base_config(args.dataset)
    train_conf = config['training']

    batch_size = args.batch_size or train_conf['batch_size']
    lr = args.lr or train_conf['lr']
    epochs = args.epochs or train_conf['epochs']
    weight_decay = train_conf['weight_decay']
    gradient_clip = train_conf.get('gradient_clip', 0.5)
    # baselines all use patience=25: these architectures barely start by epoch 1, 10 stops at the first optimum
    early_stop_patience = 25

    base_data_path = root_dir("Data")
    if args.gene_version == 'V1':
        config['data']['gene_feat_dir'] = f"{base_data_path}/Gene/GDSC_EXP_GF_V1"
        config['data']['gene_input_dim'] = 256
    else:
        config['data']['gene_feat_dir'] = f"{base_data_path}/Gene/GDSC_EXP_GF_V2"
        config['data']['gene_input_dim'] = 768

    fold_csv = args.fold_csv or os.path.join(base_data_path, "Membership", "fold_var3_fromckpt.csv")

    drug_id_col = config['data']['drug_id_col']
    cell_id_col = config['data']['cell_id_col']
    label_col = config['data']['label_col']

    print("Loading data...")
    drug_atom_dict = load_drug_atom_features(config['data']['drug_feat_path'])
    gene_emb_dict, _ = load_gene_embeddings(config['data']['gene_feat_dir'], cell_id_col)
    response_df = load_response(config['data']['response_path'], cell_id_col, drug_id_col)

    with pd.HDFStore(config['data']['drug_feat_path'], 'r') as store:
        drug_smiles_df = store['drug_smiles']
    drug_smile_dict = {str(row['DRUG_ID']): row['DRUG_SMILE']
                       for _, row in drug_smiles_df.iterrows()}

    valid_mask = (response_df[drug_id_col].isin(drug_atom_dict)
                  & response_df[cell_id_col].isin(gene_emb_dict))
    filtered_df = response_df[valid_mask].copy()
    print(f"Valid samples: {len(filtered_df)}")

    # SMILES char table is built on the whole dataset (drug strings only, no label information)
    all_smiles = [drug_smile_dict[d] for d in filtered_df[drug_id_col].unique()
                  if d in drug_smile_dict]
    chars, seq_len = build_smiles_vocab(all_smiles)
    char2idx = {c: i for i, c in enumerate(chars)}
    print(f"SMILES vocab: {len(chars)} chars, max_len={seq_len}")

    drug_onehot = {d: smiles_to_onehot(drug_smile_dict[d], char2idx, seq_len)
                   for d in filtered_df[drug_id_col].unique() if d in drug_smile_dict}

    # cell expression vectors pooled up front (941 cell lines), z-scored per fold
    pooled_gene = {cid: emb.mean(axis=0) for cid, emb in gene_emb_dict.items()}

    grp = filtered_df['DRUG_NAME'].astype(str).str.strip().str.upper()
    splits = load_fixed_fold_splits(fold_csv, grp)
    n_folds = len(splits)
    print(f"Fixed fold split: {n_folds} folds from {fold_csv}")

    b_lo, b_hi = build_censor_bounds(filtered_df, label_col)
    only_folds = ({int(x) for x in args.only_folds.split(',') if x.strip().isdigit()}
                  if args.only_folds else None)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # dir carries lr/batch so sweep runs do not overwrite each other
    if args.variant:
        # final results: <env>_<variant>_<timestamp>, same naming as train_cv.py in the main model
        output_dir = os.path.join(root_dir("Result"), "2_baselines", "tCNNS", "CV",
                                  f"{detect_env(args.env)}_{args.variant}_"
                                  f"{time.strftime('%Y%m%d_%H%M')}")
    else:
        output_dir = os.path.join(root_dir("Result"), "2_baselines", "tCNNS", "CV",
                                  f"{args.dataset}_{args.gene_version}"
                                  f"_bs{batch_size}_lr{lr:g}")
        if args.pred_clamp:
            output_dir = f"{output_dir}_clamp"
        if args.ablate:
            output_dir = f"{output_dir}_ab{args.ablate}"
    os.makedirs(output_dir, exist_ok=True)

    fold_results = []
    oof_true, oof_pred, oof_lo, oof_hi = [], [], [], []

    def make_dataset(df, g_mean, g_std):
        drugs, cells, labs, rows = [], [], [], []
        for pos, (did, cid, lab) in enumerate(zip(df[drug_id_col].values,
                                                 df[cell_id_col].values,
                                                 df[label_col].values)):
            if did not in drug_onehot or cid not in pooled_gene:
                continue
            if args.ablate == 'drug':
                drugs.append(np.zeros_like(drug_onehot[did]))
            else:
                drugs.append(drug_onehot[did])
            if args.ablate == 'gene':
                cells.append(np.zeros(config['data']['gene_input_dim'], dtype=np.float32))
            else:
                cells.append(apply_gene_scaler(pooled_gene[cid], g_mean, g_std))
            labs.append(lab)
            rows.append(pos)
        ds = TensorDataset(torch.tensor(np.stack(drugs), dtype=torch.float32),
                           torch.tensor(np.stack(cells), dtype=torch.float32),
                           torch.tensor(np.asarray(labs), dtype=torch.float32))
        return ds, np.asarray(rows, dtype=np.int64)

    for fold_idx, (train_val_idx, test_idx) in enumerate(splits):
        if only_folds is not None and (fold_idx + 1) not in only_folds:
            continue
        print(f"\n{'=' * 70}\nFold {fold_idx + 1}/{n_folds}\n{'=' * 70}")

        train_val_df = filtered_df.iloc[train_val_idx]
        tv_grp = train_val_df['DRUG_NAME'].astype(str).str.strip().str.upper().values
        unique_grp = np.unique(tv_grp)
        np.random.seed(42)
        val_grp = set(np.random.choice(unique_grp,
                                       size=max(1, int(len(unique_grp) * args.val_ratio)),
                                       replace=False))
        val_mask = train_val_df['DRUG_NAME'].astype(str).str.strip().str.upper().isin(val_grp).values
        train_df, val_df = train_val_df[~val_mask], train_val_df[val_mask]
        test_df = filtered_df.iloc[test_idx]
        print(f"  train {len(train_df)} / val {len(val_df)} / test {len(test_df)} "
              f"({test_df['DRUG_NAME'].nunique()} test drugs)")

        # standardization stats are fit on the training-fold cells only, val/test are just transformed
        tr_cells = sorted(set(train_df[cell_id_col].values))
        g_mean, g_std = fit_gene_scaler([pooled_gene[c] for c in tr_cells if c in pooled_gene])

        train_ds, _ = make_dataset(train_df, g_mean, g_std)
        val_ds, _ = make_dataset(val_df, g_mean, g_std)
        test_ds, test_rows = make_dataset(test_df, g_mean, g_std)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        model = tCNNS(len(chars), seq_len, config['data']['gene_input_dim'],
                      dropout=config['model']['dropout']).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                               factor=0.5, patience=5)
        criterion = nn.MSELoss()

        best_val_r2, best_state, patience_counter = -float('inf'), None, 0

        for epoch in range(1, epochs + 1):
            model.train()
            loss_sum = 0.0
            for drug, cell, labels in train_loader:
                drug, cell, labels = drug.to(device), cell.to(device), labels.to(device)
                optimizer.zero_grad()
                loss = criterion(model(drug, cell), labels)
                loss.backward()
                if gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                optimizer.step()
                loss_sum += loss.item() * len(labels)

            val_trues, val_preds = _predict(model, val_loader, device)
            val_r2 = r2_score(val_trues, val_preds) if len(val_trues) > 1 else 0.0
            scheduler.step(val_r2)

            if val_r2 > best_val_r2:
                best_val_r2 = val_r2
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            print(f"Epoch {epoch:3d} | Train MSE {loss_sum / len(train_ds):.4f} | "
                  f"Val R² {val_r2:.4f}{' | * best' if patience_counter == 0 else ''}")

            if patience_counter >= early_stop_patience:
                print(f"Early stopping at epoch {epoch}")
                break

        if best_state:
            model.load_state_dict(best_state)

        test_trues, test_preds = _predict(model, test_loader, device)
        fold_results.append({
            'fold': fold_idx + 1,
            'n_test': int(len(test_trues)),
            'best_val_r2': best_val_r2,
            'raw_r2': float(r2_score(test_trues, test_preds)),
        })
        oof_true.extend(test_trues.tolist())
        oof_pred.extend(test_preds.tolist())
        oof_lo.extend(b_lo[test_idx][test_rows].tolist())
        oof_hi.extend(b_hi[test_idx][test_rows].tolist())

        if best_state:
            torch.save(best_state, os.path.join(output_dir, f'best_model_fold{fold_idx + 1}.pth'))

    oof_true = np.asarray(oof_true, dtype=np.float64)
    oof_pred = np.asarray(oof_pred, dtype=np.float64)
    oof_lo = np.asarray(oof_lo, dtype=np.float64)
    oof_hi = np.asarray(oof_hi, dtype=np.float64)

    raw_metrics = pooled_metrics(oof_true, oof_pred)
    clamp_metrics = pooled_metrics(oof_true, winsorize_pred(oof_pred, oof_lo, oof_hi))

    pd.DataFrame(fold_results).to_csv(os.path.join(output_dir, 'cv_folds.csv'), index=False)
    pd.DataFrame([{
        'model': 'tCNNS',
        'gene_version': args.gene_version,
        'n_total': int(len(oof_true)),
        'raw_r2': raw_metrics['r2'], 'raw_rmse': raw_metrics['rmse'],
        'raw_mae': raw_metrics['mae'],
        'raw_pearson': raw_metrics['pearson'], 'raw_spearman': raw_metrics['spearman'],
        'clamp_r2': clamp_metrics['r2'], 'clamp_rmse': clamp_metrics['rmse'],
        'clamp_mae': clamp_metrics['mae'],
        'clamp_pearson': clamp_metrics['pearson'], 'clamp_spearman': clamp_metrics['spearman'],
    }]).to_csv(os.path.join(output_dir, 'final_performance.csv'), index=False)

    print("\n" + "=" * 70)
    print(f"tCNNS pooled results, n={int(len(oof_true))}")
    print(f"  RAW   : R²={raw_metrics['r2']:.4f}  RMSE={raw_metrics['rmse']:.4f}  "
          f"Pearson={raw_metrics['pearson']:.4f}  Spearman={raw_metrics['spearman']:.4f}")
    print(f"  CLAMP : R²={clamp_metrics['r2']:.4f}  RMSE={clamp_metrics['rmse']:.4f}  "
          f"Pearson={clamp_metrics['pearson']:.4f}  Spearman={clamp_metrics['spearman']:.4f}")
    print("=" * 70)


if __name__ == '__main__':
    main()
