#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TransCDR training (Xia et al., BMC Biology 2024, 22:227) on the main model's fixed var3 5-fold split.

architecture (faithful to model_helper.GCN_encoder in the repo and to the paper):
- drug encoder : 3 GCN layers (hidden 64) + WeightedSumAndMax readout -> Linear(256)
                 (the official code uses dgllife; this reimplements the same structure and
                  readout with torch_geometric, shares the graph features with
                  GraphDRP/GraTransDRP and adds no dgllife dependency)
- cell encoder : mean GeneFormer expression embedding -> MLP(512) -> 256
- fusion       : concat[256, 256] -> Dense(256) -> ReLU -> Dense(1)
deviations (required by the unified protocol; see Supplementary Material):
- cell input uses the mean expression embedding and does not read the RNA-seq/mutation/methylation
  tri-modal CSVs (this dataset has no mutation or methylation data);
- output regresses raw LN_IC50 directly (paper uses sigmoid + normalization, so numbers are not directly comparable to the main model);
- no pretraining/transfer and no SMILES augmentation.
split: read fold_var3_fromckpt.csv and map samples to folds by drug name;
evaluation: winsorize each fold's test predictions by concentration window, pool, then report
R²/RMSE/MAE/Pearson/Spearman.
Output: Result/TransCDR/CV/{dataset}_{gene_version}/
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
from torch.utils.data import DataLoader
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_max_pool
from torch_geometric.utils import softmax as graph_softmax
from sklearn.metrics import r2_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from bca_hfp.config.base_config import get_base_config, root_dir, detect_env
from bca_hfp.data.loader import load_drug_atom_features, load_gene_embeddings, load_response
from bca_hfp.training.train_holdout import (load_fixed_fold_splits, build_censor_bounds,
                                            winsorize_pred, pooled_metrics,
                                            fit_gene_scaler, apply_gene_scaler)
from bca_hfp.utils.reproducibility import set_seed

from benchmarks.gratransdrp.gratransdrp_model import smile_to_graph


class WeightedSumAndMax(nn.Module):
    """equivalent of dgllife WeightedSumAndMax: graph-level weighted sum + per-graph max pooling."""

    def __init__(self, in_feats):
        super().__init__()
        self.gate = nn.Linear(in_feats, 1)

    def forward(self, node_feats, batch):
        w = graph_softmax(self.gate(node_feats), batch, dim=0)         # softmax within each graph
        weighted = torch.zeros(batch.max().item() + 1, node_feats.size(1),
                               dtype=node_feats.dtype, device=node_feats.device)
        weighted.index_add_(0, batch, w * node_feats)
        return torch.cat([weighted, global_max_pool(node_feats, batch)], dim=-1)


class DrugGCNEncoder(nn.Module):
    """3 GCN layers (64) + WeightedSumAndMax -> 256, same as dgllife GCN_encoder."""

    def __init__(self, in_feats=78, hidden=64, n_layers=3, out_dim=256, dropout=0.1):
        super().__init__()
        self.convs = nn.ModuleList(
            [GCNConv(in_feats if i == 0 else hidden, hidden) for i in range(n_layers)]
        )
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.readout = WeightedSumAndMax(hidden)
        self.transform = nn.Linear(hidden * 2, out_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv in self.convs:
            x = self.dropout(self.act(conv(x, edge_index)))
        return self.transform(self.readout(x, batch))


class GeneEncoder(nn.Module):
    """cell expression vector -> MLP -> 256 dims (replaces the paper's multi-omics concat)."""

    def __init__(self, input_dim, out_dim=256, hidden_dim=512, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.mlp(x)


class TransCDRModel(nn.Module):
    def __init__(self, gene_input_dim, dropout=0.1):
        super().__init__()
        self.drug_encoder = DrugGCNEncoder(out_dim=256, dropout=dropout)
        self.gene_encoder = GeneEncoder(gene_input_dim, out_dim=256, dropout=dropout)
        self.predictor = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(self, drug_batch, gene_feat):
        d = self.drug_encoder(drug_batch)
        g = self.gene_encoder(gene_feat)
        return self.predictor(torch.cat([d, g], dim=1)).squeeze(-1)


class TransCDRDataset(torch.utils.data.Dataset):
    def __init__(self, drug_ids, cell_ids, labels, drug_smile_dict, gene_emb_dict,
                 gene_scaler=None, ablate=''):
        self.samples = []
        self.row_idx = []          # index into the passed df, used to align the per-row censor window
        self.graph_cache = {}
        skipped = 0

        for pos, (did, cid, lab) in enumerate(zip(drug_ids, cell_ids, labels)):
            if cid not in gene_emb_dict or did not in drug_smile_dict:
                skipped += 1
                continue

            smile = drug_smile_dict[did]
            if smile not in self.graph_cache:
                c_size, features, edge_index = smile_to_graph(smile)
                if c_size == 0 or len(features) == 0:
                    self.graph_cache[smile] = None
                else:
                    x = torch.tensor(np.asarray(features), dtype=torch.float32)
                    if ablate == 'drug':
                        x = torch.zeros_like(x)      # ablation: zero atom features, keep topology only
                    self.graph_cache[smile] = Data(x=x, edge_index=edge_index,
                                                   num_nodes=c_size)
            graph = self.graph_cache[smile]
            if graph is None:
                skipped += 1
                continue

            gene_vec = gene_emb_dict[cid].mean(axis=0)
            if gene_scaler is not None:
                gene_vec = apply_gene_scaler(gene_vec, *gene_scaler)
            if ablate == 'gene':
                gene_vec = np.zeros_like(gene_vec)

            self.samples.append((graph, gene_vec, lab))
            self.row_idx.append(pos)

        print(f"TransCDR valid samples: {len(self.samples)} (skipped {skipped})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        graph, gene_pooled, label = self.samples[idx]
        return graph, torch.tensor(gene_pooled, dtype=torch.float32), \
            torch.tensor(label, dtype=torch.float32)


def transcdr_collate_fn(batch):
    graphs, gene_feats, labels = zip(*batch)
    return (Batch.from_data_list(list(graphs)),
            torch.stack(gene_feats),
            torch.tensor(labels, dtype=torch.float32))


def _predict(model, loader, device):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for drug_batch, gene_feats, labels in loader:
            out = model(drug_batch.to(device), gene_feats.to(device))
            preds.extend(out.cpu().numpy().flatten().tolist())
            trues.extend(labels.numpy().flatten().tolist())
    return np.asarray(trues, dtype=np.float64), np.asarray(preds, dtype=np.float64)


def main():
    parser = argparse.ArgumentParser(description='TransCDR training (fair 5-fold)')
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
                        help='diagnostic ablation: drug=zero atom features, gene=zero cell vector')
    parser.add_argument('--variant', type=str, default='',
                        help='variant id (e.g. v1_base4_TransCDR). when set the output dir is '
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
        output_dir = os.path.join(root_dir("Result"), "2_baselines", "TransCDR", "CV",
                                  f"{detect_env(args.env)}_{args.variant}_"
                                  f"{time.strftime('%Y%m%d_%H%M')}")
    else:
        output_dir = os.path.join(root_dir("Result"), "2_baselines", "TransCDR", "CV",
                                  f"{args.dataset}_{args.gene_version}"
                                  f"_bs{batch_size}_lr{lr:g}")
        if args.pred_clamp:
            output_dir = f"{output_dir}_clamp"
        if args.ablate:
            output_dir = f"{output_dir}_ab{args.ablate}"
    os.makedirs(output_dir, exist_ok=True)

    fold_results = []
    oof_true, oof_pred, oof_lo, oof_hi = [], [], [], []

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
        scaler = fit_gene_scaler([gene_emb_dict[c].mean(axis=0)
                                  for c in tr_cells if c in gene_emb_dict])

        train_ds = TransCDRDataset(train_df[drug_id_col].tolist(), train_df[cell_id_col].tolist(),
                                   train_df[label_col].values, drug_smile_dict, gene_emb_dict,
                                   gene_scaler=scaler, ablate=args.ablate)
        val_ds = TransCDRDataset(val_df[drug_id_col].tolist(), val_df[cell_id_col].tolist(),
                                 val_df[label_col].values, drug_smile_dict, gene_emb_dict,
                                 gene_scaler=scaler, ablate=args.ablate)
        test_ds = TransCDRDataset(test_df[drug_id_col].tolist(), test_df[cell_id_col].tolist(),
                                  test_df[label_col].values, drug_smile_dict, gene_emb_dict,
                                  gene_scaler=scaler, ablate=args.ablate)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  collate_fn=transcdr_collate_fn)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                collate_fn=transcdr_collate_fn)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                                 collate_fn=transcdr_collate_fn)

        model = TransCDRModel(config['data']['gene_input_dim'],
                              dropout=config['model']['dropout']).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                               factor=0.5, patience=5)
        criterion = nn.MSELoss()

        best_val_r2, best_state, patience_counter = -float('inf'), None, 0

        for epoch in range(1, epochs + 1):
            model.train()
            loss_sum = 0.0
            for drug_batch, gene_feats, labels in train_loader:
                drug_batch = drug_batch.to(device)
                gene_feats, labels = gene_feats.to(device), labels.to(device)
                optimizer.zero_grad()
                loss = criterion(model(drug_batch, gene_feats), labels)
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
        row_idx = np.asarray(test_ds.row_idx, dtype=np.int64)
        oof_true.extend(test_trues.tolist())
        oof_pred.extend(test_preds.tolist())
        oof_lo.extend(b_lo[test_idx][row_idx].tolist())
        oof_hi.extend(b_hi[test_idx][row_idx].tolist())

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
        'model': 'TransCDR',
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
    print(f"TransCDR pooled results, n={int(len(oof_true))}")
    print(f"  RAW   : R²={raw_metrics['r2']:.4f}  RMSE={raw_metrics['rmse']:.4f}  "
          f"Pearson={raw_metrics['pearson']:.4f}  Spearman={raw_metrics['spearman']:.4f}")
    print(f"  CLAMP : R²={clamp_metrics['r2']:.4f}  RMSE={clamp_metrics['rmse']:.4f}  "
          f"Pearson={clamp_metrics['pearson']:.4f}  Spearman={clamp_metrics['spearman']:.4f}")
    print("=" * 70)


if __name__ == '__main__':
    main()
