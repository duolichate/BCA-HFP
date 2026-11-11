#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GraphDRP training aligned with the main model's fixed var3 5-fold split (fold_var3_fromckpt.csv).

split: read the fold CSV and map samples to folds by drug name (GROUP), same protocol as
train_cv.py --fold_csv in the main model.
evaluation: winsorize each fold's test predictions by concentration window, pool, then report
R²/RMSE/MAE/Pearson/Spearman.
Output: Result/GraphDRP/CV/{dataset}_{model}/
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
import torch
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from bca_hfp.config.base_config import get_base_config, root_dir, detect_env
from bca_hfp.data.loader import load_drug_atom_features, load_gene_embeddings, load_response
from bca_hfp.training.train_holdout import (load_fixed_fold_splits, build_censor_bounds,
                                            winsorize_pred, pooled_metrics,
                                            fit_gene_scaler, apply_gene_scaler)
from bca_hfp.utils.reproducibility import set_seed

from benchmarks.gratransdrp.gratransdrp_model import smile_to_graph

from benchmarks.graphdrp.graphdrp.models.gcn import GCNNet
from benchmarks.graphdrp.graphdrp.models.gat import GATNet
from benchmarks.graphdrp.graphdrp.models.gat_gcn import GAT_GCN
from benchmarks.graphdrp.graphdrp.models.ginconv import GINConvNet
from torch_geometric.data import Data, Batch


class GraphDRPSimpleDataset(torch.utils.data.Dataset):
    def __init__(self, drug_ids, cell_ids, labels, drug_smile_dict, gene_emb_dict,
                 gene_scaler=None, ablate=''):
        self.samples = []
        self.smile_graph_cache = {}
        self.row_idx = []  # index into the passed df, used to align the per-row censor window
        skipped = 0

        for pos, (did, cid, lab) in enumerate(zip(drug_ids, cell_ids, labels)):
            if cid not in gene_emb_dict:
                skipped += 1
                continue
            if did not in drug_smile_dict:
                skipped += 1
                continue

            smile = drug_smile_dict[did]
            if smile not in self.smile_graph_cache:
                c_size, features, edge_index = smile_to_graph(smile)
                if c_size == 0 or len(features) == 0:
                    skipped += 1
                    continue

                x_tensor = torch.tensor(features, dtype=torch.float32)
                if x_tensor.dim() == 3 and x_tensor.size(1) == 1:
                    x_tensor = x_tensor.squeeze(1)
                if ablate == 'drug':
                    x_tensor = torch.zeros_like(x_tensor)   # ablation: zero out atom features

                graph = Data(
                    x=x_tensor,
                    edge_index=edge_index,
                    num_nodes=c_size
                )
                self.smile_graph_cache[smile] = graph
            else:
                graph = self.smile_graph_cache[smile]

            raw_gene_emb = gene_emb_dict[cid]
            gene_pooled = raw_gene_emb.mean(axis=0)
            if gene_scaler is not None:
                gene_pooled = apply_gene_scaler(gene_pooled, *gene_scaler)
            if ablate == 'gene':
                gene_pooled = np.zeros_like(gene_pooled)
            self.samples.append((graph, gene_pooled, lab))
            self.row_idx.append(pos)

        print(f"GraphDRP valid samples: {len(self.samples)} (skipped {skipped})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        graph, gene_pooled, label = self.samples[idx]
        g = graph.clone()
        g.target = torch.tensor(gene_pooled, dtype=torch.float32)
        return g, torch.tensor(label, dtype=torch.float32)


def gra_collate_fn(batch):
    graphs = []
    labels = []
    targets = []

    for graph, label in batch:
        graphs.append(graph)
        targets.append(graph.target)
        labels.append(label)

    batch_graph = Batch.from_data_list(graphs)
    batch_graph.target = torch.stack(targets, dim=0)

    labels = torch.tensor(labels, dtype=torch.float32)
    return batch_graph, labels


def _predict(model, loader, device):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for graphs, labels in loader:
            graphs = graphs.to(device)
            labels = labels.to(device)
            output, _ = model(graphs)
            preds.extend(output.cpu().numpy().flatten())
            trues.extend(labels.cpu().numpy().flatten())
    return np.array(trues, dtype=np.float64), np.array(preds, dtype=np.float64)


def main():
    parser = argparse.ArgumentParser(description='GraphDRP training (fair 5-fold)')
    parser.add_argument('--dataset', type=str, default='GDSC', choices=['GDSC', 'CCLE'])
    parser.add_argument('--gene_version', type=str, default='V1', choices=['V1', 'V2'])
    parser.add_argument('--model', type=str, default='gcn', choices=['gcn', 'gat', 'gat_gcn', 'ginconv'])
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--fold_csv', type=str, default='',
                        help='external fixed split CSV; default Data/Membership/fold_var3_fromckpt.csv')
    parser.add_argument('--val_ratio', type=float, default=0.125,
                        help='val fraction of the non-test samples per fold (matches main model --val_ratio default)')
    parser.add_argument('--pred_clamp', action='store_true',
                        help='winsorize test predictions by concentration window (same as main model --pred_clamp)')
    parser.add_argument('--only_folds', type=str, default='',
                        help='run only these folds (comma separated, 1-based)')
    parser.add_argument('--ablate', type=str, default='', choices=['', 'drug', 'gene'],
                        help='diagnostic ablation: drug=zero atom features, gene=zero cell vector')
    parser.add_argument('--variant', type=str, default='',
                        help='variant id (e.g. v1_base2_GraphDRP). when set the output dir is '
                             '<env>_<variant>_<timestamp> instead of bs/lr')
    parser.add_argument('--env', type=str, default='auto',
                        choices=['auto', 'RTX5090', 'TeslaT4', '7900XT', 'cpu', 'loc', 'srv'],
                        help='machine tag (default: detect from GPU name)')
    args = parser.parse_args()

    set_seed(42)

    config = get_base_config(args.dataset)
    train_conf = config['training']

    batch_size = args.batch_size if args.batch_size is not None else train_conf['batch_size']
    lr = args.lr if args.lr is not None else train_conf['lr']
    epochs = args.epochs if args.epochs is not None else train_conf['epochs']
    weight_decay = train_conf['weight_decay']
    gradient_clip = train_conf.get('gradient_clip', 0.5)
    # baselines all use patience=25: these architectures barely start by epoch 1, 10 stops at the first optimum
    early_stop_patience = 25

    print(f"Training config: batch_size={batch_size}, lr={lr}, epochs={epochs}, weight_decay={weight_decay}")

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
    drug_smile_dict = {str(row['DRUG_ID']): row['DRUG_SMILE'] for _, row in drug_smiles_df.iterrows()}

    valid_drug_mask = response_df[drug_id_col].isin(drug_atom_dict)
    valid_cell_mask = response_df[cell_id_col].isin(gene_emb_dict)
    valid_mask = valid_drug_mask & valid_cell_mask
    filtered_df = response_df[valid_mask].copy()
    print(f"Valid samples: {len(filtered_df)}")

    grp = filtered_df['DRUG_NAME'].astype(str).str.strip().str.upper()
    splits = load_fixed_fold_splits(fold_csv, grp)
    n_folds = len(splits)
    print(f"Fixed fold split: {n_folds} folds from {fold_csv}")

    b_lo, b_hi = build_censor_bounds(filtered_df, label_col)

    only_folds = ({int(x) for x in args.only_folds.split(',') if x.strip().isdigit()}
                  if args.only_folds else None)

    model_map = {
        'gcn': GCNNet,
        'gat': GATNet,
        'gat_gcn': GAT_GCN,
        'ginconv': GINConvNet
    }
    model_cls = model_map[args.model]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    fold_results = []
    oof_true, oof_pred, oof_lo, oof_hi = [], [], [], []

    # dir carries lr/batch so sweep runs do not overwrite each other
    # dir name must match tune_baselines.dir_label() (no model segment) or the sweep cannot read cv_folds.csv
    if args.variant:
        # final results: <env>_<variant>_<timestamp>, same naming as train_cv.py in the main model
        output_dir = os.path.join(root_dir("Result"), "2_baselines", "GraphDRP", "CV",
                                  f"{detect_env(args.env)}_{args.variant}_"
                                  f"{time.strftime('%Y%m%d_%H%M')}")
    else:
        output_dir = os.path.join(root_dir("Result"), "2_baselines", "GraphDRP", "CV",
                                  f"{args.dataset}_{args.gene_version}"
                                  f"_bs{batch_size}_lr{lr:g}")
        if args.pred_clamp:
            output_dir = f"{output_dir}_clamp"
        if args.ablate:
            output_dir = f"{output_dir}_ab{args.ablate}"
    os.makedirs(output_dir, exist_ok=True)

    for fold_idx, (train_val_idx, test_idx) in enumerate(splits):
        if only_folds is not None and (fold_idx + 1) not in only_folds:
            continue
        print(f"\n{'=' * 60}")
        print(f"Fold {fold_idx + 1}/{n_folds}")
        print(f"{'=' * 60}")

        train_val_df = filtered_df.iloc[train_val_idx]
        tv_grp = train_val_df['DRUG_NAME'].astype(str).str.strip().str.upper().values
        unique_grp = np.unique(tv_grp)
        np.random.seed(42)
        val_grp = set(np.random.choice(unique_grp, size=max(1, int(len(unique_grp) * args.val_ratio)),
                                       replace=False))
        val_mask = train_val_df['DRUG_NAME'].astype(str).str.strip().str.upper().isin(val_grp).values
        train_df = train_val_df[~val_mask]
        val_df = train_val_df[val_mask]
        test_df = filtered_df.iloc[test_idx]
        print(f"  train {len(train_df)} / val {len(val_df)} / test {len(test_df)} "
              f"({test_df['DRUG_NAME'].nunique()} test drugs)")

        # standardization stats are fit on the training-fold cells only, val/test are just transformed
        tr_cells = sorted(set(train_df[cell_id_col].values))
        scaler = fit_gene_scaler([gene_emb_dict[c].mean(axis=0)
                                  for c in tr_cells if c in gene_emb_dict])

        train_dataset = GraphDRPSimpleDataset(
            train_df[drug_id_col].tolist(), train_df[cell_id_col].tolist(),
            train_df[label_col].values, drug_smile_dict, gene_emb_dict,
            gene_scaler=scaler, ablate=args.ablate)
        val_dataset = GraphDRPSimpleDataset(
            val_df[drug_id_col].tolist(), val_df[cell_id_col].tolist(),
            val_df[label_col].values, drug_smile_dict, gene_emb_dict,
            gene_scaler=scaler, ablate=args.ablate)
        test_dataset = GraphDRPSimpleDataset(
            test_df[drug_id_col].tolist(), test_df[cell_id_col].tolist(),
            test_df[label_col].values, drug_smile_dict, gene_emb_dict,
            gene_scaler=scaler, ablate=args.ablate)

        # model has BatchNorm: a trailing batch of 1 row raises "Expected more than 1 value per channel"
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  collate_fn=gra_collate_fn, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                collate_fn=gra_collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                 collate_fn=gra_collate_fn)

        model = model_cls(
            num_features_xd=78,
            num_features_xt=config['data']['gene_input_dim'],
            output_dim=128,
            dropout=0.2
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5, verbose=True)
        criterion = torch.nn.MSELoss()

        best_val_r2 = -float('inf')
        best_state = None
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            model.train()
            train_preds, train_trues = [], []
            train_loss = 0.0
            for graphs, labels in train_loader:
                graphs = graphs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                output, _ = model(graphs)
                loss = criterion(output, labels.view(-1, 1))
                loss.backward()
                if gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                optimizer.step()
                train_loss += loss.item() * len(graphs)
                train_preds.extend(output.cpu().detach().numpy().flatten())
                train_trues.extend(labels.cpu().numpy().flatten())

            val_trues, val_preds = _predict(model, val_loader, device)
            val_r2 = r2_score(val_trues, val_preds)
            val_pearson = pearsonr(val_trues, val_preds)[0] if len(val_trues) > 1 else 0.0

            scheduler.step(val_r2)

            if val_r2 > best_val_r2:
                best_val_r2 = val_r2
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
                print(f"  ★ New best val R²: {best_val_r2:.4f}")
            else:
                patience_counter += 1

            print(f"Epoch {epoch:3d} | Val R²={val_r2:.4f} | Val Pearson={val_pearson:.4f}")

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
        # align censor windows via dataset.row_idx (when the fold dropped some samples)
        row_idx = np.array(test_dataset.row_idx, dtype=np.int64)
        oof_true.extend(test_trues.tolist())
        oof_pred.extend(test_preds.tolist())
        oof_lo.extend(b_lo[test_idx][row_idx].tolist())
        oof_hi.extend(b_hi[test_idx][row_idx].tolist())

        if best_state:
            torch.save(best_state, os.path.join(output_dir, f'best_model_fold{fold_idx + 1}.pth'))

    oof_true = np.array(oof_true, dtype=np.float64)
    oof_pred = np.array(oof_pred, dtype=np.float64)
    oof_lo = np.array(oof_lo, dtype=np.float64)
    oof_hi = np.array(oof_hi, dtype=np.float64)

    raw_metrics = pooled_metrics(oof_true, oof_pred)
    clamp_metrics = pooled_metrics(oof_true, winsorize_pred(oof_pred, oof_lo, oof_hi))

    pd.DataFrame(fold_results).to_csv(os.path.join(output_dir, 'cv_folds.csv'), index=False)
    pd.DataFrame([{
        'model': args.model,
        'gene_version': args.gene_version,
        'n_total': int(len(oof_true)),
        'raw_r2': raw_metrics['r2'], 'raw_rmse': raw_metrics['rmse'],
        'raw_pearson': raw_metrics['pearson'], 'raw_spearman': raw_metrics['spearman'],
        'clamp_r2': clamp_metrics['r2'], 'clamp_rmse': clamp_metrics['rmse'],
        'clamp_pearson': clamp_metrics['pearson'], 'clamp_spearman': clamp_metrics['spearman'],
    }]).to_csv(os.path.join(output_dir, 'final_performance.csv'), index=False)

    print("\n" + "=" * 60)
    print(f"GraphDRP ({args.model}) pooled results, n={int(len(oof_true))}")
    print(f"  RAW    : R²={raw_metrics['r2']:.4f}  RMSE={raw_metrics['rmse']:.4f}  "
          f"Pearson={raw_metrics['pearson']:.4f}  Spearman={raw_metrics['spearman']:.4f}")
    print(f"  CLAMP  : R²={clamp_metrics['r2']:.4f}  RMSE={clamp_metrics['rmse']:.4f}  "
          f"Pearson={clamp_metrics['pearson']:.4f}  Spearman={clamp_metrics['spearman']:.4f}")
    print("=" * 60)


if __name__ == '__main__':
    main()