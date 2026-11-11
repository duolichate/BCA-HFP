#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PaccMann training (Oskooei et al., 2019) on the main model's fixed var3 5-fold split.

architecture (faithful to PaccMannV2 in paccmann_predictor / Figure 2 in the paper):
- SMILES -> token embedding (dim 16) -> 3 parallel Conv2d(k=[3,5,11], 64 filters, keeps sequence length)
- molecule stream: 4 context-attention heads each attend over the embedding and the 3 conv outputs,
                context is the cell expression vector; 4 heads per layer -> 4*16 + 4*64*3 dims
- gene stream:     cell expression vector as reference, SMILES sequence as context, 2 heads per layer,
                average_seq=False -> G dims per head
- concat -> BatchNorm -> Dense(1024) -> Dense(512) -> Dense(1)
deviations (required by the unified protocol; see Supplementary Material):
- cell input uses mean GeneFormer expression embeddings (256/768 dims) instead of RPKM over 2128 genes;
- output regresses raw LN_IC50 directly (paper uses sigmoid + min-max scaling);
- no SMILES augmentation / start-stop tokens, the split is read from fold_var3_fromckpt.csv.

Output: Result/PaccMann/CV/{dataset}_{gene_version}/
"""

import sys as _sys
# windows console defaults to GBK, printing "R²" (U+00B2) raises UnicodeEncodeError and kills the run;
# switch stdout/stderr to utf-8 (unencodable chars replaced), with or without a console.
if hasattr(_sys.stdout, "reconfigure"):
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import sys
import os
import re
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


# ---------------------------------------------------------------- SMILES tokenization
_SMILES_TOKEN = re.compile(
    r"(\[[^\]]+\]|Br|Cl|Si|Se|Na|Li|Ca|Mg|Al|@@|@|"
    r"[A-Z]|[a-z]|[0-9]|[%\-\+\.=#\(\)\/\\])"
)


def tokenize_smiles(smiles):
    """SMILES -> token list (atoms/bonds/ring-closure digits become separate tokens), same granularity as pytoda."""
    return _SMILES_TOKEN.findall(smiles)


def build_vocab(smiles_list, pad_token='<PAD>', unk_token='<UNK>'):
    tokens = set()
    max_len = 1
    for s in smiles_list:
        toks = tokenize_smiles(s)
        tokens.update(toks)
        max_len = max(max_len, len(toks))
    vocab = [pad_token, unk_token] + sorted(tokens)
    return vocab, max_len


# ---------------------------------------------------------------- model
class ContextAttention(nn.Module):
    """PaccMann context attention (PyTorch version of Figure 2C in the paper)."""

    def __init__(self, ref_dim, ref_len, ctx_dim, ctx_len, attention_size=64, temperature=1.0):
        super().__init__()
        self.ref_len, self.ctx_len = ref_len, ctx_len
        self.ref_proj = nn.Linear(ref_dim, attention_size)
        self.ctx_proj = nn.Linear(ctx_dim, attention_size)
        # when context is a sequence, project its length to the reference length first (same as the official code)
        self.ctx_len_proj = nn.Linear(ctx_len, ref_len) if ctx_len > 1 else nn.Identity()
        self.alpha = nn.Linear(attention_size, 1, bias=False)
        self.temperature = temperature

    def forward(self, ref, ctx, average_seq=True):
        """ref: [B, T, H], ctx: [B, C, G]; returns (encoding, attention weights)."""
        ref_att = self.ref_proj(ref)                                  # [B, T, A]
        ctx_att = self.ctx_proj(ctx).permute(0, 2, 1)                  # [B, A, C]
        ctx_att = self.ctx_len_proj(ctx_att).permute(0, 2, 1)          # [B, T, A]
        alphas = torch.softmax(
            self.alpha(torch.tanh(ref_att + ctx_att)) / self.temperature, dim=1
        ).squeeze(-1)                                                  # [B, T]
        out = ref * alphas.unsqueeze(-1)
        out = out.sum(1) if average_seq else out.squeeze(-1)
        return out, alphas


class PaccMann(nn.Module):
    """PaccMannV2: SMILES convolutions + two-way context attention over cell expression."""

    def __init__(self, vocab_size, smiles_len, gene_dim,
                 embed_size=16, filters=(64, 64, 64), kernels=(3, 5, 11),
                 molecule_heads=(4, 4, 4, 4), gene_heads=(2, 2, 2, 2),
                 dense_sizes=(1024, 512), dropout=0.5, batch_norm=True,
                 attention_size=64):
        super().__init__()
        assert len(filters) == len(kernels) == len(molecule_heads) - 1
        assert len(gene_heads) == len(molecule_heads)

        self.smiles_embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, f, (k, embed_size), padding=(k // 2, 0)),
                nn.ReLU(),
                nn.Dropout(dropout),
            ) for f, k in zip(filters, kernels)
        ])
        self.conv_norm = nn.ModuleList([
            nn.BatchNorm1d(f) if batch_norm else nn.Identity() for f in filters
        ])

        smiles_dims = [embed_size] + list(filters)
        self.mol_attn = nn.ModuleList([
            ContextAttention(smiles_dims[layer], smiles_len, 1, gene_dim,
                             attention_size=attention_size)
            for layer in range(len(molecule_heads))
            for _ in range(molecule_heads[layer])
        ])
        self.gene_attn = nn.ModuleList([
            ContextAttention(1, gene_dim, smiles_dims[layer], smiles_len,
                             attention_size=1)
            for layer in range(len(gene_heads))
            for _ in range(gene_heads[layer])
        ])

        hidden = (molecule_heads[0] * embed_size
                  + sum(h * f for h, f in zip(molecule_heads[1:], filters))
                  + sum(gene_heads) * gene_dim)
        sizes = [hidden] + list(dense_sizes)
        self.batch_norm = nn.BatchNorm1d(hidden) if batch_norm else nn.Identity()
        self.dense = nn.Sequential(*[
            nn.Sequential(nn.Linear(sizes[i], sizes[i + 1]),
                          nn.BatchNorm1d(sizes[i + 1]) if batch_norm else nn.Identity(),
                          nn.ReLU(), nn.Dropout(dropout))
            for i in range(len(sizes) - 1)
        ])
        self.final_dense = nn.Linear(sizes[-1], 1)
        self.molecule_heads = tuple(molecule_heads)
        self.gene_heads = tuple(gene_heads)

    def forward(self, smiles, gene):
        emb = self.smiles_embedding(smiles)                            # [B, T, E]
        encoded = [emb]
        for conv, norm in zip(self.convs, self.conv_norm):
            x = conv(emb.unsqueeze(1)).squeeze(-1)                     # [B, F, T]
            encoded.append(norm(x).permute(0, 2, 1))                   # [B, T, F]

        gep = gene.unsqueeze(-1)                                       # [B, G, 1]
        encodings, mol_offset, gene_offset = [], 0, 0
        for layer in range(len(self.molecule_heads)):
            for _ in range(self.molecule_heads[layer]):
                e, _ = self.mol_attn[mol_offset](encoded[layer], gep)
                encodings.append(e)
                mol_offset += 1
            for _ in range(self.gene_heads[layer]):
                e, _ = self.gene_attn[gene_offset](gep, encoded[layer], average_seq=False)
                encodings.append(e)
                gene_offset += 1

        h = self.batch_norm(torch.cat(encodings, dim=1))
        h = self.dense(h)
        return self.final_dense(h).squeeze(-1)


# ---------------------------------------------------------------- train/eval
def _predict(model, loader, device):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for smiles, gene, labels in loader:
            out = model(smiles.to(device), gene.to(device))
            preds.extend(out.cpu().numpy().flatten().tolist())
            trues.extend(labels.numpy().flatten().tolist())
    return np.asarray(trues, dtype=np.float64), np.asarray(preds, dtype=np.float64)


def main():
    parser = argparse.ArgumentParser(description='PaccMann training (fair 5-fold)')
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
                        help='variant id (e.g. v1_base5_PaccMann). when set the output dir is '
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

    all_smiles = [drug_smile_dict[d] for d in filtered_df[drug_id_col].unique()
                  if d in drug_smile_dict]
    vocab, smiles_len = build_vocab(all_smiles)
    token2idx = {t: i for i, t in enumerate(vocab)}
    print(f"SMILES vocab: {len(vocab)} tokens, max_len={smiles_len}")

    def encode_smiles(s):
        ids = [token2idx.get(t, 1) for t in tokenize_smiles(s)[:smiles_len]]
        return ids + [0] * (smiles_len - len(ids))

    drug_ids_arr = {d: np.asarray(encode_smiles(drug_smile_dict[d]), dtype=np.int64)
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
        output_dir = os.path.join(root_dir("Result"), "2_baselines", "PaccMann", "CV",
                                  f"{detect_env(args.env)}_{args.variant}_"
                                  f"{time.strftime('%Y%m%d_%H%M')}")
    else:
        output_dir = os.path.join(root_dir("Result"), "2_baselines", "PaccMann", "CV",
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
            if did not in drug_ids_arr or cid not in pooled_gene:
                continue
            if args.ablate == 'drug':
                drugs.append(np.zeros_like(drug_ids_arr[did]))   # all PAD tokens
            else:
                drugs.append(drug_ids_arr[did])
            if args.ablate == 'gene':
                cells.append(np.zeros(config['data']['gene_input_dim'], dtype=np.float32))
            else:
                cells.append(apply_gene_scaler(pooled_gene[cid], g_mean, g_std))
            labs.append(lab)
            rows.append(pos)
        ds = TensorDataset(torch.tensor(np.stack(drugs), dtype=torch.long),
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

        # PaccMann has BatchNorm, a trailing batch of 1 sample raises -> drop_last
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        model = PaccMann(len(vocab), smiles_len, config['data']['gene_input_dim'],
                         dropout=config['model']['dropout']).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                               factor=0.5, patience=5)
        criterion = nn.MSELoss()

        best_val_r2, best_state, patience_counter = -float('inf'), None, 0

        for epoch in range(1, epochs + 1):
            model.train()
            loss_sum = 0.0
            for smiles, gene, labels in train_loader:
                smiles, gene, labels = smiles.to(device), gene.to(device), labels.to(device)
                optimizer.zero_grad()
                loss = criterion(model(smiles, gene), labels)
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
        'model': 'PaccMann',
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
    print(f"PaccMann pooled results, n={int(len(oof_true))}")
    print(f"  RAW   : R²={raw_metrics['r2']:.4f}  RMSE={raw_metrics['rmse']:.4f}  "
          f"Pearson={raw_metrics['pearson']:.4f}  Spearman={raw_metrics['spearman']:.4f}")
    print(f"  CLAMP : R²={clamp_metrics['r2']:.4f}  RMSE={clamp_metrics['rmse']:.4f}  "
          f"Pearson={clamp_metrics['pearson']:.4f}  Spearman={clamp_metrics['spearman']:.4f}")
    print("=" * 70)


if __name__ == '__main__':
    main()
