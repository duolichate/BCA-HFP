# bca_hfp/training/train_holdout.py
"""Hold-out validation training."""

import sys
import os
import time
import argparse
import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from bca_hfp.config.base_config import get_base_config, PROJECT_ROOT, root_dir
from bca_hfp.data.loader import (load_drug_atom_features, load_gene_embeddings,
                                 load_response, load_morgan_fingerprints, PHYSCHEM_DESC_DIM)
from bca_hfp.data.dataset import DrugGeneDataset, BaselineDataset, collate_fn, DrugGroupedBatchSampler
from bca_hfp.models import create_model
from bca_hfp.training.trainer import VariableLengthTrainer, FixedLengthTrainer
from bca_hfp.utils.reproducibility import set_seed


def split_data_random(df, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_state=42):
    """Random split without considering drug/cell grouping."""
    train_df, temp_df = train_test_split(df, test_size=(1 - train_ratio), random_state=random_state)
    val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(temp_df, test_size=(1 - val_ratio_adjusted), random_state=random_state)
    return train_df, val_df, test_df


def split_data_by_cell(df, cell_col, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_state=42):
    """Split by cell line with GroupShuffleSplit; no cell leakage between sets.

    Restored to the original two-step GroupShuffleSplit scheme so that the
    partition reproduces the paper's LDO/LCO splits (random_state=42).
    """
    groups = df[cell_col].values
    indices = np.arange(len(df))

    gss_test = GroupShuffleSplit(n_splits=1, test_size=test_ratio, random_state=random_state)
    train_val_idx, test_idx = next(gss_test.split(indices, groups=groups))

    train_val_df = df.iloc[train_val_idx]
    val_ratio_in_train_val = val_ratio / (train_ratio + val_ratio)
    gss_val = GroupShuffleSplit(n_splits=1, test_size=val_ratio_in_train_val, random_state=random_state)
    train_rel_idx, val_rel_idx = next(
        gss_val.split(np.arange(len(train_val_df)), groups=train_val_df[cell_col].values)
    )

    train_idx = train_val_idx[train_rel_idx]
    val_idx = train_val_idx[val_rel_idx]

    print(f"Split by cell: train={len(train_idx)} ({len(set(groups[train_idx]))} cells), "
          f"val={len(val_idx)} ({len(set(groups[val_idx]))} cells), "
          f"test={len(test_idx)} ({len(set(groups[test_idx]))} cells)")
    return df.iloc[train_idx], df.iloc[val_idx], df.iloc[test_idx]


def split_data_by_drug(df, drug_col, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_state=42):
    """Split by drug with GroupShuffleSplit (leave-drug-out); no drug leakage between sets.

    Restored to the original two-step GroupShuffleSplit scheme so that the
    partition reproduces the paper's LDO split (random_state=42).
    """
    groups = df[drug_col].values
    indices = np.arange(len(df))

    gss_test = GroupShuffleSplit(n_splits=1, test_size=test_ratio, random_state=random_state)
    train_val_idx, test_idx = next(gss_test.split(indices, groups=groups))

    train_val_df = df.iloc[train_val_idx]
    val_ratio_in_train_val = val_ratio / (train_ratio + val_ratio)
    gss_val = GroupShuffleSplit(n_splits=1, test_size=val_ratio_in_train_val, random_state=random_state)
    train_rel_idx, val_rel_idx = next(
        gss_val.split(np.arange(len(train_val_df)), groups=train_val_df[drug_col].values)
    )

    train_idx = train_val_idx[train_rel_idx]
    val_idx = train_val_idx[val_rel_idx]

    print(f"Split by drug: train={len(train_idx)} ({len(set(groups[train_idx]))} drugs), "
          f"val={len(val_idx)} ({len(set(groups[val_idx]))} drugs), "
          f"test={len(test_idx)} ({len(set(groups[test_idx]))} drugs)")
    return df.iloc[train_idx], df.iloc[val_idx], df.iloc[test_idx]


def load_fixed_fold_splits(fold_csv, drug_name_series):
    """Map samples to fixed folds by drug name (GROUP), as in train_cv.py --fold_csv.

    Key is GROUP (drug name), uppercased and stripped. Returns
    [(train_val_idx, test_idx), ...] as numpy indices, folds in ascending order.
    Every drug name must be covered, otherwise SystemExit.
    """
    fm = pd.read_csv(fold_csv)
    key = 'GROUP' if 'GROUP' in fm.columns else ('DRUG_NAME' if 'DRUG_NAME' in fm.columns else None)
    if key is None:
        raise SystemExit(f"--fold_csv has no GROUP/DRUG_NAME column: {list(fm.columns)}")
    fm[key] = fm[key].astype(str).str.strip().str.upper()
    g2f = dict(zip(fm[key], fm['fold'].astype(int)))
    grp = drug_name_series.astype(str).str.strip().str.upper()
    f = grp.map(g2f)
    if f.isna().any():
        missing = sorted(set(grp[f.isna()]))
        raise SystemExit(f"--fold_csv does not cover {int(f.isna().sum())} rows / {len(missing)} drug names "
                         f"(first few: {missing[:5]})")
    splits = []
    for k in sorted(fm['fold'].unique()):
        te = (f == k).values
        splits.append((np.where(~te)[0], np.where(te)[0]))
    return splits


def build_censor_bounds(filtered_df, label_col):
    """Per-row log-space concentration window [b_lo, b_hi].

    Matches the censoring columns consumed by VariableLengthTrainer._clamp_pred;
    invalid windows become [-1e6, 1e6].
    """
    need = {'MIN_CONC', 'MAX_CONC'}
    if not need.issubset(set(filtered_df.columns)):
        raise SystemExit('response table lacks MIN_CONC / MAX_CONC; cannot build the concentration window')
    lo = pd.to_numeric(filtered_df['MIN_CONC'], errors='coerce').values.astype(np.float64)
    hi = pd.to_numeric(filtered_df['MAX_CONC'], errors='coerce').values.astype(np.float64)
    ok = np.isfinite(lo) & np.isfinite(hi) & (lo > 0) & (hi > 0)
    b_lo = np.where(ok, np.log(np.maximum(lo, 1e-12)), -1e6)
    b_hi = np.where(ok, np.log(np.maximum(hi, 1e-12)), 1e6)
    return b_lo, b_hi


def winsorize_pred(pred, b_lo, b_hi):
    """Per-row window +/-2, then clamp to [ln0.001, ln100]."""
    p = np.asarray(pred, dtype=np.float64)
    out = np.maximum(np.minimum(p, b_hi + 2.0), b_lo - 2.0)
    return np.clip(out, float(np.log(0.001)), float(np.log(100.0)))


def fit_gene_scaler(gene_vectors):
    """Per-dimension mean/std of the training-fold gene vectors, for baseline input scaling.

    GeneFormer embeddings averaged over genes have a per-dim std near 5e-3, which
    leaves Conv/Linear activations close to zero. Statistics come from the training
    fold only; val/test are transformed, so no leakage.
    """
    X = np.asarray(gene_vectors, dtype=np.float64)
    mean = X.mean(axis=0).astype(np.float32)
    std = X.std(axis=0).astype(np.float32)
    std = np.where(std < 1e-8, np.float32(1.0), std)
    return mean, std


def apply_gene_scaler(x, mean, std):
    """z-score using the statistics from fit_gene_scaler."""
    return ((np.asarray(x, dtype=np.float32) - mean) / std).astype(np.float32)


def pooled_metrics(y_true, y_pred):
    """R2/RMSE/MAE/Pearson/Spearman over pooled predictions."""
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    from scipy.stats import pearsonr, spearmanr
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return {
        'n': int(len(y_true)),
        'r2': float(r2_score(y_true, y_pred)),
        'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'mae': float(mean_absolute_error(y_true, y_pred)),
        'pearson': float(pearsonr(y_true, y_pred)[0]),
        'spearman': float(spearmanr(y_true, y_pred)[0]),
    }


def _file_info(path):
    """File size and modification time, for the data provenance header."""
    if not os.path.exists(path):
        return f"{path}   << NOT FOUND >>"
    st = os.stat(path)
    mtime = time.strftime('%Y-%m-%d %H:%M', time.localtime(st.st_mtime))
    return f"{path}   ({st.st_size / 1024 ** 2:.1f} MB, mtime {mtime})"


def _dir_info(path):
    """Number of .pt files in the directory."""
    if not os.path.isdir(path):
        return f"{path}   << NOT FOUND >>"
    n = len([f for f in os.listdir(path) if f.endswith('.pt')])
    return f"{path}   ({n} .pt files)"


def print_run_header(config, args, fp_dim, response_df, filtered_df,
                     train_df, val_df, test_df, drug_col, cell_col):
    """Print the run configuration and data provenance before training."""
    tr, md = config['training'], config['model']
    line = '=' * 92
    print(line)
    print('RUN CONFIG')
    print(line)
    print(f"  time            : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  run             : {args.dataset} | {args.model_type} | split={args.split_method} | "
          f"gene_version={args.gene_version} | layer={args.layer}")
    print(f"  seed            : {args.seed}   (training randomness: weight init / shuffling)")
    print(f"  split_seed      : {args.split_seed}   (leave-drug-out split seed; 42 = paper split)")
    print(f"  reg_norm        : {args.reg_norm}")
    print('-' * 92)
    print('  DATA PROVENANCE')
    print(f"    drug_feat   : {_file_info(config['data']['drug_feat_path'])}")
    print(f"    gene_feat   : {_dir_info(config['data']['gene_feat_dir'])}")
    print(f"    response    : {_file_info(config['data']['response_path'])}")
    print(f"    samples     : raw={len(response_df)}  valid={len(filtered_df)}")
    print(f"    split       : train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")
    print(f"    groups      : drugs {train_df[drug_col].nunique()}/"
          f"{val_df[drug_col].nunique()}/{test_df[drug_col].nunique()} | "
          f"cells {train_df[cell_col].nunique()}/"
          f"{val_df[cell_col].nunique()}/{test_df[cell_col].nunique()}  (train/val/test)")
    print('-' * 92)
    print('  TRAINING')
    print(f"    batch_size={tr['batch_size']}  grad_accum={tr['gradient_accumulation_steps']}  "
          f"epochs={tr['epochs']}  lr={tr['lr']:.1e}  weight_decay={tr['weight_decay']}")
    print(f"    grad_clip={tr['gradient_clip']}  early_stop_patience={tr['early_stop_patience']}  "
          f"scheduler={tr['scheduler']}(factor={tr['reduce_factor']}, patience={tr['reduce_patience']})")
    print(f"    num_workers={tr['num_workers']}  atom_mask_prob={tr['atom_mask_prob']}")
    print(f"    modality_dropout : drug_global={tr['drug_global_dropout']}  "
          f"global_gene={tr['global_gene_dropout']}   (per-sample, gate-mask removal)")
    if args.eval_test_each_epoch:
        print(f"    eval_test_each_epoch : ON  (test pass each epoch, logged only)")
    print('-' * 92)
    print('  MODEL')
    print(f"    feature_dim={md['feature_dim']}  num_heads={md['num_heads']}  dropout={md['dropout']}  "
          f"num_fusion_layers={md['num_fusion_layers']}")
    print(f"    hidden_dims={md['hidden_dims']}  gene_input_dim={config['data']['gene_input_dim']}  "
          f"drug_global_dim={fp_dim}")
    print(f"    decouple_atom : {bool(md.get('decouple_atom'))}   "
          f"(number of fusion modalities = {4 if md.get('decouple_atom') else 3})")
    print(f"    regression_head_norm : {args.reg_norm}")
    print(f"    gate_temperature     : init={tr['gate_temperature']} (learnable, clamp [0.05, 20])")
    print('-' * 92)
    print('  OUTPUT')
    print(f"    model_save_dir  : {config['output']['model_save_dir']}")
    print(f"    result_save_dir : {config['output']['result_save_dir']}")
    print(line)


def run_holdout(args):
    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 1e-6:
        raise ValueError(f'train/val/test ratios must sum to 1, got '
                         f'{args.train_ratio}/{args.val_ratio}/{args.test_ratio}')
    set_seed(args.seed)
    config = get_base_config(args.dataset)

    base_data_path = root_dir("Data")
    if args.gene_version == 'V1':
        config['data']['gene_feat_dir'] = f"{base_data_path}/GDSC_EXP_GF_V1"
        config['data']['gene_input_dim'] = 256
    else:
        config['data']['gene_feat_dir'] = f"{base_data_path}/GDSC_EXP_GF_V2"
        config['data']['gene_input_dim'] = 768

    if args.h5:
        config['data']['drug_feat_path'] = args.h5
        config['data']['response_path'] = args.h5

    if args.layer == 1:
        hidden_dims = [64]
    elif args.layer == 2:
        hidden_dims = [256, 128]
    else:
        hidden_dims = [512, 256, 128]
    config['model']['hidden_dims'] = hidden_dims
    config['model']['feature_dim'] = 256
    config['model']['num_heads'] = 8
    config['model']['dropout'] = 0.2  # fixed so the norm layer is the only variable
    config['model']['num_fusion_layers'] = 1
    config['model']['reg_norm'] = args.reg_norm  # 'layernorm' (default) or 'batchnorm'
    config['model']['decouple_atom'] = bool(args.decouple_atom)
    # single-variable fusion switches; defaults: query_source='atom', add_atom_modality=False
    if args.query_source is not None:
        config['model']['query_source'] = args.query_source
    if args.add_atom_modality:
        config['model']['add_atom_modality'] = True
    # drop the fusion residual: f_fused = sum_k a_k * m_k (Eq.(7))
    config['model']['no_residual'] = bool(args.no_residual)
    config['model']['drop_drug_global_modality'] = bool(args.drop_drug_global_modality)

    # lr schedule: fixed = epoch milestones, no val read
    if args.lr_schedule is not None:
        config['training']['scheduler'] = ('fixed' if args.lr_schedule == 'fixed'
                                           else 'reduce_on_plateau')
    # SWA: average the last K epochs (0 = off)
    config['training']['swa_last_k'] = int(args.swa_last_k or 0)
    # loss: smoothl1 (default) | mse
    if args.loss is not None:
        config['training']['loss'] = args.loss
    # drug mean auxiliary loss weight (0 = off)
    config['training']['drug_mean_weight'] = float(args.drug_mean_weight or 0.0)
    # best_balance selection rule
    config['training']['balance_rule'] = args.balance_rule

    # modality dropout; precedence: dedicated flag > --modal_dropout > base_config
    base_dg = config['training']['drug_global_dropout']
    base_gg = config['training']['global_gene_dropout']
    if args.modal_dropout is not None:
        base_dg = base_gg = args.modal_dropout
    drug_global_dropout = (args.drug_global_dropout if args.drug_global_dropout is not None
                           else base_dg)
    global_gene_dropout = (args.global_gene_dropout if args.global_gene_dropout is not None
                           else base_gg)
    config['training']['drug_global_dropout'] = drug_global_dropout
    config['training']['global_gene_dropout'] = global_gene_dropout

    # fixed-epoch evaluation: --max_epochs overrides epochs, --no_early_stop disables early stopping
    if args.max_epochs is not None:
        config['training']['epochs'] = args.max_epochs
    if args.no_early_stop:
        config['training']['early_stop_patience'] = 0

    # output dir: Result/holdout/HO/{dataset}_{model_type}_{layer}_{gene_version}_{split}_fp{dim}/
    if args.layer == 1:
        mlp_layer_str = '1l'
    elif args.layer == 2:
        mlp_layer_str = '2l'
    else:
        mlp_layer_str = '3l'
    fp_dim = 2048
    md_str = (f"{drug_global_dropout:g}" if drug_global_dropout == global_gene_dropout
              else f"{drug_global_dropout:g}-{global_gene_dropout:g}")
    subdir_name = (f"{args.dataset}_{args.model_type}_{mlp_layer_str}_{args.gene_version}_"
                   f"{args.split_method}_fp{fp_dim}_{args.reg_norm}_md{md_str}_seed{args.seed}")
    if args.split_seed != args.seed:
        # same training seed with a different split needs a distinct dir
        subdir_name = f"{subdir_name}_sp{args.split_seed}"
    if (abs(args.train_ratio - 0.8) > 1e-9 or abs(args.val_ratio - 0.1) > 1e-9
            or abs(args.test_ratio - 0.1) > 1e-9):
        subdir_name = (f"{subdir_name}_tr{args.train_ratio:g}"
                       f"_va{args.val_ratio:g}_te{args.test_ratio:g}")
    if args.decouple_atom:
        # decoupled version has a fourth modality
        subdir_name = f"{subdir_name}_dec"
    else:
        # single-variable variants: qm = Morgan query, am = extra atom modality
        if args.query_source == 'morgan':
            subdir_name = f"{subdir_name}_qm"
        if args.add_atom_modality:
            subdir_name = f"{subdir_name}_am"
    if args.no_early_stop:
        # fixed-epoch run uses a different selection rule
        subdir_name = f"{subdir_name}_ne{config['training']['epochs']}"
    if args.no_residual:
        # no-residual version has a different architecture
        subdir_name = f"{subdir_name}_nores"
    if args.drop_drug_global_modality:
        # with query=morgan the Morgan vector would otherwise appear twice
        subdir_name = f"{subdir_name}_nodg"
    if args.lr_schedule == 'fixed':
        subdir_name = f"{subdir_name}_lrfix"
    if int(args.swa_last_k or 0) > 0:
        subdir_name = f"{subdir_name}_swa{int(args.swa_last_k)}"
    if args.use_desc:
        subdir_name = f"{subdir_name}_desc"
    if args.loss == 'mse':
        subdir_name = f"{subdir_name}_mse"
    if float(args.drug_mean_weight or 0) > 0:
        subdir_name = f"{subdir_name}_dml{float(args.drug_mean_weight):g}"
    elif args.group_by_drug:
        subdir_name = f"{subdir_name}_gbd"
    if args.tag:
        subdir_name = f"{subdir_name}_{args.tag}"
    output_root = os.path.join(root_dir("Result"), "holdout", "HO", subdir_name)
    config['output']['model_save_dir'] = os.path.join(output_root, 'models')
    config['output']['attention_csv_dir'] = os.path.join(output_root, 'attentions')
    config['output']['result_save_dir'] = os.path.join(output_root, 'results')

    config['data']['drug_global_dim'] = fp_dim + (PHYSCHEM_DESC_DIM if args.use_desc else 0)

    print("Generating drug global features...")
    drug_global_dict = load_morgan_fingerprints(
        config['data']['drug_feat_path'],
        fp_bits=fp_dim,
        fp_radius=2,
        use_desc=args.use_desc
    )

    print("=" * 60)
    print("Loading data...")
    print("=" * 60)

    drug_atom_dict = load_drug_atom_features(config['data']['drug_feat_path'])
    gene_emb_dict, gene_names = load_gene_embeddings(config['data']['gene_feat_dir'], config['data']['cell_id_col'])
    response_df = load_response(config['data']['response_path'],
                                config['data']['cell_id_col'],
                                config['data']['drug_id_col'])

    drug_col = config['data']['drug_id_col']
    cell_col = config['data']['cell_id_col']
    label_col = config['data']['label_col']

    valid_drug_mask = response_df[drug_col].isin(drug_atom_dict)
    valid_cell_mask = response_df[cell_col].isin(gene_emb_dict)
    valid_mask = valid_drug_mask & valid_cell_mask
    filtered_df = response_df[valid_mask].copy()
    print(f"Valid samples: {len(filtered_df)}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if args.split_method == 'random':
        train_df, val_df, test_df = split_data_random(
            filtered_df, train_ratio=args.train_ratio, val_ratio=args.val_ratio,
            test_ratio=args.test_ratio, random_state=args.split_seed
        )
    elif args.split_method == 'cell':
        train_df, val_df, test_df = split_data_by_cell(
            filtered_df, cell_col, train_ratio=args.train_ratio, val_ratio=args.val_ratio,
            test_ratio=args.test_ratio, random_state=args.split_seed
        )
    else:
        train_df, val_df, test_df = split_data_by_drug(
            filtered_df, drug_col, train_ratio=args.train_ratio, val_ratio=args.val_ratio,
            test_ratio=args.test_ratio, random_state=args.split_seed
        )

    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    print_run_header(config, args, fp_dim, response_df, filtered_df,
                     train_df, val_df, test_df, drug_col, cell_col)

    aux_on = config['training']['drug_mean_weight'] > 0
    group_on = aux_on or bool(args.group_by_drug)
    drug_to_idx = ({d: i for i, d in enumerate(sorted(filtered_df[drug_col].unique()))}
                   if aux_on else None)

    if args.model_type == 'attention':
        train_dataset = DrugGeneDataset(
            train_df[drug_col].tolist(), train_df[cell_col].tolist(),
            train_df[label_col].values, drug_atom_dict, gene_emb_dict, drug_global_dict,
            drug_to_idx=drug_to_idx
        )
        val_dataset = DrugGeneDataset(
            val_df[drug_col].tolist(), val_df[cell_col].tolist(),
            val_df[label_col].values, drug_atom_dict, gene_emb_dict, drug_global_dict
        )
        test_dataset = DrugGeneDataset(
            test_df[drug_col].tolist(), test_df[cell_col].tolist(),
            test_df[label_col].values, drug_atom_dict, gene_emb_dict, drug_global_dict
        )

        from functools import partial
        collate = partial(collate_fn, atom_mask_prob=config['training'].get('atom_mask_prob', 0.0))
        train_collate = partial(collate_fn, atom_mask_prob=config['training'].get('atom_mask_prob', 0.0),
                                with_drug_idx=aux_on)
        _bs = config['training']['batch_size']
        if group_on:
            train_sampler = DrugGroupedBatchSampler(
                train_dataset.drug_ids, batch_size=_bs,
                drugs_per_batch=args.aux_drugs_per_batch, seed=args.split_seed)
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_sampler=train_sampler,
                collate_fn=train_collate, num_workers=config['training']['num_workers']
            )
            print(f"  [drug-mean aux] drugs/batch={train_sampler.drugs_per_batch}  "
                  f"samples/drug={train_sampler.samples_per_drug}  steps/epoch={train_sampler.steps}")
        else:
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=_bs,
                shuffle=True, collate_fn=train_collate, num_workers=config['training']['num_workers']
            )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=config['training']['batch_size'],
            shuffle=False, collate_fn=collate, num_workers=config['training']['num_workers']
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=config['training']['batch_size'],
            shuffle=False, collate_fn=collate, num_workers=config['training']['num_workers']
        )

        model = create_model('attention', config).to(device)
        trainer = VariableLengthTrainer(model, config, device)
    else:
        from bca_hfp.data.loader import pool_drug_atom_features, pool_gene_embeddings
        drug_pooled = pool_drug_atom_features(drug_atom_dict)
        gene_pooled = pool_gene_embeddings(gene_emb_dict)

        train_dataset = BaselineDataset(
            train_df[drug_col].tolist(), train_df[cell_col].tolist(),
            train_df[label_col].values, drug_pooled, gene_pooled
        )
        val_dataset = BaselineDataset(
            val_df[drug_col].tolist(), val_df[cell_col].tolist(),
            val_df[label_col].values, drug_pooled, gene_pooled
        )
        test_dataset = BaselineDataset(
            test_df[drug_col].tolist(), test_df[cell_col].tolist(),
            test_df[label_col].values, drug_pooled, gene_pooled
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=config['training']['batch_size'],
            shuffle=True, num_workers=config['training']['num_workers']
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=config['training']['batch_size'],
            shuffle=False, num_workers=config['training']['num_workers']
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=config['training']['batch_size'],
            shuffle=False, num_workers=config['training']['num_workers']
        )

        model = create_model('baseline', config).to(device)
        trainer = FixedLengthTrainer(model, config, device)

    save_dir = config['output']['model_save_dir']
    name = f"{args.model_type}_{args.split_method}"
    train_kwargs = dict(
        num_epochs=config['training']['epochs'],
        save_dir=save_dir, fold_idx=None, name=name
    )
    if args.model_type == 'attention':
        train_kwargs['test_loader'] = test_loader
        train_kwargs['eval_test_each_epoch'] = args.eval_test_each_epoch
    history, best_metrics, best_val_path, best_balance_path = trainer.train(
        train_loader, val_loader, **train_kwargs
    )

    results = []
    if best_val_path and os.path.exists(best_val_path):
        checkpoint = torch.load(best_val_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        _, test_metrics = trainer.validate(test_loader)
        print(f"\nTest metrics (best val model): R²={test_metrics['r2']:.4f}, RMSE={test_metrics['rmse']:.4f}, Pearson={test_metrics['pearson']:.4f}")
        test_metrics['model_type'] = 'best_val'
        results.append(test_metrics)

    if best_balance_path and os.path.exists(best_balance_path):
        checkpoint = torch.load(best_balance_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        _, test_metrics = trainer.validate(test_loader)
        print(f"Test metrics (best balance model): R²={test_metrics['r2']:.4f}, RMSE={test_metrics['rmse']:.4f}, Pearson={test_metrics['pearson']:.4f}")
        test_metrics['model_type'] = 'best_balance'
        results.append(test_metrics)

    if results:
        metrics_df = pd.DataFrame(results)
        result_dir = config['output']['result_save_dir']
        os.makedirs(result_dir, exist_ok=True)
        metrics_path = os.path.join(result_dir, f"{args.model_type}_{args.split_method}_holdout_results.csv")
        metrics_df.to_csv(metrics_path, index=False)
        print(f"\nHold-out results saved to {metrics_path}")


def main():
    parser = argparse.ArgumentParser(description='Hold-out validation training')
    parser.add_argument('--dataset', type=str, default='GDSC', choices=['GDSC'])
    parser.add_argument('--model_type', type=str, default='attention', choices=['attention', 'baseline'])
    parser.add_argument('--split_method', type=str, default='drug', choices=['random', 'cell', 'drug'])
    parser.add_argument('--gene_version', type=str, default='V1', choices=['V1', 'V2'])
    parser.add_argument('--layer', type=int, default=3, choices=[1, 2, 3])
    parser.add_argument('--reg_norm', type=str, default='layernorm',
                        choices=['batchnorm', 'layernorm'],
                        help='Regression head norm layer; layernorm avoids OOD running stats under leave-drug-out')
    parser.add_argument('--modal_dropout', type=float, default=None,
                        help='per-sample modality dropout; sets both drug_global and global_gene; '
                             'default comes from base_config')
    parser.add_argument('--drug_global_dropout', type=float, default=None,
                        help='Morgan fingerprint modality dropout; takes precedence over --modal_dropout')
    parser.add_argument('--global_gene_dropout', type=float, default=None,
                        help='global gene modality dropout; takes precedence over --modal_dropout')
    parser.add_argument('--seed', type=int, default=42,
                        help='training seed for weight init and shuffling')
    parser.add_argument('--split_seed', type=int, default=42,
                        help='leave-drug-out split seed, independent of --seed; default 42 is the paper split')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='training fraction (default 0.8); GroupShuffleSplit works on drug/cell counts')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='validation fraction (default 0.1)')
    parser.add_argument('--test_ratio', type=float, default=0.1,
                        help='test fraction (default 0.1)')
    parser.add_argument('--h5', type=str, default=None,
                        help='override drug_feat_path and response_path')
    parser.add_argument('--decouple_atom', action='store_true',
                        help='decouple the atom branch: atom features become a fourth gated modality and '
                             'the query comes from the Morgan projection')
    parser.add_argument('--query_source', type=str, default=None, choices=['atom', 'morgan'],
                        help='fusion query source: atom = pooled atom features (Eq.(6), default); '
                             'morgan = Morgan projection')
    parser.add_argument('--add_atom_modality', action='store_true',
                        help='add pooled atom features as a fourth gated modality')
    parser.add_argument('--no_residual', action='store_true',
                        help='drop the query residual: out = norm(fused), matching Eq.(7) '
                             'f_fused = sum_k a_k * m_k')
    parser.add_argument('--drop_drug_global_modality', action='store_true',
                        help='do not use the Morgan vector as the drug_global modality; '
                             'it still enters through query and residual')
    parser.add_argument('--lr_schedule', type=str, default=None, choices=['plateau', 'fixed'],
                        help='lr schedule: plateau = ReduceLROnPlateau on val; '
                             'fixed = epoch milestones, no val read')
    parser.add_argument('--swa_last_k', type=int, default=0,
                        help='SWA: average the weights of the last K epochs (0 = off)')
    parser.add_argument('--use_desc', action='store_true',
                        help='append 8 physicochemical descriptors after the Morgan fingerprint '
                             '(MW / logP / TPSA / HBD / HBA / aromatic rings / rotatable bonds / FractionCSP3)')
    parser.add_argument('--loss', type=str, default=None, choices=['smoothl1', 'mse'],
                        help='loss: smoothl1 (default) | mse')
    parser.add_argument('--drug_mean_weight', type=float, default=0.0,
                        help='drug mean auxiliary loss weight: L + w * |mean(pred_d) - mean(true_d)|; 0 = off')
    parser.add_argument('--aux_drugs_per_batch', type=int, default=8,
                        help='drugs per batch in grouped sampling (samples per drug = batch_size / this)')
    parser.add_argument('--group_by_drug', action='store_true',
                        help='enable drug-grouped sampling only, without auxiliary losses')
    parser.add_argument('--balance_rule', type=str, default='gated', choices=['gated', 'gap', 'val_loss'],
                        help='best_balance rule: gated = signed gap, val_r2 > 0 (default); '
                             'gap = |train_r2 - val_r2|; val_loss = val loss')
    parser.add_argument('--tag', type=str, default='',
                        help='suffix appended to the output directory name')
    parser.add_argument('--eval_test_each_epoch', action='store_true',
                        help='extra test pass each epoch, logged to history only')
    parser.add_argument('--no_early_stop', action='store_true',
                        help='disable early stopping; use with --max_epochs and report _final.pth')
    parser.add_argument('--max_epochs', type=int, default=None,
                        help='override total epochs (default from base_config)')
    args = parser.parse_args()

    run_holdout(args)


if __name__ == '__main__':
    main()
