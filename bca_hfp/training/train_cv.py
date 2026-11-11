# bca_hfp/training/train_cv.py
"""5-fold cross-validation training."""

import sys
import os
import time
import argparse
import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import KFold, GroupKFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from bca_hfp.config.base_config import get_base_config, PROJECT_ROOT, root_dir, detect_env
from bca_hfp.data.loader import (load_drug_atom_features, load_gene_embeddings,
                                 load_response, load_morgan_fingerprints, PHYSCHEM_DESC_DIM,
                                 pool_fgbert_global)
from bca_hfp.data.dataset import (DrugGeneDataset, BaselineDataset, collate_fn,
                                  DrugGroupedBatchSampler, MixedBatchSampler)
from bca_hfp.models import create_model
from bca_hfp.training.trainer import VariableLengthTrainer, FixedLengthTrainer
from bca_hfp.utils.reproducibility import set_seed


class _TeeStream:
    """Write console output to a log file (UTF-8) and to the original stream.

    Logging from Python avoids encoding damage when stdout is redirected by
    PowerShell on Windows.
    """

    def __init__(self, log, stream):
        self.log = log
        self.stream = stream

    def write(self, s):
        self.log.write(s)
        self.log.flush()
        try:
            self.stream.write(s)
            self.stream.flush()
        except Exception:
            pass

    def flush(self):
        self.log.flush()
        try:
            self.stream.flush()
        except Exception:
            pass

    def isatty(self):
        return False


def _install_log(path):
    """Open the log file and tee sys.stdout / sys.stderr to it."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    log = open(path, 'a', encoding='utf-8', buffering=1)
    sys.stdout = _TeeStream(log, sys.__stdout__)
    sys.stderr = _TeeStream(log, sys.__stderr__)
    return log


def _detect_env(choice='auto'):
    """Machine tag (TeslaT4 / 7900XT / RTX5090 / cpu), see base_config.detect_env.

    Used as a prefix for directory and log names.
    """
    return detect_env(choice)


def run_cross_validation(args):
    # log and model dirs share one timestamp
    stamp = time.strftime('%Y%m%d_%H%M')
    _env = _detect_env(getattr(args, 'env', 'auto'))
    _log_path = os.path.join(root_dir("Result"), "4_training_logs",
                             f"{_env}_{args.variant or 'run'}_{stamp}.txt")
    _install_log(_log_path)
    print(f"[log] {_log_path}  env={_env}")

    set_seed(args.seed)
    config = get_base_config(args.dataset)

    base_data_path = root_dir("Data")
    if args.gene_version == 'V1':
        config['data']['gene_feat_dir'] = f"{base_data_path}/Gene/GDSC_EXP_GF_V1"
        config['data']['gene_input_dim'] = 256
    else:
        config['data']['gene_feat_dir'] = f"{base_data_path}/Gene/GDSC_EXP_GF_V2"
        config['data']['gene_input_dim'] = 768

    # gene embedding dir can be overridden from the command line
    if args.gene_feat_dir:
        config['data']['gene_feat_dir'] = args.gene_feat_dir

    # --h5 overrides both the drug feature and the response table
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

    # fusion variant switches; unset means the paper defaults
    config['model']['decouple_atom'] = bool(args.decouple_atom)
    config['model']['center_keys'] = bool(args.center_keys)
    config['model']['center_query'] = bool(args.center_query)
    config['model']['atom_cond'] = bool(args.atom_cond)
    config['model']['atom_cond_pos'] = args.atom_cond_pos
    config['model']['key_cond'] = bool(args.key_cond)
    config['model']['gate_content'] = bool(args.gate_content)
    config['model']['gate_hidden'] = int(args.gate_hidden or 0)
    config['model']['g2d_pool'] = str(args.g2d_pool or 'mean')
    if args.query_source is not None:
        config['model']['query_source'] = args.query_source
    if args.add_atom_modality:
        config['model']['add_atom_modality'] = True
    config['model']['no_residual'] = bool(args.no_residual)
    config['model']['drop_drug_global_modality'] = bool(args.drop_drug_global_modality)

    config['model']['prior_dim'] = (int(args.prior_dim)
                                    if (args.interaction_prior and args.model_type == 'attention')
                                    else 0)
    if args.interaction_prior and args.model_type != 'attention':
        raise SystemExit('--interaction_prior requires --model_type attention')

    # modality dropout strength; precedence: dedicated flag > --modal_dropout > base_config
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

    # a smaller batch with proportionally more accumulation keeps the effective batch size
    if args.batch_size is not None:
        config['training']['batch_size'] = int(args.batch_size)
    if args.grad_accum is not None:
        config['training']['gradient_accumulation_steps'] = int(args.grad_accum)
    # each worker re-imports torch on Windows
    if args.num_workers is not None:
        config['training']['num_workers'] = int(args.num_workers)

    # fixed-epoch evaluation: --max_epochs overrides total epochs, --no_early_stop disables early stopping
    if args.max_epochs is not None:
        config['training']['epochs'] = args.max_epochs
    if args.no_early_stop:
        config['training']['early_stop_patience'] = 0

    # lr schedule: plateau (watches val) | fixed (epoch milestones, no val read)
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
    # within-drug loss weight (MSE after removing the drug mean, 0 = off)
    config['training']['within_weight'] = float(args.within_weight or 0.0)
    # query contrastive loss weight (in-batch InfoNCE, 0 = off) and temperature
    config['training']['query_contrast_weight'] = float(args.query_contrast or 0.0)
    config['training']['query_contrast_tau'] = float(args.query_contrast_tau or 0.1)
    # gate entropy regularization (0 = off); counteracts gate collapse
    config['training']['gate_entropy'] = float(args.gate_entropy or 0.0)
    config['training']['drug_bias_l2'] = float(args.drug_bias_l2 or 0.0)
    # best_balance selection rule
    config['training']['balance_rule'] = args.balance_rule

    # --- censoring / RMSE weighting / Tobit / inference clamp, all off by default ---
    cens_path = args.censoring_csv
    _cens_on = (float(args.tobit_sigma or 0) > 0) or bool(args.rmse_weight) or bool(args.pred_clamp)
    if args.within_target and _cens_on:
        raise SystemExit('--within_target targets within-drug deviation z, but the censor window '
                         '(b_lo/b_hi) is on the absolute ln(IC50) scale; cannot combine')
    _has_map = bool(cens_path) and os.path.exists(cens_path)
    if _cens_on and not _has_map:
        print("  [info] no --censoring_csv; using MIN_CONC / MAX_CONC / RMSE from the response table")
    use_cens = _cens_on
    if cens_path and not _cens_on:
        print(f"  [warn] censoring disabled, ignoring censor map {cens_path}")
    config['training']['use_cens'] = use_cens
    config['training']['tobit_sigma'] = float(args.tobit_sigma or 0.0)
    config['training']['rmse_weight'] = bool(args.rmse_weight)
    config['training']['pred_clamp'] = bool(args.pred_clamp)
    # soft penalty on predictions outside the per-drug window +/-2 (0 = off)
    config['training']['win_penalty'] = float(args.win_penalty or 0.0)

    split_seed = args.split_seed if args.split_seed is not None else args.seed

    # output dir name: md strength + variant + epochs + folds + tag
    if args.layer == 1:
        mlp_layer_str = '1l'
    elif args.layer == 2:
        mlp_layer_str = '2l'
    else:
        mlp_layer_str = '3l'
    fp_dim = int(getattr(args, 'fp_bits', 2048) or 2048)
    md_str = (f"{drug_global_dropout:g}" if drug_global_dropout == global_gene_dropout
              else f"{drug_global_dropout:g}-{global_gene_dropout:g}")
    subdir_name = (f"{args.dataset}_{args.model_type}_{mlp_layer_str}_{args.gene_version}_"
                   f"{args.split_method}_fp{fp_dim}_{args.reg_norm}_md{md_str}_seed{args.seed}_cv{args.n_splits}")
    if abs(args.val_ratio - 0.125) > 1e-9:
        subdir_name = f"{subdir_name}_vr{args.val_ratio:g}"
    if split_seed != args.seed:
        subdir_name = f"{subdir_name}_sp{split_seed}"
    if args.decouple_atom:
        subdir_name = f"{subdir_name}_dec"
    else:
        if args.query_source == 'morgan':
            subdir_name = f"{subdir_name}_qm"
        elif args.query_source == 'global_gene':
            subdir_name = f"{subdir_name}_qg"
        elif args.query_source == 'gene_pooled':
            subdir_name = f"{subdir_name}_qp"
        if args.add_atom_modality:
            subdir_name = f"{subdir_name}_am"
    if args.no_early_stop:
        subdir_name = f"{subdir_name}_ne{config['training']['epochs']}"
    if args.no_residual:
        subdir_name = f"{subdir_name}_nores"
    if args.key_cond:
        subdir_name = f"{subdir_name}_kc"
    if args.gate_content:
        subdir_name = f"{subdir_name}_gc"
    if float(args.gate_entropy or 0) > 0:
        subdir_name = f"{subdir_name}_ge{float(args.gate_entropy):g}"
    if args.g2d_pool and args.g2d_pool != 'mean':
        subdir_name = f"{subdir_name}_g2d{args.g2d_pool}"
    if args.drop_drug_global_modality:
        subdir_name = f"{subdir_name}_nodg"
    if args.lr_schedule == 'fixed':
        subdir_name = f"{subdir_name}_lrfix"
    if int(args.swa_last_k or 0) > 0:
        subdir_name = f"{subdir_name}_swa{int(args.swa_last_k)}"
    if args.use_desc:
        subdir_name = f"{subdir_name}_desc"
    if args.fgbert_global:
        subdir_name = f"{subdir_name}_fg"
    if float(args.tobit_sigma or 0) > 0:
        subdir_name = f"{subdir_name}_tobit{float(args.tobit_sigma):g}"
    if args.rmse_weight:
        subdir_name = f"{subdir_name}_rw"
    if args.pred_clamp:
        subdir_name = f"{subdir_name}_pc"
    if args.group_by_name:
        subdir_name = f"{subdir_name}_gbn"
    if args.loss == 'mse':
        subdir_name = f"{subdir_name}_mse"
    if float(args.drug_mean_weight or 0) > 0:
        subdir_name = f"{subdir_name}_dml{float(args.drug_mean_weight):g}"
    elif args.group_by_drug:
        subdir_name = f"{subdir_name}_gbd"
    if float(args.within_weight or 0) > 0:
        subdir_name = f"{subdir_name}_win{float(args.within_weight):g}"
    if config['training']['batch_size'] != 32:
        subdir_name = f"{subdir_name}_bs{config['training']['batch_size']}"
    if args.tag:
        subdir_name = f"{subdir_name}_{args.tag}"
    # --variant selects <env>_<variant>_<stamp>, otherwise the full auto name is used
    if args.variant:
        subdir_name = f"{_env}_{args.variant}_{stamp}"
    else:
        subdir_name = f"{_env}_{subdir_name}_{stamp}"
    output_root = os.path.join(root_dir("Result"), "1_main_models", subdir_name)
    config['output']['model_save_dir'] = os.path.join(output_root, 'models')
    config['output']['attention_csv_dir'] = os.path.join(output_root, 'attentions')
    config['output']['result_save_dir'] = os.path.join(output_root, 'results')

    config['data']['drug_global_dim'] = fp_dim + (PHYSCHEM_DESC_DIM if args.use_desc else 0)

    print("=" * 92)
    print("RUN CONFIG (CV)")
    print("=" * 92)
    print(f"  seed={args.seed}  split_seed={split_seed}  n_splits={args.n_splits}  reg_norm={args.reg_norm}")
    print(f"  split: test=1/{args.n_splits}  val_ratio={args.val_ratio} (of the remainder)  "
          f"drug-level ratio approx. "
          f"{(1 - 1 / args.n_splits) * (1 - args.val_ratio):.2f}:"
          f"{(1 - 1 / args.n_splits) * args.val_ratio:.2f}:"
          f"{1 / args.n_splits:.2f}")
    print(f"  h5={config['data']['drug_feat_path']}")
    print(f"  gene_feat_dir={config['data']['gene_feat_dir']}")
    print(f"  epochs={config['training']['epochs']}  early_stop_patience={config['training']['early_stop_patience']}")
    print(f"  batch_size={config['training']['batch_size']}  "
          f"grad_accum={config['training']['gradient_accumulation_steps']}  "
          f"effective batch={config['training']['batch_size'] * config['training']['gradient_accumulation_steps']}")
    print(f"  modality_dropout: drug_global={drug_global_dropout} global_gene={global_gene_dropout}")
    print(f"  query_source={config['model'].get('query_source')}  add_atom_modality={config['model'].get('add_atom_modality')}"
          f"  decouple_atom={config['model']['decouple_atom']}  no_residual={config['model']['no_residual']}"
          f"  center_keys={config['model'].get('center_keys', False)}"
          f"  center_query={config['model'].get('center_query', False)}"
          f"  atom_cond={config['model'].get('atom_cond', False)}"
          f"(pos={config['model'].get('atom_cond_pos', 'post')})"
          f"  key_cond={config['model'].get('key_cond', False)}"
          f"  gate_content={config['model'].get('gate_content', False)}"
          f"  gate_hidden={config['model'].get('gate_hidden', 0) or 'auto'}"
          f"  gate_entropy={config['training'].get('gate_entropy', 0.0):g}"
          f"  g2d_pool={config['model'].get('g2d_pool', 'mean')}"
          f"  drop_drug_global_modality={config['model']['drop_drug_global_modality']}")
    print(f"  loss={config['training'].get('loss', 'smoothl1')}  "
          f"drug_mean_weight={config['training']['drug_mean_weight']}  "
          f"within_weight={config['training']['within_weight']}  "
          f"query_contrast={config['training']['query_contrast_weight']}"
          f"(tau={config['training']['query_contrast_tau']:g})  "
          f"sampler={args.sampler}  drug_bias_l2={config['training']['drug_bias_l2']:g}  "
          f"balance_rule={config['training']['balance_rule']}")
    print(f"  censoring: use_cens={use_cens}  tobit_sigma={config['training']['tobit_sigma']:g}  "
          f"rmse_weight={config['training']['rmse_weight']}  pred_clamp={config['training']['pred_clamp']}"
          f"  win_penalty={config['training']['win_penalty']:g}"
          f"  fgbert_global={args.fgbert_global}  use_desc={args.use_desc}")
    print(f"  output_root={output_root}")
    print("=" * 92)

    print("=" * 60)
    print("Loading data...")
    print("=" * 60)

    drug_atom_dict = load_drug_atom_features(config['data']['drug_feat_path'])
    # FG-BERT drug-level pooling (mean/max of atom embeddings), concatenated into drug_global
    fgbert_pool = None
    if args.fgbert_global:
        fgbert_pool = pool_fgbert_global(drug_atom_dict, mode='meanmax')
        _fg_dim = int(next(iter(fgbert_pool.values())).shape[0])
        print(f"  FG-BERT drug-level pooling (meanmax): {len(fgbert_pool)} drugs, dim={_fg_dim}")
        config['data']['drug_global_dim'] = (fp_dim
                                             + (PHYSCHEM_DESC_DIM if args.use_desc else 0)
                                             + _fg_dim)

    print("Generating drug global features...")
    drug_global_dict = load_morgan_fingerprints(
        config['data']['drug_feat_path'],
        fp_bits=fp_dim,
        fp_radius=2,
        use_desc=args.use_desc,
        fgbert_dict=fgbert_pool
    )
    config['data']['drug_global_dim'] = int(next(iter(drug_global_dict.values())).shape[0])
    print(f"  drug_global_dim = {config['data']['drug_global_dim']}")

    gene_emb_dict, gene_names = load_gene_embeddings(config['data']['gene_feat_dir'], config['data']['cell_id_col'])
    response_df = load_response(config['data']['response_path'],
                                config['data']['cell_id_col'],
                                config['data']['drug_id_col'])
    # keep the original row index: the censor map aligns row by row with the response table
    response_df['_row'] = np.arange(len(response_df))

    drug_col = config['data']['drug_id_col']
    cell_col = config['data']['cell_id_col']
    label_col = config['data']['label_col']

    valid_drug_mask = response_df[drug_col].isin(drug_atom_dict)
    valid_cell_mask = response_df[cell_col].isin(gene_emb_dict)
    valid_mask = valid_drug_mask & valid_cell_mask
    filtered_df = response_df[valid_mask].copy()
    print(f"Valid samples: {len(filtered_df)}")

    if args.group_by_name:
        filtered_df['_group'] = (filtered_df['DRUG_NAME'].astype(str)
                                 .str.strip().str.upper())
        _n_multi = int((filtered_df.groupby('_group')[drug_col].nunique() > 1).sum())
        print(f"  [group_by_name] grouped by drug name: {filtered_df['_group'].nunique()} groups; "
              f"{_n_multi} names map to multiple DRUG_IDs (merged)")
    else:
        filtered_df['_group'] = filtered_df[drug_col]

    # --- censor columns: b_lo/b_hi (observable concentration window) + cens flag + RMSE weight w ---
    # prefer MIN_CONC / MAX_CONC / RMSE from the response table, else fall back to --censoring_csv
    if use_cens:
        _self_sufficient = {'MIN_CONC', 'MAX_CONC', 'RMSE'}.issubset(set(filtered_df.columns))
        if _self_sufficient:
            print("  censor window source: MIN_CONC / MAX_CONC / RMSE from the response table")
            _lo = pd.to_numeric(filtered_df['MIN_CONC'], errors='coerce').values.astype(np.float64)
            _hi = pd.to_numeric(filtered_df['MAX_CONC'], errors='coerce').values.astype(np.float64)
            _y = pd.to_numeric(filtered_df[label_col], errors='coerce').values.astype(np.float64)
            _rmse = pd.to_numeric(filtered_df['RMSE'], errors='coerce').values.astype(np.float64)
            _valid = np.isfinite(_lo) & np.isfinite(_hi) & (_lo > 0) & (_hi > 0)
            if not _valid.all():
                print(f"  [warn] {int((~_valid).sum())} rows have an invalid concentration window; "
                      f"using [-1e6, 1e6, w=1, cens=0]")
            b_lo = np.where(_valid, np.log(np.maximum(_lo, 1e-12)), -1e6)
            b_hi = np.where(_valid, np.log(np.maximum(_hi, 1e-12)), 1e6)
            _fl = np.zeros(len(filtered_df), dtype=np.float32)
            _fl[_valid & (_y >= b_hi)] = 1.0
            _fl[_valid & (_y <= b_lo)] = -1.0
            _w_raw = 1.0 / (1.0 + (np.nan_to_num(_rmse, nan=0.1) / 0.1) ** 2)
            _w = np.clip(_w_raw / _w_raw.mean(), 0.2, 2.0)
            filtered_df['b_lo'] = b_lo
            filtered_df['b_hi'] = b_hi
            filtered_df['cens_flag'] = _fl
            filtered_df['cens_w'] = _w.astype(np.float32)
        else:
            if not cens_path or not os.path.exists(cens_path):
                raise SystemExit('response table lacks MIN_CONC/MAX_CONC/RMSE and no usable --censoring_csv was given')
            _cens_df = pd.read_csv(cens_path)
            if len(_cens_df) != len(response_df):
                raise SystemExit(f"censor map has {len(_cens_df)} rows but the response table has "
                                 f"{len(response_df)}; cannot align by row order")
            _dl = np.abs(_cens_df['LN_IC50'].values.astype(np.float64)
                         - response_df[label_col].values.astype(np.float64))
            _ok = float(np.mean(_dl < 1e-6))
            print(f"  censor map row check: {_ok * 100:.2f}% of rows match LN_IC50 (max|delta|={_dl.max():.2e})")
            if _ok < 0.99:
                raise SystemExit("censor map rows do not match the response table; regenerate censoring_map.csv")
            _rows = filtered_df['_row'].values
            filtered_df['b_lo'] = _cens_df['b_lo'].values[_rows]
            filtered_df['b_hi'] = _cens_df['b_hi'].values[_rows]
            filtered_df['cens_flag'] = _cens_df['cens'].values[_rows].astype(np.float32)
            _w_raw = _cens_df['w_rmse'].values.astype(np.float64)
            _w = np.clip(_w_raw / _w_raw.mean(), 0.2, 2.0)
            filtered_df['cens_w'] = _w[_rows].astype(np.float32)
            _nan = int(filtered_df[['b_lo', 'b_hi', 'cens_w']].isna().any(axis=1).sum())
            if _nan > 0:
                print(f"  [warn] {_nan} rows have NaN censor fields; filled with [-1e6, 1e6, w=1, cens=0]")
                filtered_df['b_lo'] = filtered_df['b_lo'].fillna(-1e6)
                filtered_df['b_hi'] = filtered_df['b_hi'].fillna(1e6)
                filtered_df['cens_w'] = filtered_df['cens_w'].fillna(1.0)
                filtered_df['cens_flag'] = filtered_df['cens_flag'].fillna(0.0)
        _fl = filtered_df['cens_flag'].values
        print(f"  censored samples: right={int((_fl > 0).sum())} left={int((_fl < 0).sum())} "
              f"in-window={int((_fl == 0).sum())}")
        print(f"  RMSE weight: raw_mean={_w_raw.mean():.4f} -> clipped mean={float(_w.mean()):.4f} "
              f"min={_w.min():.3f} max={_w.max():.3f}")

    def _cens_of(sub):
        """Return (N,4) = [b_lo, b_hi, cens_flag, w], row-aligned with the dataset."""
        if not use_cens:
            return None
        return sub[['b_lo', 'b_hi', 'cens_flag', 'cens_w']].values.astype(np.float32)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # integer drug index used by the drug mean / within-drug / contrastive losses;
    # with --group_by_name the index is built on the normalized drug name
    aux_on = config['training']['drug_mean_weight'] > 0
    within_on = config['training']['within_weight'] > 0
    qc_on = config['training']['query_contrast_weight'] > 0
    bias_on = config['training']['drug_bias_l2'] > 0
    need_drug_idx = aux_on or within_on or qc_on or bias_on
    # grouped sampling serves the batch-level drug mean and contrastive losses only
    group_on = (aux_on or within_on or qc_on) or bool(args.group_by_drug)
    if need_drug_idx:
        _grp_to_idx = {g: i for i, g in enumerate(sorted(filtered_df['_group'].unique()))}
        _id_to_grp = dict(zip(filtered_df[drug_col], filtered_df['_group']))
        drug_to_idx = {d: _grp_to_idx[g] for d, g in _id_to_grp.items()}
        config['model']['drug_bias'] = bool(bias_on)
        config['model']['num_drugs'] = len(_grp_to_idx)
    else:
        drug_to_idx = None

    all_fold_metrics = []

    # scaffold split: group by Bemis-Murcko scaffold so no scaffold crosses train/test;
    # acyclic molecules (empty scaffold) form individual groups
    scaffold_group = None
    if args.split_method == 'scaffold':
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
        ds = pd.read_hdf(config['data']['drug_feat_path'], 'drug_smiles')
        smi_col = next((c for c in ['DRUG_SMILE', 'SMILES', 'smiles', 'drug_smiles']
                        if c in ds.columns), None)
        if smi_col is None:
            raise KeyError(f'drug_smiles table has no SMILES column: {list(ds.columns)}')
        smi_map = {str(k): v for k, v in zip(ds['DRUG_ID'], ds[smi_col])}

        def _scaf(d):
            smi = smi_map.get(d)
            mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
            if mol is None:
                return f'__no_scaffold__{d}'
            try:
                s = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
            except Exception:
                s = ''
            return s if s else f'__no_scaffold__{d}'

        scaffold_group = {d: _scaf(d) for d in filtered_df[drug_col].unique()}
        _vc = pd.Series(list(scaffold_group.values())).value_counts()
        print(f"scaffold split: {len(scaffold_group)} drugs -> {len(_vc)} scaffold groups, "
              f"largest group = {int(_vc.iloc[0])} drugs")
        _n_nos = sum(1 for v in scaffold_group.values() if str(v).startswith('__no_scaffold__'))
        if _n_nos:
            print(f"  [warn] {_n_nos}/{len(scaffold_group)} drugs have no scaffold; each forms its own group")
        if _n_nos > 0.5 * len(scaffold_group):
            raise RuntimeError(
                f"scaffold split failed: {_n_nos}/{len(scaffold_group)} drugs have no Bemis-Murcko "
                f"scaffold. Check the SMILES column and DRUG_ID in the h5 drug_smiles table.")

    if args.fold_csv:
        # external fixed split: read a drug -> fold map instead of running GroupKFold,
        # whose result differs across sklearn versions
        fm = pd.read_csv(args.fold_csv)
        key = 'GROUP' if 'GROUP' in fm.columns else ('DRUG_NAME' if 'DRUG_NAME' in fm.columns
                                                    else drug_col)
        fm[key] = fm[key].astype(str).str.strip().str.upper()
        g2f = dict(zip(fm[key], fm['fold'].astype(int)))
        _keys = (filtered_df['_group'].astype(str).str.strip().str.upper()
                 if args.group_by_name else filtered_df[drug_col].astype(str))
        _fold = _keys.map(g2f)
        if _fold.isna().any():
            raise SystemExit(f"--fold_csv does not cover {int(_fold.isna().sum())} rows "
                             f"(check key {key} against --group_by_name)")
        splits = []
        for k in sorted(fm['fold'].unique()):
            te = (_fold == k).values
            splits.append((np.where(~te)[0], np.where(te)[0]))
        print(f"  [fold_csv] using external split {args.fold_csv}: "
              f"{len(splits)} folds, drug groups per fold="
              f"{[int((_fold==k).sum()) for k in sorted(fm['fold'].unique())]}")
    elif args.split_method == 'random':
        kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=split_seed)
        splits = list(kf.split(filtered_df))
    elif args.split_method == 'cell':
        gkf = GroupKFold(n_splits=args.n_splits)
        groups = filtered_df[cell_col].values
        splits = list(gkf.split(filtered_df, groups=groups))
    elif args.split_method == 'scaffold':
        groups = filtered_df[drug_col].map(scaffold_group).values
        splits = list(GroupKFold(n_splits=args.n_splits).split(filtered_df, groups=groups))
    else:
        gkf = GroupKFold(n_splits=args.n_splits)
        groups = (filtered_df['_group'].values if args.group_by_name
                  else filtered_df[drug_col].values)
        splits = list(gkf.split(filtered_df, groups=groups))

    oof_frames = {}  # tag -> per-fold prediction frames, merged into out-of-fold metrics
    result_dir = config['output']['result_save_dir']
    os.makedirs(result_dir, exist_ok=True)

    only_folds = ({int(x) for x in args.only_folds.split(',') if x.strip().isdigit()}
                  if args.only_folds else None)
    if only_folds:
        print(f"  [only_folds] running folds {sorted(only_folds)} only")

    for fold_idx, (train_val_idx, test_idx) in enumerate(splits):
        if only_folds is not None and (fold_idx + 1) not in only_folds:
            continue
        print(f"\n{'=' * 60}")
        print(f"Fold {fold_idx + 1}/{args.n_splits}")
        print(f"{'=' * 60}")

        train_val_df = filtered_df.iloc[train_val_idx].copy()
        test_df = filtered_df.iloc[test_idx].copy()

        if args.split_method == 'random':
            from sklearn.model_selection import train_test_split
            train_idx_inner, val_idx_inner = train_test_split(
                range(len(train_val_df)), test_size=args.val_ratio, random_state=split_seed
            )
            train_df = train_val_df.iloc[train_idx_inner].copy()
            val_df = train_val_df.iloc[val_idx_inner].copy()
        elif args.split_method == 'cell':
            unique_cells = train_val_df[cell_col].unique()
            np.random.seed(split_seed)
            val_cells = np.random.choice(unique_cells, size=int(len(unique_cells) * args.val_ratio), replace=False)
            val_mask = train_val_df[cell_col].isin(val_cells)
            train_df = train_val_df[~val_mask].copy()
            val_df = train_val_df[val_mask].copy()
        elif args.split_method == 'scaffold':
            tv_scaf = train_val_df[drug_col].map(scaffold_group)
            unique_scaf = tv_scaf.unique()
            np.random.seed(split_seed)
            val_scaf = np.random.choice(
                unique_scaf, size=max(1, int(len(unique_scaf) * args.val_ratio)), replace=False)
            val_mask = tv_scaf.isin(val_scaf)
            train_df = train_val_df[~val_mask].copy()
            val_df = train_val_df[val_mask].copy()
        else:
            # --group_by_name: split on drug name so both DRUG_IDs of a drug stay together
            _split_key = '_group' if args.group_by_name else drug_col
            unique_drugs = train_val_df[_split_key].unique()
            np.random.seed(split_seed)
            val_drugs = np.random.choice(unique_drugs, size=int(len(unique_drugs) * args.val_ratio), replace=False)
            val_mask = train_val_df[_split_key].isin(val_drugs)
            train_df = train_val_df[~val_mask].copy()
            val_df = train_val_df[val_mask].copy()

        if args.within_target:
            # consistent with drug-level grouping: both DRUG_IDs of a drug share mu_d
            tr_mu = train_df.groupby('_group')[label_col].mean()
            mu_map = dict(tr_mu)
            for _sub in (val_df, test_df):
                for _d, _m in _sub.groupby('_group')[label_col].mean().items():
                    mu_map.setdefault(_d, _m)
            for _sub in (train_df, val_df, test_df):
                _mu = _sub['_group'].map(mu_map).astype(float)
                _sub['__mu_d'] = _mu.values
                _sub[label_col] = _sub[label_col].astype(float) - _mu.values
            print(f"  [within_target] target is now within-drug deviation z = LN_IC50 - mu_d; "
                  f"{len(tr_mu)} training drugs, mean mu_d {float(tr_mu.mean()):+.4f}, "
                  f"sd(z) {float(train_df[label_col].std()):.4f}")

        group_col = cell_col if args.split_method == 'cell' else (
            '_group' if (args.split_method == 'drug' and args.group_by_name) else drug_col)
        n_tr = train_df[group_col].nunique() if args.split_method != 'random' else len(train_df)
        n_va = val_df[group_col].nunique() if args.split_method != 'random' else len(val_df)
        n_te = test_df[group_col].nunique() if args.split_method != 'random' else len(test_df)
        unit = 'cell lines' if args.split_method == 'cell' else 'drugs'
        print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        print(f"  {unit}: train={n_tr}, val={n_va}, test={n_te}  "
              f"(val_ratio={args.val_ratio} of the remaining {n_tr + n_va} {unit})")

        if args.model_type == 'attention':
            # interaction prior: fitted on this fold's training pairs only. With --within_target
            # the labels are already z, so __mu_d is added back before fitting
            prior_dict = None
            if args.interaction_prior:
                from bca_hfp.data.interaction_prior import fit_interaction_prior
                if getattr(args, 'within_target', False) and '__mu_d' in train_df.columns:
                    _pairs = train_df[[drug_col, cell_col, '__mu_d']].copy()
                    _pairs[label_col] = (train_df[label_col].values
                                         + train_df['__mu_d'].values)
                else:
                    _pairs = train_df[[drug_col, cell_col, label_col]].copy()
                prior_dict, _pinfo = fit_interaction_prior(
                    _pairs, config['data']['drug_feat_path'],
                    label_col, drug_col, cell_col,
                    K=int(args.prior_dim), als_lam=float(args.prior_als_lam),
                    seed=int(split_seed),
                    fp_bits=int(getattr(args, 'fp_bits', 2048) or 2048),
                    verbose=True)

            train_dataset = DrugGeneDataset(
                train_df[drug_col].tolist(), train_df[cell_col].tolist(),
                train_df[label_col].values, drug_atom_dict, gene_emb_dict, drug_global_dict,
                drug_to_idx=drug_to_idx, prior_dict=prior_dict,
                cens_arr=_cens_of(train_df)
            )
            val_dataset = DrugGeneDataset(
                val_df[drug_col].tolist(), val_df[cell_col].tolist(),
                val_df[label_col].values, drug_atom_dict, gene_emb_dict, drug_global_dict,
                prior_dict=prior_dict, cens_arr=_cens_of(val_df)
            )
            test_dataset = DrugGeneDataset(
                test_df[drug_col].tolist(), test_df[cell_col].tolist(),
                test_df[label_col].values, drug_atom_dict, gene_emb_dict, drug_global_dict,
                prior_dict=prior_dict, cens_arr=_cens_of(test_df)
            )

            from functools import partial
            _has_prior = prior_dict is not None
            collate = partial(collate_fn, atom_mask_prob=config['training'].get('atom_mask_prob', 0.0),
                              with_prior=_has_prior, with_cens=use_cens)
            train_collate = partial(collate_fn, atom_mask_prob=config['training'].get('atom_mask_prob', 0.0),
                                    with_drug_idx=need_drug_idx, with_prior=_has_prior,
                                    with_cens=use_cens)
            _bs = config['training']['batch_size']
            _mode = args.sampler
            if _mode == 'auto':
                _mode = 'grouped' if group_on else 'random'
            if _mode == 'grouped':
                # drug-grouped sampling: drugs_per_batch drugs x (batch_size // drugs_per_batch) samples
                train_sampler = DrugGroupedBatchSampler(
                    train_df['_group'].tolist() if args.group_by_name else train_dataset.drug_ids,
                    batch_size=_bs,
                    drugs_per_batch=args.aux_drugs_per_batch, seed=split_seed)
                train_loader = torch.utils.data.DataLoader(
                    train_dataset, batch_sampler=train_sampler,
                    collate_fn=train_collate, num_workers=config['training']['num_workers']
                )
                print(f"  [sampler=grouped] drugs/batch={train_sampler.drugs_per_batch}  "
                      f"samples/drug={train_sampler.samples_per_drug}  steps/epoch={train_sampler.steps}")
            elif _mode == 'mixed':
                # mixed sampling: random shuffle, with one drug-grouped batch every group_every batches
                train_sampler = MixedBatchSampler(
                    len(train_dataset), train_dataset.drug_ids, batch_size=_bs,
                    drugs_per_batch=args.aux_drugs_per_batch,
                    group_every=args.mixed_group_every, seed=split_seed)
                train_loader = torch.utils.data.DataLoader(
                    train_dataset, batch_sampler=train_sampler,
                    collate_fn=train_collate, num_workers=config['training']['num_workers']
                )
                print(f"  [sampler=mixed] 1 of every {train_sampler.group_every} batches is drug-grouped "
                      f"(drugs/batch={train_sampler.grouped.drugs_per_batch}), "
                      f"the rest random; steps/epoch={train_sampler.steps}")
            else:
                train_loader = torch.utils.data.DataLoader(
                    train_dataset, batch_size=_bs,
                    shuffle=True, collate_fn=train_collate, num_workers=config['training']['num_workers'],
                    # drop_last=False: keep the trailing partial batch; SafeBatchNorm1d handles batch=1
                    drop_last=False
                )
                print("  [sampler=random] uniform random shuffle")
            _rem = len(train_dataset) % _bs
            print(f"  train_size={len(train_dataset)}  batch_size={_bs}  "
                  f"last_batch={_rem if _rem > 0 else _bs}"
                  f"{'  [remainder=1 -> SafeBatchNorm1d]' if _rem == 1 else ''}")
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
                shuffle=True, num_workers=config['training']['num_workers'],
                # baseline uses native BatchNorm1d, which rejects batch=1; attention uses SafeBatchNorm1d
                drop_last=True
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
        if args.model_type == 'attention':
            history, best_metrics, best_val_path, best_balance_path = trainer.train(
                train_loader, val_loader, config['training']['epochs'],
                save_dir, fold_idx, name,
                test_loader=test_loader,
                eval_test_each_epoch=args.eval_test_each_epoch,
            )
        else:
            history, best_metrics, best_val_path, best_balance_path = trainer.train(
                train_loader, val_loader, config['training']['epochs'],
                save_dir, fold_idx, name
            )

        def _eval_ckpt(path, tag):
            if not path or not os.path.exists(path):
                return
            checkpoint = torch.load(path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            _, m = trainer.validate(test_loader)
            print(f"Test metrics ({tag}): R²={m['r2']:.4f}, RMSE={m['rmse']:.4f}, Pearson={m['pearson']:.4f}")
            m['model_type'] = tag
            m['fold'] = fold_idx + 1
            m['select_epoch'] = checkpoint.get('epoch', -1)
            all_fold_metrics.append(m)

            preds = trainer.last_preds
            if preds is None or len(preds) != len(test_df):
                print(f"  [warn] fold{fold_idx + 1} {tag}: {0 if preds is None else len(preds)} predictions "
                      f"!= {len(test_df)} test samples, skipping prediction dump")
                return
            _rec = {
                'DRUG_ID': test_df[drug_col].values,
                'GDSC_ID': test_df[cell_col].values,
                'true_label': test_df[label_col].values,
                'predicted': preds,
                'fold': fold_idx + 1,
                'model_type': tag,
            }
            if args.within_target and '__mu_d' in test_df.columns:
                # with within-drug targets, labels and predictions are z; mu_d recovers absolute ln(IC50)
                _rec['mu_d'] = test_df['__mu_d'].values
            df_pred = pd.DataFrame(_rec)
            df_pred.to_csv(os.path.join(
                result_dir, f"{name}_fold{fold_idx + 1}_{tag}_test_predictions.csv"), index=False)
            oof_frames.setdefault(tag, []).append(df_pred)

        final_path = os.path.join(save_dir, f"{name}_fold{fold_idx + 1}_final.pth")
        _eval_ckpt(best_val_path, 'best_val')
        # best_balance falls back to final when no candidate was found
        if best_balance_path is None:
            print(f"  [warn] fold{fold_idx + 1}: no best_balance candidate "
                  f"(rule={args.balance_rule}), using final")
        _eval_ckpt(best_balance_path or final_path, 'best_balance')
        # fixed-epoch evaluation: last-epoch (_final) test, independent of best-val selection
        _eval_ckpt(final_path, 'final')
        # SWA weights, present only when --swa_last_k > 0
        _eval_ckpt(getattr(trainer, 'swa_model_path', None), 'swa')

    if all_fold_metrics:
        # merge per-fold test predictions (held-out drugs do not overlap) into out-of-fold metrics
        for tag, frames in oof_frames.items():
            df_oof = pd.concat(frames, ignore_index=True)
            m = trainer._compute_metrics(df_oof['true_label'].values, df_oof['predicted'].values)
            m['model_type'] = f'oof_{tag}'
            m['fold'] = 'all'
            m['select_epoch'] = -1
            all_fold_metrics.append(m)
            print(f"\nOOF ({tag}, {len(df_oof)} samples, {df_oof['DRUG_ID'].nunique()} drugs): "
                  f"R²={m['r2']:+.4f}  RMSE={m['rmse']:.4f}  "
                  f"MAE={m['mae']:.4f}  Pearson={m['pearson']:.4f}")
            if args.within_target:
                print("  [within_target] these metrics are on within-drug z and are not comparable "
                      "to absolute IC50 metrics; add a drug-level model before comparing.")

        metrics_df = pd.DataFrame(all_fold_metrics)
        metrics_df['env'] = _env
        metrics_path = os.path.join(result_dir, f"{args.model_type}_{args.split_method}_cv_results.csv")
        metrics_df.to_csv(metrics_path, index=False)
        print(f"\nCross-validation results saved to {metrics_path}")

        for mt in ['best_val', 'best_balance', 'final', 'swa']:
            subset = metrics_df[metrics_df['model_type'] == mt]
            if len(subset) > 0:
                print(f"\n{mt} model summary:")
                print(f"  Mean R²: {subset['r2'].mean():.4f} ± {subset['r2'].std():.4f}")
                print(f"  Mean RMSE: {subset['rmse'].mean():.4f} ± {subset['rmse'].std():.4f}")
                print(f"  Mean Pearson: {subset['pearson'].mean():.4f} ± {subset['pearson'].std():.4f}")


def main():
    parser = argparse.ArgumentParser(description='5-fold cross-validation training')
    parser.add_argument('--dataset', type=str, default='GDSC', choices=['GDSC'])
    parser.add_argument('--model_type', type=str, default='attention', choices=['attention', 'baseline'])
    parser.add_argument('--split_method', type=str, default='drug',
                        choices=['random', 'cell', 'drug', 'scaffold'])
    parser.add_argument('--gene_version', type=str, default='V1', choices=['V1', 'V2'])
    parser.add_argument('--layer', type=int, default=3, choices=[1, 2, 3])
    parser.add_argument('--reg_norm', type=str, default='layernorm',
                        choices=['batchnorm', 'layernorm'],
                        help='Regression head norm layer; layernorm avoids OOD running stats under leave-drug-out')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--split_seed', type=int, default=None,
                        help='split seed for leave-drug-out; defaults to --seed')
    parser.add_argument('--n_splits', type=int, default=5,
                        help='number of CV folds (default 5)')
    parser.add_argument('--val_ratio', type=float, default=0.125,
                        help='fraction of the train+val part held out for validation (default 0.125); '
                             'test is always 1/n_splits, so a 6:2:2 drug-level split needs '
                             '--n_splits 5 --val_ratio 0.25')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='override batch_size from base_config')
    parser.add_argument('--grad_accum', type=int, default=None,
                        help='override gradient_accumulation_steps')
    parser.add_argument('--num_workers', type=int, default=None,
                        help='override DataLoader num_workers')
    parser.add_argument('--only_folds', type=str, default='',
                        help='comma-separated folds to run, e.g. "1" or "1,2"; '
                             'the OOF output then covers only those folds')
    parser.add_argument('--h5', type=str, default='',
                        help='override both the drug feature and the response table paths')
    parser.add_argument('--gene_feat_dir', type=str, default='',
                        help='override the gene embedding directory (default depends on --gene_version)')
    parser.add_argument('--modal_dropout', type=float, default=None,
                        help='set both drug_global and global_gene modality dropout')
    parser.add_argument('--drug_global_dropout', type=float, default=None,
                        help='drug_global (Morgan) modality dropout; takes precedence over --modal_dropout')
    parser.add_argument('--global_gene_dropout', type=float, default=None,
                        help='global_gene modality dropout; takes precedence over --modal_dropout')
    parser.add_argument('--decouple_atom', action='store_true',
                        help='decoupled query: Morgan as query plus an extra atom modality')
    parser.add_argument('--center_keys', action='store_true',
                        help='center gene keys across genes (key only, value unchanged); '
                             'identity under softmax')
    parser.add_argument('--center_query', action='store_true',
                        help='center drug queries: subtract the mean query direction, keeping only '
                             'each drug deviation')
    parser.add_argument('--atom_cond', action='store_true',
                        help='FiLM conditioning of atom features from the drug global vector '
                             '(Morgan + descriptors); zero-init, so the start equals the baseline')
    parser.add_argument('--atom_cond_pos', type=str, default='post',
                        choices=['pre', 'mid', 'post'],
                        help='where conditioning is applied: pre = before query centering, '
                             'mid = after centering, post = after norm restoration (default)')
    parser.add_argument('--key_cond', action='store_true',
                        help='multiplicative drug-conditioned gene key: key * (1 + gamma); '
                             'zero-init, and additive offsets cancel in softmax')
    parser.add_argument('--g2d_pool', type=str, default='mean',
                        choices=['mean', 'meanmax', 'd2g'],
                        help='gene pooling for gene->drug: mean | meanmax = Linear([mean;max]) '
                             'initialized to mean | d2g = weight genes by drug->gene attention')
    parser.add_argument('--gate_content', action='store_true',
                        help='content-aware gating: gate input becomes [query; attended modality pools]; '
                             'new columns zero-init')
    parser.add_argument('--gate_entropy', type=float, default=0.0,
                        help='gate entropy regularization weight (0 = off): maximize per-sample gate '
                             'entropy normalized by log K')
    parser.add_argument('--gate_hidden', type=int, default=0,
                        help='gate MLP hidden width (0 = d_model//2)')
    parser.add_argument('--win_penalty', type=float, default=0.0,
                        help='soft penalty on predictions outside the per-drug concentration window '
                             '+/-2 (0 = off); unlike --pred_clamp it enters the training objective')
    parser.add_argument('--env', type=str, default='auto',
                        choices=['auto', 'TeslaT4', '7900XT', 'RTX5090', 'cpu', 'loc', 'srv'],
                        help='machine tag used as directory/log prefix and written to the result CSV; '
                             'auto derives it from the GPU name; loc/srv are legacy aliases')
    parser.add_argument('--fold_csv', type=str, default='',
                        help='external fixed split CSV with a GROUP/DRUG_NAME or DRUG_ID column '
                             'and a fold column')
    parser.add_argument('--query_source', type=str, default=None,
                        choices=['atom', 'atommean', 'morgan', 'global_gene', 'gene_pooled'],
                        help='fusion query source: atom = pooled atom attention output, modulated by the '
                             'cell line; atommean = plain atom mean pooling (Eq.(6), no cell line info); '
                             'morgan = Morgan projection; global_gene = cell line global gene; '
                             'gene_pooled = drug-conditioned gene pooling')
    parser.add_argument('--add_atom_modality', action='store_true',
                        help='add pooled atom features as a fourth gated modality')
    parser.add_argument('--no_residual', action='store_true',
                        help='drop the query residual in gated fusion (Eq.(7))')
    parser.add_argument('--drop_drug_global_modality', action='store_true',
                        help='do not use the Morgan vector as the drug_global modality; '
                             'it still enters through query and residual')
    parser.add_argument('--no_early_stop', action='store_true',
                        help='disable early stopping and run all epochs')
    parser.add_argument('--max_epochs', type=int, default=None,
                        help='override total epochs (use with --no_early_stop)')
    parser.add_argument('--lr_schedule', type=str, default=None, choices=['plateau', 'fixed'],
                        help='lr schedule: plateau = ReduceLROnPlateau on val; '
                             'fixed = epoch milestones, no val read')
    parser.add_argument('--swa_last_k', type=int, default=0,
                        help='SWA: average the weights of the last K epochs (0 = off)')
    parser.add_argument('--query_contrast', type=float, default=0.0,
                        help='query contrastive loss weight (in-batch InfoNCE, 0 = off); '
                             'typically 0.05 to 0.3')
    parser.add_argument('--query_contrast_tau', type=float, default=0.1,
                        help='softmax temperature of the query contrastive loss')
    parser.add_argument('--fp_bits', type=int, default=2048,
                        help='Morgan fingerprint bits (default 2048)')
    parser.add_argument('--use_desc', action='store_true',
                        help='append 8 physicochemical descriptors after the Morgan fingerprint '
                             '(MW / logP / TPSA / HBD / HBA / aromatic rings / rotatable bonds / FractionCSP3)')
    parser.add_argument('--fgbert_global', action='store_true',
                        help='append the FG-BERT drug-level pooled vector (mean||max of atom embeddings, '
                             '512 dims); order: Morgan, descriptors, FG-BERT')
    parser.add_argument('--censoring_csv', type=str, default='',
                        help='censor map csv (row,b_lo,b_hi,cens,w) aligned row by row with the response '
                             'table; empty = no censoring')
    parser.add_argument('--tobit_sigma', type=float, default=0.0,
                        help='if > 0, use the Tobit censored likelihood with this sigma instead of MSE/SmoothL1')
    parser.add_argument('--rmse_weight', action='store_true',
                        help='reweight the regression loss by 1/(1+(RMSE/0.1)^2), clipped to [0.2, 2.0]')
    parser.add_argument('--pred_clamp', action='store_true',
                        help='inference clamp: [b_lo-2, b_hi+2], then [ln0.001, ln100]')
    parser.add_argument('--group_by_name', action='store_true',
                        help='group by normalized drug name instead of DRUG_ID; some GDSC2 drugs have '
                             'two DRUG_IDs, which would split one compound across folds')
    parser.add_argument('--loss', type=str, default=None, choices=['smoothl1', 'mse'],
                        help='loss: smoothl1 (default) | mse')
    parser.add_argument('--drug_mean_weight', type=float, default=0.0,
                        help='drug mean auxiliary loss weight: L + w * |mean(pred_d) - mean(true_d)|; 0 = off')
    parser.add_argument('--interaction_prior', action='store_true',
                        help='add an interaction prior: K-dim drug x cell coefficients fitted on the '
                             'training fold (data/interaction_prior.py), added through a zero-init '
                             'projection; intended for use with --within_target')
    parser.add_argument('--prior_dim', type=int, default=16,
                        help='interaction prior dimension K (SVD basis), default 16')
    parser.add_argument('--prior_als_lam', type=float, default=5.0,
                        help='shrinkage for the additive model (mu + a_d + b_c) in the interaction prior')
    parser.add_argument('--within_target', action='store_true',
                        help='change the target to within-drug deviation z = LN_IC50 - mu_d '
                             '(mu_d estimated on this fold training drugs); reported R2/RMSE are '
                             'within-drug and not comparable to absolute IC50 metrics')
    parser.add_argument('--within_weight', type=float, default=0.0,
                        help='within-drug loss weight: MSE after subtracting each drug batch mean; 0 = off')
    parser.add_argument('--aux_drugs_per_batch', type=int, default=8,
                        help='drugs per batch in grouped sampling (samples per drug = batch_size / this)')
    parser.add_argument('--group_by_drug', action='store_true',
                        help='enable drug-grouped sampling only, without auxiliary losses')
    parser.add_argument('--sampler', type=str, default='auto',
                        choices=['auto', 'grouped', 'random', 'mixed'],
                        help='training sampler: auto (grouped when a batch-level drug loss is active, '
                             'else random) | grouped | random | mixed (one grouped batch every N)')
    parser.add_argument('--mixed_group_every', type=int, default=4,
                        help='for --sampler mixed, insert one drug-grouped batch every N batches')
    parser.add_argument('--drug_bias_l2', type=float, default=0.0,
                        help='L2 strength on the learnable per-drug bias (0 = off)')
    parser.add_argument('--balance_rule', type=str, default='gated', choices=['gated', 'gap', 'val_loss'],
                        help='best_balance rule: gated = signed gap, val_r2 > 0 (default); '
                             'gap = |train_r2 - val_r2|; val_loss = val loss')
    parser.add_argument('--eval_test_each_epoch', action='store_true',
                        help='extra test evaluation each epoch, logged only')
    parser.add_argument('--variant', type=str, default='',
                        help='variant tag; when set the output dir becomes <env>_<variant>_<timestamp>')
    parser.add_argument('--tag', type=str, default='',
                        help='suffix appended to the output directory name')
    args = parser.parse_args()

    run_cross_validation(args)


if __name__ == '__main__':
    main()
