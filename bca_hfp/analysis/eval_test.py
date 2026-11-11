# bca_hfp/analysis/eval_test.py
"""rerun the test set from a trained checkpoint -- mirrors the evaluation of train_holdout.py.

purpose:
  when the test evaluation at the end of training fails (OOM, interruption, an accidental Ctrl-C)
  there is no need to retrain, just recompute the test metrics from the saved checkpoint.

arguments that must match training (otherwise metrics are not comparable, or loading fails):
  --reg_norm      BN/LN state_dicts differ, a wrong choice fails immediately (the script says so)
  --h5            if training used a replaced h5 (e.g. randatom) evaluation must use the same one
  --layer / --gene_version / --fp_bits / --seed

usage:
python bca_hfp/analysis/eval_test.py \\
  --model_path  <...>/models/attention_drug_best_val.pth \\
  --reg_norm    batchnorm \\
  --output_dir  <...>/diagnostics_eval
"""

import sys
import os
import time
import argparse
import numpy as np
import torch
import pandas as pd
from functools import partial

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from bca_hfp.config.base_config import get_base_config, PROJECT_ROOT, root_dir
from bca_hfp.data.loader import (load_drug_atom_features, load_gene_embeddings,
                                 load_response, load_morgan_fingerprints, PHYSCHEM_DESC_DIM)
from bca_hfp.data.dataset import DrugGeneDataset, collate_fn
from bca_hfp.models import create_model
from bca_hfp.training.train_holdout import split_data_by_drug
from bca_hfp.training.metrics import compute_metrics
from bca_hfp.utils.reproducibility import set_seed


def main():
    parser = argparse.ArgumentParser(description='rerun the test set from a checkpoint')
    parser.add_argument('--model_path', type=str, required=True, help='checkpoint (.pth)')
    parser.add_argument('--dataset', type=str, default='GDSC', choices=['GDSC'])
    parser.add_argument('--gene_version', type=str, default='V1', choices=['V1', 'V2'])
    parser.add_argument('--layer', type=int, default=3, choices=[1, 2, 3])
    parser.add_argument('--reg_norm', type=str, default='batchnorm',
                        choices=['batchnorm', 'layernorm'],
                        help='must match training, otherwise the state_dict does not fit')
    parser.add_argument('--fp_bits', type=int, default=2048)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42, help='same as the training seed (determines the split)')
    parser.add_argument('--h5', type=str, default=None,
                        help='override drug_feat_path and response_path; '
                             'if training used a replaced h5 (e.g. randatom) pass the same one')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='optional: save per-sample val/test prediction csv')
    parser.add_argument('--query_source', type=str, default='atom',
                        choices=['atom', 'morgan', 'global_gene', 'gene_pooled'],
                        help='fusion query source, must match training (default atom = the setting used in the paper)')
    parser.add_argument('--add_atom_modality', action='store_true',
                        help='whether training added the 4th atom modality (also inferred from the gate_mlp output width)')
    parser.add_argument('--no_residual', action='store_true',
                        help='whether training dropped the fusion residual (not inferable from the state_dict, must be given)')
    parser.add_argument('--drop_drug_global_modality', action='store_true',
                        help='whether training dropped the drug_global modality (gate width is then 3, '
                             'indistinguishable from "3 modalities without atom", must be given)')
    parser.add_argument('--use_desc', action='store_true',
                        help='whether training appended physicochemical descriptors to the Morgan fingerprint (must match training)')
    args = parser.parse_args()

    set_seed(args.seed)

    # ---------- 1. config (reproduces train_holdout.run_holdout item by item) ----------
    config = get_base_config(args.dataset)
    base_data_path = root_dir('Data')
    if args.gene_version == 'V1':
        config['data']['gene_feat_dir'] = f'{base_data_path}/GDSC_EXP_GF_V1'
        config['data']['gene_input_dim'] = 256
    else:
        config['data']['gene_feat_dir'] = f'{base_data_path}/GDSC_EXP_GF_V2'
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
    config['model']['dropout'] = 0.2
    config['model']['num_fusion_layers'] = 1
    config['model']['reg_norm'] = args.reg_norm
    config['model']['query_source'] = args.query_source
    config['model']['add_atom_modality'] = bool(args.add_atom_modality)
    config['model']['no_residual'] = bool(args.no_residual)
    config['model']['drop_drug_global_modality'] = bool(args.drop_drug_global_modality)
    config['model']['decouple_atom'] = False  # expressed by the two explicit flags above, avoids the implicit meaning of the old flag
    config['data']['drug_global_dim'] = args.fp_bits + (PHYSCHEM_DESC_DIM if args.use_desc else 0)

    h5_path = config['data']['drug_feat_path']
    print('=' * 78)
    print(f'model_path : {args.model_path}')
    print(f'drug_feat  : {h5_path}   (mtime {time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(h5_path)))})')
    print(f'reg_norm   : {args.reg_norm}    seed={args.seed}    layer={args.layer}')
    print('=' * 78)

    # ---------- 2. data ----------
    drug_global_dict = load_morgan_fingerprints(h5_path, fp_bits=args.fp_bits, fp_radius=2,
                                                use_desc=args.use_desc)
    drug_atom_dict = load_drug_atom_features(h5_path)
    gene_emb_dict, _ = load_gene_embeddings(config['data']['gene_feat_dir'],
                                            config['data']['cell_id_col'])
    response_df = load_response(config['data']['response_path'],
                                config['data']['cell_id_col'],
                                config['data']['drug_id_col'])

    drug_col = config['data']['drug_id_col']
    cell_col = config['data']['cell_id_col']
    label_col = config['data']['label_col']

    valid_mask = response_df[drug_col].isin(drug_atom_dict) & response_df[cell_col].isin(gene_emb_dict)
    filtered_df = response_df[valid_mask].copy()
    print(f'Valid samples: {len(filtered_df)}')

    _, val_df, test_df = split_data_by_drug(
        filtered_df, drug_col, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1,
        random_state=args.seed
    )
    print(f'Val : {len(val_df)} samples / {val_df[drug_col].nunique()} drugs')
    print(f'Test: {len(test_df)} samples / {test_df[drug_col].nunique()} drugs')

    collate = partial(collate_fn, atom_mask_prob=config['training'].get('atom_mask_prob', 0.0))

    def _make_loader(df):
        ds = DrugGeneDataset(
            df[drug_col].tolist(), df[cell_col].tolist(),
            df[label_col].values, drug_atom_dict, gene_emb_dict, drug_global_dict
        )
        return torch.utils.data.DataLoader(
            ds, batch_size=args.batch_size, shuffle=False,
            collate_fn=collate, num_workers=args.num_workers
        )

    val_loader = _make_loader(val_df)
    test_loader = _make_loader(test_df)

    # ---------- 3. model + checkpoint ----------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
    state = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
    # the number of fusion modalities can be inferred from the gate_mlp output width (4 = a 4th atom modality);
    # the query source (atom/morgan) and the residual flag cannot be seen in the state_dict, pass them on the command line.
    _n_gate = next((int(tuple(v.shape)[0]) for k, v in state.items()
                    if k.endswith('gate_mlp.2.weight')), 3)
    if _n_gate == 4:
        config['model']['add_atom_modality'] = True
        print('[info] 4-modality checkpoint detected -> add_atom_modality=True')
    elif _n_gate == 3 and config['model']['drop_drug_global_modality']:
        # still 3 modalities after dropping drug_global => they are [gene_pooled, global_gene, drug_atom]
        config['model']['add_atom_modality'] = True
        print('[info] 3 modalities + drop_drug_global -> add_atom_modality=True')
    print(f'[info] query_source={config["model"]["query_source"]}  '
          f'no_residual={config["model"]["no_residual"]}  '
          f'drop_drug_global={config["model"]["drop_drug_global_modality"]}  (must match training)')
    model = create_model('attention', config).to(device)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as e:
        print('\n[ERROR] state_dict does not match strictly. the usual cause is --reg_norm differing from training')
        print('        (BatchNorm and LayerNorm differ in parameter names/count), or a wrong --layer/--gene_version.')
        print('        original error:')
        raise
    model.eval()
    print(f'Loaded checkpoint: {args.model_path}')

    # ---------- 4. inference ----------
    def _infer(loader):
        ps, ts = [], []
        with torch.no_grad():
            for atom, atom_mask, gene, gene_mask, labels, drug_global in loader:
                out = model(gene.to(device), atom.to(device), atom_mask.to(device),
                            gene_mask.to(device), drug_global=drug_global.to(device))
                ps.extend(out.detach().cpu().float().numpy())
                ts.extend(labels.numpy())
        return np.array(ts, dtype=float), np.array(ps, dtype=float)

    y_test, p_test = _infer(test_loader)
    y_val, p_val = _infer(val_loader)

    m_raw = compute_metrics(y_test, p_test)
    m_val = compute_metrics(y_val, p_val)

    # recalibration: fit y ~ a*p + b on the validation set (val labels only, test labels untouched)
    #   a = cov(p_val, y_val) / var(p_val)
    #   b = mean(y_val) - a * mean(p_val)
    var_p = float(np.var(p_val))
    a = float(np.cov(p_val, y_val, bias=True)[0, 1] / var_p) if var_p > 1e-12 else 1.0
    b = float(y_val.mean() - a * p_val.mean())
    m_cal = compute_metrics(y_test, a * p_test + b)

    print('\n' + '=' * 78)
    print(f"[VAL ] raw      : R²={m_val['r2']:+.4f}  Pearson={m_val['pearson']:+.4f}  "
          f"MAE={m_val['mae']:.4f}")
    print(f"[TEST] raw      : R²={m_raw['r2']:+.4f}  RMSE={m_raw['rmse']:.4f}  "
          f"Pearson={m_raw['pearson']:+.4f}  MAE={m_raw['mae']:.4f}")
    print(f"[TEST] rescaled : R²={m_cal['r2']:+.4f}  RMSE={m_cal['rmse']:.4f}  "
          f"MAE={m_cal['mae']:.4f}")
    print(f"       (a={a:+.4f}, b={b:+.4f}, fitted on validation; Pearson²={m_raw['pearson'] ** 2:+.4f} is the recalibration ceiling)")
    print('=' * 78)

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        tag = os.path.splitext(os.path.basename(args.model_path))[0]

        val_out = val_df[[drug_col, cell_col, label_col]].copy().reset_index(drop=True)
        val_out = val_out.rename(columns={label_col: 'true_label'})
        val_out['predicted'] = p_val
        val_path = os.path.join(args.output_dir, f'val_predictions_{tag}.csv')
        val_out.to_csv(val_path, index=False)
        print(f'Val predictions saved to : {val_path}')

        out_df = test_df[[drug_col, cell_col, label_col]].copy().reset_index(drop=True)
        out_df = out_df.rename(columns={label_col: 'true_label'})
        out_df['predicted'] = p_test
        out_df['predicted_rescaled'] = a * p_test + b
        csv_path = os.path.join(args.output_dir, f'test_predictions_{tag}.csv')
        out_df.to_csv(csv_path, index=False)
        print(f'Test predictions saved to: {csv_path}')


if __name__ == '__main__':
    main()
