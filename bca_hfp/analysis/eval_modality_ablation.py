# bca_hfp/analysis/eval_modality_ablation.py
"""inference only: export per-sample gate weights + the metric drop when each of the three modalities is removed (R3-3 / interpretability evidence).

no training, no hyperparameter change, no new checkpoint. the model structure is restored from the config
saved inside the checkpoint, the test drug set of that fold is reproduced with the same GroupKFold rule as
training, then:

  1) normal forward pass, record the gate weights of every GatedFusionLayer (per sample) and the predictions;
  2) remove gene_pooled / global_gene / drug_global one at a time via the modality_present mask
     (the gate weight is exactly 0, the features are not zeroed), recompute the predictions;
  3) post-process all predictions with the training concentration window (pred_clamp), then compute R2/RMSE/Pearson/Spearman.

output:
  gate_weights_<which>.csv       per sample: drug / cell line / label / prediction / per-layer per-modality weights
  modality_ablation_<which>.csv  metrics per condition + difference against full
  modality_ablation_summary.csv  summary (gate mean/std/argmax share)

usage (E8 scaffold fold 3):
python bca_hfp/analysis/eval_modality_ablation.py \
  --model_path /home/xxin/my_model/Result/My_model/CV/GDSC_attention_3l_V1_scaffold_fp2048_batchnorm_md0_seed42_cv5_ne20_lrfix_swa5_desc_pc_gbn_mse_dml0.5_e8scaf/models/attention_scaffold_fold3_best_val.pth \
  --fold 3 --split_method scaffold --output_dir /home/xxin/my_model/Result/diag/modality_ablation_fold3

smoke test on a small sample first (200 rows):
  same as above without --output_dir, add --limit 200
"""
import argparse
import os
import sys
from functools import partial

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from bca_hfp.data.dataset import DrugGeneDataset, collate_fn
from bca_hfp.data.loader import (load_drug_atom_features, load_gene_embeddings,
                                 load_morgan_fingerprints, load_response)
from bca_hfp.models import create_model
from bca_hfp.utils.reproducibility import set_seed


def build_scaffold_groups(filtered_df, drug_col, h5_path):
    """same scaffold split as train_cv.py: Bemis-Murcko scaffold grouping."""
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    ds = pd.read_hdf(h5_path, 'drug_smiles')
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
    vc = pd.Series(list(scaffold_group.values())).value_counts()
    n_nos = sum(1 for v in scaffold_group.values() if str(v).startswith('__no_scaffold__'))
    print(f'scaffold split: {len(scaffold_group)} drugs -> {len(vc)} scaffold groups, '
          f'largest group = {int(vc.iloc[0])} drugs')
    if n_nos:
        print(f'  [warn] {n_nos}/{len(scaffold_group)} drugs have no usable scaffold, each became its own group')
    if n_nos > 0.5 * len(scaffold_group):
        raise RuntimeError('scaffold split failed: more than half of the drugs have no Bemis-Murcko scaffold')
    return filtered_df[drug_col].map(scaffold_group).values


def add_cens_columns(df, label_col, use_cens):
    """reproduce the censoring / concentration window fields of train_cv.py (b_lo, b_hi, cens_flag, cens_w)."""
    if not use_cens:
        return df
    need = {'MIN_CONC', 'MAX_CONC', 'RMSE'}
    if not need.issubset(set(df.columns)):
        raise SystemExit('response table lacks MIN_CONC / MAX_CONC / RMSE, cannot reproduce the window clamp')
    lo = pd.to_numeric(df['MIN_CONC'], errors='coerce').values.astype(np.float64)
    hi = pd.to_numeric(df['MAX_CONC'], errors='coerce').values.astype(np.float64)
    y = pd.to_numeric(df[label_col], errors='coerce').values.astype(np.float64)
    rmse = pd.to_numeric(df['RMSE'], errors='coerce').values.astype(np.float64)
    ok = np.isfinite(lo) & np.isfinite(hi) & (lo > 0) & (hi > 0)
    b_lo = np.where(ok, np.log(np.maximum(lo, 1e-12)), -1e6)
    b_hi = np.where(ok, np.log(np.maximum(hi, 1e-12)), 1e6)
    flag = np.zeros(len(df), dtype=np.float32)
    flag[ok & (y >= b_hi)] = 1.0
    flag[ok & (y <= b_lo)] = -1.0
    w_raw = 1.0 / (1.0 + (np.nan_to_num(rmse, nan=0.1) / 0.1) ** 2)
    w = np.clip(w_raw / w_raw.mean(), 0.2, 2.0)
    df = df.copy()
    df['b_lo'] = b_lo
    df['b_hi'] = b_hi
    df['cens_flag'] = flag
    df['cens_w'] = w.astype(np.float32)
    return df


def clamp_pred(pred, cens, enabled):
    """same as VariableLengthTrainer._clamp_pred: per-row window ±2, then clamp to [ln0.001, ln100]."""
    if not enabled or cens is None:
        return pred
    out = torch.max(torch.min(pred, cens[:, 1] + 2.0), cens[:, 0] - 2.0)
    return torch.clamp(out, min=float(np.log(0.001)), max=float(np.log(100.0)))


def metrics(y, p):
    return {
        'n': int(len(y)),
        'R2': float(r2_score(y, p)),
        'RMSE': float(np.sqrt(mean_squared_error(y, p))),
        'MAE': float(mean_absolute_error(y, p)),
        'Pearson': float(pearsonr(y, p)[0]),
        'Spearman': float(spearmanr(y, p)[0]),
    }


def main():
    ap = argparse.ArgumentParser(description='Eval-only: gate weights + modality ablation')
    ap.add_argument('--model_path', required=True, help='*_best_val.pth of one fold (contains config)')
    ap.add_argument('--fold', type=int, required=True, help='1-based fold index')
    ap.add_argument('--n_splits', type=int, default=5)
    ap.add_argument('--split_method', type=str, default='scaffold',
                    choices=['scaffold', 'drug'], help='must match training')
    ap.add_argument('--no_group_by_name', action='store_true',
                    help='disable grouping by drug name (on by default, as in E7/E8)')
    ap.add_argument('--fold_csv', type=str, default='',
                    help='per-drug fold table used in training (GROUP,DRUG_ID,fold); when given it is used as is, GroupKFold is not recomputed')
    ap.add_argument('--which', type=str, default='test', choices=['test'],
                    help='only the test fold is supported (leave-drug / leave-scaffold evaluation)')
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--limit', type=int, default=0, help='when >0 run only the first N rows, for a smoke test')
    ap.add_argument('--clamp', type=str, default='auto', choices=['auto', 'on', 'off'],
                    help='auto=follow the pred_clamp setting of the checkpoint')
    ap.add_argument('--modalities', type=str, default='gene_pooled,global_gene,drug_global',
                    help='modality names to remove one by one, comma separated')
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--gate_mode', type=str, default='learned',
                    choices=['learned', 'uniform', 'frozen', 'both'],
                    help='add fixed-gate conditions: uniform=equal weights; frozen=constants from --gate_frozen; both=both')
    ap.add_argument('--gate_frozen', type=str, default='',
                    help='per-modality constants for the frozen mode, comma separated (same order as the gate_weights CSV columns)')
    ap.add_argument('--h5', default='', help='override drug_feat_path from the config (optional)')
    ap.add_argument('--gene_feat_dir', default='', help='override gene_feat_dir from the config (optional)')
    ap.add_argument('--save_preds', action='store_true',
                    help='also save per-sample predictions of every condition (for the paired per-drug comparison)')
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(0)

    ckpt = torch.load(args.model_path, map_location='cpu', weights_only=False)
    config = ckpt.get('config') if isinstance(ckpt, dict) else None
    if config is None:
        # SWA and final store weights only; reuse a checkpoint with a config in the same dir (best_val preferred)
        here = os.path.dirname(os.path.abspath(args.model_path))
        cands = sorted(fn for fn in os.listdir(here)
                       if fn.endswith('.pth') and fn != os.path.basename(args.model_path))
        cands.sort(key=lambda fn: (0 if 'best_val' in fn else 1, fn))
        for fn in cands:
            other = torch.load(os.path.join(here, fn), map_location='cpu', weights_only=False)
            if isinstance(other, dict) and other.get('config'):
                config = other['config']
                print(f'  [info] {os.path.basename(args.model_path)} has no config, reuse {fn} from the same dir')
                break
        if config is None:
            raise SystemExit(f'{args.model_path} has no config, and no .pth with a config in the same dir')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        free, total = torch.cuda.mem_get_info()
        print(f'GPU free {free / 2**30:.1f} GiB / {total / 2**30:.1f} GiB')

    data_cfg = config['data']
    drug_col, cell_col, label_col = data_cfg['drug_id_col'], data_cfg['cell_id_col'], data_cfg['label_col']
    h5_path = args.h5 or data_cfg['drug_feat_path']
    use_desc = int(data_cfg['drug_global_dim']) > 2048
    fp_bits = 2048 if use_desc else int(data_cfg['drug_global_dim'])
    pred_clamp = bool(config['training'].get('pred_clamp', False))
    if args.clamp != 'auto':
        pred_clamp = (args.clamp == 'on')
    print(f'checkpoint epoch={ckpt.get("epoch")} '
          f'query_source={config["model"].get("query_source")} '
          f'add_atom={config["model"].get("add_atom_modality")} '
          f'drop_drug_global={config["model"].get("drop_drug_global_modality")} '
          f'num_fusion_layers={config["model"].get("num_fusion_layers")} '
          f'use_desc={use_desc} fp_bits={fp_bits} pred_clamp={pred_clamp}')

    # ---------- data ----------
    print('Loading data...')
    drug_atom_dict = load_drug_atom_features(h5_path)
    gene_emb_dict, _ = load_gene_embeddings(args.gene_feat_dir or data_cfg['gene_feat_dir'], cell_col)
    response_df = load_response(h5_path, cell_col, drug_col)
    drug_global_dict = load_morgan_fingerprints(h5_path, fp_bits=fp_bits, fp_radius=2,
                                                use_desc=use_desc)

    valid = (response_df[drug_col].isin(drug_atom_dict)
             & response_df[cell_col].isin(gene_emb_dict))
    filtered_df = response_df[valid].copy()
    filtered_df['_group'] = (filtered_df['DRUG_NAME'].astype(str).str.strip().str.upper()
                             if not args.no_group_by_name else filtered_df[drug_col])
    filtered_df = add_cens_columns(filtered_df, label_col, use_cens=True)

    # ---------- reproduce the test set of that fold ----------
    if args.fold_csv:
        # same as train_cv.py: fold numbers map by the grouping key (DRUG_NAME by default), not by DRUG_ID
        fm = pd.read_csv(args.fold_csv)
        key = 'GROUP' if 'GROUP' in fm.columns else ('DRUG_NAME' if 'DRUG_NAME' in fm.columns
                                                    else drug_col)
        fm[key] = fm[key].astype(str).str.strip().str.upper()
        g2f = dict(zip(fm[key], fm['fold'].astype(int)))
        _keys = (filtered_df['_group'].astype(str).str.strip().str.upper()
                 if not args.no_group_by_name else filtered_df[drug_col].astype(str))
        _fold = _keys.map(g2f)
        n_miss = int(_fold.isna().sum())
        if n_miss:
            raise SystemExit(f'fold_csv does not cover {n_miss} rows (check grouping key {key} against --group_by_name)')
        test_idx = np.where(_fold.values == args.fold)[0]
    elif args.split_method == 'scaffold':
        groups = build_scaffold_groups(filtered_df, drug_col, h5_path)
        splits = list(GroupKFold(n_splits=args.n_splits).split(filtered_df, groups=groups))
        _, test_idx = splits[args.fold - 1]
    else:
        groups = (filtered_df['_group'] if not args.no_group_by_name
                  else filtered_df[drug_col]).values
        splits = list(GroupKFold(n_splits=args.n_splits).split(filtered_df, groups=groups))
        _, test_idx = splits[args.fold - 1]
    eval_df = filtered_df.iloc[test_idx].copy()
    if args.limit > 0:
        eval_df = eval_df.iloc[:args.limit].copy()
    print(f'fold{args.fold} test: {len(eval_df)} samples, {eval_df[drug_col].nunique()} drugs, '
          f'{eval_df[cell_col].nunique()} cell lines')

    dataset = DrugGeneDataset(
        eval_df[drug_col].tolist(), eval_df[cell_col].tolist(),
        eval_df[label_col].values, drug_atom_dict, gene_emb_dict, drug_global_dict,
        cens_arr=eval_df[['b_lo', 'b_hi', 'cens_flag', 'cens_w']].values.astype(np.float32))
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=partial(collate_fn, with_cens=True), num_workers=args.num_workers)

    # ---------- model ----------
    model = create_model('attention', config).to(device)
    missing, unexpected = model.load_state_dict(ckpt['model_state_dict'], strict=False)
    if unexpected:
        raise RuntimeError(f'checkpoint has unmatched parameters: {unexpected[:5]}')
    if missing:
        print(f'  [warn] {len(missing)} missing parameters: {missing[:5]}')
    model.eval()

    mods_to_drop = [m.strip() for m in args.modalities.split(',') if m.strip()]
    gate_extra = {'uniform': ['gate_uniform'], 'frozen': ['gate_frozen'],
                  'both': ['gate_uniform', 'gate_frozen']}.get(args.gate_mode, [])
    conditions = ([('full', None)] + [(g, None) for g in gate_extra]
                  + [(f'no_{m}', {m}) for m in mods_to_drop])
    frozen_vec = None
    if args.gate_mode in ('frozen', 'both'):
        frozen_vec = [float(v) for v in args.gate_frozen.split(',') if v.strip()]
        if not frozen_vec:
            raise SystemExit('--gate_mode frozen/both needs --gate_frozen')

    preds = {name: [] for name, _ in conditions}
    gate_rows = []
    gate_names = None

    with torch.no_grad():
        for batch in loader:
            atom, atom_mask, gene, gene_mask, labels, drug_global, cens = batch
            atom, atom_mask = atom.to(device), atom_mask.to(device)
            gene, gene_mask = gene.to(device), gene_mask.to(device)
            drug_global, cens = drug_global.to(device), cens.to(device)
            for name, drops in conditions:
                # fixed-gate conditions: override only in the matching condition, others keep the learned gates
                for layer in model.fusion_layers:
                    layer.gate_override = None
                if name == 'gate_uniform':
                    for layer in model.fusion_layers:
                        layer.gate_override = 'uniform'
                elif name == 'gate_frozen':
                    for layer in model.fusion_layers:
                        layer.gate_override = frozen_vec
                out = model(gene, atom, atom_mask, gene_mask, drug_global=drug_global,
                            drop_modalities=drops)
                out = clamp_pred(out, cens, pred_clamp)
                preds[name].extend(out.cpu().float().numpy())
                if name == 'full':
                    gate_names = list(model._last_modality_names)
                    if frozen_vec is not None and len(frozen_vec) != len(gate_names):
                        raise SystemExit(f'--gate_frozen length {len(frozen_vec)} does not match '
                                         f'the number of modalities {len(gate_names)}: {gate_names}')
                    for layer_idx, layer in enumerate(model.fusion_layers):
                        if layer.last_gate_weights is None:
                            continue
                        w = layer.last_gate_weights.numpy()
                        for i in range(w.shape[0]):
                            row = {'layer': layer_idx}
                            for j, mn in enumerate(gate_names):
                                row[mn] = float(w[i, j])
                            gate_rows.append(row)

    # ---------- per-sample CSV ----------
    gate_df = pd.DataFrame(gate_rows)
    base = pd.DataFrame({
        'DRUG_ID': eval_df[drug_col].values,
        'GDSC_ID': eval_df[cell_col].values,
        'true_label': eval_df[label_col].values,
        'predicted': preds['full'],
        'fold': args.fold,
    })
    if gate_names:
        gate_cols = [c for c in gate_df.columns if c != 'layer']
        for c in gate_cols:
            base[c] = gate_df[c].values
        base['argmax_modality'] = gate_df[gate_cols].values.argmax(axis=1)
        base['argmax_modality'] = base['argmax_modality'].map(
            {i: n for i, n in enumerate(gate_cols)})
    gate_path = os.path.join(args.output_dir, f'gate_weights_{args.which}_fold{args.fold}.csv')
    base.to_csv(gate_path, index=False)

    if args.save_preds:
        wide = pd.DataFrame({
            'DRUG_ID': eval_df[drug_col].values,
            'GROUP': eval_df['_group'].values,
            'GDSC_ID': eval_df[cell_col].values,
            'fold': args.fold,
            'true_label': eval_df[label_col].values,
        })
        for name, _ in conditions:
            wide['pred_' + name] = np.asarray(preds[name])
        pred_path = os.path.join(args.output_dir, f'preds_all_conditions_fold{args.fold}.csv')
        wide.to_csv(pred_path, index=False)
        print(f'saved per-sample predictions: {pred_path}')

    # ---------- metrics ----------
    y = base['true_label'].values
    rows = []
    for name, _ in conditions:
        m = metrics(y, np.asarray(preds[name]))
        m['condition'] = name
        if name != 'full':
            m['dR2'] = m['R2'] - rows[0]['R2']
            m['dRMSE'] = m['RMSE'] - rows[0]['RMSE']
            m['dPearson'] = m['Pearson'] - rows[0]['Pearson']
            m['mean_abs_delta_pred'] = float(
                np.mean(np.abs(np.asarray(preds[name]) - np.asarray(preds['full']))))
        else:
            m['dR2'] = 0.0
            m['dRMSE'] = 0.0
            m['dPearson'] = 0.0
            m['mean_abs_delta_pred'] = 0.0
        rows.append(m)
    abl = pd.DataFrame(rows)[['condition', 'n', 'R2', 'dR2', 'RMSE', 'dRMSE',
                              'MAE', 'Pearson', 'dPearson', 'Spearman',
                              'mean_abs_delta_pred']]
    abl_path = os.path.join(args.output_dir, f'modality_ablation_{args.which}_fold{args.fold}.csv')
    abl.to_csv(abl_path, index=False)

    # ---------- summary ----------
    summary = {'fold': args.fold, 'n_samples': len(base),
               'n_drugs': int(base['DRUG_ID'].nunique()),
               'n_cells': int(base['GDSC_ID'].nunique())}
    if gate_names:
        for mn in gate_names:
            summary[f'gate_mean_{mn}'] = float(gate_df[mn].mean())
            summary[f'gate_std_{mn}'] = float(gate_df[mn].std())
            summary[f'gate_argmax_share_{mn}'] = float((base['argmax_modality'] == mn).mean())
    for m in rows:
        tag = m['condition']
        summary[f'R2_{tag}'] = m['R2']
        summary[f'dR2_{tag}'] = m['dR2']
    sum_path = os.path.join(args.output_dir, f'modality_ablation_{args.which}_summary.csv')
    pd.DataFrame([summary]).to_csv(sum_path, index=False)

    # ---------- print ----------
    print('\n' + '=' * 78)
    print(f'Gate weights (fold {args.fold}, mean over {len(base)} samples)')
    for mn in (gate_names or []):
        print(f'  {mn:<12} mean={summary[f"gate_mean_{mn}"]:.4f} '
              f'std={summary[f"gate_std_{mn}"]:.4f} '
              f'argmax={summary[f"gate_argmax_share_{mn}"] * 100:.1f}%')
    print('\nModality ablation (predictions post-clamped, same as training)')
    for m in rows:
        print(f"  {m['condition']:<18} R2={m['R2']:+.4f} (d={m['dR2']:+.4f})  "
              f"RMSE={m['RMSE']:.4f}  Pearson={m['Pearson']:.4f}  "
              f"dPred={m['mean_abs_delta_pred']:.4f}")
    print('=' * 78)
    print(f'saved:\n  {gate_path}\n  {abl_path}\n  {sum_path}')


if __name__ == '__main__':
    main()
