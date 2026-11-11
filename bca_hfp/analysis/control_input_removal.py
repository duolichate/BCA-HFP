# bca_hfp/analysis/control_input_removal.py
"""R3-9 control three: input-removal test.

remove key inputs (genes / drug atoms) from the sample one at a time (position zeroed + mask False), observe
the prediction shift, and compare against removing the same number of random genes; if the shift caused by a
key gene/atom is clearly larger than the random control, its contribution is not generic or replaceable.
CSVs only, no figures.

removing a gene = zero its GeneFormer embedding and set gene_mask=False, so it drops out of the key/value of
the cross-attention and of the global_gene mean. atoms work the same way. drug_global (Morgan fingerprint)
does not change when an atom is removed, so a single atom's shift is diluted by the drug_global-dominated
baseline.

usage (MG-132 x BC-1):
  python bca_hfp/analysis/control_input_removal.py \
    --model_path <...>/attention_drug_fold2_best_val.pth \
    --drug_id 1862 --cell BC-1 --genes RAB21,SURF6,PSMB5,PSMB2 \
    --top_genes 10 --top_atoms 0 --n_ctrl 20 --output_dir <dir> \
    --h5 ... --gene_feat_dir ...
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from bca_hfp.analysis.control_utils import (load_model_from_ckpt, load_control_data, run_attention,
                                        gene_importance, load_symbol_maps, load_cell_gene_orders,
                                        resolve_cell_id, norm_id)


def main():
    ap = argparse.ArgumentParser(description='input-removal test (R3-9c)')
    ap.add_argument('--model_path', required=True)
    ap.add_argument('--h5', default='')
    ap.add_argument('--gene_feat_dir', default='')
    ap.add_argument('--drug_id', required=True, help='drug DRUG_ID (MG-132=1862)')
    ap.add_argument('--cell', required=True, help='cell line (CELL_LINES or GDSC_ID)')
    ap.add_argument('--genes', default='RAB21,SURF6,PSMB5,PSMB2,PSMA2',
                    help='key gene symbols to remove, comma separated')
    ap.add_argument('--top_genes', type=int, default=10, help='also append the top-k genes by attention')
    ap.add_argument('--top_atoms', type=int, default=0, help='remove the top-k atoms by attention, 0 = skip')
    ap.add_argument('--n_ctrl', type=int, default=20, help='number of random control genes')
    ap.add_argument('--seed', type=int, default=0, help='sampling seed for the random control')
    ap.add_argument('--output_dir', required=True)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    model, config, device, ckpt = load_model_from_ckpt(args.model_path)
    drug_atom, gene_emb, drug_global, response, cols = load_control_data(
        config, args.h5 or None, args.gene_feat_dir or None)
    cell_col, drug_col, label_col = cols

    cell_id = resolve_cell_id(response, cell_col, args.cell)
    drug_id = norm_id(args.drug_id)
    order_map = load_cell_gene_orders(args.gene_feat_dir or config['data']['gene_feat_dir'],
                                      cell_col, only_cells={norm_id(cell_id)})
    gene_ensgs = order_map.get(norm_id(cell_id))
    sym2ens, ens2sym = load_symbol_maps()

    # single-sample tensors
    atom0 = torch.tensor(drug_atom[drug_id], dtype=torch.float32).to(device)
    atom_mask0 = torch.ones(atom0.shape[0], dtype=torch.bool, device=device)
    gene0 = torch.tensor(gene_emb[cell_id], dtype=torch.float32).to(device)
    gene_mask0 = torch.ones(gene0.shape[0], dtype=torch.bool, device=device)
    dg0 = torch.tensor(drug_global[drug_id], dtype=torch.float32).to(device)

    def forward1(atom, atom_mask, gene, gene_mask, dg):
        with torch.no_grad():
            return model(gene.unsqueeze(0), atom.unsqueeze(0),
                         atom_mask.unsqueeze(0), gene_mask.unsqueeze(0),
                         drug_global=dg.unsqueeze(0)).item()

    pred_base, d2g, g2d = run_attention(model, drug_atom, gene_emb, drug_global, drug_id, cell_id, device)
    print(f'baseline pred={pred_base:.4f}  pred_clamp={bool(config["training"].get("pred_clamp", False))}')

    # target genes: given symbols + top-k by attention (deduplicated)
    ens2idx = {norm_id(g): i for i, g in enumerate(gene_ensgs)}
    targets = []  # (name, idx)
    seen_idx = set()
    skipped = []
    for sym in args.genes.split(','):
        sym = sym.strip().upper()
        ens = sym2ens.get(sym)
        if ens and norm_id(ens) in ens2idx:
            idx = ens2idx[norm_id(ens)]
            if idx not in seen_idx:
                targets.append((sym, idx)); seen_idx.add(idx)
        else:
            skipped.append(sym)
    if skipped:
        print(f'genes outside the top-2048 expressed genes of this cell, skipped: {skipped}')

    gene_imp = gene_importance(d2g)
    top_order = np.argsort(gene_imp)[::-1]
    for r, idx in enumerate(top_order[:args.top_genes], 1):
        if idx not in seen_idx:
            sym = ens2sym.get(norm_id(gene_ensgs[idx]), gene_ensgs[idx])
            targets.append((f'top{r}_{sym}', idx)); seen_idx.add(idx)

    rng = np.random.default_rng(args.seed)
    ctrl_idx = [i for i in range(len(gene_ensgs)) if i not in seen_idx]
    ctrl_pick = rng.choice(ctrl_idx, size=min(args.n_ctrl, len(ctrl_idx)), replace=False)

    rows = []
    for name, idx in targets:
        g, gm = gene0.clone(), gene_mask0.clone()
        g[idx] = 0.0; gm[idx] = False
        p = forward1(atom0, atom_mask0, g, gm, dg0)
        rows.append({'removed_type': 'gene', 'removed_name': name, 'pred_baseline': pred_base,
                     'pred_removed': p, 'delta': p - pred_base})

    for idx in ctrl_pick:
        g, gm = gene0.clone(), gene_mask0.clone()
        g[idx] = 0.0; gm[idx] = False
        p = forward1(atom0, atom_mask0, g, gm, dg0)
        sym = ens2sym.get(norm_id(gene_ensgs[idx]), gene_ensgs[idx])
        rows.append({'removed_type': 'gene_ctrl', 'removed_name': sym, 'pred_baseline': pred_base,
                     'pred_removed': p, 'delta': p - pred_base})

    if args.top_atoms > 0:
        # atom importance = gene->drug attention summed over the gene axis, then averaged over heads
        # (same direction as the atom map in Fig5). the drug->gene direction runs a softmax over genes per
        # row, so summing over genes is always 1 and carries no atom signal, g2d must be used.
        atom_imp = g2d.mean(axis=0).sum(axis=0)  # (n_atoms,)
        for r, idx in enumerate(np.argsort(atom_imp)[::-1][:args.top_atoms], 1):
            a, am = atom0.clone(), atom_mask0.clone()
            a[idx] = 0.0; am[idx] = False
            p = forward1(a, am, gene0, gene_mask0, dg0)
            rows.append({'removed_type': 'atom', 'removed_name': f'atom_{idx}', 'pred_baseline': pred_base,
                         'pred_removed': p, 'delta': p - pred_base})

        atom_rng = np.random.default_rng(args.seed + 1)
        atom_ctrl_pick = atom_rng.choice(len(atom_imp), size=min(args.n_ctrl, len(atom_imp)), replace=False)
        for idx in atom_ctrl_pick:
            a, am = atom0.clone(), atom_mask0.clone()
            a[idx] = 0.0; am[idx] = False
            p = forward1(a, am, gene0, gene_mask0, dg0)
            rows.append({'removed_type': 'atom_ctrl', 'removed_name': f'atom_{idx}',
                         'pred_baseline': pred_base, 'pred_removed': p, 'delta': p - pred_base})

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.output_dir, f'input_removal_{drug_id}_{cell_id}.csv'), index=False)

    # short summary
    gene_rm = df[df['removed_type'] == 'gene']
    ctrl = df[df['removed_type'] == 'gene_ctrl']
    print('=' * 60)
    print('|Δpred| after gene removal:')
    if len(gene_rm):
        print(gene_rm.assign(abs_delta=gene_rm['delta'].abs())
              .sort_values('abs_delta', ascending=False)[['removed_name', 'delta', 'abs_delta']]
              .to_string(index=False))
    if len(ctrl):
        print(f'random control |Δpred| mean={ctrl["delta"].abs().mean():.4f}  '
              f'max={ctrl["delta"].abs().max():.4f}  (n={len(ctrl)})')
    if 'atom' in set(df['removed_type']):
        at = df[df['removed_type'] == 'atom']
        print(f'atom removal |Δpred| mean={at["delta"].abs().mean():.4f} (n={len(at)})')
    print(f'\ndetail: {os.path.join(args.output_dir, f"input_removal_{drug_id}_{cell_id}.csv")}')


if __name__ == '__main__':
    main()