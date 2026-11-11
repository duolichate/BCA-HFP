# bca_hfp/analysis/control_stability.py
"""R3-9 control two: attention stability across drugs / cell lines / random seeds.

the reviewer asked to show that the interpretable pattern (MG-132 -> proteasome genes etc.) is not an
accident of one random seed or one cell line. three axes:
  seeds      : same (drug, cell) sample, Spearman between the per-gene attention vectors of different seed checkpoints.
  cell_lines : same drug, several cell lines, is the attention rank of the target genes stably near the top.
  drugs      : same cell line, several drugs, is the target-gene rank high for MG-132 only.

writes intermediate CSVs, no figures. per-gene attention = drug->gene attention (averaged over the 8 heads)
summed over the atom axis, same definition as gene_attn in fig_drug2gene.py.

usage:
  seeds axis:
    python bca_hfp/analysis/control_stability.py --axis seeds \
      --model_paths <seed42_fold>/best_val.pth,<seed43_fold>/best_val.pth \
      --drug_id 1862 --cell BC-1 --output_dir <dir> --h5 ... --gene_feat_dir ...
  cell lines axis:
    python ... --axis cell_lines --drug_id 1862 --output_dir <dir> ...
  drugs axis:
    python ... --axis drugs --cell BC-1 --output_dir <dir> ...
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from bca_hfp.analysis.control_utils import (load_model_from_ckpt, load_control_data, run_attention,
                                        gene_importance, load_symbol_maps, load_cell_gene_orders,
                                        resolve_cell_id, norm_id)

DEFAULT_GENES = 'PSMB5,PSMB2,PSMA2,PSMB1,RAB21,SURF6'


def importance_vector(model, drug_atom, gene_emb, drug_global, drug_id, cell_id, device):
    _, d2g, _g2d = run_attention(model, drug_atom, gene_emb, drug_global, drug_id, cell_id, device)
    return gene_importance(d2g)  # (n_genes,)


def gene_rank_vec(imp):
    """turn per-gene attention into ranks (1 = largest attention); ranks are more outlier-resistant."""
    s = pd.Series(imp)
    return s.rank(ascending=False, method='average').values


def main():
    ap = argparse.ArgumentParser(description='attention stability across drugs / cell lines / seeds (R3-9b)')
    ap.add_argument('--axis', required=True, choices=['seeds', 'cell_lines', 'drugs'])
    ap.add_argument('--model_path', default='', help='single model (for the cell_lines/drugs axis)')
    ap.add_argument('--model_paths', default='', help='comma separated seed models (for the seeds axis)')
    ap.add_argument('--h5', default='')
    ap.add_argument('--gene_feat_dir', default='')
    ap.add_argument('--drug_id', default='1862', help='target drug (seeds/cell_lines axis)')
    ap.add_argument('--cell', default='', help='target cell line (seeds/drugs axis)')
    ap.add_argument('--n_cells', type=int, default=0, help='max cell lines on the cell_lines axis, 0 = no limit')
    ap.add_argument('--n_drugs', type=int, default=0, help='max drugs on the drugs axis, 0 = no limit')
    ap.add_argument('--genes', default=DEFAULT_GENES, help='target gene symbols, comma separated')
    ap.add_argument('--top_k', type=int, default=20, help='threshold for counting a rank as near the top')
    ap.add_argument('--output_dir', required=True)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    sym2ens, _ = load_symbol_maps()
    target_ens = {s: sym2ens.get(s.strip().upper()) for s in args.genes.split(',') if s.strip()}
    target_ens = {k: v for k, v in target_ens.items() if v}

    if args.axis == 'seeds':
        run_seeds(args, target_ens, args.top_k)
    elif args.axis == 'cell_lines':
        run_cell_lines(args, target_ens, args.top_k)
    else:
        run_drugs(args, target_ens, args.top_k)


def run_seeds(args, target_ens, top_k):
    paths = [p for p in args.model_paths.split(',') if p.strip()]
    if len(paths) < 2:
        raise SystemExit('the seeds axis needs at least two models in --model_paths')

    model0, config, device, _ = load_model_from_ckpt(paths[0])
    drug_atom, gene_emb, drug_global, response, cols = load_control_data(
        config, args.h5 or None, args.gene_feat_dir or None)
    cell_col, drug_col, _ = cols
    cell_id = resolve_cell_id(response, cell_col, args.cell)
    drug_id = norm_id(args.drug_id)
    order_map = load_cell_gene_orders(args.gene_feat_dir or config['data']['gene_feat_dir'],
                                      cell_col, only_cells={norm_id(cell_id)})
    gene_ensgs = order_map.get(norm_id(cell_id))
    _, ens2sym = load_symbol_maps()

    vecs = {}
    for p in paths:
        m, _, _, _ = load_model_from_ckpt(p)
        vecs[p] = importance_vector(m, drug_atom, gene_emb, drug_global, drug_id, cell_id, device)

    mat = pd.DataFrame(vecs, index=gene_ensgs)
    mat.index.name = 'gene_ensg'
    mat.insert(0, 'symbol', [ens2sym.get(norm_id(g), g) for g in mat.index])
    mat.to_csv(os.path.join(args.output_dir, f'seeds_attn_{drug_id}_{cell_id}.csv'))

    cols = list(vecs.keys())
    rho_rows = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            r, _ = spearmanr(mat[a], mat[b])
            rho_rows.append((os.path.basename(a).split('_')[0], os.path.basename(b).split('_')[0], r, a, b))
    rho_df = pd.DataFrame(rho_rows, columns=['model_a', 'model_b', 'spearman', 'path_a', 'path_b'])
    rho_df.to_csv(os.path.join(args.output_dir, 'seeds_spearman.csv'), index=False)
    print('per-gene attention Spearman between seeds:')
    print(rho_df[['model_a', 'model_b', 'spearman']].to_string(index=False))


def run_cell_lines(args, target_ens, top_k):
    model, config, device, _ = load_model_from_ckpt(args.model_path)
    drug_atom, gene_emb, drug_global, response, cols = load_control_data(
        config, args.h5 or None, args.gene_feat_dir or None)
    cell_col, drug_col, _ = cols
    drug_id = norm_id(args.drug_id)
    order_map = load_cell_gene_orders(args.gene_feat_dir or config['data']['gene_feat_dir'], cell_col)

    cells = [c for c, g in order_map.items() if c in gene_emb]
    sub = response[response[drug_col].astype(str).apply(norm_id) == drug_id]
    cells = [c for c in cells if c in set(sub[cell_col].astype(str).apply(norm_id))]
    if args.n_cells > 0:
        cells = cells[:args.n_cells]
    print(f'cell_lines axis: drug {drug_id} covers {len(cells)} cell lines')

    rows = []
    for c in cells:
        imp = importance_vector(model, drug_atom, gene_emb, drug_global, drug_id, c, device)
        gene_ensgs = order_map[c]
        rank = gene_rank_vec(imp)
        pos = {norm_id(g): r for g, r in zip(gene_ensgs, rank)}
        for sym, ens in target_ens.items():
            if norm_id(ens) in pos:
                rows.append((c, sym, pos[norm_id(ens)]))

    df = pd.DataFrame(rows, columns=['cell_id', 'gene', 'attn_rank'])
    df.to_csv(os.path.join(args.output_dir, 'cell_lines_target_rank.csv'), index=False)

    summ = []
    for sym in target_ens:
        g = df[df['gene'] == sym]
        if len(g) == 0:
            continue
        summ.append({
            'gene': sym, 'n_cells': len(g), 'median_rank': float(g['attn_rank'].median()),
            'mean_rank': float(g['attn_rank'].mean()),
            'top_k_hit_frac': float((g['attn_rank'] <= top_k).mean()),
        })
    summ = pd.DataFrame(summ).sort_values('median_rank')
    summ.to_csv(os.path.join(args.output_dir, 'cell_lines_summary.csv'), index=False)
    print('target-gene attention rank across cell lines (top_k=%d):' % top_k)
    print(summ.to_string(index=False))


def run_drugs(args, target_ens, top_k):
    model, config, device, _ = load_model_from_ckpt(args.model_path)
    drug_atom, gene_emb, drug_global, response, cols = load_control_data(
        config, args.h5 or None, args.gene_feat_dir or None)
    cell_col, drug_col, _ = cols
    cell_id = resolve_cell_id(response, cell_col, args.cell)
    order_map = load_cell_gene_orders(args.gene_feat_dir or config['data']['gene_feat_dir'],
                                      cell_col, only_cells={norm_id(cell_id)})
    gene_ensgs = order_map.get(norm_id(cell_id))
    _, ens2sym = load_symbol_maps()

    sub = response[response[cell_col].astype(str).apply(norm_id) == norm_id(cell_id)]
    drugs = [norm_id(d) for d in sorted(set(sub[drug_col].astype(str).apply(norm_id)))
             if norm_id(d) in drug_atom and norm_id(d) in drug_global]
    if args.n_drugs > 0:
        drugs = drugs[:args.n_drugs]
    print(f'drugs axis: cell {cell_id} covers {len(drugs)} drugs')

    ref = norm_id(args.drug_id)
    mat = {}
    for d in drugs:
        mat[d] = importance_vector(model, drug_atom, gene_emb, drug_global, d, cell_id, device)
    mat = pd.DataFrame(mat, index=gene_ensgs)
    mat.index.name = 'gene_ensg'
    mat.insert(0, 'symbol', [ens2sym.get(norm_id(g), g) for g in mat.index])
    mat.to_csv(os.path.join(args.output_dir, f'drugs_attn_{cell_id}.csv'))
    print('per-gene attention matrix written (rows=genes, columns=drugs), for local heatmaps')

    rank_mat = mat.drop(columns=['symbol']).rank(ascending=False, method='average')
    rows = []
    if ref in rank_mat.columns:
        for sym, ens in target_ens.items():
            e = norm_id(ens)
            if e in rank_mat.index:
                rows.append({
                    'gene': sym, 'rank_in_ref': float(rank_mat.loc[e, ref]),
                    'median_rank_other_drugs': float(rank_mat.loc[e, rank_mat.columns != ref].median()),
                    'min_rank_other_drugs': float(rank_mat.loc[e, rank_mat.columns != ref].min()),
                })
    tgt = pd.DataFrame(rows)
    tgt.to_csv(os.path.join(args.output_dir, 'drugs_target_rank.csv'), index=False)
    print('target-gene rank (ref=%s, 1 = largest attention):' % ref)
    print(tgt.to_string(index=False))

    # overall agreement with the other drugs (shows whether the pattern is drug specific)
    if ref in rank_mat.columns:
        others = [c for c in rank_mat.columns if c != ref]
        rho_list = []
        for o in others:
            r, _ = spearmanr(rank_mat[ref], rank_mat[o])
            rho_list.append(r)
        print(f'attention-rank Spearman of the reference drug {ref} against the other {len(others)} drugs: '
              f'mean={np.mean(rho_list):+.3f}  std={np.std(rho_list):.3f}')


if __name__ == '__main__':
    main()