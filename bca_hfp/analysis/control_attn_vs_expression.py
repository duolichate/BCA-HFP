# bca_hfp/analysis/control_attn_vs_expression.py
"""R3-9 control one: gene attention vs gene expression rank.

the reviewer asked to rule out the trivial explanation that cross-attention merely tracks highly expressed
genes. for a single (drug, cell line) sample: forward pass, take the gene->drug per-gene attention, and
correlate it with the expression of each gene in that cell line (value + rank) using Spearman/Pearson.
a correlation near 0 means attention is not a simple mirror of expression. key genes such as RAB21/SURF6
and the proteasome (PSMB*) are listed as well, as controls. CSVs only, no figures.

usage (MG-132 x BC-1, with the E7 fold2 model):
  python bca_hfp/analysis/control_attn_vs_expression.py \
    --model_path <..._e7>/models/attention_drug_fold2_best_val.pth \
    --h5 <data dir>/GDSC_RES_FGBERT_full.h5 \
    --gene_feat_dir <data dir>/GDSC_EXP_GF_V1 \
    --expr_csv <data dir>/GDSC_EXP_ensembl.csv \
    --drug_id 1862 --cell BC-1 --output_dir <output dir>
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from bca_hfp.analysis.control_utils import (load_model_from_ckpt, load_control_data, run_attention,
                                        gene_importance, load_symbol_maps, load_cell_gene_orders,
                                        resolve_cell_id, norm_id, PROJECT_ROOT)
from bca_hfp.config.base_config import root_dir


def main():
    ap = argparse.ArgumentParser(description='gene attention vs expression rank (R3-9a)')
    ap.add_argument('--model_path', required=True, help='*_best_val.pth of one fold (contains config)')
    ap.add_argument('--h5', default='', help='override drug_feat_path from the config')
    ap.add_argument('--gene_feat_dir', default='', help='override gene_feat_dir from the config')
    ap.add_argument('--expr_csv', default=os.path.join(root_dir('Data'), 'Gene', 'GDSC_EXP_ensembl.csv'),
                    help='expression matrix CSV (rows=cell lines, columns=ENSG)')
    ap.add_argument('--drug_id', required=True, help='drug DRUG_ID (MG-132=1862)')
    ap.add_argument('--cell', required=True, help='cell line (GDSC_ID or CELL_LINES, e.g. BC-1)')
    ap.add_argument('--top_k', type=int, default=30, help='how many top attention genes to print')
    ap.add_argument('--genes', default='RAB21,SURF6,PSMB5,PSMB2,PSMA2',
                    help='extra gene symbols to annotate, comma separated')
    ap.add_argument('--output_dir', required=True)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    model, config, device, ckpt = load_model_from_ckpt(args.model_path)

    drug_atom, gene_emb, drug_global, response, cols = load_control_data(
        config, args.h5 or None, args.gene_feat_dir or None)
    cell_col, drug_col, label_col = cols

    cell_id = resolve_cell_id(response, cell_col, args.cell)
    if cell_id is None or cell_id not in gene_emb:
        raise SystemExit(f'no gene embedding for cell line {args.cell} (cell_id={cell_id})')
    drug_id = norm_id(args.drug_id)
    if drug_id not in drug_atom or drug_id not in drug_global:
        raise SystemExit(f'no features for drug {args.drug_id}')

    # gene order from that cell's own .pt (differs per cell)
    order_map = load_cell_gene_orders(args.gene_feat_dir or config['data']['gene_feat_dir'],
                                      cell_col, only_cells={norm_id(cell_id)})
    gene_ensgs = order_map.get(norm_id(cell_id))
    if gene_ensgs is None:
        raise SystemExit(f'no gene_names for cell {cell_id} in the .pt files; check gene_feat_dir')

    pred, d2g, g2d = run_attention(model, drug_atom, gene_emb, drug_global, drug_id, cell_id, device)
    imp = gene_importance(d2g)  # length should equal len(gene_ensgs)

    if len(imp) != len(gene_ensgs):
        raise SystemExit(f'attention has {len(imp)} genes but gene_names has {len(gene_ensgs)}')

    _, ens2sym = load_symbol_maps()

    # expression: read only the 2048 ENSG columns of that cell instead of the whole table
    header = pd.read_csv(args.expr_csv, nrows=0)
    avail_cols = [c for c in header.columns if c.startswith('ENSG')]
    need = ['GDSC_ID'] + [c for c in avail_cols if c in set(gene_ensgs)]
    expr = pd.read_csv(args.expr_csv, usecols=need)
    expr['GDSC_ID'] = expr['GDSC_ID'].astype(str).apply(norm_id)
    row = expr[expr['GDSC_ID'] == norm_id(cell_id)]
    if len(row) == 0:
        raise SystemExit(f'cell {cell_id} not found in the expression matrix {args.expr_csv}')

    expr_vals = {g: row.iloc[0][g] for g in gene_ensgs if g in row.columns}
    n_hit = len(expr_vals)
    print(f'{len(gene_ensgs)} genes in total, {n_hit} found in the expression matrix')

    rows = []
    for i, g in enumerate(gene_ensgs):
        sym = ens2sym.get(norm_id(g), g)
        ev = expr_vals.get(g, np.nan)
        rows.append((i, g, sym, ev, imp[i]))

    df = pd.DataFrame(rows, columns=['pos', 'gene_ensg', 'symbol', 'expr_value', 'attention'])
    df['expr_rank'] = df['expr_value'].rank(ascending=False, method='average')
    df['attn_rank'] = df['attention'].rank(ascending=False, method='average')

    # correlation: only over genes that have both values
    sub = df.dropna(subset=['expr_value'])
    rho_val, p_val = spearmanr(sub['attention'], sub['expr_value'])
    r_val, pv_val = pearsonr(sub['attention'], sub['expr_value'])
    rho_rank, p_rank = spearmanr(sub['attention'], sub['expr_rank'])

    print('=' * 60)
    print(f'sample: drug={drug_id} cell={cell_id}  pred={pred:.4f}  n_genes={len(sub)}')
    print(f'attention vs expression value  Spearman={rho_val:+.3f} (p={p_val:.2e})  Pearson={r_val:+.3f}')
    print(f'attention vs expression rank   Spearman={rho_rank:+.3f} (p={p_rank:.2e})')
    print('-' * 60)
    print('top-%d genes by attention:' % args.top_k)
    top = df.sort_values('attention', ascending=False).head(args.top_k)
    print(top[['gene_ensg', 'symbol', 'expr_value', 'expr_rank', 'attn_rank', 'attention']].to_string(index=False))

    key = [g.strip().upper() for g in args.genes.split(',') if g.strip()]
    print('-' * 60)
    print('key genes (symbol -> expression rank / attention rank):')
    for sym in key:
        hit = df[df['symbol'] == sym]
        if len(hit) == 0:
            print(f'  {sym:<8} not in the top-2048 genes of this cell')
            continue
        r = hit.iloc[0]
        print(f'  {sym:<8} expr_rank={int(r["expr_rank"]):>4}  attn_rank={int(r["attn_rank"]):>4}  '
              f'expr={r["expr_value"]:.3f}  attn={r["attention"]:.5f}')

    out_csv = os.path.join(args.output_dir, f'attn_vs_expr_{drug_id}_{cell_id}.csv')
    df.sort_values('attention', ascending=False).to_csv(out_csv, index=False)
    summ = pd.DataFrame([{
        'drug_id': drug_id, 'cell_id': cell_id, 'pred': pred, 'n_genes': len(sub),
        'spearman_vs_value': rho_val, 'p_value': p_val,
        'pearson_vs_value': r_val, 'spearman_vs_rank': rho_rank, 'p_rank': p_rank,
        'direction': 'drug2gene',
    }])
    summ.to_csv(os.path.join(args.output_dir, f'attn_vs_expr_summary_{drug_id}_{cell_id}.csv'), index=False)
    print(f'\nper-gene table: {out_csv}')


if __name__ == '__main__':
    main()