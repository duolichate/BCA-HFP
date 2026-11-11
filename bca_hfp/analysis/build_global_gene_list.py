# bca_hfp/analysis/build_global_gene_list.py
"""collect the "gene universe" from the GeneFormer embedding directory and write a global gene list.

purpose
----
each cell line in `Data/GDSC_EXP_GF_V*.pt` keeps only 2048 tokens and the gene sets differ between cell
lines (tokens are the top 2048 genes by expression). a non-sequence baseline (e.g. gradient boosting)
needs a **fixed column set**; the set is picked here and the per-gene statistics are written out.

output
----
1) <out_dir>/global_gene_names_<prefix>.txt   the selected N Ensembl ids, one per line (no header)
2) <out_dir>/gene_stats_<prefix>.csv          every gene seen so far plus its stats:
      n_cells / frac_cells / mean_rank / best_rank / present_in_exp / mean_expr / var_expr
3) terminal: cell lines, tokens, union / intersection size, coverage distribution, selection rule

meaning of rank: GeneFormer token i is the i-th highest expressed gene of that cell line (1-based),
so a small mean_rank means the gene is highly expressed in most cell lines.

usage
----
# only the cell lines present in the response table (same as main-model training, recommended)
python bca_hfp/analysis/build_global_gene_list.py \
  --pt_dir ../Data/GDSC_EXP_GF_V1 \
  --h5 ../Data/GDSC_RES_FGBERT.h5 \
  --exp_csv ../Data/GDSC_EXP_ensembl.csv \
  --top_n 2048 --rank_by freq \
  --out_dir ../Result/Interpretability --prefix GDSC

# no cell-line restriction and no expression table (token statistics only)
python bca_hfp/analysis/build_global_gene_list.py --pt_dir ../Data/GDSC_EXP_GF_V2 --top_n 2048

the generated global_gene_names_GDSC.txt can be passed to table_a_rebuild.py as --gene_names;
if you don't want to maintain a file, table_a_rebuild.py has the equivalent --auto_genes 2048 (picks by mean expression).
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser(description='collect the GeneFormer token gene universe and write a global gene list')
    ap.add_argument('--pt_dir', required=True, help='e.g. ../Data/GDSC_EXP_GF_V1')
    ap.add_argument('--h5', default='', help='response h5; when given only cell lines present in it are counted')
    ap.add_argument('--exp_csv', default='', help='expression table; when given adds mean_expr / var_expr and present_in_exp')
    ap.add_argument('--top_n', type=int, default=2048)
    ap.add_argument('--rank_by', choices=['freq', 'mean_expr'], default='freq',
                    help='selection rule: freq=most frequent (default) | mean_expr=highest mean expression over cell lines')
    ap.add_argument('--min_cells', type=int, default=1, help='keep only genes present in at least this many cell lines')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--prefix', default='GDSC')
    ap.add_argument('--limit', type=int, default=0, help='when >0 read only the first N files (debug)')
    ap.add_argument('--filename_key', action='store_true',
                    help='fall back to the file name when metadata has no GDSC_ID. in this repo the .pt names are '
                         'counters (0.pt, 1.pt, ...), not GDSC_IDs, so by default such files are skipped')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.pt_dir, '*.pt')))
    if not files:
        raise SystemExit(f'no .pt file under {args.pt_dir}')
    if args.limit > 0:
        files = files[:args.limit]

    keep_cells = None
    if args.h5:
        with pd.HDFStore(args.h5, 'r') as store:
            cells_in_resp = set(store['original_data']['GDSC_ID'].astype(str).unique())
        keep_cells = cells_in_resp
        print(f'[h5] {len(cells_in_resp)} cell lines in the response table, only those are counted')

    import torch

    per_cell = {}          # GDSC_ID -> gene_names
    dup_ids, no_key, normed = 0, 0, 0
    token_lens, emb_dims = set(), set()

    def norm_id(v):
        """normalize GDSC_ID: '910921.0' -> '910921'; 'nan'/None -> '' (file unusable)."""
        s = str(v).strip()
        if s.lower() in ('', 'nan', 'none', '__missing__'):
            return ''
        if s.endswith('.0'):
            return s[:-2]
        return s

    for f in files:
        d = torch.load(f, map_location='cpu', weights_only=False)
        meta = (d.get('metadata') or {}) if isinstance(d, dict) else {}
        raw_cid = meta.get('GDSC_ID', '')
        cid = norm_id(raw_cid)
        genes = d.get('gene_names') if isinstance(d, dict) else None
        emb = d.get('embeddings') if isinstance(d, dict) else None
        if genes is None:
            continue
        if not cid:
            if not args.filename_key:
                no_key += 1
                continue
            cid = os.path.splitext(os.path.basename(f))[0].replace('DATA.', '').strip()
            no_key += 1
        elif str(raw_cid).strip() != cid:
            normed += 1
        if keep_cells is not None and cid not in keep_cells:
            continue
        if cid in per_cell:
            dup_ids += 1
            continue
        per_cell[cid] = list(genes)
        token_lens.add(len(genes))
        if emb is not None:
            emb_dims.add(tuple(emb.shape))

    if not per_cell:
        raise SystemExit('no cell line read (check that the GDSC_ID values in --h5 match the metadata)')

    print(f'[files] read {len(files)} files -> {len(per_cell)} cell lines'
          f' (duplicate GDSC_ID {dup_ids}, skipped for nan/empty GDSC_ID {no_key}, '
          f'normalized from a decimal form (e.g. 123.0) {normed})')
    if dup_ids or no_key:
        print('  [warn] the dir has duplicate or unusable .pt files (nan/empty GDSC_ID): duplicates were skipped, '
              'unusable ones ignored. cleaning them up first keeps the statistics from being skewed.')
    print(f'[shape] token counts {sorted(token_lens)}  embedding shapes {sorted(emb_dims)[:2]}')
    if len(token_lens) != 1:
        print('  [warn] token counts differ between cell lines, each is counted with its own length')

    # ---------- per-gene statistics ----------
    vocab = {}
    for genes in per_cell.values():
        for g in genes:
            if g not in vocab:
                vocab[g] = len(vocab)
    n_genes = len(vocab)
    cnt = np.zeros(n_genes, dtype=np.int32)
    rank_sum = np.zeros(n_genes, dtype=np.int64)
    rank_min = np.full(n_genes, np.iinfo(np.int32).max, dtype=np.int32)

    for genes in per_cell.values():
        for i, g in enumerate(genes, start=1):
            j = vocab[g]
            cnt[j] += 1
            rank_sum[j] += i
            if i < rank_min[j]:
                rank_min[j] = i

    ids = np.array(list(vocab.keys()), dtype=object)
    n_cells = len(per_cell)
    stats = pd.DataFrame({
        'gene_id': ids,
        'n_cells': cnt,
        'frac_cells': cnt / n_cells,
        'mean_rank': rank_sum / np.maximum(cnt, 1),
        'best_rank': rank_min,
    })

    # coverage distribution
    cover = stats['n_cells'].values
    print(f'\n[universe] {n_cells} cell lines; {n_genes} distinct genes; '
          f'{sorted(token_lens)[0]} tokens per cell line')
    print(f'  in 100% of cell lines: {int((cover == n_cells).sum())}')
    for thr in (0.9, 0.75, 0.5, 0.25):
        print(f'  in >= {int(thr * 100):>3d}% of cell lines: {int((cover >= thr * n_cells).sum()):>5d}')
    print(f'  in 1 cell line only: {int((cover == 1).sum())}')
    print(f'  intersection (shared by all cell lines) = '
          f'{len(set.intersection(*[set(g) for g in per_cell.values()]))}')

    # ---------- optional: expression table ----------
    if args.exp_csv and os.path.exists(args.exp_csv):
        head = pd.read_csv(args.exp_csv, nrows=0).columns.tolist()
        gcols = [c for c in head if c.upper().startswith('ENSG')]
        want = [g for g in stats.gene_id if g in set(gcols)]
        print(f'\n[exp] expression table has {len(gcols)} gene columns; {len(want)} of them are in the universe')
        if want:
            exp = pd.read_csv(args.exp_csv, usecols=['GDSC_ID'] + want,
                              dtype={c: np.float32 for c in want})
            mu = exp[want].mean(axis=0)
            var = exp[want].var(axis=0)
            stats['present_in_exp'] = stats.gene_id.isin(want)
            stats['mean_expr'] = stats.gene_id.map(mu)
            stats['var_expr'] = stats.gene_id.map(var)
        else:
            stats['present_in_exp'] = False
    else:
        stats['present_in_exp'] = np.nan

    # ---------- select genes ----------
    stats = stats.sort_values(['n_cells', 'mean_rank'], ascending=[False, True])
    if args.rank_by == 'mean_expr':
        if 'mean_expr' not in stats.columns:
            raise SystemExit('--rank_by mean_expr needs --exp_csv')
        stats = stats.dropna(subset=['mean_expr']).sort_values('mean_expr', ascending=False)
    picked = stats[stats.n_cells >= args.min_cells].head(args.top_n)

    out_txt = os.path.join(args.out_dir, f'global_gene_names_{args.prefix}.txt')
    with open(out_txt, 'w') as fh:
        fh.write('\n'.join(picked.gene_id.tolist()) + '\n')
    out_csv = os.path.join(args.out_dir, f'gene_stats_{args.prefix}.csv')
    stats.to_csv(out_csv, index=False)

    print(f'\n[pick] rule={args.rank_by}, top_n={args.top_n}, min_cells={args.min_cells}')
    print(f'  picked {len(picked)} genes; median cell-line coverage '
          f'{int(picked.n_cells.median())}/{n_cells}; median mean_rank {picked.mean_rank.median():.1f}')
    print(f'  full stats -> {out_csv}')
    print(f'  gene list -> {out_txt}')
    print('\nfirst 5:', ', '.join(picked.gene_id.head(5).tolist()))


if __name__ == '__main__':
    main()
