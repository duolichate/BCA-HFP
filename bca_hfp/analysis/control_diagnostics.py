# bca_hfp/analysis/control_diagnostics.py
"""R3-9 v2: a stricter version of the attention diagnostics.

the first version (control_stability / control_attn_vs_expression / control_input_removal) had two issues that
distort the conclusions, both are rewritten here:

  1) the cross-drug analysis used every drug the cell line covers, including the drugs held out for that
     checkpoint. the model never saw them, so explaining their attention is meaningless. this version keeps
     only the drugs inside the training folds via --fold_csv/--fold.
  2) only one per-gene Spearman was reported. attention has a gene-level component shared by all drugs,
     which pushes the correlation above 0.9 while being unrelated to the drug. everything is split into
     three numbers here:
         raw      : raw per-gene Spearman
         residual : per-gene Spearman after subtracting the across-drug mean
         jaccard  : Jaccard of the top-k gene sets (together with the random expectation)
     a gene-label permutation null is reported as well.

other fixes:
  - gene importance averages over the atom axis (not sums), so values are comparable between drugs with
    different atom counts;
  - per-head output is kept (--per_head) to double-check the multi-head specialization claim;
  - new sens_vs_res: take the top and bottom deciles by measured ln(IC50) and test whether attention
    separates sensitive from resistant cells;
  - removal: remove every gene one at a time to get the full |Δpred| distribution instead of 50 random draws.

usage:
  cross drug (training folds only):
    python bca_hfp/analysis/control_diagnostics.py --axis drugs \
      --model_path <fold2.pth> --fold 2 --fold_csv Result/oof/fold_membership.csv \
      --drug_id 1862 --cell BC-1 --output_dir Result/diag/controls_v2 \
      --h5 ~/my_model/Data/Drug/GDSC_RES_FGBERT.h5 \
      --gene_feat_dir ~/my_model/Data/Gene/GDSC_EXP_GF_V1

  cross cell line (same drug):
    python bca_hfp/analysis/control_diagnostics.py --axis cell_lines --model_path ... \
      --drug_id 1862 --n_cells 40 --output_dir ...

  sensitive vs resistant (drug in the training fold, cell lines by the top/bottom deciles of measured ln(IC50)):
    python bca_hfp/analysis/control_diagnostics.py --axis sens_vs_res --model_path ... \
      --fold 2 --fold_csv ... --max_drugs 60 --n_per_group 10 --output_dir ...

  remove genes one by one:
    python bca_hfp/analysis/control_diagnostics.py --axis removal --model_path ... \
      --drug_id 1862 --cell BC-1 --output_dir ...
"""

import argparse
import itertools
import os
import pickle
import sys

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from bca_hfp.analysis.control_utils import (load_model_from_ckpt, load_control_data, run_attention,
                                        load_symbol_maps, load_cell_gene_orders,
                                        resolve_cell_id, norm_id)

TARGET_GENES = 'PSMB5,PSMB2,PSMA2,PSMB1,RAB21,SURF6,MYC'


# ---------------------------------------------------------------- helpers

def importance(d2g, per_head=False):
    """drug->gene attention (8, atoms, genes) -> per-gene importance.

    average over the atom axis first (not sum), so values are comparable between drugs with different atom
    counts, then average over heads. per_head=True keeps the head axis and returns (8, genes).
    """
    per = d2g.mean(axis=1)           # (8, genes)
    return per if per_head else per.mean(axis=0)


def load_fold_map(path):
    if not path:
        return None
    df = pd.read_csv(path)
    return {norm_id(r.DRUG_ID): int(r.fold) for r in df.itertuples()}


def get_order_map(gene_feat_dir, cell_col, cells=None, cache=''):
    """per-cell gene_names.

    only a full scan is cached on disk: with cells given it is a partial scan whose result must not be
    reused as the full cache, so the cache file is neither read nor written in that case.
    """
    if cells is not None:
        return load_cell_gene_orders(gene_feat_dir, cell_col, only_cells=cells)
    if cache and os.path.exists(cache):
        with open(cache, 'rb') as f:
            return pickle.load(f)
    om = load_cell_gene_orders(gene_feat_dir, cell_col)
    if cache:
        with open(cache, 'wb') as f:
            pickle.dump(om, f)
    return om


def topk_set(vec_index, values, k):
    order = np.argsort(values)[::-1][:k]
    return set(vec_index[i] for i in order)


def decompose(mat, ref_col, k=20, n_null=1000, seed=0):
    """mat: DataFrame(genes x conditions). returns the three components + the null distribution."""
    cols = [c for c in mat.columns if c != ref_col]
    ref = mat[ref_col].to_numpy(dtype=float)
    resid_df = mat.sub(mat.mean(axis=1), axis=0)

    raw, res = [], []
    for c in cols:
        raw.append(spearmanr(ref, mat[c].to_numpy(dtype=float))[0])
        res.append(spearmanr(resid_df[ref_col].to_numpy(dtype=float),
                             resid_df[c].to_numpy(dtype=float))[0])

    ref_set = topk_set(mat.index, ref, k)
    jac = []
    for c in cols:
        s = topk_set(mat.index, mat[c].to_numpy(dtype=float), k)
        jac.append(len(ref_set & s) / len(ref_set | s))

    # null: shuffle the gene labels of the reference drug, then correlate with the other conditions
    rng = np.random.default_rng(seed)
    null_rho = []
    for _ in range(n_null):
        perm = rng.permutation(len(ref))
        null_rho.append(np.mean([spearmanr(ref[perm], mat[c].to_numpy(dtype=float))[0]
                                 for c in cols]))
    return {
        'n_other': len(cols),
        'raw_mean': float(np.mean(raw)), 'raw_std': float(np.std(raw)),
        'residual_mean': float(np.mean(res)), 'residual_std': float(np.std(res)),
        'jaccard_mean': float(np.mean(jac)), 'jaccard_max': float(np.max(jac)),
        'jaccard_random_expect': k / (2 * len(mat) - k),
        'null_raw_mean': float(np.mean(null_rho)), 'null_raw_std': float(np.std(null_rho)),
    }


def print_decomp(tag, d):
    print(f'\n[{tag}] compared against {d["n_other"]} other conditions')
    print(f'  raw      Spearman = {d["raw_mean"]:+.4f} ± {d["raw_std"]:.4f}')
    print(f'  residual Spearman = {d["residual_mean"]:+.4f} ± {d["residual_std"]:.4f}   '
          f'(the across-condition common component removed)')
    print(f'  top-20   Jaccard  = {d["jaccard_mean"]:.4f} (max {d["jaccard_max"]:.4f}), '
          f'random expectation {d["jaccard_random_expect"]:.4f}')
    print(f'  permutation null raw = {d["null_raw_mean"]:+.4f} ± {d["null_raw_std"]:.4f}')


def save_json(obj, path):
    import json
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------- axes

def axis_drugs(args):
    model, config, device, _ = load_model_from_ckpt(args.model_path)
    drug_atom, gene_emb, drug_global, response, cols = load_control_data(
        config, args.h5 or None, args.gene_feat_dir or None)
    cell_col, drug_col, _ = cols
    cell_id = resolve_cell_id(response, cell_col, args.cell)
    if cell_id is None:
        raise SystemExit(f'cell line {args.cell} not found')

    gene_ensgs = get_order_map(args.gene_feat_dir or config['data']['gene_feat_dir'],
                               cell_col, cells={norm_id(cell_id)},
                               cache=args.order_cache).get(norm_id(cell_id))
    if gene_ensgs is None:
        raise SystemExit('this cell line has no gene_names')

    sub = response[response[cell_col].apply(norm_id) == norm_id(cell_id)]
    drugs = sorted({norm_id(d) for d in sub[drug_col]}
                   & set(drug_atom) & set(drug_global))

    fmap = load_fold_map(args.fold_csv)
    if fmap is not None and args.fold > 0:
        before = len(drugs)
        drugs = [d for d in drugs if fmap.get(d, -1) != args.fold]
        print(f'training-fold filter: {before} -> {len(drugs)} drugs (fold {args.fold} excluded)')
    elif args.fold > 0:
        print('[warn] no --fold_csv given, held-out drugs cannot be excluded; the cross-drug result will be optimistic')
    if args.max_drugs > 0:
        drugs = drugs[:args.max_drugs]

    mat = {}
    heads = {}
    for d in drugs:
        _, d2g, _ = run_attention(model, drug_atom, gene_emb, drug_global, d, cell_id, device)
        mat[d] = importance(d2g)
        if args.per_head:
            heads[d] = importance(d2g, per_head=True)
    mat = pd.DataFrame(mat, index=gene_ensgs)
    mat.index.name = 'gene_ensg'
    mat.to_csv(os.path.join(args.output_dir, f'v2_drugs_matrix_{cell_id}.csv'))

    ref = norm_id(args.drug_id)
    ref = ref if ref in mat.columns else mat.columns[0]
    d = decompose(mat, ref, k=args.top_k, n_null=args.n_null)
    print_decomp(f'cross drug cell={cell_id} ref={ref}', d)
    save_json(d, os.path.join(args.output_dir, f'v2_drugs_decomp_{cell_id}.json'))

    if args.per_head:
        rows = []
        for hi in range(next(iter(heads.values())).shape[0]):
            hm = pd.DataFrame({k: v[hi] for k, v in heads.items()}, index=gene_ensgs)
            dh = decompose(hm, ref, k=args.top_k, n_null=max(200, args.n_null // 5))
            rows.append({'head': hi + 1, **dh})
        per = pd.DataFrame(rows)
        per.to_csv(os.path.join(args.output_dir, f'v2_drugs_per_head_{cell_id}.csv'), index=False)
        print('\nper-head decomposition (raw / residual / Jaccard):')
        print(per[['head', 'raw_mean', 'residual_mean', 'jaccard_mean']].to_string(index=False))


def axis_cell_lines(args):
    model, config, device, _ = load_model_from_ckpt(args.model_path)
    drug_atom, gene_emb, drug_global, response, cols = load_control_data(
        config, args.h5 or None, args.gene_feat_dir or None)
    cell_col, drug_col, _ = cols
    drug_id = norm_id(args.drug_id)

    sub = response[response[drug_col].apply(norm_id) == drug_id]
    cells = sorted({norm_id(c) for c in sub[cell_col]} & set(gene_emb))
    rng = np.random.default_rng(args.seed)
    if args.n_cells > 0 and len(cells) > args.n_cells:
        cells = sorted(rng.choice(cells, size=args.n_cells, replace=False))
    print(f'cross cell line: drug {drug_id}, {len(cells)} cell lines (random)')

    om = get_order_map(args.gene_feat_dir or config['data']['gene_feat_dir'],
                       cell_col, cells=set(cells), cache=args.order_cache)

    mat = {}
    for c in cells:
        gn = om.get(c)
        if gn is None:
            continue
        _, d2g, _ = run_attention(model, drug_atom, gene_emb, drug_global, drug_id, c, device)
        mat[c] = pd.Series(importance(d2g), index=gn)
    mat = pd.DataFrame(mat)
    mat.index.name = 'gene_ensg'
    mat = mat.dropna(axis=0, how='any')
    mat.to_csv(os.path.join(args.output_dir, f'v2_cells_matrix_{drug_id}.csv'))

    ref = args.cell if args.cell in mat.columns else mat.columns[0]
    d = decompose(mat, ref, k=args.top_k, n_null=args.n_null)
    print_decomp(f'cross cell line drug={drug_id} ref={ref}', d)
    save_json(d, os.path.join(args.output_dir, f'v2_cells_decomp_{drug_id}.json'))


def axis_sens_vs_res(args):
    model, config, device, _ = load_model_from_ckpt(args.model_path)
    drug_atom, gene_emb, drug_global, response, cols = load_control_data(
        config, args.h5 or None, args.gene_feat_dir or None)
    cell_col, drug_col, label_col = cols
    if label_col not in response.columns:
        raise SystemExit(f'response table lacks the label column {label_col}')

    fmap = load_fold_map(args.fold_csv)
    sub = response.dropna(subset=[label_col]).copy()
    sub['_d'] = sub[drug_col].apply(norm_id)
    sub['_c'] = sub[cell_col].apply(norm_id)

    rows = []
    rng = np.random.default_rng(args.seed)
    drugs = [d for d in sorted(set(sub['_d']))
             if d in drug_atom and d in drug_global
             and (fmap is None or args.fold <= 0 or fmap.get(d, -1) != args.fold)]
    if args.max_drugs > 0:
        drugs = drugs[:args.max_drugs]
    print(f'sens/res axis: {len(drugs)} candidate drugs (fold filter={args.fold})')

    om = get_order_map(args.gene_feat_dir or config['data']['gene_feat_dir'], cell_col,
                       cache=args.order_cache)

    for di, d in enumerate(drugs, 1):
        g = sub[sub['_d'] == d]
        g = g[g['_c'].isin(gene_emb)]
        if len(g) < 2 * args.n_per_group:
            continue
        g = g.sort_values(label_col)
        lo = g.head(args.n_per_group)['_c'].tolist()     # most sensitive
        hi = g.tail(args.n_per_group)['_c'].tolist()     # most resistant

        vec_map = {}
        for c in lo + hi:
            gn = om.get(c)
            if gn is None:
                continue
            _, d2g, _ = run_attention(model, drug_atom, gene_emb, drug_global, d, c, device)
            vec_map[norm_id(c)] = pd.Series(importance(d2g), index=gn)
        lo_ids = [norm_id(x) for x in lo if norm_id(x) in vec_map]
        hi_ids = [norm_id(x) for x in hi if norm_id(x) in vec_map]
        if len(lo_ids) < 2 or len(hi_ids) < 2:
            continue

        ml = pd.concat([vec_map[c] for c in lo_ids], axis=1).mean(axis=1)
        mh = pd.concat([vec_map[c] for c in hi_ids], axis=1).mean(axis=1)
        # each cell line has its own input gene set (2048 by expression), align the group means to the common genes first
        comm = ml.index.intersection(mh.index)
        if len(comm) < 100:
            continue
        ml, mh = ml.reindex(comm), mh.reindex(comm)
        rho = spearmanr(ml, mh)[0]
        jac = len(topk_set(ml.index, ml.to_numpy(), args.top_k) &
                  topk_set(mh.index, mh.to_numpy(), args.top_k))
        jac = jac / (2 * args.top_k - jac)

        # baseline: correlation between any two cell lines inside the group, to tell whether the "sens vs res"
        # difference is just an ordinary cell-line difference
        rho_pair = []
        for a, b in itertools.combinations(list(vec_map), 2):
            s = pd.concat([vec_map[a], vec_map[b]], axis=1).dropna()
            if len(s) > 100:
                rho_pair.append(spearmanr(s.iloc[:, 0], s.iloc[:, 1])[0])
        rho_base = float(np.mean(rho_pair)) if rho_pair else np.nan

        # stricter null: also "group mean vs group mean", but with the labels of the 2n cells shuffled.
        # comparing a group-mean correlation against a single-pair correlation is biased (averaging removes
        # noise and raises the correlation), so it has to be compared with an isomorphic permutation distribution.
        all_ids = lo_ids + hi_ids
        n_lo = len(lo_ids)
        null_rho = []
        for _ in range(args.n_perm):
            perm = rng.permutation(all_ids)
            ga, gb = perm[:n_lo], perm[n_lo:]
            ma = pd.concat([vec_map[c] for c in ga], axis=1).mean(axis=1)
            mb = pd.concat([vec_map[c] for c in gb], axis=1).mean(axis=1)
            cc = ma.index.intersection(mb.index)
            if len(cc) < 100:
                continue
            null_rho.append(spearmanr(ma.reindex(cc), mb.reindex(cc))[0])
        rho_null = float(np.mean(null_rho)) if null_rho else np.nan

        diff = (mh - ml).abs()
        rows.append({'drug': d, 'n_sens': len(lo_ids), 'n_res': len(hi_ids),
                     'spearman_lo_hi': rho, 'jaccard_top': jac,
                     'spearman_random_pairs': rho_base,
                     'spearman_label_perm': rho_null,
                     'rho_delta': rho - rho_null,
                     'mean_abs_diff': float(diff.mean()),
                     'max_abs_diff': float(diff.max())})

    per = pd.DataFrame(rows)
    per.to_csv(os.path.join(args.output_dir, 'v2_sensres_per_drug.csv'), index=False)
    if len(per) == 0:
        print('no drug qualifies, loosen --n_per_group or check the fold filter')
        return
    print(f'\ndone with {len(per)} drugs. sensitive vs resistant group:')
    print(per[['spearman_lo_hi', 'spearman_label_perm', 'rho_delta',
               'jaccard_top', 'mean_abs_diff']].describe().to_string())
    print(f'\n  sens vs res  between-group Spearman median = {per["spearman_lo_hi"].median():+.4f}')
    print(f'  label-permutation null, between-group Spearman median = {per["spearman_label_perm"].median():+.4f}'
          f'   (same group-mean comparison, so the "averaging removes noise" bias is gone)')
    print(f'  rho_delta median = {per["rho_delta"].median():+.4f}   '
          f'attention separates sens/res only if this is clearly away from 0.')


def axis_removal(args):
    model, config, device, _ = load_model_from_ckpt(args.model_path)
    drug_atom, gene_emb, drug_global, response, cols = load_control_data(
        config, args.h5 or None, args.gene_feat_dir or None)
    cell_col, drug_col, _ = cols
    cell_id = resolve_cell_id(response, cell_col, args.cell)
    drug_id = norm_id(args.drug_id)

    gene_ensgs = get_order_map(args.gene_feat_dir or config['data']['gene_feat_dir'],
                               cell_col, cells={norm_id(cell_id)},
                               cache=args.order_cache).get(norm_id(cell_id))
    _, ens2sym = load_symbol_maps()
    _, d2g, _ = run_attention(model, drug_atom, gene_emb, drug_global, drug_id, cell_id, device)
    imp = importance(d2g)
    attn_rank = pd.Series(imp).rank(ascending=False).to_numpy()

    atom = torch.tensor(drug_atom[drug_id], dtype=torch.float32).unsqueeze(0).to(device)
    atom_mask = torch.ones(1, atom.shape[1], dtype=torch.bool, device=device)
    gene0 = torch.tensor(gene_emb[cell_id], dtype=torch.float32).unsqueeze(0).to(device)
    mask0 = torch.ones(1, gene0.shape[1], dtype=torch.bool, device=device)
    dg0 = torch.tensor(drug_global[drug_id], dtype=torch.float32).unsqueeze(0).to(device)

    def fwd(g, gm):
        with torch.no_grad():
            return float(model(g, atom, atom_mask, gm, drug_global=dg0).item())

    base = fwd(gene0, mask0)
    rows = []
    for idx in range(gene0.shape[1]):
        g, gm = gene0.clone(), mask0.clone()
        g[0, idx] = 0.0
        gm[0, idx] = False
        rows.append({'idx': idx,
                     'symbol': ens2sym.get(norm_id(gene_ensgs[idx]), gene_ensgs[idx]),
                     'attn_rank': int(attn_rank[idx]),
                     'delta_abs': abs(fwd(g, gm) - base)})
    df = pd.DataFrame(rows).sort_values('delta_abs', ascending=False)
    df.to_csv(os.path.join(args.output_dir, f'v2_removal_genes_{drug_id}_{cell_id}.csv'),
              index=False)

    print(f'\nremoved every one of the {len(df)} genes (baseline pred={base:.4f}):')
    print(f'  |Δpred| median = {df.delta_abs.median():.3e}  '
          f'p95 = {df.delta_abs.quantile(.95):.3e}  max = {df.delta_abs.max():.3e}')
    print('  top-10 genes by removal effect:')
    print(df.head(10)[['symbol', 'attn_rank', 'delta_abs']].to_string(index=False))

    keys = [s.strip().upper() for s in args.genes.split(',') if s.strip()]
    hit = df[df.symbol.str.upper().isin(keys)]
    if len(hit):
        print('\n  requested genes:')
        print(hit[['symbol', 'attn_rank', 'delta_abs']].to_string(index=False))

    # where the 20 highest-attention genes rank by removal effect
    top20 = df[df.attn_rank <= 20]['delta_abs']
    print(f'\n  |Δpred| of the top-20 attention genes: median {top20.median():.3e}, max {top20.max():.3e};'
          f' median rank within the full gene distribution {int((df.delta_abs > top20.median()).sum())}')


def axis_key_norm(args):
    """diagnose where the common component comes from: does the norm of the gene-side key set the attention level.

    cross-attention scores q·k and the softmax is very sensitive to a large inner product; if some genes have
    a clearly larger key norm, attention collapses onto them regardless of the drug. this computes the key norm
    of every gene in the 8 heads, correlates it with its across-drug mean attention, and gives a
    label-permutation null.
    """
    model, config, device, _ = load_model_from_ckpt(args.model_path)
    drug_atom, gene_emb, drug_global, response, cols = load_control_data(
        config, args.h5 or None, args.gene_feat_dir or None)
    cell_col, drug_col, _ = cols
    cell_id = resolve_cell_id(response, cell_col, args.cell)
    gene_ensgs = get_order_map(args.gene_feat_dir or config['data']['gene_feat_dir'],
                               cell_col, cells={norm_id(cell_id)},
                               cache=args.order_cache).get(norm_id(cell_id))
    _, ens2sym = load_symbol_maps()

    att = model.cross_attn_drug2gene
    gene = torch.tensor(gene_emb[cell_id], dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        key = att.input_norm(att.gene_proj(gene))            # (1, n_genes, d)
    d_model = key.shape[-1]
    n_heads = att.cross_attn.num_heads
    hd = d_model // n_heads
    W = att.cross_attn.in_proj_weight                        # (3d, d), Q/K/V in that order
    Wk = W[d_model:2 * d_model]

    knorms = []
    for h in range(n_heads):
        Wh = Wk[h * hd:(h + 1) * hd]
        kh = key[0] @ Wh.t()                                 # (n_genes, hd)
        knorms.append(kh.norm(dim=-1).detach().cpu().numpy())
    knorms = np.stack(knorms)                                # (n_heads, n_genes)

    # across-drug mean attention
    sub = response[response[cell_col].apply(norm_id) == norm_id(cell_id)]
    drugs = sorted({norm_id(d) for d in sub[drug_col]} & set(drug_atom) & set(drug_global))
    fmap = load_fold_map(args.fold_csv)
    if fmap is not None and args.fold > 0:
        drugs = [d for d in drugs if fmap.get(d, -1) != args.fold]
    if args.max_drugs > 0:
        drugs = drugs[:args.max_drugs]
    print(f'key_norm: cell={cell_id}, averaging attention over {len(drugs)} drugs')

    imps = []
    for d in drugs:
        _, d2g, _ = run_attention(model, drug_atom, gene_emb, drug_global, d, cell_id, device)
        imps.append(importance(d2g))
    mean_attn = np.mean(imps, axis=0)

    rng = np.random.default_rng(args.seed)
    rows = []
    for h in range(n_heads):
        rho = spearmanr(knorms[h], mean_attn)[0]
        null = [abs(spearmanr(rng.permutation(knorms[h]), mean_attn)[0])
                for _ in range(max(200, args.n_null // 5))]
        rows.append({'head': h + 1, 'spearman_keynorm_vs_attn': rho,
                     'abs_null_p95': float(np.percentile(null, 95))})
    df = pd.DataFrame(rows)
    kmean = knorms.mean(axis=0)
    rho_all = spearmanr(kmean, mean_attn)[0]
    df.to_csv(os.path.join(args.output_dir, f'v2_keynorm_per_head_{cell_id}.csv'), index=False)

    out = pd.DataFrame({
        'gene_ensg': gene_ensgs,
        'symbol': [ens2sym.get(norm_id(g), g) for g in gene_ensgs],
        'key_norm_mean': kmean,
        'mean_attn': mean_attn,
        'attn_rank': pd.Series(mean_attn).rank(ascending=False).to_numpy(),
    })
    for h in range(n_heads):
        out[f'key_norm_h{h + 1}'] = knorms[h]
    out.sort_values('mean_attn', ascending=False).to_csv(
        os.path.join(args.output_dir, f'v2_keynorm_{cell_id}.csv'), index=False)

    print('\nper-head key norm vs mean attention Spearman:')
    print(df.to_string(index=False))
    print(f'\nhead-mean key norm vs mean attention: Spearman = {rho_all:+.4f}')
    print('  |rho| near 1 -> attention is mostly set by the key norm (drug independent); '
          'near 0 -> unrelated to the norm, look at the q·k geometry or the architecture')


def main():
    ap = argparse.ArgumentParser(description='R3-9 v2 strict attention diagnostics')
    ap.add_argument('--axis', required=True,
                    choices=['drugs', 'cell_lines', 'sens_vs_res', 'removal', 'key_norm'])
    ap.add_argument('--model_path', required=True)
    ap.add_argument('--h5', default='')
    ap.add_argument('--gene_feat_dir', default='')
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--drug_id', default='1862')
    ap.add_argument('--cell', default='')
    ap.add_argument('--fold', type=int, default=0, help='held-out fold of this checkpoint (1-5), 0 = no filter')
    ap.add_argument('--fold_csv', default='', help='fold_membership.csv')
    ap.add_argument('--n_cells', type=int, default=40)
    ap.add_argument('--max_drugs', type=int, default=0)
    ap.add_argument('--n_per_group', type=int, default=10, help='cell lines per sensitive / resistant group')
    ap.add_argument('--top_k', type=int, default=20)
    ap.add_argument('--n_null', type=int, default=1000)
    ap.add_argument('--n_perm', type=int, default=200,
                    help='label permutations for sens_vs_res (group-mean vs group-mean null)')
    ap.add_argument('--per_head', action='store_true')
    ap.add_argument('--genes', default=TARGET_GENES)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--order_cache', default='', help='pkl path of the gene_names cache')
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    {'drugs': axis_drugs, 'cell_lines': axis_cell_lines,
     'sens_vs_res': axis_sens_vs_res, 'removal': axis_removal,
     'key_norm': axis_key_norm}[args.axis](args)


if __name__ == '__main__':
    main()