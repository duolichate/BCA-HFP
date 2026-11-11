# -*- coding: utf-8 -*-
"""AZD6738 x KARPAS-299 attention across seeds (seed 42 vs seed 43, five folds).

seed 42 reads the existing npz (CHEK1 rank 7/395/1355/612/558 already checked against the paper);
seed 43 is re-exported by a forward pass using the config stored in its checkpoint, same drug-cell
pairs in all five folds.

output: per-fold CHEK1 rank, Spearman on the per-gene attention vector, top-k Jaccard.
usage (cwd must be 4_DrugResponse_BCAHFP):
  python -m bca_hfp.analysis.cross_seed_attention
"""
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from bca_hfp.figures.plot_attention_g1 import extract  # noqa: E402

DRUG_ID = '1917'          # AZD6738
CELL_ID = '907273'        # KARPAS-299
FOLDS = [1, 2, 3, 4, 5]
TARGET = 'CHEK1'

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
S42_DIR = os.path.join(ROOT, '5_Result', '3_diagnostics', '1_model_level',
                       'r36_v1gc_s42_5090_20261108', 'attention')
S43_RUN = os.path.join(ROOT, '5_Result', '1_main_models',
                       'RTX5090_v1_gc_c5_s43_20261107_0211')
S43_DIR = os.path.join(ROOT, '5_Result', '3_diagnostics', '1_model_level',
                       's43_attn_20261110', 'attention')
OUT_CSV = os.path.join(ROOT, 'results', 'tables', 'S13_cross_seed_attention.csv')
ENSG_PKL = os.path.join(ROOT, '3_Gene_Feature', 'geneformer',
                        'gene_dictionaries_30m', 'gene_name_id_dict_gc30M.pkl')


def load_ensg2sym():
    with open(ENSG_PKL, 'rb') as f:
        m = pickle.load(f)
    return {str(v).split('.')[0]: str(k).upper() for k, v in m.items()}


def score_vec(d2g):
    """(H, A, G) -> (G,), per-gene attention score averaged over heads and atoms."""
    return d2g.mean(axis=(0, 1))


def rank_of_no_nan(v, idx):
    return int(np.argsort(np.argsort(-v))[idx]) + 1


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b)


def spearman(x, y):
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    e2s = load_ensg2sym()
    os.makedirs(S43_DIR, exist_ok=True)
    rows = []
    vecs = {}

    for fold in FOLDS:
        # ---- seed 42 ----
        p42 = os.path.join(S42_DIR, 'attention_%s_%s_fold%d.npz'
                           % (DRUG_ID, CELL_ID, fold))
        z42 = np.load(p42, allow_pickle=True)
        d2g42 = z42['d2g']
        names42 = [str(x) for x in z42['gene_names']]

        # ---- seed 43 ----
        p43 = os.path.join(S43_DIR, 'attention_%s_%s_fold%d.npz'
                           % (DRUG_ID, CELL_ID, fold))
        if os.path.exists(p43):
            z43 = np.load(p43, allow_pickle=True)
        else:
            d = extract(DRUG_ID, CELL_ID, fold, S43_RUN)
            np.savez_compressed(p43, **d)
            z43 = np.load(p43, allow_pickle=True)
        d2g43 = z43['d2g']
        names43 = [str(x) for x in z43['gene_names']]
        assert names42 == names43, 'gene order differs between the two seeds'

        s42, s43 = score_vec(d2g42), score_vec(d2g43)
        syms = [e2s.get(n, n) for n in names42]
        ti = syms.index(TARGET)

        vecs[fold] = (s42, s43, syms)

        o42 = np.argsort(-s42)
        o43 = np.argsort(-s43)
        rec = dict(fold=fold,
                   CHEK1_rank_s42=rank_of_no_nan(s42, ti),
                   CHEK1_rank_s43=rank_of_no_nan(s43, ti),
                   spearman_all_genes=spearman(s42, s43),
                   jaccard_top20=jaccard(o42[:20], o43[:20]),
                   jaccard_top50=jaccard(o42[:50], o43[:50]),
                   jaccard_top100=jaccard(o42[:100], o43[:100]),
                   top10_overlap=len(set(o42[:10]) & set(o43[:10])),
                   d2g_shape='%dx%dx%d' % d2g42.shape)
        rows.append(rec)
        print('[fold %d] CHEK1 %4d vs %4d | Spearman %.3f | J@20 %.3f '
              'J@50 %.3f J@100 %.3f | top10 overlap %d/10'
              % (fold, rec['CHEK1_rank_s42'], rec['CHEK1_rank_s43'],
                 rec['spearman_all_genes'], rec['jaccard_top20'],
                 rec['jaccard_top50'], rec['jaccard_top100'],
                 rec['top10_overlap']))

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False, encoding='utf-8-sig')

    print('\n== summary (five folds) ==')
    for c in ('CHEK1_rank_s42', 'CHEK1_rank_s43', 'spearman_all_genes',
              'jaccard_top20', 'jaccard_top50', 'jaccard_top100'):
        v = df[c].values
        print('  %-20s %s' % (c, ' '.join('%6.3f' % x if isinstance(x, float)
                                          else '%6d' % x for x in v)))
    print('  %-20s median Spearman %.3f | median J@20 %.3f'
          % ('median', df['spearman_all_genes'].median(),
             df['jaccard_top20'].median()))

    # whether both seeds rank the same genes near the CHEK1 top
    print('\n== fold1 top-10 genes (both seeds side by side) ==')
    s42, s43, syms = vecs[1]
    o42, o43 = np.argsort(-s42), np.argsort(-s43)
    for i in range(12):
        print('  %2d  s42: %-12s  s43: %-12s' % (
            i + 1, syms[o42[i]], syms[o43[i]]))
    print('\nwrote %s' % OUT_CSV)


if __name__ == '__main__':
    main()
