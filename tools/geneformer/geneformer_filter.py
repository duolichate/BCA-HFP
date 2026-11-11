#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
gene filtering: threshold on the specificity score, housekeeping genes are not added back
input: CSV written by geneformer_ensembl.py (cols 0-4 are metadata, cols 5+ are expression values with Ensembl IDs as column names)
output: new filtered CSV (only genes with specificity score > threshold)
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

_ROOT = os.environ.get('BCA_HFP_ROOT',
                 os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_GENE = os.path.join(_ROOT, 'Data', 'Gene')


# ==================== configuration ====================
# input: output file of geneformer_ensembl.py
INPUT_CSV = os.path.join(_GENE, 'GDSC_EXP_ensembl.csv')

# output: filtered CSV
OUTPUT_CSV = os.path.join(_GENE, 'GDSC_EXP_ensembl_filtered.csv')

# cache: global medians (avoids recomputation)
CACHE_DIR = os.path.join(_GENE, 'GDSC_EXP_GF_V1', 'cache')
CACHE_FILE = os.path.join(CACHE_DIR, "global_medians.pkl")

# specificity score threshold (1.5-2.0 recommended, 2.0 for read counts)
SPECIFICITY_THRESHOLD = 2.0

# metadata column names (must match your CSV)
META_COLS = ['GDSC_ID', 'CCLE_ID', 'CELL_LINES', 'CELL_LINES_STD', 'ORGAN']


# ===============================================


def load_csv(csv_path):
    """read the CSV, split metadata and expression matrix"""
    print(f"read: {csv_path}")
    df = pd.read_csv(csv_path)
    gene_cols = [c for c in df.columns if c not in META_COLS]
    print(f"   shape: {df.shape}, gene columns: {len(gene_cols)}")
    meta_df = df[META_COLS].copy()
    exp_df = df[gene_cols].copy().apply(pd.to_numeric, errors='coerce').fillna(0)
    return meta_df, exp_df, gene_cols


def compute_global_medians(exp_df, cache_file):
    """per-gene median over nonzero values (cached)"""
    if os.path.exists(cache_file):
        print(f"load from cache: {cache_file}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    print("computing global nonzero medians...")
    exp_matrix = exp_df.values
    medians = np.zeros(exp_matrix.shape[1])
    for i in tqdm(range(exp_matrix.shape[1]), desc="  computing medians"):
        vals = exp_matrix[:, i][exp_matrix[:, i] > 0]
        medians[i] = np.median(vals) if len(vals) > 0 else 1.0

    medians_series = pd.Series(medians, index=exp_df.columns)
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    with open(cache_file, 'wb') as f:
        pickle.dump(medians_series, f)
    print(f"   cache saved: {cache_file}")
    return medians_series


def filter_by_specificity(meta_df, exp_df, medians, threshold=2.0):
    """
    filter on the specificity score, keep only genes with score > threshold
    housekeeping genes are not added back
    """
    print(f"filtering by specificity score (threshold: {threshold})...")

    exp_matrix = exp_df.values
    medians_arr = np.where(medians.values == 0, 1.0, medians.values)
    gene_names = exp_df.columns.tolist()

    filtered_rows = []
    final_gene_cols = None
    gene_counts = []

    for i in tqdm(range(len(meta_df)), desc="  processing cell lines"):
        scores = exp_matrix[i, :] / medians_arr
        high_score_mask = scores > threshold
        selected_indices = np.where(high_score_mask)[0]
        gene_counts.append(len(selected_indices))

        # if no gene passes the threshold for this cell line, keep the top-scoring gene to avoid an empty row
        if len(selected_indices) == 0:
            selected_indices = [np.argmax(scores)]

        filtered_rows.append(exp_matrix[i, selected_indices])
        if final_gene_cols is None:
            final_gene_cols = [gene_names[j] for j in selected_indices]

    filtered_exp_df = pd.DataFrame(
        filtered_rows,
        index=meta_df.index,
        columns=final_gene_cols
    )

    print(f"   done, mean genes kept: {np.mean(gene_counts):.0f}")
    print(f"   gene count range: {min(gene_counts)} ~ {max(gene_counts)}")
    return filtered_exp_df


def save_csv(meta_df, exp_df, output_path):
    """save the filtered CSV"""
    df_out = pd.concat([meta_df.reset_index(drop=True),
                        exp_df.reset_index(drop=True)], axis=1)
    df_out.to_csv(output_path, index=False)
    print(f"   saved: {output_path}")


def main():
    print("=" * 70)
    print("gene specificity filtering (threshold only, housekeeping genes not added back)")
    print("=" * 70)

    meta_df, exp_df, _ = load_csv(INPUT_CSV)
    medians = compute_global_medians(exp_df, CACHE_FILE)
    filtered_exp = filter_by_specificity(meta_df, exp_df, medians, SPECIFICITY_THRESHOLD)
    save_csv(meta_df, filtered_exp, OUTPUT_CSV)

    print("\n" + "=" * 70)
    print("done")
    print(f"next: point the input file in geneformer_loom.py to:")
    print(f"  {OUTPUT_CSV}")
    print("=" * 70)


if __name__ == '__main__':
    main()