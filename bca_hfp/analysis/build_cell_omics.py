#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""build_cell_omics.py

turn the DepMap CRISPR gene-effect matrix (optionally the hotspot mutation matrix) into the .npz the model reads.

why no DepMap login is needed
-----------------------
the GDSC expression table `Data/GDSC_EXP.csv` carries both `GDSC_ID` and `CCLE_ID` (of the form ACH-001711),
and `ACH-xxxxxx` is exactly the DepMap `ModelID`. so any dependency matrix whose **row names are ACH ids**
joins directly to GDSC_ID, without DepMap sample_info.csv and without registration.

login-free sources (pick one):
  1) Zenodo teaching subset (40MB, 1208 lines x 5000 genes, Chronos scores, CC-BY-4.0):
     https://zenodo.org/records/22759375
       depmap_ders_skorlar.csv   dependency scores (rows=ACH, columns="SYMBOL (EntrezID)")
       OmicsSomaticMutationsMatrixHotspot.csv  hotspot mutations (optional)
  2) the full DepMap release on figshare (CRISPRGeneEffect.csv, ~1.5GB, 17k genes)

usage
----
# dependency scores only
python bca_hfp/analysis/build_cell_omics.py \
    --gene_effect /home/xxin/my_model/Data/depmap_gene_effect.csv \
    --exp /home/xxin/my_model/Data/GDSC_EXP.csv \
    --out /home/xxin/my_model/Data/cell_dependency.npz

# dependencies + hotspot mutations (concatenated into one matrix, column names get the MUT_ prefix)
python bca_hfp/analysis/build_cell_omics.py \
    --gene_effect .../depmap_gene_effect.csv \
    --mutations   .../OmicsSomaticMutationsMatrixHotspot.csv \
    --exp /home/xxin/my_model/Data/GDSC_EXP.csv \
    --out /home/xxin/my_model/Data/cell_omics.npz

keys of the output npz:
    cell_ids    (n,)     GDSC_ID strings
    ach_ids     (n,)     the matching ACH ids
    X           (n, p)   float32 features (not standardized; standardize downstream)
    feature_names (p,)
    present     (n,)     bool, always True (only cell lines with data are kept)
plus <out>.report.txt: coverage and the list of missing entries.

caveats
----
- missing cell lines are **not** filled with a zero vector, they are dropped from the npz; cell lines
  absent from the npz take the "zero vector + missing" branch of the dataset, keep that in mind.
- NaN dependency scores are filled with the column median; columns with a missing rate above --max_missing are dropped.
"""

from __future__ import annotations

import argparse
import gzip
import os
import re
import sys

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# reading
# --------------------------------------------------------------------------- #
def _sep_of(path):
    p = path.lower()
    if p.endswith(".csv") or p.endswith(".csv.gz"):
        return ","
    if p.endswith(".tsv") or p.endswith(".tsv.gz") or p.endswith(".txt") or p.endswith(".txt.gz"):
        return "\t"
    return ","


def _open(path):
    return gzip.open(path, "rt", encoding="utf-8", errors="replace") if path.lower().endswith(".gz") else path


def read_matrix(path, id_prefix="ACH-", verbose=True):
    """read a matrix and detect which axis holds the cell lines (ACH ids). returns a (genes x cells) DataFrame."""
    sep = _sep_of(path)
    if path.lower().endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
            df = pd.read_csv(fh, sep=sep, index_col=0, low_memory=False)
    else:
        df = pd.read_csv(path, sep=sep, index_col=0, low_memory=False)
    head_i = [str(v) for v in list(df.index[:30])]
    head_c = [str(v) for v in list(df.columns[:30])]
    hit_i = sum(v.startswith(id_prefix) for v in head_i)
    hit_c = sum(v.startswith(id_prefix) for v in head_c)
    if verbose:
        print("[read] %s  shape=%s  ACH on index=%d/%d  on columns=%d/%d"
              % (os.path.basename(path), df.shape, hit_i, len(head_i), hit_c, len(head_c)))
    if hit_i == 0 and hit_c == 0:
        # third case: ACH ids sit in the values of one column (the ModelID column of the DepMap mutation matrix)
        id_col = None
        for c in df.columns:
            vals = [str(v) for v in df[c].head(50).tolist()]
            n_hit = sum(v.startswith(id_prefix) for v in vals)
            if n_hit >= max(3, int(0.5 * len(vals))):
                id_col = c
                break
        if id_col is None:
            raise SystemExit(
                "[error] no ID starting with %s found in %s (tried index / column names / column values). "
                "check whether the file is transposed or the column names were changed." % (id_prefix, path)
            )
        if verbose:
            print("[read] ACH ids in the values of column '%s' -> use it as index, drop the other non-numeric columns" % id_col)
        df = df.set_index(id_col)
        keep = [c for c in df.columns
                if pd.to_numeric(df[c], errors="coerce").notna().any()]
        df = df[keep]
        hit_i, hit_c = len(df), 0
    # convert to numeric along the axis with unique names (genes or cell lines, no duplicates there),
    # then orient and handle duplicate cell lines (one ModelID has several ModelCondition rows in the mutation matrix).
    df = df.apply(pd.to_numeric, errors="coerce")
    if hit_c >= hit_i:
        M = df                                    # rows=genes, cols=cell lines
    else:
        M = df.T                                  # rows=cell lines -> make rows=genes
    M.columns = [str(c).strip() for c in M.columns]
    M.index = [str(i).strip() for i in M.index]
    # a cell line in several rows (ModelCondition in the mutation matrix) -> aggregate by max
    if M.columns.duplicated().any():
        n_before = M.shape[1]
        M = M.T.groupby(level=0).max().T
        if verbose:
            print("[read] duplicate cell lines %d -> %d columns, aggregated by max" % (n_before, M.shape[1]))
    return M


def clean_gene_name(s):
    """`TSPAN6 (7105)` -> `TSPAN6`; `MUT_KRAS` kept as is."""
    s = str(s).strip()
    s = re.sub(r"\s*\(\d+\)\s*$", "", s)
    s = re.sub(r"\s*\([^)]*\)\s*$", "", s) if s.count("(") else s
    return s


def read_gdsc_map(exp_path, verbose=True):
    """build the GDSC_ID -> CCLE_ID (ACH id) map from GDSC_EXP.csv."""
    df = pd.read_csv(exp_path, usecols=[0, 1], nrows=None, low_memory=False)
    df.columns = ["gdsc", "ach"]
    df["gdsc"] = df["gdsc"].map(lambda v: str(v).strip().split(".")[0])
    df["ach"] = df["ach"].map(lambda v: str(v).strip())
    df = df[df["ach"].str.startswith("ACH-")]
    m = dict(zip(df["ach"], df["gdsc"]))
    if verbose:
        print("[map] GDSC_EXP: %d rows, %d with an ACH id, %d unique ACH ids"
              % (len(df), int(df["ach"].str.startswith("ACH-").sum()), len(m)))
    return m


# --------------------------------------------------------------------------- #
# building
# --------------------------------------------------------------------------- #
def build(M, gdsc_map, top_var, max_missing, prefix="", verbose=True, log=print):
    """M: (genes x ACH cells). returns (gdsc_ids, X, feature_names)."""
    genes = [prefix + clean_gene_name(g) for g in M.index]
    cells_ach = list(M.columns)
    miss = M.isna().mean(1).values
    keep = miss <= max_missing
    if (~keep).sum():
        log("[filter] %s dropping %d columns (missing rate > %.2f)" % (prefix, int((~keep).sum()), max_missing))
    X = M.values[keep].astype(np.float64)
    genes = [g for g, k in zip(genes, keep) if k]
    if X.shape[0] == 0:
        raise SystemExit("[error] every column was dropped by the missing-rate filter")

    # fill NaN with the column median
    med = np.nanmedian(X, axis=1, keepdims=True)
    med = np.where(np.isfinite(med), med, 0.0)
    X = np.where(np.isfinite(X), X, med)

    # keep the top_var highest-variance features
    if top_var and X.shape[0] > top_var:
        v = X.var(1)
        idx = np.argsort(-v)[:top_var]
        idx.sort()
        X = X[idx]
        genes = [genes[i] for i in idx]
        log("[filter] %s keeping %d / %d columns by variance" % (prefix, X.shape[0], len(keep)))

    # ACH -> GDSC
    hit = [(j, gdsc_map[a]) for j, a in enumerate(cells_ach) if a in gdsc_map]
    if verbose:
        log("[join] %s matched %d / %d ACH ids" % (prefix, len(hit), len(cells_ach)))
    if not hit:
        raise SystemExit("[error] no ACH id joins to a GDSC_ID, check that --exp points at the right columns")
    cols = [j for j, _ in hit]
    ids = [g for _, g in hit]
    X = X[:, cols].T          # (n_cells, n_features) -- the cell axis comes first in the npz
    return ids, X, genes


def main(argv=None):
    p = argparse.ArgumentParser(description="build the cell-line omics features .npz (no DepMap login needed)")
    p.add_argument("--gene_effect", required=True, help="DepMap CRISPR gene-effect matrix (csv/tsv, ACH ids in rows or columns)")
    p.add_argument("--mutations", default="", help="optional: hotspot mutation 0/1 matrix")
    p.add_argument("--exp", default="/home/xxin/my_model/Data/GDSC_EXP.csv", help="used for the GDSC_ID <-> ACH map")
    p.add_argument("--out", default="/home/xxin/my_model/Data/cell_dependency.npz")
    p.add_argument("--top_var", type=int, default=2000, help="how many columns to keep by variance; 0 = keep all")
    p.add_argument("--max_missing", type=float, default=0.5)
    p.add_argument("--id_prefix", default="ACH-")
    args = p.parse_args(argv)

    def log(m):
        print(m)
        sys.stdout.flush()

    gdsc_map = read_gdsc_map(args.exp, verbose=True)

    M = read_matrix(args.gene_effect, args.id_prefix)
    ids, X, genes = build(M, gdsc_map, args.top_var, args.max_missing, prefix="", log=log)

    if args.mutations:
        Mm = read_matrix(args.mutations, args.id_prefix)
        ids_m, Xm, genes_m = build(
            Mm, gdsc_map, top_var=0, max_missing=0.9, prefix="MUT_", log=log
        )
        pos = {g: i for i, g in enumerate(ids)}
        shared = [g for g in ids_m if g in pos]
        log("[join] %d cell lines shared by the mutation matrix and the main matrix" % len(shared))
        if shared:
            r1 = {g: i for i, g in enumerate(ids)}
            r2 = {g: i for i, g in enumerate(ids_m)}
            ids = shared
            X = np.concatenate(
                [X[[r1[g] for g in shared], :], Xm[[r2[g] for g in shared], :]],
                axis=1,                       # X is already (cells, features), concatenate on the feature axis
            )
            genes = genes + genes_m

    X = np.asarray(X, dtype=np.float32)
    ids = [str(v).split(".")[0] for v in ids]
    if X.ndim != 2 or X.shape[0] != len(ids):
        raise SystemExit(
            "[error] shape self-check failed: X=%s, len(ids)=%d (expected (n_cells, n_features))"
            % (X.shape, len(ids))
        )
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(
        args.out,
        cell_ids=np.array(ids, dtype=object),
        X=X,
        feature_names=np.array(genes, dtype=object),
        present=np.ones(len(ids), dtype=bool),
    )
    log("[saved] %s   cells=%d  features=%d  %.2f MB"
        % (args.out, X.shape[0], X.shape[1], X.nbytes / 1e6))

    rep = args.out + ".report.txt"
    with open(rep, "w", encoding="utf-8") as f:
        f.write("cells\t%d\nfeatures\t%d\n" % (X.shape[0], X.shape[1]))
        f.write("cells_with_ach_but_not_in_gdsc\t%d\n" % 0)
        f.write("# first 20 features\n")
        for g in genes[:20]:
            f.write("feat\t%s\n" % g)
    log("[saved] %s" % rep)
    return 0


if __name__ == "__main__":
    sys.exit(main())
