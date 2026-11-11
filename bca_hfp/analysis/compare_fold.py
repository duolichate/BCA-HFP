#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""single-fold comparison: put base(OOF) and a few predictors on the same fold and the same pairs, compare within-drug metrics.

why: ib0 only ran fold 1, so whether it beats cellmean / the two-stage baseline has to be judged
on the fold-1 subset, not on another fold or on global numbers.

usage:
  $PY bca_hfp/analysis/compare_fold.py \
      --base /home/xxin/my_model/Result/diag/oof_swa_v3.csv \
      --fold 1 --add-cellmean \
      --pred twostage=/path/pred_r16_a0.4.csv \
      --pred ib0=/path/attention_drug_fold1_ib0_test_predictions.csv
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

D_COL, C_COL, Y_COL, F_COL = "DRUG_ID", "GDSC_ID", "true_label", "fold"
P_CANDS = ["predicted", "yhat", "pred", "pred_label", "y_pred", "prediction"]
D_NAME, C_NAME = ["DRUG_ID", "DRUG_ID", "drug_id", "DRUG"], ["GDSC_ID", "GDSC_ID", "cell_line", "COSMIC_ID", "GENE"]


def _pick(df, cands):
    for c in cands:
        if c in df.columns:
            return c
    return None


def log(*a):
    print(*a); sys.stdout.flush()


def load_base(path):
    df = pd.read_csv(path)
    pc = _pick(df, P_CANDS)
    if F_COL not in df.columns or pc is None:
        raise SystemExit("[error] base is missing fold or prediction column: %s" % list(df.columns))
    return df.rename(columns={pc: "__base"})[[D_COL, C_COL, F_COL, Y_COL, "__base"]]


def load_pred(path, name):
    df = pd.read_csv(path)
    dc = _pick(df, D_NAME); cc = _pick(df, C_NAME); pc = _pick(df, P_CANDS)
    if dc is None or cc is None or pc is None:
        raise SystemExit("[error] %s column names not recognized: %s" % (name, list(df.columns)))
    out = df.rename(columns={dc: D_COL, cc: C_COL, pc: ("pred_" + name)})[[D_COL, C_COL, "pred_" + name]]
    out[D_COL] = out[D_COL].astype(str); out[C_COL] = out[C_COL].astype(str)
    return out


def dedupe(df, keys, how):
    if how == "none":
        return df
    agg = {c: ("mean" if c not in keys else "first") for c in df.columns if c not in keys}
    return df.groupby(keys, as_index=False).agg(agg)


def add_cellmean(df, fold_col=F_COL, cell_col=C_COL, y_col=Y_COL):
    out = np.full(len(df), np.nan)
    fold = df[fold_col].values; cell = df[cell_col].values; y = df[y_col].values
    for f in np.unique(fold):
        tr = fold != f
        if not tr.any():
            continue
        m = pd.Series(y[tr]).groupby(cell[tr]).mean()
        te = ~tr
        got = pd.Series(cell[te]).map(m).values
        out[te] = got
    if np.isnan(out).any():
        out = np.where(np.isnan(out), np.nanmean(y), out)
    return out


def within_r2(drug, y, p):
    y = np.asarray(y, float); p = np.asarray(p, float); d = np.asarray(drug)
    ok = np.isfinite(y) & np.isfinite(p)
    y, p, d = y[ok], p[ok], d[ok]
    _, inv = np.unique(d, return_inverse=True)
    cnt = np.bincount(inv); sy = np.bincount(inv, y); sp = np.bincount(inv, p)
    yc = y - (sy / cnt)[inv]; pc = p - (sp / cnt)[inv]
    sst = float((yc ** 2).sum()); sse = float(((pc - yc) ** 2).sum())
    r2 = 1.0 - sse / sst if sst > 0 else float("nan")
    rho = float(np.corrcoef(pc, yc)[0, 1]) if yc.std() > 0 and pc.std() > 0 else float("nan")
    return r2, rho, int(ok.sum()), int(len(cnt))


def per_drug_rho(drug, y, p, min_n=3):
    y = np.asarray(y, float); p = np.asarray(p, float); d = np.asarray(drug)
    vals = []
    for _, idx in pd.Series(d).groupby(d).groups.items():
        if len(idx) < min_n:
            continue
        yy = y[idx] - y[idx].mean(); pp = p[idx] - p[idx].mean()
        if yy.std() == 0 or pp.std() == 0:
            continue
        vals.append(float(np.corrcoef(pp, yy)[0, 1]))
    return (float(np.median(vals)) if vals else float("nan")), len(vals)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--base-name", default="model")
    ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--pred", nargs="*", default=[])
    ap.add_argument("--add-cellmean", action="store_true")
    ap.add_argument("--dedupe", choices=["mean", "none"], default="mean")
    args = ap.parse_args(argv)

    base = load_base(args.base)
    base[D_COL] = base[D_COL].astype(str); base[C_COL] = base[C_COL].astype(str)
    if args.add_cellmean:
        base["pred_cellmean"] = add_cellmean(base)

    preds = {}
    for spec in args.pred:
        if "=" not in spec:
            raise SystemExit("[error] --pred expects name=path: %s" % spec)
        name, path = spec.split("=", 1)
        if not os.path.exists(path):
            raise SystemExit("[error] file not found: %s" % path)
        preds[name] = load_pred(path, name)

    keys = [D_COL, C_COL]
    m = dedupe(base, keys, args.dedupe)
    for name, d in preds.items():
        d = dedupe(d, keys, args.dedupe)
        before = len(m)
        m = m.merge(d, on=keys, how="inner")
        if len(m) < before:
            log("[warn] %s: %d fewer pairs after merge" % (name, before - len(m)))

    sub = m[m[F_COL] == args.fold].reset_index(drop=True)
    if len(sub) == 0:
        raise SystemExit("[error] fold=%d has no data" % args.fold)

    pcols = [c for c in sub.columns if c.startswith("pred_")]
    names = [c[5:] for c in pcols]
    if ("__base" in sub.columns):
        pass

    log("=" * 78)
    log("fold %d within-fold comparison  n_pairs=%d  n_drugs=%d  (dedupe=%s)" %
        (args.fold, len(sub), sub[D_COL].nunique(), args.dedupe))
    log("=" * 78)
    log("%-14s %10s %10s %14s %8s" % ("name", "withinR2", "withinP", "perDrugRhoMed", "n_drug"))
    rows = []
    entries = [(args.base_name, sub["__base"].values)] + [(n, sub["pred_" + n].values) for n in names]
    for nm, pv in entries:
        if nm == args.base_name and args.base_name in names:
            pass
        r2, rho, n_ok, n_d = within_r2(sub[D_COL].values, sub[Y_COL].values, pv)
        med, k = per_drug_rho(sub[D_COL].values, sub[Y_COL].values, pv)
        rows.append((nm, r2, rho, med))
        log("%-14s %+10.4f %+10.4f %+14.4f %8d" % (nm, r2, rho, med, k))

    log("-" * 78)
    log("delta vs %s (same fold, same pairs):" % args.base_name)
    b = rows[0]
    for nm, r2, rho, med in rows[1:]:
        log("  %-12s dWithinR2=%+.4f  dWithinP=%+.4f  dPerDrugRhoMed=%+.4f" % (nm, r2 - b[1], rho - b[2], med - b[3]))
    log("")
    log("reading: ib0 counts as a win only if withinR2 beats cellmean and is close to or above twostage")
    return 0


if __name__ == "__main__":
    sys.exit(main())
