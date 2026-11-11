# -*- coding: utf-8 -*-
"""
bootstrap_ci.py -- drug-block bootstrap confidence intervals for OOF drug-response predictions
=================================================================================

why not the mean±sd of the 5 folds
-------------------------
under LDO the 5 folds give only 5 numbers and the between-fold sd is often around 0.2, so the 95% CI
of the mean is as wide as ±0.11 and no 0.02~0.03 improvement can be resolved. the real independent unit
is the **drug** (all pairs of one drug share the same model bias), so drugs are resampled as blocks here
and the paired difference is taken on the same resampled drugs.

usage
----
  python bca_hfp/analysis/bootstrap_ci.py \
      --base  Result/diag/oof_swa_v3.csv \
      --pred  cellmean=Result/diag/pred_cellmean.csv \
      --pred  twostage=Result/diag/pred_twostage.csv \
      --n-boot 1000 --output-dir Result/diag/bootstrap

every csv needs the columns DRUG_ID, GDSC_ID and a prediction column (--pred-col, default predicted;
yhat / pred / pred_label are also accepted). the --base csv additionally needs true_label (or y / LN_IC50).

output
----
  bootstrap_summary.txt   point estimate + 95% CI + paired differences + per-fold reference
  bootstrap_draws.csv     metrics of every resample (for custom plots)

all metrics are computed on pairs, but the resampling unit is the drug; within metrics are centered per drug first, then pooled.
"""
import argparse
import os

import numpy as np
import pandas as pd

KEY = ["DRUG_ID", "GDSC_ID"]
_DEDUPE = "mean"
Y_CANDS = ["true_label", "y", "label", "LN_IC50", "ic50"]
P_CANDS = ["predicted", "yhat", "pred", "pred_label", "y_pred"]


def log(msg=""):
    print(msg)


def _pick(df, cands, what):
    for c in cands:
        if c in df.columns:
            return c
    raise SystemExit("[error] %s column not found: tried %s, have %s"
                     % (what, cands, list(df.columns)))


def load_frame(path, pred_col=None, need_y=False):
    df = pd.read_csv(path)
    df[KEY[0]] = df[KEY[0]].astype(str)
    df[KEY[1]] = df[KEY[1]].astype(str)
    if pred_col is not None:
        if pred_col not in df.columns:
            raise SystemExit("[error] %s has no column %s" % (path, pred_col))
        p = pred_col
    else:
        p = _pick(df, P_CANDS, "prediction")
    out = {KEY[0]: df[KEY[0]], KEY[1]: df[KEY[1]], "pred": df[p].astype(float)}
    if need_y:
        yc = _pick(df, Y_CANDS, "label")
        out["y"] = df[yc].astype(float)
    if "fold" in df.columns:
        out["fold"] = df["fold"].values
    out = pd.DataFrame(out)
    n0 = len(out)
    if not out.duplicated(KEY).any():
        return out
    if _DEDUPE == "none":
        log("[warn] %s has %d rows with duplicate keys, --dedupe none: kept as is"
            % (os.path.basename(path), n0 - len(out.drop_duplicates(KEY))))
        return out
    if _DEDUPE == "first":
        out = out.drop_duplicates(KEY, keep="first")
        how = "keeping the first row"
    else:
        agg = {c: "mean" for c in out.columns if c not in KEY}
        out = out.groupby(KEY, as_index=False).agg(agg)
        how = "averaging"
    log("[warn] %s has duplicate (DRUG_ID, GDSC_ID) keys, deduplicated by %s: %d -> %d"
        % (os.path.basename(path), how, n0, len(out)))
    return out


def merge_all(base, base_name, others):
    """align on (DRUG_ID, GDSC_ID), keep only pairs present in every predictor."""
    df = base.rename(columns={"pred": "pred_" + base_name})
    for name, d in others.items():
        df = df.merge(d.rename(columns={"pred": "pred_" + name}), on=KEY, how="inner")
    return df


def drug_stats(y, p, didx, n_drugs):
    """per-drug sufficient statistics, so any weighted resample needs only an O(1) reduction."""
    n = np.bincount(didx, minlength=n_drugs).astype(np.float64)
    sy = np.bincount(didx, weights=y, minlength=n_drugs)
    sp = np.bincount(didx, weights=p, minlength=n_drugs)
    sy2 = np.bincount(didx, weights=y * y, minlength=n_drugs)
    sp2 = np.bincount(didx, weights=p * p, minlength=n_drugs)
    syp = np.bincount(didx, weights=y * p, minlength=n_drugs)
    sse_raw = np.bincount(didx, weights=(p - y) ** 2, minlength=n_drugs)  # not centered
    return dict(n=n, sy=sy, sp=sp, sy2=sy2, sp2=sp2, syp=syp, sse_raw=sse_raw)


def metrics(stats, w):
    """compute all metrics for drug weights w (resample counts)."""
    n = stats["n"]
    N = float(np.sum(w * n))
    if N <= 2:
        return None
    sy = float(np.sum(w * stats["sy"]))
    sp = float(np.sum(w * stats["sp"]))
    sy2 = float(np.sum(w * stats["sy2"]))
    sp2 = float(np.sum(w * stats["sp2"]))
    syp = float(np.sum(w * stats["syp"]))
    sse_raw = float(np.sum(w * stats["sse_raw"]))

    # ---- per pair (pooled) ----
    sst = sy2 - sy * sy / N
    sse = sse_raw - (sp - sy) ** 2 / N  # equivalent to removing the global mean offset
    r2 = 1.0 - sse / sst if sst > 1e-12 else np.nan
    cov = syp - sp * sy / N
    vp = sp2 - sp * sp / N
    vy = sst
    pear = cov / np.sqrt(vp * vy) if vp > 1e-12 and vy > 1e-12 else np.nan

    # ---- within drug (centered per drug, then pooled) ----
    nz = np.maximum(n, 1.0)
    sst_w = stats["sy2"] - np.where(n > 0, stats["sy"] ** 2 / nz, 0.0)
    sse_w = (stats["sse_raw"]
             - np.where(n > 0, (stats["sp"] - stats["sy"]) ** 2 / nz, 0.0))
    vp_w = stats["sp2"] - np.where(n > 0, stats["sp"] ** 2 / nz, 0.0)
    cov_w = stats["syp"] - np.where(n > 0, stats["sp"] * stats["sy"] / nz, 0.0)
    SW_SST = float(np.sum(w * sst_w))
    SW_SSE = float(np.sum(w * sse_w))
    SW_VP = float(np.sum(w * vp_w))
    SW_COV = float(np.sum(w * cov_w))
    wr2 = 1.0 - SW_SSE / SW_SST if SW_SST > 1e-12 else np.nan
    wpear = SW_COV / np.sqrt(SW_VP * SW_SST) if SW_VP > 1e-12 and SW_SST > 1e-12 else np.nan

    # ---- drug level ----
    with np.errstate(divide="ignore", invalid="ignore"):
        my = stats["sy"] / nz
        mp = stats["sp"] / nz
    return dict(r2=r2, pearson=pear, within_r2=wr2, within_pearson=wpear,
                mean_y=sy / N, mean_p=sp / N)


def per_drug_rho(y, p, didx, n_drugs, min_n=3):
    """per-drug Pearson; nan when there are too few pairs or zero variance."""
    rho = np.full(n_drugs, np.nan)
    for d in range(n_drugs):
        m = didx == d
        if m.sum() < min_n:
            continue
        yy = y[m]
        pp = p[m]
        if np.std(yy) < 1e-9 or np.std(pp) < 1e-9:
            continue
        rho[d] = np.corrcoef(yy, pp)[0, 1]
    return rho


def rho_median(rho, w):
    ok = np.isfinite(rho) & (w > 0)
    if ok.sum() == 0:
        return np.nan
    vals = np.repeat(rho[ok], w[ok].astype(int))
    return float(np.median(vals)) if len(vals) else np.nan


def ci_of(arr):
    a = np.asarray([v for v in arr if np.isfinite(v)], dtype=float)
    if a.size == 0:
        return np.nan, np.nan, np.nan
    return float(np.mean(a)), float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def p_two_sided(diffs):
    a = np.asarray([v for v in diffs if np.isfinite(v)], dtype=float)
    if a.size == 0:
        return np.nan
    frac = float((a > 0).mean())
    return 2.0 * min(frac, 1.0 - frac)


def main(argv=None):
    ap = argparse.ArgumentParser(description="drug-block bootstrap confidence intervals for OOF metrics")
    ap.add_argument("--base", required=True, help="reference csv (the model), must contain true_label")
    ap.add_argument("--base-name", default="model")
    ap.add_argument("--pred", action="append", default=[],
                    help="name=path, repeatable; paired difference against base")
    ap.add_argument("--pred-col", default=None, help="prediction column name (same in every file)")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-drug-n", type=int, default=3,
                    help="minimum pairs per drug for the per-drug correlation")
    ap.add_argument("--add-cellmean", action="store_true",
                    help="add a cell-line mean baseline (leave-fold, no leakage) as an extra predictor")
    ap.add_argument("--dedupe", default="mean", choices=["mean", "first", "none"],
                    help="handling of duplicate (DRUG_ID,GDSC_ID) keys: mean=default first=keep first row none=keep as is")
    ap.add_argument("--ref", default=None,
                    help="reference name for the paired difference (default = base); pass cellmean to compare twostage vs cellmean")
    ap.add_argument("--hybrid", action="append", default=[],
                    help="name=alt: hybrid predictor from the drug-level part of base plus the within-drug part of alt")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args(argv)

    global _DEDUPE
    _DEDUPE = args.dedupe
    os.makedirs(args.output_dir, exist_ok=True)
    base = load_frame(args.base, args.pred_col, need_y=True)
    n0 = len(base)

    if args.add_cellmean:
        # leave-fold cell-line mean: mean y of that cell line over the **other folds** (its own fold excluded)
        tot_s = base.groupby(KEY[1])["y"].transform("sum")
        tot_n = base.groupby(KEY[1])["y"].transform("count")
        fs = base.groupby([KEY[1], "fold"])["y"].transform("sum")
        fn = base.groupby([KEY[1], "fold"])["y"].transform("count")
        o_s = tot_s - fs
        o_n = tot_n - fn
        cm = np.where(o_n > 0, o_s / np.maximum(o_n, 1), tot_s / np.maximum(tot_n, 1))
        others_cm = pd.DataFrame({KEY[0]: base[KEY[0]], KEY[1]: base[KEY[1]],
                                  "pred": cm})
        log("[base] cell-line mean (leave-fold) baseline built")
    else:
        others_cm = None

    others = {}
    if others_cm is not None:
        others["cellmean"] = others_cm
    for spec in args.pred:
        if "=" not in spec:
            raise SystemExit("[error] --pred expects name=path, got %r" % spec)
        name, path = spec.split("=", 1)
        others[name.strip()] = load_frame(path, args.pred_col)

    df = merge_all(base, args.base_name, others)
    log("[data] base=%d merged=%d (dropped %d pairs: predictor coverage differs)"
        % (n0, len(df), n0 - len(df)))
    if len(df) < 1000:
        raise SystemExit("[error] too few samples after the merge, check that the keys agree")

    drugs = np.array(sorted(df[KEY[0]].unique()))
    dmap = {d: i for i, d in enumerate(drugs)}
    didx = df[KEY[0]].map(dmap).values.astype(np.int64)
    n_drugs = len(drugs)
    y = df["y"].values.astype(np.float64)
    log("[data] drugs=%d pairs=%d" % (n_drugs, len(df)))

    names = [args.base_name] + list(others.keys())
    preds = {args.base_name: df["pred_" + args.base_name].values.astype(np.float64)}
    for nm in others:
        preds[nm] = df["pred_" + nm].values.astype(np.float64)

    # ---- hybrid predictor: drug-level part of base (per-drug mean) + within-drug part of alt (minus its own per-drug mean) ----
    def _drug_level(p):
        n = np.bincount(didx, minlength=n_drugs).astype(np.float64)
        s = np.bincount(didx, weights=p, minlength=n_drugs)
        return (s / np.maximum(n, 1.0))[didx]

    for spec in args.hybrid:
        if "=" not in spec:
            raise SystemExit("[error] --hybrid expects name=alt, got %r" % spec)
        nm, alt = spec.split("=", 1)
        nm, alt = nm.strip(), alt.strip()
        if alt not in preds:
            raise SystemExit("[error] --hybrid %r is not among the predictors: %s" % (alt, names))
        hp = _drug_level(preds[args.base_name]) + (preds[alt] - _drug_level(preds[alt]))
        preds[nm] = hp
        names.append(nm)
        log("[hybrid] %s = drug level (%s) + within drug (%s)" % (nm, args.base_name, alt))

    ref_name = (args.ref or args.base_name).strip()
    if ref_name not in preds:
        raise SystemExit("[error] --ref %r is not among the predictors: %s" % (ref_name, names))

    stats = {nm: drug_stats(y, preds[nm], didx, n_drugs) for nm in names}
    rho_d = {nm: per_drug_rho(y, preds[nm], didx, n_drugs, args.min_drug_n)
             for nm in names}

    w_all = np.ones(n_drugs)
    point = {}
    for nm in names:
        m = metrics(stats[nm], w_all)
        m["rho_median"] = rho_median(rho_d[nm], w_all)
        point[nm] = m

    fold_tbl = {}
    if "fold" in df.columns:
        for f in sorted(df["fold"].unique()):
            m_ = (df["fold"] == f).values
            if m_.sum() < 50:
                continue
            for nm in names:
                # renumber drugs inside the fold
                uq = np.unique(didx[m_])
                sub = {k: v[uq] for k, v in stats[nm].items()}
                mm = metrics(sub, np.ones(len(uq)))
                fold_tbl.setdefault(nm, []).append(mm)

    # ---- bootstrap: metrics for every method and the paired differences in one loop ----
    rng = np.random.RandomState(args.seed)
    keys_ = ("r2", "pearson", "within_r2", "within_pearson", "rho_median")
    draws = {nm: {k: [] for k in keys_} for nm in names}
    diff_draws = {nm: {k: [] for k in ("r2", "pearson", "within_r2", "within_pearson")}
                  for nm in names if nm != ref_name}
    boot_rows = []
    for b in range(args.n_boot):
        w = np.bincount(rng.randint(0, n_drugs, size=n_drugs),
                        minlength=n_drugs).astype(np.float64)
        if w.sum() == 0:
            continue
        cur = {}
        for nm in names:
            m = metrics(stats[nm], w)
            if m is None:
                cur = None
                break
            m["rho_median"] = rho_median(rho_d[nm], w)
            cur[nm] = m
            for k in keys_:
                draws[nm][k].append(m[k])
            boot_rows.append(dict(boot=b, name=nm,
                                  within_r2=m["within_r2"],
                                  within_pearson=m["within_pearson"]))
        if cur is None:
            continue
        mb = cur[ref_name]
        for nm in names:
            if nm == ref_name:
                continue
            for k in diff_draws[nm]:
                diff_draws[nm][k].append(cur[nm][k] - mb[k])

    # ---- output ----
    out = []
    out.append("=" * 92)
    out.append("drug-block bootstrap  (n_drugs=%d, n_pairs=%d, B=%d, seed=%d)"
               % (n_drugs, len(df), args.n_boot, args.seed))
    out.append("=" * 92)
    out.append("%-14s %22s %22s %22s" % ("", "R2(pair)", "Pearson(pair)", "R2(within)"))
    out.append("%-14s %22s %22s %22s" % ("name", "point   95%CI", "point   95%CI", "point   95%CI"))
    for nm in names:
        mu_r, lo_r, hi_r = ci_of(draws[nm]["r2"])
        mu_p, lo_p, hi_p = ci_of(draws[nm]["pearson"])
        mu_w, lo_w, hi_w = ci_of(draws[nm]["within_r2"])
        out.append("%-14s %8.4f [%+.3f,%+.3f] %8.4f [%+.3f,%+.3f] %8.4f [%+.3f,%+.3f]"
                   % (nm, point[nm]["r2"], lo_r, hi_r,
                      point[nm]["pearson"], lo_p, hi_p,
                      point[nm]["within_r2"], lo_w, hi_w))
    out.append("")
    out.append("%-14s %22s %22s %22s" % ("", "within Pearson", "per-drug rho median", "(point R2 within)"))
    for nm in names:
        mu_wp, lo_wp, hi_wp = ci_of(draws[nm]["within_pearson"])
        mu_rm, lo_rm, hi_rm = ci_of(draws[nm]["rho_median"])
        out.append("%-14s %8.4f [%+.3f,%+.3f] %8.4f [%+.3f,%+.3f] %8.4f"
                   % (nm, point[nm]["within_pearson"], lo_wp, hi_wp,
                      point[nm]["rho_median"], lo_rm, hi_rm, point[nm]["within_r2"]))
    out.append("")
    out.append("paired difference (minus %s, subtracted on the same resampled drugs):" % ref_name)
    out.append("%-14s %26s %26s %10s" % ("vs base", "dR2(pair)", "dR2(within)", "p(within)"))
    for nm in names:
        if nm == ref_name:
            continue
        d = diff_draws[nm]
        mu1, lo1, hi1 = ci_of(d.get("r2", []))
        mu2, lo2, hi2 = ci_of(d.get("within_r2", []))
        pv = p_two_sided(d.get("within_r2", []))
        out.append("%-14s %+8.4f [%+.3f,%+.3f] %+8.4f [%+.3f,%+.3f] %10.3f"
                   % (nm, mu1, lo1, hi1, mu2, lo2, hi2, pv))
    out.append("")
    out.append("reading: a win needs the 95% interval of the difference to exclude 0 (= p<0.05); an interval containing 0 is noise.")
    out.append("")

    if fold_tbl:
        out.append("reference: 5-fold mean±sd (old definition, low resolution, only to show the difference)")
        out.append("%-14s %22s %22s" % ("name", "R2(pair)", "R2(within)"))
        for nm in names:
            rs = [m["r2"] for m in fold_tbl.get(nm, []) if m]
            ws = [m["within_r2"] for m in fold_tbl.get(nm, []) if m]
            if not rs:
                continue
            out.append("%-14s %+8.4f ± %-8.4f %+8.4f ± %-8.4f"
                       % (nm, float(np.mean(rs)), float(np.std(rs)),
                          float(np.mean(ws)), float(np.std(ws))))
        out.append("")

    txt = "\n".join(out)
    log(txt)
    p1 = os.path.join(args.output_dir, "bootstrap_summary.txt")
    with open(p1, "w", encoding="utf-8") as fh:
        fh.write(txt + "\n")
    pd.DataFrame(boot_rows).to_csv(
        os.path.join(args.output_dir, "bootstrap_draws.csv"), index=False)
    log("[saved] %s" % p1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
