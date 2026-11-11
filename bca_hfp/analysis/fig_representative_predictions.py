#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""pick representative cases from the per-sample OOF predictions and draw a scatter.

reads the pooled OOF table from pool_oof_stats.py (columns: DRUG_ID, GROUP, fold, GDSC_ID, y, p),
picks the smallest-error samples at the resistant end (high ln IC50, below the clipping window)
and at the sensitive end (low ln IC50), highlights them on the full predicted-vs-observed scatter
and labels the drug name.
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def pick(df, lo, hi, n):
    sub = df[(df.y >= lo) & (df.y <= hi)]
    return sub.nsmallest(n, "abs_resid")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof_csv", required=True, help="pooled OOF csv from pool_oof_stats.py")
    ap.add_argument("--out_png", required=True, help="output PNG path")
    ap.add_argument("--res_lo", type=float, default=3.0, help="resistant-end ln IC50 lower bound")
    ap.add_argument("--res_hi", type=float, default=4.5, help="resistant-end ln IC50 upper bound (stays below the clip)")
    ap.add_argument("--sens_hi", type=float, default=-1.5, help="sensitive-end ln IC50 upper bound")
    ap.add_argument("--n_pair", type=int, default=2, help="cases per resistant / sensitive end")
    ap.add_argument("--dpi", type=int, default=180)
    args = ap.parse_args()

    df = pd.read_csv(args.oof_csv)
    df["abs_resid"] = (df.y - df.p).abs()

    res = pick(df, args.res_lo, args.res_hi, args.n_pair)
    sens = pick(df, -1e6, args.sens_hi, args.n_pair)
    picks = pd.concat([res, sens])

    cols = ["DRUG_ID", "GROUP", "fold", "GDSC_ID", "y", "p", "abs_resid"]
    print("selected:")
    print(picks[cols].to_string(index=False))

    fig, ax = plt.subplots(figsize=(8, 7.2))
    ax.scatter(df.p, df.y, s=1.5, c="#c9c9c9", alpha=0.25, linewidths=0,
               label=f"all (n={len(df)})")
    lim = [float(df.y.min()), float(df.y.max())]
    ax.plot(lim, lim, "k--", lw=1.0, label="y = p")

    ax.scatter(res.p, res.y, s=110, c="#d62728", edgecolors="black",
               linewidths=0.8, zorder=5, label="resistant")
    ax.scatter(sens.p, sens.y, s=110, c="#1f77b4", edgecolors="black",
               linewidths=0.8, zorder=5, label="sensitive")
    for _, r in picks.iterrows():
        ax.annotate(str(r.GROUP), (r.p, r.y), textcoords="offset points",
                    xytext=(8, 6), fontsize=9, zorder=6)

    ax.set_xlabel("Predicted ln IC50")
    ax.set_ylabel("Observed ln IC50")
    ax.set_title("Representative predictions (OOF, drug split)")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()

    os.makedirs(os.path.dirname(args.out_png), exist_ok=True)
    plt.savefig(args.out_png, dpi=args.dpi)
    print("saved:", args.out_png)


if __name__ == "__main__":
    main()