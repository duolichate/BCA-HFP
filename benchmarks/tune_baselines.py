#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run the full sweep over the baselines in sequence (unified protocol, offset in time so the
main model and the baselines never contend for GPU memory):
  each model first tries 3 lrs (paper value /3, x1, x3) on fold 1 only and picks the best by
  val best_val_r2, then runs the full 5 folds with that lr (--pred_clamp) and writes to
  5_Result/2_baselines/<Model>/CV/.

usage:
  python benchmarks/tune_baselines.py                  # all 5 models in sequence
  python benchmarks/tune_baselines.py --dry-run        # print the commands only
  python benchmarks/tune_baselines.py --only tcnns transcdr
  python benchmarks/tune_baselines.py --envs D:\\anaconda3\\envs

conda envs root resolution order: --envs > BENCH_ENVS env var > $CONDA_PREFIX/../envs >
common default paths. paths differ per machine, the old hardcoded C:\\Users\\dell\\.conda\\envs
breaks as soon as the box changes.

the full 5-fold run passes --variant, output dir is <env>_<variant>_<timestamp> (env is
detected by each script from the GPU name).
"""

import argparse
import csv
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_ROOT = os.path.join(os.path.dirname(ROOT), "5_Result", "2_baselines")

DATASET = "GDSC"
GENE = "V1"
EPOCHS = "200"

# (dir_name, model_label, conda_env, batch_size, base_lr)
MODELS = [
    ("tcnns",       "tCNNS",       "bench-tcnns",       100,  1e-4),
    ("transcdr",    "TransCDR",    "bench-transcdr",    64,   1e-5),
    ("paccmann",    "PaccMann",    "bench-paccmann",    256,  0.01),
    ("gratransdrp", "GraTransDRP", "bench-gratransdrp", 1024, 1e-4),
    ("graphdrp",    "GraphDRP",    "bench-graphdrp",    1024, 1e-3),
]

# variant ids for the official 5-fold results (same as base1-6 in the run notes; base1 goes to DeepDR)
VARIANT = {
    "GraphDRP":    "v1_base2_GraphDRP",
    "GraTransDRP": "v1_base3_GraTransDRP",
    "TransCDR":    "v1_base4_TransCDR",
    "PaccMann":    "v1_base5_PaccMann",
    "tCNNS":       "v1_base6_tCNNS",
}


def envs_root(cli_value=''):
    """locate the conda envs dir used by the per-model environments."""
    for cand in (cli_value, os.environ.get("BENCH_ENVS", "")):
        if cand:
            return cand
    prefix = os.environ.get("CONDA_PREFIX", "")
    if prefix:
        cand = os.path.join(os.path.dirname(prefix), "envs")
        if os.path.isdir(cand):
            return cand
    for cand in (os.path.expanduser("~/anaconda3/envs"),
                 os.path.expanduser("~/miniconda3/envs"),
                 r"D:\ProgramData\anaconda3\envs"):
        if os.path.isdir(cand):
            return cand
    return ""


def dir_label(batch, lr):
    """dir name for sweep intermediates, must match the non-variant branch in each train_*.py."""
    return f"{DATASET}_{GENE}_bs{batch}_lr{lr:g}"


def read_best_val_r2(path):
    with open(path, newline="") as f:
        row = next(csv.DictReader(f))
    return float(row["best_val_r2"])


def build_cmd(m, batch, lr, envs, only_folds=None, pred_clamp=False, variant=''):
    dirn, _label, env, _bs, _lr0 = m
    py = os.path.join(envs, env, "python.exe")
    script = os.path.join(ROOT, "benchmarks", dirn, f"train_{dirn}.py")
    cmd = [py, script, "--dataset", DATASET, "--gene_version", GENE,
           "--batch_size", str(batch), "--lr", repr(lr), "--epochs", EPOCHS]
    if only_folds:
        cmd += ["--only_folds", only_folds]
    if pred_clamp:
        cmd += ["--pred_clamp"]
    if variant:
        cmd += ["--variant", variant]
    return cmd


def run_one(m, dry_run, envs):
    dirn, label, env, batch, lr0 = m
    lrs = [lr0 / 3.0, lr0, lr0 * 3.0]
    best_lr, best_r2 = None, float("-inf")

    for lr in lrs:
        outdir = os.path.join(RESULT_ROOT, label, "CV", dir_label(batch, lr))
        cmd = build_cmd(m, batch, lr, envs, only_folds="1")
        print(f"[{label}] TUNE lr={lr:g}", flush=True)
        if dry_run:
            print("    " + " ".join(cmd), flush=True)
            continue
        r = subprocess.run(cmd, cwd=ROOT)
        if r.returncode != 0:
            print(f"[{label}] TUNE lr={lr:g} FAILED rc={r.returncode}", flush=True)
            continue
        val = read_best_val_r2(os.path.join(outdir, "cv_folds.csv"))
        print(f"[{label}] TUNE lr={lr:g} best_val_r2={val:.6f}", flush=True)
        if val > best_r2:
            best_r2, best_lr = val, lr

    if dry_run or best_lr is None:
        if not dry_run and best_lr is None:
            print(f"[{label}] SKIP 5-fold (no valid lr)", flush=True)
        return

    print(f"[{label}] SELECT lr={best_lr:g} (val best_val_r2={best_r2:.6f})", flush=True)
    variant = VARIANT.get(label, '')
    cmd = build_cmd(m, batch, best_lr, envs, pred_clamp=True, variant=variant)
    print(f"[{label}] RUN 5-fold: {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, cwd=ROOT)
    print(f"[{label}] 5-fold done rc={r.returncode}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", nargs="*", default=None, help="run only these models (default: all)")
    ap.add_argument("--envs", default="", help="conda envs root (default: autodetect)")
    args = ap.parse_args()

    envs = envs_root(args.envs)
    if not envs:
        raise SystemExit("conda envs dir not found, pass --envs or set BENCH_ENVS")
    print(f"envs root: {envs}", flush=True)

    todo = MODELS
    if args.only:
        by_name = {m[1].lower(): m for m in MODELS}
        todo = [by_name[n.lower()] for n in args.only if n.lower() in by_name]

    for m in todo:
        run_one(m, args.dry_run, envs)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
