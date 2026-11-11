"""Queue launcher for the training runs and ablations behind the paper.

Keeps at most MAX_PARALLEL train_cv processes alive and starts the next job as
soon as a slot frees up. Each job writes to its own results directory, so a
long queue can run unattended.

Usage: python scripts/run_queue.py
"""
import os
import subprocess
import sys
import time
from datetime import datetime

from bca_hfp.config.base_config import PROJECT_ROOT

LOG = os.path.join(PROJECT_ROOT, 'scripts', 'run_queue.log')
FOLD_CSV = os.path.join(PROJECT_ROOT, 'data', 'fold_drug_grouped_5fold.csv')
MAX_PARALLEL = 3
POLL_SEC = 60


def log(msg):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write(line + '\n')
    print(line, flush=True)


def build(variant, layer='3', reg_norm='batchnorm', seed='42', split_seed=None,
          gate_content=True, use_desc=True, split_method='drug', group_by_name=True,
          fold_csv=FOLD_CSV, fp_bits=None, gate_hidden=None):
    a = ['--dataset', 'GDSC', '--env', 'auto', '--variant', variant,
         '--layer', layer, '--gene_version', 'V1', '--split_method', split_method]
    if group_by_name:
        a += ['--group_by_name']
    a += ['--reg_norm', reg_norm, '--n_splits', '5', '--seed', seed]
    if split_seed:
        a += ['--split_seed', split_seed]
    a += ['--batch_size', '32', '--grad_accum', '4', '--max_epochs', '20',
          '--no_early_stop', '--loss', 'mse', '--drug_mean_weight', '0.5',
          '--drug_global_dropout', '0.0', '--global_gene_dropout', '0.0',
          '--pred_clamp']
    if use_desc:
        a += ['--use_desc']
    a += ['--swa_last_k', '5', '--num_workers', '4']
    if fold_csv:
        a += ['--fold_csv', fold_csv]
    if gate_content:
        a += ['--gate_content']
    if fp_bits:
        a += ['--fp_bits', str(fp_bits)]
    if gate_hidden:
        a += ['--gate_hidden', str(gate_hidden)]
    return a


QUEUE = [
    # reported runs (seeds 42 and 43 on the same folds)
    ('v1_gc_c5_s42', build('v1_gc_c5_s42', seed='42', split_seed='42')),
    ('v1_gc_c5_s43', build('v1_gc_c5_s43', seed='43', split_seed='42')),
    # fingerprint length ablation
    ('v1_gc_fp256', build('v1_gc_fp256', fp_bits=256)),
    ('v1_gc_fp512', build('v1_gc_fp512', fp_bits=512)),
    ('v1_gc_fp1024', build('v1_gc_fp1024', fp_bits=1024)),
    ('v1_gc_fp4096', build('v1_gc_fp4096', fp_bits=4096)),
    # fusion depth
    ('v1_gc_l1', build('v1_gc_l1', layer='1')),
    ('v1_gc_l2', build('v1_gc_l2', layer='2')),
    # scaffold split instead of drug-name split
    ('v1_gc_scaf', build('v1_gc_scaf', split_method='scaffold',
                         group_by_name=False, fold_csv=None)),
    # capacity control and normalisation variant
    ('v1_gc_cap', build('v1_gc_cap', gate_hidden=506)),
    ('v1_gc_ln', build('v1_gc_ln', reg_norm='layernorm')),
]


def running():
    try:
        import psutil
    except ImportError:
        return 0
    n = 0
    for p in psutil.process_iter(['cmdline']):
        try:
            cmd = ' '.join(p.info['cmdline'] or [])
        except Exception:
            continue
        if 'bca_hfp.training.train_cv' in cmd:
            n += 1
    return n


def main():
    log(f"queue begin: {len(QUEUE)} jobs, max_parallel={MAX_PARALLEL}")
    while QUEUE:
        n = running()
        if n < MAX_PARALLEL:
            variant, job = QUEUE.pop(0)
            p = subprocess.Popen([sys.executable, '-m', 'bca_hfp.training.train_cv'] + job,
                                 cwd=PROJECT_ROOT,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            log(f"started {variant} pid={p.pid} (running before={n})")
            time.sleep(20)
        else:
            time.sleep(POLL_SEC)
    log("queue empty: all done")


if __name__ == '__main__':
    main()
