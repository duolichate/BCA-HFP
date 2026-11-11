# bca_hfp/analysis/parse_cv_log.py
"""parse the train_cv terminal log into a training-history CSV.

why: train_cv overwrites attention_drug_training_history.csv every fold, so only the last fold
survives. the log keeps every epoch of every fold, so history can be rebuilt from it.

usage:
python bca_hfp/analysis/parse_cv_log.py \
  --log ../Result/v4_E7.txt \
  --out ../Result/diag/history_e7_seed42.csv
"""

import argparse
import os
import re

import pandas as pd

# Fold 4 | Epoch  12 |    767s | LR 1.00e-05 | TrLoss 2.8662 | TrR2 +0.7128 |
# VaLoss 5.3740 | VaR2 -0.0890 | VaMAE 1.8471 | VaRMSE 2.3182 | VaPearson 0.5968 |
# Aux 0.8628 | patience 0/off | T 0.981
PAT = re.compile(
    r'^Fold\s+(?P<fold>\d+)\s*\|\s*Epoch\s+(?P<epoch>\d+)\s*\|\s*(?P<sec>\d+)s\s*\|'
    r'\s*LR\s+(?P<lr>[0-9.eE+-]+)\s*\|'
    r'\s*TrLoss\s+(?P<train_loss>[0-9.eE+-]+)\s*\|'
    r'\s*TrR2\s+(?P<train_r2>[+-]?[0-9.]+)\s*\|'
    r'\s*VaLoss\s+(?P<val_loss>[0-9.eE+-]+)\s*\|'
    r'\s*VaR2\s+(?P<val_r2>[+-]?[0-9.]+)\s*\|'
    r'\s*VaMAE\s+(?P<val_mae>[+-]?[0-9.]+)\s*\|'
    r'\s*VaRMSE\s+(?P<val_rmse>[+-]?[0-9.]+)\s*\|'
    r'\s*VaPearson\s+(?P<val_pearson>[+-]?[0-9.]+)\s*\|'
    r'\s*Aux\s+(?P<aux>[+-]?[0-9.]+)'
)

NUM = ('lr', 'train_loss', 'train_r2', 'val_loss', 'val_r2',
       'val_mae', 'val_rmse', 'val_pearson', 'aux')
INT = ('fold', 'epoch', 'sec')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--log', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    rows = []
    with open(args.log, 'r', errors='ignore') as f:
        for line in f:
            m = PAT.match(line.strip())
            if m:
                rows.append(m.groupdict())
    if not rows:
        raise SystemExit('no epoch line parsed, check the log format')

    df = pd.DataFrame(rows)
    for c in INT:
        df[c] = df[c].astype(int)
    for c in NUM:
        df[c] = df[c].astype(float)
    df = df.sort_values(['fold', 'epoch']).reset_index(drop=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    df.to_csv(args.out, index=False)

    print(f'[ok] {len(df)} rows -> {args.out}')
    print(f'     folds={df.fold.nunique()}  epochs/fold={df.groupby("fold").size().to_dict()}')
    print('\nbest val R2 per fold:')
    for f, s in df.groupby('fold'):
        i = s.val_r2.idxmax()
        print(f'  fold{f}: R²={s.loc[i, "val_r2"]:+.4f} @ epoch {s.loc[i, "epoch"]}  '
              f'Pearson={s.loc[i, "val_pearson"]:.4f}  RMSE={s.loc[i, "val_rmse"]:.4f}')


if __name__ == '__main__':
    main()
