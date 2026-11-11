"""
Benchmark models for comparison with BCA-HFP.
Each subdirectory contains an adapted training script aligned with the main model's
fixed 5-fold protocol (Data/Membership/fold_var3_fromckpt.csv + winsorized predictions)
plus the original upstream implementation.

- graphdrp/:    GraphDRP (Nguyen et al., 2022)          train_graphdrp.py
- gratransdrp/: GraTransDRP (Chu & Nguyen, 2023)        train_gratransdrp.py
- transcdr/:    TransCDR (Xia et al., 2024)             train_transcdr.py
- tcnns/:       tCNNS (Liu et al., 2019)                train_tcnns.py
- paccmann/:    PaccMann (Oskooei et al., 2019)         train_paccmann.py

environments (isolated from the main gene env, one conda env per model, named bench-<model>):
  neither machine can reach pypi, so always conda create --clone <main env>: clone gene on
  the server, gene5090 on the 5090 box; graph models additionally need torch_geometric
  (skip scatter/sparse if they fail to build, PyG 2.7 falls back to pure PyTorch). the clone
  shares the same torch as the main model, so numbers stay comparable.
  bash benchmarks/run_baselines.sh        # switch to each model's env and train in sequence

common protocol:
- split: fold_var3_fromckpt.csv maps samples to 5 folds by drug name (GROUP), val is cut from
  the non-test samples by drug name within each fold (0.125)
- evaluation: winsorize each fold's test predictions by concentration window, concatenate,
  then report R²/RMSE/MAE/Pearson/Spearman
- output: Result/2_baselines/<Model>/CV/<dataset>_<variant>/{cv_folds.csv, final_performance.csv}
"""
