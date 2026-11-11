# BCA-HFP

Code and result files for the manuscript **"Bidirectional Cross-Attention for Leave-Drug-Out
Anticancer Drug Response Prediction"** (submitted to *Pharmaceuticals*).

The study evaluates anticancer drug response prediction under a leave-drug-out protocol on
GDSC2: drugs are grouped by name (verified against InChIKey) and entire drug groups are held
out, so a test drug is never seen during training.

Repository: <https://github.com/duolichate/BCA-HFP> · Processed data: Zenodo
(`10.5281/zenodo.XXXXXXX`, see [Data](#data))

## Repository contents

| Path | Contents |
|---|---|
| `bca_hfp/` | Model, training and evaluation code (`models/`, `training/`, `data/`, `config/`, `utils/`) |
| `benchmarks/` | Training/inference wrappers for the five published baselines (GraphDRP, GraTransDRP, PaccMann, TransCDR, tCNNS) |
| `tools/` | Feature extraction scripts: Geneformer tokenisation/embedding for cell lines, and drug feature construction (fgBERT atom features + Morgan fingerprints) |
| `data/fold_drug_grouped_5fold.csv` | The five-fold drug-grouped split used in the paper (`GROUP,DRUG_ID,fold`, 242 drugs) |
| `results/seed42/`, `results/seed43/` | Per-sample out-of-fold test predictions for the two reported seeds, per-fold metrics (`attention_drug_cv_results.csv`) and the per-epoch training history |
| `results/tables/` | Per-fold and per-drug metric tables behind the main text and Supplementary Tables S1–S13 |
| `environment/env_rtx5090_20261112.txt` | GPU, driver, CUDA and full package inventory of the RTX 5090 workstation used for the reported runs |
| `scripts/run_queue.py` | Batch launcher with the command line of every run behind the paper (two reported seeds, fingerprint-length and depth ablations, scaffold split, capacity and normalisation controls) |
| `logs/` | Full training logs for the two reported runs (resolved configuration, per-epoch losses, per-fold test metrics) |

## Data

The GDSC2 dataset is publicly available from the Genomics of Drug Sensitivity in Cancer
consortium (<https://www.cancerrxgene.org/>). It is not redistributed here.

The processed files that the pipeline consumes directly (response table with fgBERT drug
features, basal expression matrices, gene lists, fold assignments and the environment dump)
are archived on Zenodo:

> **Zenodo DOI:** `10.5281/zenodo.XXXXXXX` <!-- replace with the published DOI -->
>
> Archive: `BCA-HFP_processed_data.zip` (≈445 MB)

Unpack the archive into a `Data/` directory next to `bca_hfp/`:

```bash
unzip BCA-HFP_processed_data.zip -d Data
```

Download the release and place the files so that the tree looks like:

```
Data/
├── Drug/
│   ├── GDSC_RES_FGBERT.h5        # response table + fgBERT drug features
│   └── GDSC_RES.csv
└── Gene/
    ├── GDSC_EXP.csv              # basal expression matrix (cell line x gene)
    ├── GDSC_EXP_ensembl.csv
    ├── global_gene_names_GDSC.txt
    └── GDSC_EXP_GF_V1/           # Geneformer V1 embeddings, one .pt file per cell line
```

`GDSC_EXP_GF_V1/` can be regenerated from `GDSC_EXP.csv` with `tools/geneformer/` and the
public Geneformer V1 checkpoint (<https://huggingface.co/ctheodoris/Geneformer>); see
`geneformer_extract_V1.py`. The embeddings themselves are not distributed here, because
their redistribution is governed by the Geneformer licence.

Set `BCA_HFP_ROOT` if the data directory lives outside the repository root:

```bash
export BCA_HFP_ROOT=/path/to/project      # optional, defaults to the repository root
```

## Environment

Python 3.11.18 with torch 2.7.1 (CUDA 12.8), rdkit 2026.9.1 and pandas 3.1.0. The full
package inventory of the workstation used for the reported runs is in
`environment/env_rtx5090_20261112.txt`, which also lists the pinned versions of the
core packages:

```bash
conda create -n bca_hfp python=3.11
conda activate bca_hfp
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
pip install h5py==3.16.0 numpy==2.4.6 pandas==3.1.0 rdkit==2026.9.1 scikit-learn==1.9.1 \
            scipy==1.17.1 tables==3.11.1 tqdm==4.70.1
```

The two reported runs were trained on a single RTX 5090; earlier hardware-comparison runs
used a Tesla T4 and a Radeon RX 7900 XT.

## Reproducing the reported runs

```bash
python -m bca_hfp.training.train_cv \
  --dataset GDSC --model_type attention \
  --gene_version V1 --layer 3 --reg_norm batchnorm \
  --split_method drug --group_by_name \
  --seed 42 --split_seed 42 --n_splits 5 \
  --max_epochs 20 --no_early_stop --batch_size 32 --grad_accum 4 \
  --gate_content --use_desc --drug_mean_weight 0.5 \
  --balance_rule gated --swa_last_k 5 --fp_bits 2048 --pred_clamp \
  --fold_csv data/fold_drug_grouped_5fold.csv
```

Both reported seeds use the same split file; only `--seed` changes (43 for the second run,
with `--split_seed 42` so the folds stay fixed).

`logs/RTX5090_v1_gc_c5_s42_20261106_1822.txt` and its seed-43 counterpart record the fully
resolved configuration of the two reported runs, including every flag listed above.

## Notes on the numbers in the paper

- Two seeds are reported: 42 and 43, both with the same five drug-grouped folds.
- Fold assignments are defined by drug name, and names mapping to the same InChIKey were
  merged before splitting (233 name groups over 242 DRUG_IDs).
- Predictions are stored unclamped; the values in the paper are obtained by clamping each
  prediction to the observed assay window of the corresponding drug
  (`MIN_CONC`/`MAX_CONC` columns of the response table). Both `preclamp` and `postclamp`
  metric rows are kept in `results/tables/A1_macro_summary.csv`.
- Pooled R² over all out-of-fold predictions is 0.413 (seed 42) and 0.409 (seed 43).

## Third-party code

`benchmarks/` contains code adapted from the public releases of the five baselines
(GraphDRP, GraTransDRP, PaccMann, TransCDR, tCNNS). Licence and attribution files are kept
where the original release provided them (`benchmarks/paccmann/LICENSE` and the README files
in each subdirectory); those terms apply to the corresponding files, not to the MIT licence
below. `benchmarks/tcnns/data/preprocess.py` is the original Python 2 preprocessing script
from the tCNNS release, kept for reference and not used by the pipeline here.

## Licence

MIT for the code authored in this repository; see `LICENSE`.

## Citation

If you use this code or the archived data, please cite the manuscript above and the data
record:

- Code: <https://github.com/duolichate/BCA-HFP>
- Processed data: Zenodo, `10.5281/zenodo.XXXXXXX` <!-- replace with the published DOI -->
