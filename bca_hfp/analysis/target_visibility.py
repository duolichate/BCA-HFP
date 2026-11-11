# -*- coding: utf-8 -*-
"""Compute target-gene visibility and pathway-member visibility in the 2048-gene input panel."""
import pandas as pd, numpy as np, csv, json
from collections import defaultdict

CELLS_FILE = "5_Result/3_diagnostics/1_model_level/r36_v1gc_s42_5090_20261108/raw_predictions/oof_raw_predictions.csv"
PC_FILE = "1_Data/Gene/per_cell_gene_names_GDSC.csv"
GMT = {"KEGG": "1_Data/Raw/c2.cp.kegg.v7.5.1.symbols.gmt",
       "Hallmark": "1_Data/Raw/h.all.v7.5.1.symbols.gmt"}

# symbol -> ensembl from the two expression-matrix headers (same column order)
with open("1_Data/Gene/GDSC_EXP.csv", encoding="utf-8") as f: h_sym = next(csv.reader(f))
with open("1_Data/Gene/GDSC_EXP_ensembl.csv", encoding="utf-8") as f: h_ens = next(csv.reader(f))
sym2ens = {s.upper(): e for s, e in zip(h_sym[5:], h_ens[5:])}
ens2sym = {v: k for k, v in sym2ens.items()}

# ---- analysis cells and per-cell panels ----
oof = pd.read_csv(CELLS_FILE, usecols=['DRUG_ID', 'GDSC_ID', 'GROUP', 'y', 'p_raw', 'p_win'])
cells = list(pd.unique(oof.GDSC_ID))
pc = pd.read_csv(PC_FILE)
gc = [c for c in pc.columns if c.startswith("gene_")]
pc = pc[pc.GDSC_ID.notna()]
sets = {}
for gid, grp in pc.groupby('GDSC_ID'):
    sets.setdefault(int(gid), set()).update(grp[gc].values.ravel().tolist())
sets = {k: {x for x in v if isinstance(x, str)} for k, v in sets.items()}
panel = {c: sets[c] for c in cells if c in sets}
print("cells with panel:", len(panel), "/", len(cells), " median panel:", int(np.median([len(v) for v in panel.values()])))

# ---- gene sets ----
def load_gmt(path):
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) > 2:
                out[p[0]] = {g.upper() for g in p[2:] if g}
    return out

gsets = {}
for name, path in GMT.items():
    gsets[name] = load_gmt(path)
    print(name, "sets:", len(gsets[name]), " median size:", int(np.median([len(v) for v in gsets[name].values()])))

# gene -> pathways containing it (symbol)
gene2paths = defaultdict(list)
for src, dd in gsets.items():
    for pname, members in dd.items():
        for g in members:
            gene2paths[g].append((src, pname))

# ---- targets from screened compounds ----
sc = pd.read_csv("1_Data/Raw/screened_compounds_rel_8.5_inchi.csv")
drugs = oof[['DRUG_ID', 'GROUP']].drop_duplicates()
m = drugs.merge(sc[['DRUG_ID', 'TARGET', 'TARGET_PATHWAY']], on='DRUG_ID', how='left').drop_duplicates('GROUP')
print("drug groups:", len(m), " annotated:", int(m.TARGET.notna().sum()))

all_targets = sorted({p.strip().upper() for t in m.TARGET.dropna().astype(str) for p in t.split(',')
                      if p.strip().upper() in sym2ens})
print("target symbols mapped:", len(all_targets))

# ---- per-target statistics ----
cell_list = list(panel)
rows = []
for g in all_targets:
    e = sym2ens[g]
    hit = np.array([e in panel[c] for c in cell_list])
    pres = hit.mean()
    paths = gene2paths.get(g, [])
    path_cov = []
    for src, pname in paths:
        members = gsets[src][pname]
        mem_ens = {sym2ens[mm] for mm in members if mm in sym2ens}
        if len(mem_ens) < 5: continue
        covs = [len(mem_ens & panel[c]) / len(mem_ens) for c in cell_list]
        path_cov.append((src, pname, len(mem_ens), float(np.mean(covs))))
    rows.append(dict(target=g, ensembl=e, presence=pres,
                     n_pathways=len(path_cov),
                     path_cov_mean=float(np.mean([x[3] for x in path_cov])) if path_cov else np.nan,
                     path_cov_min=float(np.min([x[3] for x in path_cov])) if path_cov else np.nan,
                     path_cov_max=float(np.max([x[3] for x in path_cov])) if path_cov else np.nan,
                     best_pathway=max(path_cov, key=lambda x: x[3])[1] if path_cov else ""))
tgt = pd.DataFrame(rows).sort_values('presence')
tgt['n_drugs'] = [int(m.TARGET.astype(str).str.upper().str.contains(g).sum()) for g in tgt.target]
print("\n=== target presence (n=%d) ===" % len(tgt))
print(tgt.presence.describe().round(4).to_dict())
for thr in (0.0, 0.01, 0.1, 0.5, 0.9):
    print("  presence > %.2f : %d" % (thr, int((tgt.presence > thr).sum())))
print("\n=== pathway-member coverage (all targets) ===")
print(tgt.path_cov_mean.describe().round(4).to_dict())
print("  pathways per target: median %d, max %d" % (tgt.n_pathways.median(), tgt.n_pathways.max()))
print(tgt.sort_values('presence').head(5)[['target','presence','path_cov_mean','path_cov_max','n_drugs']].to_string(index=False))
print(tgt.sort_values('presence').tail(5)[['target','presence','path_cov_mean','path_cov_max','n_drugs']].to_string(index=False))

# ---- per-drug statistics ----
def per_drug_stats(col):
    out = {}
    for g, d in oof.groupby('GROUP'):
        if len(d) < 50 or d[col].std() < 1e-9:
            out[g] = (np.nan, np.nan); continue
        r = np.corrcoef(d.y, d[col])[0, 1]
        yd = d.y - d.y.mean(); pd_ = d[col] - d[col].mean()
        r2c = 1 - ((yd - pd_) ** 2).sum() / (yd ** 2).sum()
        out[g] = (r, r2c)
    return out

pre = per_drug_stats('p_raw'); post = per_drug_stats('p_win')
print("\nper-drug median pearson pre=%.3f post=%.3f" % (pd.Series({k: v[0] for k, v in pre.items()}).median(),
                                                        pd.Series({k: v[0] for k, v in post.items()}).median()))

prow = []
for _, r in m.iterrows():
    tg = [p.strip().upper() for p in str(r.TARGET).split(',')] if pd.notna(r.TARGET) else []
    tg = [g for g in tg if g in sym2ens]
    sub = tgt[tgt.target.isin(tg)]
    prow.append(dict(GROUP=r.GROUP, target=r.TARGET if pd.notna(r.TARGET) else "",
                     n_tgt=len(tg),
                     tgt_presence=float(sub.presence.mean()) if len(sub) else np.nan,
                     path_cov=float(sub.path_cov_mean.mean()) if len(sub) and sub.path_cov_mean.notna().any() else np.nan,
                     pearson_pre=pre.get(r.GROUP, (np.nan, np.nan))[0],
                     pearson_post=post.get(r.GROUP, (np.nan, np.nan))[0],
                     r2c_pre=pre.get(r.GROUP, (np.nan, np.nan))[1]))
pdr = pd.DataFrame(prow)
print("per-drug with target mapped:", pdr.n_tgt.gt(0).sum(), " with pathway cov:", int(pdr.path_cov.notna().sum()))


def spearman(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 10: return np.nan, np.nan, len(x)
    rx = pd.Series(x).rank().values; ry = pd.Series(y).rank().values
    r = np.corrcoef(rx, ry)[0, 1]
    # permutation p
    rng = np.random.default_rng(11); cnt = 0; B = 20000
    for _ in range(B):
        if abs(np.corrcoef(rx, rng.permutation(ry))[0, 1]) >= abs(r): cnt += 1
    return r, (cnt + 1) / (B + 1), len(x)


tests = {
    "target_presence vs pearson_pre": spearman(pdr.tgt_presence, pdr.pearson_pre),
    "path_cov vs pearson_pre": spearman(pdr.path_cov, pdr.pearson_pre),
    "target_presence vs r2c_pre": spearman(pdr.tgt_presence, pdr.r2c_pre),
    "path_cov vs r2c_pre": spearman(pdr.path_cov, pdr.r2c_pre),
    "target_presence vs path_cov (genes)": spearman(tgt.presence, tgt.path_cov_mean),
}
for k, v in tests.items():
    print("%-38s rho=%+.3f  p=%.4f  n=%d" % (k, v[0], v[1], v[2]))

tgt.to_csv("tmp_targets_full.csv", index=False)
pdr.to_csv("tmp_per_drug_targets.csv", index=False)
json.dump({k: {"rho": None if not np.isfinite(v[0]) else float(v[0]),
               "p": None if not np.isfinite(v[1]) else float(v[1]),
               "n": int(v[2])} for k, v in tests.items()},
          open("tmp_target_tests.json", "w"), ensure_ascii=False, indent=1)
summary = dict(
    n_cells=len(cell_list),
    n_targets=len(tgt),
    presence_median=float(tgt.presence.median()),
    presence_gt50=int((tgt.presence > 0.5).sum()),
    presence_gt90=int((tgt.presence > 0.9).sum()),
    presence_zero=int((tgt.presence == 0).sum()),
    path_cov_median=float(tgt.path_cov_mean.median()),
    path_cov_q25=float(tgt.path_cov_mean.quantile(0.25)),
    path_cov_q75=float(tgt.path_cov_mean.quantile(0.75)),
    per_drug_path_cov_median=float(pdr.path_cov.median()),
    n_drugs_mapped=int(pdr.n_tgt.gt(0).sum()),
    n_drugs=len(pdr))
json.dump(summary, open("tmp_target_summary.json", "w"), ensure_ascii=False, indent=1)
print("\nsummary:", summary)
print("saved tmp_targets_full.csv / tmp_per_drug_targets.csv / tmp_target_tests.json")
