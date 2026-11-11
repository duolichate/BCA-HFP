# -*- coding: utf-8 -*-
"""Recompute: (1) window baselines, (2) censoring strata, (3) paired drug-block bootstrap.
Writes tmp_tables.json and prints markdown tables.
"""
import pandas as pd, numpy as np, json

RNG = np.random.default_rng(20261109)
B = 2000
GLOBAL_LO, GLOBAL_HI = np.log(0.001), np.log(100)

# ---------------- data ----------------
bca = pd.read_csv("5_Result/3_diagnostics/1_model_level/r36_v1gc_s42_5090_20261108/raw_predictions/oof_raw_predictions.csv",
                  usecols=['DRUG_ID','GDSC_ID','fold','GROUP','y','p_raw','p_win'])
tc = pd.read_csv("5_Result/3_diagnostics/4_baselines/r36_tcnns_oof/oof_tcnns.csv")
res = pd.read_csv("1_Data/Drug/GDSC_RES.csv", usecols=['GDSC_ID','DRUG_ID','DRUG_NAME','MIN_CONC','MAX_CONC'])

df = bca.merge(tc[['DRUG_ID','GDSC_ID','y','p_raw','p_win','lo','hi']], on=['DRUG_ID','GDSC_ID'],
               how='inner', suffixes=('', '_tc'))
df = df.merge(res[['DRUG_ID','GDSC_ID','MIN_CONC','MAX_CONC']], on=['DRUG_ID','GDSC_ID'], how='left')
assert len(df) == 198606, len(df)
assert np.abs(df.y - df.y_tc).max() < 1e-5, np.abs(df.y - df.y_tc).max()
my_lo, my_hi = np.log(df.MIN_CONC.values), np.log(df.MAX_CONC.values)
print("window match (BCA table vs tCNNS table): max|dlo|=%.4f max|dhi|=%.4f" %
      (np.abs(my_lo - df.lo.values).max(), np.abs(my_hi - df.hi.values).max()))

y = df.y.values.astype(float)
lo, hi = my_lo, my_hi
g = df.GROUP.values
groups = np.array(sorted(pd.unique(g)))
print("n =", len(df), "groups =", len(groups))


def clamp(p, slack=2.0):
    return np.clip(np.clip(p, lo - slack, hi + slack), GLOBAL_LO, GLOBAL_HI)


models = {
    'window_hi':   (hi.copy(),            clamp(hi.copy())),
    'window_mid':  ((lo + hi) / 2,        clamp((lo + hi) / 2)),
    'tcnns':       (df.p_raw_tc.values,   clamp(df.p_raw_tc.values)),
    'bca':         (df.p_raw.values,      clamp(df.p_raw.values)),
}
# sanity: clamped columns from files
print("check BCA p_win vs recomputed clamp: max diff %.2e" % np.abs(df.p_win.values - models['bca'][1]).max())
print("check tCNNS p_win vs recomputed clamp: max diff %.2e" % np.abs(df.p_win_tc.values - models['tcnns'][1]).max())


# ---------------- metrics ----------------
def metrics(y, p):
    N = len(y); sy = y.sum(); sy2 = (y*y).sum(); sp = p.sum(); sp2 = (p*p).sum(); syp = (y*p).sum()
    sse = sy2 - 2*syp + sp2
    sst = sy2 - sy*sy/N
    r2 = 1 - sse/sst
    r = (N*syp - sy*sp)/np.sqrt((N*sy2 - sy*sy)*(N*sp2 - sp*sp))
    return dict(n=N, r2=r2, pearson=r, rmse=np.sqrt(sse/N), mae=np.abs(y-p).mean(),
                bias=(p-y).mean(), sse=sse, sst=sst)


def level_within(y, p, g):
    d = pd.DataFrame({'y': y, 'p': p, 'g': g})
    agg = d.groupby('g').agg(n=('y','size'), yb=('y','mean'), pb=('p','mean'))
    yw = (agg.n*agg.yb).sum()/agg.n.sum()
    num = (agg.n*(agg.yb-agg.pb)**2).sum()
    den = (agg.n*(agg.yb-yw)**2).sum()
    r2_level = 1 - num/den
    # within
    res = d.join(agg, on='g')
    yc = res.y - res.yb
    pc = res.p - res.pb
    r2_within = 1 - ((yc-pc)**2).sum()/(yc**2).sum()
    return r2_level, r2_within


rows = []
for name, (p_raw, p_win) in models.items():
    for cond, p in (('pre', p_raw), ('post', p_win)):
        m = metrics(y, p)
        r2l, r2w = level_within(y, p, g)
        rows.append(dict(model=name, cond=cond, r2=m['r2'], pearson=m['pearson'], rmse=m['rmse'],
                         mae=m['mae'], bias=m['bias'], r2_level=r2l, r2_within=r2w))
        print(f"{name:11s} {cond:4s} R2={m['r2']:.4f} r={m['pearson']:.4f} RMSE={m['rmse']:.4f} "
              f"MAE={m['mae']:.4f} bias={m['bias']:+.4f} level={r2l:.4f} within={r2w:.4f}")
meta = pd.DataFrame(rows)

# ---------------- bootstrap sufficient statistics ----------------
def suff(y, p, g):
    d = pd.DataFrame({'y': y, 'p': p, 'g': g})
    a = d.groupby('g').agg(n=('y','size'), sy=('y','sum'), sp=('p','sum'))
    a['sy2'] = d.groupby('g').y.apply(lambda s: (s*s).sum())
    a['sp2'] = d.groupby('g').p.apply(lambda s: (s*s).sum())
    a['syp'] = (d.y*d.p).groupby(d.g).sum()
    a['yb'] = a.sy/a.n
    a['pb'] = a.sp/a.n
    tmp = d.assign(yc=d.y-d.g.map(a.yb), pc=d.p-d.g.map(a.pb))
    a['ssew'] = ((tmp.yc-tmp.pc)**2).groupby(d.g).sum()
    a['sstw'] = (tmp.yc**2).groupby(d.g).sum()
    return a.reindex(groups)


S = {k: suff(y, v[0], g) for k, v in models.items()}
Ss = {k: suff(y, v[1], g) for k, v in models.items()}


def boot_r2(a, K):
    n = a.n.values; sy = a.sy.values; sy2 = a.sy2.values; sp = a.sp.values
    sp2 = a.sp2.values; syp = a.syp.values
    N = K @ n
    Sy = K @ sy; Sy2 = K @ sy2; Sp = K @ sp; Sp2 = K @ sp2; Syp = K @ syp
    sse = Sy2 - 2*Syp + Sp2
    sst = Sy2 - Sy*Sy/N
    return 1 - sse/sst


def boot_level_within(a, K):
    n = a.n.values; yb = a.yb.values; pb = a.pb.values
    N = K @ n
    yw = (K @ (n*yb))/N
    num = K @ (n*(yb-pb)**2)
    den = K @ (n*yb**2) - N*yw**2
    lvl = 1 - num/den
    ssew = K @ a.ssew.values
    sstw = K @ a.sstw.values
    return lvl, 1 - ssew/sstw


K = RNG.multinomial(len(groups), np.full(len(groups), 1/len(groups)), size=B)   # k_g ~ Poisson(1)
boot = {}
for name in models:
    for cond, a in (('pre', S[name]), ('post', Ss[name])):
        r2 = boot_r2(a, K)
        lvl, win_ = boot_level_within(a, K)
        boot[(name, cond)] = dict(r2=r2, level=lvl, within=win_)

boot_rows = []
for (name, cond), d in boot.items():
    row = dict(model=name, cond=cond)
    for metric, arr in (('r2', d['r2']), ('r2_level', d['level']), ('r2_within', d['within'])):
        row[metric+'_lo'] = float(np.percentile(arr, 2.5)); row[metric+'_hi'] = float(np.percentile(arr, 97.5))
    boot_rows.append(row)
boot_df = pd.DataFrame(boot_rows)
print("\n--- bootstrap CI (B=%d, drug-block) ---" % B)
print(boot_df.to_string(index=False))

# ---------------- paired diffs ----------------
pairs = [('bca','window_hi'), ('tcnns','window_hi'), ('bca','tcnns')]
diff_rows = []
for a_name, b_name in pairs:
    for cond in ('pre','post'):
        dA = boot[(a_name, cond)]; dB = boot[(b_name, cond)]
        for metric in ('r2','level','within'):
            d = dA[metric] - dB[metric]
            diff_rows.append(dict(pair=f"{a_name}-{b_name}", cond=cond, metric=metric,
                                  point=None, lo=float(np.percentile(d,2.5)), hi=float(np.percentile(d,97.5)),
                                  p_neg=float(np.mean(d<0))))
diff_df = pd.DataFrame(diff_rows)
# attach point estimates
pt = {(r.model, r.cond): r for r in meta.itertuples()}
for i, r in diff_df.iterrows():
    if r.metric == 'r2':
        diff_df.loc[i,'point'] = pt[(r.pair.split('-')[0], r.cond)].r2 - pt[(r.pair.split('-')[1], r.cond)].r2
    elif r.metric == 'level':
        diff_df.loc[i,'point'] = pt[(r.pair.split('-')[0], r.cond)].r2_level - pt[(r.pair.split('-')[1], r.cond)].r2_level
    else:
        diff_df.loc[i,'point'] = pt[(r.pair.split('-')[0], r.cond)].r2_within - pt[(r.pair.split('-')[1], r.cond)].r2_within
print("\n--- paired drug-block delta (B=%d) ---" % B)
print(diff_df.round(4).to_string(index=False))

# ---------------- censoring strata ----------------
inw = (y >= lo) & (y <= hi); right = y > hi; left = y < lo
print("\nstrata: in=%d right=%d left=%d" % (inw.sum(), right.sum(), left.sum()))
strat_rows = []
for name, (p_raw, p_win) in models.items():
    for cond, p in (('pre', p_raw), ('post', p_win)):
        for sname, mask in (('in_window', inw), ('right', right), ('left', left)):
            yy, pp = y[mask], p[mask]
            m = metrics(yy, pp)
            slope, intercept = np.polyfit(pp, yy, 1)
            strat_rows.append(dict(model=name, cond=cond, stratum=sname, n=int(mask.sum()),
                                   r2=m['r2'], pearson=m['pearson'], bias=m['bias'],
                                   rmse=m['rmse'], mae=m['mae'], slope=float(slope), intercept=float(intercept)))
strat_df = pd.DataFrame(strat_rows)
print("\n--- strata ---")
print(strat_df.round(4).to_string(index=False))

out = dict(meta=meta.to_dict(orient='records'),
           boot=boot_df.to_dict(orient='records'),
           diff=diff_df.to_dict(orient='records'),
           strata=strat_df.to_dict(orient='records'),
           counts=dict(in_window=int(inw.sum()), right=int(right.sum()), left=int(left.sum())))
json.dump(out, open("tmp_tables.json", "w"), ensure_ascii=False, indent=1)
print("\nsaved tmp_tables.json")
