# -*- coding: utf-8 -*-
import pandas as pd, numpy as np, json
from PIL import Image, ImageDraw
from simpleplot import Panel, save

tgt = pd.read_csv("tmp_targets_full.csv")
pdr = pd.read_csv("tmp_per_drug_targets.csv")
summ = json.load(open("tmp_target_summary.json"))
tests = json.load(open("tmp_target_tests.json"))

# ---------------- Figure 1 ----------------
W, H = 2400, 1020
img = Image.new("RGB", (W, H), "white")
d = ImageDraw.Draw(img)

ax = Panel(img, d, (250, 150, 880, 700), (0, 1), (0, 90), fs=26)
edges = np.arange(0, 1.0001, 0.05)
cnt, _ = np.histogram(tgt.presence, bins=edges)
ax.frame()
for i in range(len(cnt)):
    if cnt[i] > 0:
        ax.bar(edges[i], edges[i + 1], cnt[i])
ax.xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], fmt="{:.1f}")
ax.yticks([0, 20, 40, 60, 80])
ax.xlabel("Fraction of the 941 cell lines whose input\npanel contains the target gene", pad=86, fs=25)
ax.ylabel("Number of target genes", pad=128, fs=26)
ax.text(ax.x0 - 150, ax.y0 - 118, "A", fs=44, bold=True)
ax.text(ax.X(0.06), ax.Y(84), f"{summ['presence_zero']} of {summ['n_targets']} targets\nare absent in every cell line", fs=21)
ax.text(ax.X(0.55), ax.Y(40), f"median presence = {summ['presence_median']*100:.1f}%", fs=21)

ax2 = Panel(img, d, (1720, 150, 600, 700), (0, 1), (0, 0.55), fs=26)
sub = tgt.dropna(subset=['path_cov_mean'])
ax2.frame()
ax2.line([(0, 0), (0.55, 0.55)], fill=(150, 150, 150), width=3, dash=(22,))
ax2.line([(0, 0.1), (1, 0.1)], fill=(210, 140, 120), width=2, dash=(10,))
for _, r in sub.iterrows():
    ax2.circle(r.presence, r.path_cov_mean, r=9)
ax2.xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], fmt="{:.1f}")
ax2.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5], fmt="{:.1f}")
ax2.xlabel("Target-gene presence", pad=86, fs=26)
ax2.ylabel("Mean member coverage\nof its KEGG/Hallmark pathways", pad=150, fs=24)
ax2.text(ax2.x0 - 120, ax2.y0 - 118, "B", fs=44, bold=True)
rho = tests["target_presence vs path_cov (genes)"]
ax2.text(ax2.X(0.03), ax2.Y(0.53), f"Spearman rho = {rho['rho']:+.2f}\n(p = {rho['p']:.0e}, n = {rho['n']})", fs=21)
ax2.text(ax2.X(0.44), ax2.Y(0.54), "y = x", fs=18)
ax2.text(ax2.X(0.55), ax2.Y(0.075), "10% of pathway members", fs=17, fill=(170, 90, 70))
for g, dx, dy in (("EGFR", 0.03, 0.03), ("MTOR", 0.03, -0.03), ("PSMB5", 0.03, 0.01),
                  ("AURKA", -0.34, 0.04), ("CHEK1", -0.30, 0.035), ("WEE1", -0.26, -0.05)):
    r = sub[sub.target == g]
    if len(r):
        r = r.iloc[0]
        ax2.text(ax2.X(r.presence + dx), ax2.Y(r.path_cov_mean + dy), g, fs=20, bold=True)
save(img, "7_Document/6_review/figs/fig_target_visibility.png")

# ---------------- Figure 2 ----------------
W, H = 1500, 1120
img = Image.new("RGB", (W, H), "white")
d = ImageDraw.Draw(img)
ax = Panel(img, d, (270, 140, 980, 820), (0, 0.5), (-0.2, 1.0), fs=26)
p = pdr.dropna(subset=['path_cov', 'pearson_pre'])
ax.frame()
for _, r in p.iterrows():
    ax.circle(r.path_cov, r.pearson_pre, r=8, fill=(31, 92, 155))
sl, ic = np.polyfit(p.path_cov, p.pearson_pre, 1)
ax.line([(x, sl * x + ic) for x in np.linspace(0.02, 0.5, 50)], fill=(130, 130, 130), width=3, dash=(18,))
ax.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5], fmt="{:.1f}")
ax.yticks([-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1.0], fmt="{:.1f}")
ax.xlabel("Mean pathway-member coverage of the\ndrug's annotated target(s)", pad=90, fs=25)
ax.ylabel("Per-drug Pearson r (pre-clamp, LDO OOF)", pad=152, fs=24)
t = tests["path_cov vs pearson_pre"]
ax.text(ax.X(0.012), ax.Y(0.97),
        f"Spearman rho = {t['rho']:+.2f} (p < 1e-4, n = {t['n']})\n"
        f"partial rho = +0.33 after adjusting for\ncell-line-mean predictability", fs=22)
save(img, "7_Document/6_review/figs/fig_pathway_visibility_vs_performance.png")
print("done")
