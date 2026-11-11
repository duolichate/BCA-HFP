# -*- coding: utf-8 -*-
import os

import pandas as pd

rows = [
    ("BCA-HFP (seed 42)", 0.2301, 0.4134, 2.4728, 2.1584, 0.5234, 0.6468, "ours"),
    ("BCA-HFP (seed 43)", 0.2253, 0.4087, 2.4805, 2.1670, 0.5241, 0.6457, "ours"),
    ("tCNNS", 0.2600, 0.3845, 2.4243, 2.2109, 0.5183, 0.6324, "SOTA"),
    ("PaccMann", 0.1796, 0.3506, 2.5525, 2.2710, 0.4283, 0.6120, "SOTA"),
    ("TransCDR", 0.1090, 0.3375, 2.6602, 2.2939, 0.3416, 0.6003, "SOTA"),
    ("GraTransDRP", 0.0896, 0.2976, 2.6889, 2.3619, 0.4298, 0.5642, "SOTA"),
    ("GraphDRP", 0.0738, 0.2067, 2.7121, 2.5101, 0.3845, 0.5069, "SOTA"),
    ("GBM (GeneFormer pool 256)", 0.1571, 0.3107, 2.5874, 2.3397, 0.4554, 0.5737, "trivial baseline"),
    ("GBM (raw expression 2048)", 0.1377, 0.3022, 2.6169, 2.3540, 0.4395, 0.5682, "trivial baseline"),
    ("Cell-line mean", 0.0902, 0.3184, 2.6880, 2.3267, 0.3015, 0.6031, "trivial baseline"),
    ("Fingerprint ridge", 0.0395, 0.2274, 2.7623, 2.4768, 0.3836, 0.5220, "trivial baseline"),
    ("Nearest training drug", -0.2980, -0.0206, 3.2106, 2.8470, 0.2280, 0.3845, "trivial baseline"),
]
cols = ["model", "r2_pre", "r2_post", "rmse_pre", "rmse_post", "pearson_pre", "pearson_post", "type"]
df = pd.DataFrame(rows, columns=cols)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
out = os.path.join(ROOT, "results", "tables", "S9_full_metrics.csv")
df.to_csv(out, index=False, encoding="utf-8-sig")
print(df.to_string(index=False))
print("saved", out)
