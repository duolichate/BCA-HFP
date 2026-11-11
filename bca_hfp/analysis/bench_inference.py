# -*- coding: utf-8 -*-
"""measure pure forward throughput (CPU) for the reported inference time."""
import os
import sys
import time
from functools import partial

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from bca_hfp.data.dataset import DrugGeneDataset, collate_fn
from bca_hfp.data.loader import (load_drug_atom_features, load_gene_embeddings,
                                 load_morgan_fingerprints, load_response)
from bca_hfp.models import create_model

ROOT = r'd:\ProgramData\my_model'
RUN = os.path.join(ROOT, '5_Result', '1_main_models', 'RTX5090_v1_gc_c5_s42_20261106_1822')
CK = os.path.join(RUN, 'models', 'attention_drug_fold1_best_val.pth')
FOLD_CSV = os.path.join(ROOT, '5_Result', '3_diagnostics', '1_model_level',
                        'r36_v1gc_s42_5090_20261108', 'fold_membership.csv')
N = 3000

ckpt = torch.load(CK, map_location='cpu', weights_only=False)
cfg = ckpt['config']
d = cfg['data']
atoms = load_drug_atom_features(d['drug_feat_path'])
genes, _ = load_gene_embeddings(d['gene_feat_dir'], d['cell_id_col'])
resp = load_response(d['drug_feat_path'], d['cell_id_col'], d['drug_id_col'])
fps = load_morgan_fingerprints(d['drug_feat_path'], fp_bits=2048, fp_radius=2, use_desc=True)

resp['_g'] = resp['DRUG_NAME'].astype(str).str.strip().str.upper()
fm = pd = None
import pandas as pd  # noqa: E402
fm = pd.read_csv(FOLD_CSV)
fm['GROUP'] = fm['GROUP'].astype(str).str.strip().str.upper()
g2f = dict(zip(fm['GROUP'], fm['fold'].astype(int)))
test = resp[resp['_g'].map(g2f) == 1]
test = test[test[d['drug_id_col']].astype(str).isin(atoms)
            & test[d['cell_id_col']].astype(str).isin(genes)].head(N)
print('benchmark samples:', len(test))

ds = DrugGeneDataset(test[d['drug_id_col']].tolist(), test[d['cell_id_col']].tolist(),
                     test[d['label_col']].values, atoms, genes, fps)
loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=False,
                                     collate_fn=partial(collate_fn, atom_mask_prob=0.0),
                                     num_workers=0)
model = create_model('attention', cfg)
model.load_state_dict(ckpt['model_state_dict'], strict=False)
model.eval()

with torch.no_grad():
    for i, b in enumerate(loader):          # warm-up
        model(b[2], b[0], b[1], b[3], drug_global=b[5])
        if i >= 2:
            break
    t0 = time.time()
    seen = 0
    for b in loader:
        model(b[2], b[0], b[1], b[3], drug_global=b[5])
        seen += b[0].shape[0]
    dt = time.time() - t0

print('pairs=%d  time=%.2fs  throughput=%.1f pairs/s  latency=%.2f ms/pair'
      % (seen, dt, seen / dt, 1000 * dt / seen))
print('extrapolated full test set (198,606): %.1f min' % (198606 / (seen / dt) / 60))
