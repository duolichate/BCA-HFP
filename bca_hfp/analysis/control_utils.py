# bca_hfp/analysis/control_utils.py
"""shared helpers for the three R3-9 controls: checkpoint / data / single-sample attention / gene name maps.

the three scripts (control_attn_vs_expression / control_stability / control_input_removal) share the loading and
attention extraction here to avoid duplication. every path is overridable via argparse, nothing hardcoded.
"""

import os
import sys
import pickle

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from bca_hfp.config.base_config import PROJECT_ROOT, root_dir
from bca_hfp.data.loader import (load_drug_atom_features, load_gene_embeddings,
                                 load_morgan_fingerprints, load_response)
from bca_hfp.models import create_model

# same as vis_gene2drug.py, use the GeneFormer 30M symbol->ENSG dictionary
SYMBOL_PKL = os.path.join(root_dir('Gene_Feature'), 'geneformer',
                          'gene_dictionaries_30m', 'gene_name_id_dict_gc30M.pkl')


def norm_id(x):
    """normalize a cell/drug ID to a version-free string (910919.0 -> 910919)."""
    s = str(x).strip()
    return s.split('.')[0]


def load_model_from_ckpt(model_path, device=None):
    ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
    config = ckpt['config']
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model('attention', config).to(device)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.eval()
    return model, config, device, ckpt


def load_control_data(config, h5=None, gene_feat_dir=None):
    """load the three feature sets and the response table from the config stored in the checkpoint. returns (dicts.., column names)."""
    dc = config['data']
    h5 = h5 or dc['drug_feat_path']
    gfeat = gene_feat_dir or dc['gene_feat_dir']
    cell_col, drug_col, label_col = dc['cell_id_col'], dc['drug_id_col'], dc['label_col']

    use_desc = int(dc['drug_global_dim']) > 2048
    fp_bits = 2048 if use_desc else int(dc['drug_global_dim'])

    drug_atom = load_drug_atom_features(h5)
    gene_emb, _ = load_gene_embeddings(gfeat, cell_col)
    # some GDSC_ID values in the .pt metadata carry .0 and do not match the clean ids of the response table, normalize them here
    gene_emb = {norm_id(k): v for k, v in gene_emb.items()}
    drug_global = load_morgan_fingerprints(h5, fp_bits=fp_bits, fp_radius=2, use_desc=use_desc)
    response = load_response(h5, cell_col, drug_col)
    return drug_atom, gene_emb, drug_global, response, (cell_col, drug_col, label_col)


def run_attention(model, drug_atom, gene_emb, drug_global, drug_id, cell_id, device):
    """single-sample forward pass, returns (pred, drug2gene_attn, gene2drug_attn).

    drug2gene_attn: (8, n_atoms, n_genes), drug atom query -> gene key, used for per-gene importance;
    gene2drug_attn: (8, n_genes, n_atoms), gene query -> drug atom key.
    """
    atom = torch.tensor(drug_atom[drug_id], dtype=torch.float32).unsqueeze(0).to(device)
    atom_mask = torch.ones(1, atom.shape[1], dtype=torch.bool, device=device)
    gene = torch.tensor(gene_emb[cell_id], dtype=torch.float32).unsqueeze(0).to(device)
    gene_mask = torch.ones(1, gene.shape[1], dtype=torch.bool, device=device)
    dg = torch.tensor(drug_global[drug_id], dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        out, d2g, g2d = model(gene, atom, atom_mask, gene_mask, drug_global=dg, return_attn=True)
    return (float(out.item()), d2g[0].cpu().numpy(), g2d[0].cpu().numpy())


def gene_importance(drug2gene_attn):
    """drug2gene attention (8, n_atoms, n_genes) -> total attention per gene (n_genes,).

    cross-attention with drug atoms as query and genes as key: each atom runs a softmax over the genes
    (rows sum to 1). after averaging over heads we sum over the atom axis, which gives the attention mass
    all atoms assign to each gene. same definition as `gene_attn = weights.sum(axis=0).sum(axis=0)`
    in fig_drug2gene.py.
    """
    return drug2gene_attn.mean(axis=0).sum(axis=0)


def load_symbol_maps(path=SYMBOL_PKL):
    with open(path, 'rb') as f:
        sym2ens = pickle.load(f)
    ens2sym = {}
    for sym, ens in sym2ens.items():
        ens2sym[norm_id(ens)] = str(sym).upper()
    return sym2ens, ens2sym


def load_cell_gene_orders(gene_feat_dir, cell_id_col='GDSC_ID', only_cells=None):
    """scan the .pt files and return {cell_id: the gene_names order of that cell}.

    the gene_names of each .pt is that cell line's own 2048-gene order (it differs per cell), a single
    global order does not explain it. torch.load unpickles the whole file, so all ~1600 files are slow;
    passing the needed cell_id set in only_cells lets the loop stop early, which saves a lot of time for
    single-cell scripts.
    """
    out = {}
    want = set(only_cells) if only_cells else None
    for fname in sorted(os.listdir(gene_feat_dir)):
        if not fname.endswith('.pt'):
            continue
        try:
            data = torch.load(os.path.join(gene_feat_dir, fname), weights_only=False)
        except Exception:
            continue
        meta = data.get('metadata', {})
        cid = str(meta.get(cell_id_col))
        if cid in ('None', '') or cid.lower() == 'none':
            continue
        nid = norm_id(cid)
        if want is not None:
            if nid not in want:
                continue
        gn = data.get('gene_names')
        if gn is not None:
            out[nid] = [str(g) for g in gn]
            if want is not None and want.issubset(set(out.keys())):
                break
    return out


def _alnum(s):
    return ''.join(ch for ch in str(s) if ch.isalnum()).upper()


def resolve_cell_id(response, cell_col, cell):
    """resolve a user-given cell line name (BC-1 / BC1 / 910919) to a cell_id in response.

    name matching drops non-alphanumeric characters such as hyphens, so BC1 and BC-1 are the same cell line.
    """
    c = str(cell).strip()
    if c == '':
        return None
    m = response[response[cell_col].astype(str) == c]
    if len(m) == 0:
        m = response[response[cell_col].apply(norm_id) == norm_id(c)]
    if len(m) == 0:
        for name_col in ('CELL_LINES', 'CELL_LINES_STD'):
            if name_col in response.columns:
                m = response[response[name_col].astype(str).map(_alnum) == _alnum(c)]
                if len(m) > 0:
                    break
    if len(m) == 0:
        return None
    return str(m.iloc[0][cell_col])