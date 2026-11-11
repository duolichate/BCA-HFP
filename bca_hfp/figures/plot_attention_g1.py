# -*- coding: utf-8 -*-
"""Atom x gene cross-attention figures for G1 (counterparts of the old Fig 4-6).

The old Fig 4-6 used MG-132 x BC1 as the sample. Under leave-drug-out CV, G1
predicts MG-132 poorly (fold 1 test true=-3.17 -> pred=+1.11, per-drug RMSE 3.64),
which a reviewer would question. This version uses lestaurtinib (FLT3 inhibitor),
held out in fold 3, x MV-4-11 (FLT3-ITD sensitive line, true=-5.67 -> pred=-5.35,
almost unbiased).

Outputs (default 6_Picture/2_supplementary):
  figs2_attention_drug2gene  8-head Drug->Gene heat map (rows atoms, cols top genes)
  figs3_attention_gene2drug  8-head Gene->Drug heat map (rows top genes, cols atoms)
  figs4_attention_atom_map   atom attention of the top gene mapped onto the structure

Usage (cwd under 4_DrugResponse_BCAHFP):
  python -m bca_hfp.figures.plot_attention_g1
  python -m bca_hfp.figures.plot_attention_g1 --drug_id 1024 --cell_id 908156 --fold 3
"""
import argparse
import glob
import os
import pickle
import sys

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from matplotlib.gridspec import GridSpec

from bca_hfp.config.base_config import get_base_config, root_dir
from bca_hfp.data.loader import (load_drug_atom_features, load_morgan_fingerprints,
                                 load_response, PHYSCHEM_DESC_DIM)
from bca_hfp.models import create_model

RUN_DIR = os.path.join(root_dir('Result'), '1_main_models',
                       'RTX5090_v1_gc_c5_s42_20261106_1822')
RESULT = root_dir('Result')
ROOT = os.path.dirname(RESULT)
OUT_DIR = os.path.join(ROOT, '6_Picture', '2_supplementary')
NPZ_DIR = os.path.join(RESULT, '3_diagnostics', '1_model_level',
                       'r36_v1gc_s42_5090_20261108', 'attention')

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Liberation Serif', 'DejaVu Serif'],
    'font.size': 7,
    'axes.linewidth': 0.6,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
})

MIN_WIDTH_PX = 4200  # 7 in @ 600 dpi


def _save(fig, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    w_in = fig.get_size_inches()[0]
    dpi = max(600, MIN_WIDTH_PX / (w_in * 0.90))
    png = os.path.join(OUT_DIR, f'{name}.png')
    fig.savefig(png, dpi=dpi)
    from PIL import Image
    with Image.open(png) as im:
        w = im.size[0]
    if w < MIN_WIDTH_PX:
        dpi *= MIN_WIDTH_PX * 1.02 / w
        fig.savefig(png, dpi=dpi)
    fig.savefig(os.path.join(OUT_DIR, f'{name}.pdf'))
    plt.close(fig)
    print(f'  -> {os.path.join(OUT_DIR, name)}.png/.pdf (dpi={dpi:.0f})')


def _load_ensg_map():
    p = os.path.join(root_dir('Gene_Feature'), 'geneformer',
                     'gene_dictionaries_30m', 'gene_name_id_dict_gc30M.pkl')
    with open(p, 'rb') as f:
        sym2ens = pickle.load(f)
    return {str(v).split('.')[0]: str(k).upper() for k, v in sym2ens.items()}


_CELL_CACHE = {}


def _find_cell_pt(gene_dir, cell_id):
    key = (gene_dir, str(cell_id))
    if key in _CELL_CACHE:
        return _CELL_CACHE[key]
    for f in glob.glob(os.path.join(gene_dir, '*.pt')):
        try:
            d = torch.load(f, map_location='cpu', weights_only=False, mmap=True)
        except TypeError:
            d = torch.load(f, map_location='cpu', weights_only=False)
        if str(d['metadata'].get('GDSC_ID')) == str(cell_id):
            _CELL_CACHE[key] = d
            return d
    raise SystemExit(f'gene embedding .pt not found for cell {cell_id}')


def extract(drug_id, cell_id, fold, run_dir):
    """one forward pass; returns per-head attention and metadata."""
    ckpt_best = os.path.join(run_dir, 'models', f'attention_drug_fold{fold}_best_val.pth')
    ckpt_swa = os.path.join(run_dir, 'models', f'attention_drug_fold{fold}_swa.pth')
    cfg = torch.load(ckpt_best, map_location='cpu', weights_only=False)['config']

    # --- drug side ---
    h5 = cfg['data']['drug_feat_path']
    fp_bits = int(cfg['data']['drug_global_dim']) - PHYSCHEM_DESC_DIM
    atom_dict = load_drug_atom_features(h5)
    fg_dict = load_morgan_fingerprints(h5, fp_bits=fp_bits, fp_radius=2, use_desc=True)
    drug_key = str(drug_id)
    if drug_key not in atom_dict:
        raise SystemExit(f'{drug_id} not in the drug table')
    atoms = atom_dict[drug_key]
    gvec = fg_dict[drug_key]

    # --- cell side (reads the target cell .pt only) ---
    data = _find_cell_pt(cfg['data']['gene_feat_dir'], cell_id)
    gene_names = list(data['gene_names'])
    emb = data['embeddings'].float().numpy()
    keep = [i for i, n in enumerate(gene_names)
            if n not in {'<cls>', '<eos>', '<pad>'}]
    gene_names = [gene_names[i] for i in keep]
    emb = emb[keep]

    # --- ground-truth label ---
    resp = load_response(h5, cfg['data']['cell_id_col'], cfg['data']['drug_id_col'])
    row = resp[(resp[cfg['data']['drug_id_col']].astype(str) == drug_key)
               & (resp[cfg['data']['cell_id_col']].astype(str) == str(cell_id))]
    if row.empty:
        raise SystemExit(f'no response row for drug={drug_id} cell={cell_id}')
    y = float(row.iloc[0][cfg['data']['label_col']])

    # --- model ---
    model = create_model('attention', cfg)
    swa = torch.load(ckpt_swa, map_location='cpu', weights_only=False)
    model.load_state_dict(swa['model_state_dict'])
    model.eval()

    t_atoms = torch.tensor(atoms[None], dtype=torch.float32)
    t_amask = torch.ones(1, atoms.shape[0], dtype=torch.bool)
    t_genes = torch.tensor(emb[None], dtype=torch.float32)
    t_gmask = torch.ones(1, emb.shape[0], dtype=torch.bool)
    t_gvec = torch.tensor(gvec[None], dtype=torch.float32)
    with torch.no_grad():
        pred, d2g, g2d = model(gene_features=t_genes, drug_features=t_atoms,
                               drug_mask=t_amask, gene_mask=t_gmask,
                               drug_global=t_gvec, return_attn=True)
    pred = float(pred.item())
    d2g = d2g[0].numpy()          # (H, A, G)
    g2d = g2d[0].numpy()          # (H, G, A)
    print(f'[sample] drug={drug_id} cell={cell_id} fold={fold}')
    print(f'  atoms(feature)={atoms.shape[0]}  atoms(rdkit) see structure panel  genes={emb.shape[0]}')
    print(f'  true={y:+.4f}  pred(raw)={pred:+.4f}  resid={pred - y:+.4f}')
    print(f'  d2g={d2g.shape}  g2d={g2d.shape}')

    return dict(d2g=d2g, g2d=g2d, gene_names=np.array(gene_names),
                drug_id=drug_key, cell_id=str(cell_id), fold=fold,
                true=y, pred=pred)


def _top_genes(mat, gene_names, ensg2sym, top_n, force_idx=None):
    """mat: (H, *, G) or (H, G, *); top genes by mean weight across heads and atoms.

    force_idx: gene kept in the panel (usually the drug target); replaces the last
    slot when it falls outside top_n.
    """
    if mat.shape[1] == len(gene_names):     # (H,G,A) gene2drug
        score = mat.mean(axis=0).mean(axis=1)
    else:                                    # (H,A,G) drug2gene
        score = mat.mean(axis=0).mean(axis=0)
    idx = np.argsort(score)[::-1][:top_n]
    if force_idx is not None and force_idx not in idx:
        idx = list(idx[:top_n - 1]) + [int(force_idx)]
    syms = [ensg2sym.get(str(gene_names[i]), str(gene_names[i])) for i in idx]
    return idx, syms, score


def plot_heads(mat, gene_idx, gene_syms, title, ylabel, xlabel, name,
               transpose=False, highlight_sym=None, cmap='Reds'):
    """8-head heat map. transpose=False: (H,A,G) rows=atoms cols=genes; True: (H,G,A) rows=genes cols=atoms."""
    H = mat.shape[0]
    sub = mat[:, :, gene_idx] if not transpose else mat[:, gene_idx, :]
    vals = sub[np.isfinite(sub)]
    vmin = float(np.percentile(vals, 1))
    vmax = float(np.percentile(vals, 99))
    norm = PowerNorm(gamma=0.35, vmin=vmin, vmax=vmax)

    fig = plt.figure(figsize=(8.0, 4.3), facecolor='white')
    gs = GridSpec(2, 5, width_ratios=[1, 1, 1, 1, 0.06],
                  wspace=0.36, hspace=0.36, left=0.045, right=0.900,
                  top=0.88, bottom=0.10)
    axes = [fig.add_subplot(gs[i // 4, i % 4]) for i in range(8)]
    cax = fig.add_subplot(gs[:, 4])
    im = None
    for i, ax in enumerate(axes):
        im = ax.imshow(sub[i], cmap=cmap, norm=norm, aspect='auto',
                       interpolation='nearest')
        ax.set_box_aspect(1)                     # near-square heat map panel
        ax.set_title(f'({chr(97 + i)}) Head {i + 1}', loc='left', fontsize=6.4,
                     fontweight='bold', pad=3)
        n_row, n_col = sub[i].shape
        if transpose:
            ax.set_yticks(range(n_row))
            ax.set_yticklabels(gene_syms, fontsize=3.9)
            step = max(1, n_col // 6)
            ax.set_xticks(range(0, n_col, step))
            ax.set_xticklabels([str(t) for t in range(0, n_col, step)], fontsize=4.2)
        else:
            step = max(1, n_row // 4)            # keep ~5 atom ticks
            ax.set_yticks(range(0, n_row, step))
            ax.set_yticklabels([str(t) for t in range(0, n_row, step)], fontsize=4.2)
            ax.set_xticks(range(n_col))
            ax.set_xticklabels(gene_syms, rotation=90, fontsize=4.3)
        ax.tick_params(length=1.6, pad=1.2, width=0.4)
        # target gene label in red bold
        tick_labels = ax.get_yticklabels() if transpose else ax.get_xticklabels()
        for j, lab in enumerate(tick_labels):
            if highlight_sym and j < len(gene_syms) and gene_syms[j] == highlight_sym:
                lab.set_color('#B2182B')
                lab.set_fontweight('bold')
        for s in ax.spines.values():
            s.set_linewidth(0.5)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label('Attention weight\n(P1-P99)', fontsize=5.6, labelpad=9)
    cbar.ax.tick_params(labelsize=4.8, length=1.6)
    cbar.formatter.set_powerlimits((0, 0))
    cbar.update_ticks()
    fig.suptitle(title, fontsize=8.2, y=0.965)
    _save(fig, name)


def plot_atom_map(npz, out_name, gene_idx=None, gene_sym='', title_tag=''):
    """Map Gene->Drug attention of the target gene onto the molecule (RDKit only).

    Matches draft Figure 5: thin bonds, small font, default pale red highlight,
    no colorbar or title.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Chem.Draw import rdMolDraw2D
    from PIL import Image

    g2d = npz['g2d']                      # (H,G,A)
    if gene_idx is None:
        gene_idx = int(np.argmax(g2d.mean(axis=(0, 2))))
    w = g2d[:, gene_idx, :].mean(axis=0)  # (A,)

    h5 = get_base_config('GDSC')['data']['drug_feat_path']
    with pd.HDFStore(h5, 'r') as st:
        smi = st['drug_smiles']
        smi = smi[smi['DRUG_ID'].astype(str) == npz['drug_id']]['DRUG_SMILE'].iloc[0]
    mol = Chem.RemoveHs(Chem.MolFromSmiles(smi))
    AllChem.Compute2DCoords(mol)
    n_rd = mol.GetNumAtoms()
    print(f'[mol] rdkit heavy atoms={n_rd}  attention atoms={len(w)}')

    order = np.argsort(w)[::-1]
    top = [int(i) for i in order if i < n_rd][:5]

    W, H = 4000, 3300
    drawer = rdMolDraw2D.MolDraw2DCairo(W, H)
    opts = drawer.drawOptions()
    opts.bondLineWidth = 1.6                # thin bonds as in the draft
    opts.fixedFontSize = 44
    opts.padding = 0.015
    opts.highlightColour = (1.0, 0.78, 0.78)   # pale pink, atom labels stay visible
    opts.highlightRadius = 0.32
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol, highlightAtoms=top)
    drawer.FinishDrawing()

    png_path = os.path.join(OUT_DIR, f'{out_name}.png')
    with open(png_path, 'wb') as f:
        f.write(drawer.GetDrawingText())
    im = Image.open(png_path).convert('RGB')
    dpi = im.size[0] / 7.0
    im.save(png_path, dpi=(dpi, dpi))
    im.save(os.path.join(OUT_DIR, f'{out_name}.pdf'), 'PDF', resolution=dpi)
    print(f'  -> {png_path}|.pdf  ({im.size[0]}x{im.size[1]}, {dpi:.0f} dpi)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--drug_id', default='1917')
    ap.add_argument('--cell_id', default='949093')
    ap.add_argument('--fold', type=int, default=1)
    ap.add_argument('--folds', default='',
                    help='comma separated, e.g. 1,2,3,4,5; overrides --fold')
    ap.add_argument('--run_dir', default=RUN_DIR)
    ap.add_argument('--top_genes', type=int, default=15)
    ap.add_argument('--prefix', default='figs')
    ap.add_argument('--title_tag', default='AZD6738 × KARPAS-299')
    ap.add_argument('--target_gene', default='CHEK1',
                    help='target gene kept in the panel and marked red (ignored if absent)')
    ap.add_argument('--suffix', default='', help='output filename suffix, separates samples')
    ap.add_argument('--fig_start', type=int, default=2,
                    help='first figure number in multi-fold runs (3 numbers per fold)')
    args = ap.parse_args()

    folds = ([int(x) for x in args.folds.split(',')] if args.folds.strip()
             else [args.fold])
    ensg2sym = _load_ensg_map()
    os.makedirs(NPZ_DIR, exist_ok=True)
    k = args.fig_start

    for fold in folds:
        d = extract(args.drug_id, args.cell_id, fold, args.run_dir)
        npz_path = os.path.join(
            NPZ_DIR, f"attention_{args.drug_id}_{args.cell_id}_fold{fold}.npz")
        np.savez_compressed(npz_path, **d)
        print(f'[save] {npz_path}')

        syms_all = [ensg2sym.get(str(g), str(g)) for g in d['gene_names']]
        force_idx = None
        if args.target_gene and args.target_gene in syms_all:
            force_idx = syms_all.index(args.target_gene)
            score = d['d2g'].mean(axis=(0, 1))
            rank = int(np.argsort(np.argsort(-score))[force_idx]) + 1
            print(f'[target] {args.target_gene} ranks {rank}/{len(score)} in '
                  f'Drug->Gene gene attention')
        tag = f'{args.title_tag} (fold {fold})'

        # Drug->Gene: rows = atoms, cols = top genes
        gi, syms, _ = _top_genes(d['d2g'], d['gene_names'], ensg2sym,
                                 args.top_genes, force_idx)
        plot_heads(d['d2g'], gi, syms,
                   f'Drug→Gene cross-attention ({tag})',
                   'Atom index', 'Top-attended genes',
                   f'{args.prefix}{k}{args.suffix}',
                   highlight_sym=args.target_gene)

        # Gene->Drug: rows = top genes, cols = atoms
        gj, syms2, _ = _top_genes(d['g2d'], d['gene_names'], ensg2sym,
                                  args.top_genes, force_idx)
        plot_heads(d['g2d'], gj, syms2,
                   f'Gene→Drug cross-attention ({tag})',
                   'Top-attended genes', 'Atom index',
                   f'{args.prefix}{k + 1}{args.suffix}', transpose=True,
                   highlight_sym=args.target_gene, cmap='Blues')

        # atom map for the target gene (top-attended gene if absent)
        if force_idx is not None:
            map_idx, map_sym = int(force_idx), args.target_gene
        else:
            map_idx, map_sym = int(gj[0]), (syms2[0] if syms2 else '')
        plot_atom_map(d, f'{args.prefix}{k + 2}{args.suffix}',
                      gene_idx=map_idx, gene_sym=map_sym, title_tag=tag)
        k += 3


if __name__ == '__main__':
    main()
