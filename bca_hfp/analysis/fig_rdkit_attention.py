# bca_hfp/analysis/fig_rdkit_attention.py
"""Map cross-attention weights onto RDKit 2D molecular structure."""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
import pickle
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from bca_hfp.config.base_config import PROJECT_ROOT, root_dir

GENE_FEAT_DIR = os.path.join(root_dir("Data"), "GDSC_EXP_GF_V1")

# Gene name loading functions (directly from .pt files)

def load_ensembl_to_symbol_map():
    """Load ENSG -> Gene Symbol mapping"""
    map_file = os.path.join(root_dir("Gene_Feature"), "geneformer", "gene_dictionaries_30m", "gene_name_id_dict_gc30M.pkl")
    with open(map_file, 'rb') as f:
        symbol_to_ensembl = pickle.load(f)
    ensembl_to_symbol = {}
    for symbol, ens_id in symbol_to_ensembl.items():
        ens_no_version = ens_id.split('.')[0]
        ensembl_to_symbol[ens_no_version] = symbol.upper()
    return ensembl_to_symbol


def load_gene_names_from_pt(cell_id, gene_dir=GENE_FEAT_DIR):
    """
    Read gene name list from .pt file for a specified cell line
    Args:
        cell_id: Cell line ID (e.g., '687794')
        gene_dir: Geneformer embedding directory
    Returns:
        list: Gene name list (special tokens filtered)
    """
    special_tokens = {'<cls>', '<eos>', '<pad>'}
    for fname in os.listdir(gene_dir):
        if not fname.endswith('.pt'):
            continue
        data = torch.load(os.path.join(gene_dir, fname))
        meta = data.get('metadata', {})
        if str(meta.get('GDSC_ID')) == str(cell_id):
            gene_names = data.get('gene_names', [])
            # Filter special tokens
            filtered = [name for name in gene_names if name not in special_tokens]
            return filtered
    raise ValueError(f"Gene embedding file not found for cell line {cell_id}")


def get_cell_id_from_metadata(weight_dir, sample_idx):
    """Read cell_id from sample_metadata.csv"""
    meta_path = os.path.join(weight_dir, 'sample_metadata.csv')
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"sample_metadata.csv not found: {meta_path}")
    df = pd.read_csv(meta_path)
    row = df[df['sample_idx'] == sample_idx]
    if row.empty:
        raise ValueError(f"Metadata not found for sample_idx={sample_idx}")
    return str(row.iloc[0]['cell_id'])


# SMILES-related functions

def get_smiles_pair_from_metadata(weight_dir, sample_idx):
    """Read SMILES from sample_metadata.csv and auto-generate hydrogenated and non-hydrogenated versions"""
    meta_path = os.path.join(weight_dir, 'sample_metadata.csv')
    if not os.path.exists(meta_path):
        return None, None
    df = pd.read_csv(meta_path)
    row = df[df['sample_idx'] == sample_idx]
    if row.empty:
        return None, None
    smiles_raw = row.iloc[0]['smiles']
    if pd.isna(smiles_raw) or smiles_raw == '':
        return None, None

    mol = Chem.MolFromSmiles(smiles_raw)
    if mol is None:
        return None, None

    num_atoms = mol.GetNumAtoms()
    if num_atoms >= 60:
        smiles_h = smiles_raw
        mol_noH = Chem.RemoveHs(mol)
        smiles_noH = Chem.MolToSmiles(mol_noH, isomericSmiles=True, canonical=True)
    elif num_atoms >= 30:
        smiles_noH = smiles_raw
        mol_h = Chem.AddHs(mol)
        smiles_h = Chem.MolToSmiles(mol_h, isomericSmiles=True, canonical=True)
    else:
        mol_h = Chem.AddHs(mol)
        smiles_h = Chem.MolToSmiles(mol_h, isomericSmiles=True, canonical=True)
        mol_noH = Chem.RemoveHs(mol)
        smiles_noH = Chem.MolToSmiles(mol_noH, isomericSmiles=True, canonical=True)

    return smiles_h, smiles_noH


def get_smiles_pair_from_drug_id(drug_id, config=None):
    if config is None:
        from config.base_config import get_base_config
        config = get_base_config('GDSC')
    import pandas as pd
    with pd.HDFStore(config['data']['drug_feat_path'], 'r') as store:
        drug_smiles_df = store['drug_smiles']
    match = drug_smiles_df[drug_smiles_df['DRUG_ID'].astype(str) == str(drug_id)]
    if match.empty:
        return None, None
    smiles_raw = match.iloc[0]['DRUG_SMILE']
    mol = Chem.MolFromSmiles(smiles_raw)
    if mol is None:
        return None, None
    mol_h = Chem.AddHs(mol)
    smiles_h = Chem.MolToSmiles(mol_h, isomericSmiles=True, canonical=True)
    mol_noH = Chem.RemoveHs(mol)
    smiles_noH = Chem.MolToSmiles(mol_noH, isomericSmiles=True, canonical=True)
    return smiles_h, smiles_noH


# Auto-select target gene

def find_top_genes(weight_dir, sample_idx, top_n=10):
    """
    Find genes with highest attention from Drug->Gene attention
    Reads gene names directly from .pt files
    """
    # First get cell_id
    cell_id = get_cell_id_from_metadata(weight_dir, sample_idx)
    # Load gene names
    gene_names = load_gene_names_from_pt(cell_id)

    # Load Drug->Gene attention weights
    weight_path = os.path.join(weight_dir, f'sample_{sample_idx:06d}_drug2gene_8heads.npy')
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"Weight file not found: {weight_path}")
    weights = np.load(weight_path)
    # Sum across all heads and all atoms to get total attention per gene
    gene_attn = weights.mean(axis=0).sum(axis=0)
    top_indices = np.argsort(gene_attn)[-top_n:][::-1]

    ensembl_to_symbol = load_ensembl_to_symbol_map()
    results = []
    for idx in top_indices:
        ensg = gene_names[idx]
        symbol = ensembl_to_symbol.get(ensg, ensg)
        results.append((symbol, ensg, gene_attn[idx]))
    return results


# Plotting functions

def plot_atomic_weights_bar(weights, actual_num_atoms, target_gene, smiles, head_idx, output_dir):
    from matplotlib.cm import Reds
    import matplotlib.colors as mcolors

    raw_weights = weights

    min_w = np.min(raw_weights)
    max_w = np.max(raw_weights)
    norm = mcolors.Normalize(vmin=min_w, vmax=max_w)
    colors = [Reds(norm(w)) for w in raw_weights]

    fig_width = max(12, actual_num_atoms * 0.5)
    fig_height = max(5, actual_num_atoms * 0.08)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    bars = ax.bar(range(actual_num_atoms), raw_weights, color=colors, edgecolor='black', linewidth=0.8)

    if actual_num_atoms <= 40:
        step = 1
        label_fontsize = 9
    elif actual_num_atoms <= 60:
        step = 2
        label_fontsize = 7
    else:
        step = 3
        label_fontsize = 6

    for i, (bar, w) in enumerate(zip(bars, raw_weights)):
        if i % step == 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f'{w:.4f}', ha='center', va='bottom',
                    fontsize=label_fontsize, rotation=0)

    ax.set_xlabel('Atom Index', fontsize=12)
    ax.set_ylabel('Attention Weight (Softmax)', fontsize=12)
    ax.set_ylim(0, max(raw_weights) * 1.2)
    ax.set_title(f'Gene {target_gene} -> Drug: Atomic Attention Weights (Head {head_idx + 1})', fontsize=14)
    ax.grid(axis='y', alpha=0.3)

    tick_step = max(1, actual_num_atoms // 30)
    tick_positions = list(range(0, actual_num_atoms, tick_step))
    tick_labels = [str(i) for i in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=90, fontsize=8)

    sm = plt.cm.ScalarMappable(cmap='Reds', norm=norm)
    sm.set_array(raw_weights)
    cbar = plt.colorbar(sm, ax=ax, shrink=0.7)
    cbar.set_label('Attention Weight', fontsize=10)

    drug_name = smiles[:30] + "..." if len(smiles) > 30 else smiles
    plt.text(0.98, 0.95, f'Drug: {drug_name}', transform=ax.transAxes,
             ha='right', va='top', fontsize=9, style='italic')

    plt.tight_layout()
    output_path = os.path.join(output_dir, f'atom_weights_{target_gene}_head{head_idx + 1}.pdf')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Atomic weight bar chart saved: {output_path}")
    return output_path


from rdkit.Chem.Draw import rdMolDraw2D
from PIL import Image

import io
from PIL import Image
from rdkit.Chem.Draw import rdMolDraw2D


def plot_mol_images(mol, output_dir, target_gene, head_idx, weights=None, dpi=600, size_inches=4):
    """
    Use MolDraw2DSVG to generate SVG, then convert to JPG (high DPI) and PDF (vector) via cairosvg.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Highlight top 5 highest-weight atoms
    highlight_atoms = []
    if weights is not None and len(weights) > 0:
        top_indices = np.argsort(weights)[-5:][::-1]
        highlight_atoms = top_indices.tolist()

    # Use SVG renderer (no Cairo needed)
    drawer = rdMolDraw2D.MolDraw2DSVG(size_inches * dpi, size_inches * dpi)  # Pixel dimensions for SVG canvas
    drawer.DrawMolecule(mol, highlightAtoms=highlight_atoms)
    drawer.FinishDrawing()
    svg_text = drawer.GetDrawingText()

    # ---- Save JPG (600 DPI) ----
    try:
        import cairosvg
        # Convert SVG to PNG (bitmap)
        png_data = cairosvg.svg2png(bytestring=svg_text.encode(), dpi=dpi)
        img = Image.open(io.BytesIO(png_data))
        jpg_path = os.path.join(output_dir, f'mol_structure_{target_gene}_head{head_idx + 1}.jpg')
        img.save(jpg_path, 'JPEG', quality=95, dpi=(dpi, dpi))
        print(f"Molecular structure JPG (600 DPI) saved: {jpg_path}")
    except ImportError:
        # If cairosvg not installed, save SVG and notify
        svg_path = os.path.join(output_dir, f'mol_structure_{target_gene}_head{head_idx + 1}.svg')
        with open(svg_path, 'w') as f:
            f.write(svg_text)
        print(f"Warning: cairosvg not installed, cannot generate JPG, saved SVG: {svg_path}")
        jpg_path = None

    # ---- Save PDF (vector) ----
    try:
        import cairosvg
        pdf_path = os.path.join(output_dir, f'mol_structure_{target_gene}_head{head_idx + 1}.pdf')
        cairosvg.svg2pdf(bytestring=svg_text.encode(), write_to=pdf_path)
        print(f"Molecular structure PDF saved: {pdf_path}")
    except ImportError:
        pdf_path = None
        print("Warning: cairosvg not installed, cannot generate PDF, please install cairosvg")

    return {'jpg': jpg_path, 'pdf': pdf_path, 'svg': svg_text if jpg_path is None else None}

def plot_mol_png(mol, output_dir, target_gene, head_idx, weights=None):
    try:
        highlight_atoms = None
        if weights is not None and len(weights) > 0:
            top_indices = np.argsort(weights)[-5:][::-1]
            highlight_atoms = top_indices.tolist()

        img = Draw.MolToImage(
            mol,
            size=(500, 500),
            kekulize=True,
            wedgeBonds=True,
            fitImage=True,
            highlightAtoms=highlight_atoms
        )
        png_path = os.path.join(output_dir, f'mol_structure_{target_gene}_head{head_idx + 1}.png')
        img.save(png_path)
        print(f"Molecular structure PNG saved: {png_path}")
        return png_path
    except Exception as e:
        print(f"Warning: Failed to generate PNG: {e}")
        return None


def plot_mol_svg(mol, output_dir, target_gene, head_idx, weights=None):
    try:
        highlight_atoms = None
        if weights is not None and len(weights) > 0:
            top_indices = np.argsort(weights)[-5:][::-1]
            highlight_atoms = top_indices.tolist()

        svg = Draw.MolToSVG(
            mol,
            width=400,
            height=400,
            kekulize=True,
            highlightAtoms=highlight_atoms
        )
        svg_path = os.path.join(output_dir, f'mol_structure_{target_gene}_head{head_idx + 1}.svg')
        with open(svg_path, 'w') as f:
            f.write(svg)
        print(f"Molecular structure SVG saved: {svg_path}")
        return svg_path
    except Exception as e:
        print(f"Warning: SVG failed, trying PNG...")
        return plot_mol_png(mol, output_dir, target_gene, head_idx, weights)


def plot_rdkit_micro_attention(weight_dir, sample_idx, target_gene,
                               smiles_h, smiles_noH, output_dir, head_idx=0, dpi=600):
    os.makedirs(output_dir, exist_ok=True)

    # ----- 1. Read cell_id from metadata -----
    cell_id = get_cell_id_from_metadata(weight_dir, sample_idx)
    print(f"Cell line ID: {cell_id}")

    # ----- 2. Read gene names from .pt file -----
    gene_names = load_gene_names_from_pt(cell_id)

    # ----- 3. Load ENSG -> Symbol mapping -----
    ensembl_to_symbol = load_ensembl_to_symbol_map()
    symbol_to_ensg = {v: k for k, v in ensembl_to_symbol.items()}

    original_target = target_gene
    if not target_gene.startswith('ENSG'):
        ensg = symbol_to_ensg.get(target_gene)
        if ensg is not None:
            target_gene = ensg
            print(f"Converted symbol {original_target} to ENSG: {target_gene}")

    if target_gene not in gene_names:
        print(f"Error: Gene {target_gene} not in this cell line's gene list")
        print(f"   Note: This cell line has {len(gene_names)} genes in total")
        return None

    gene_idx = gene_names.index(target_gene)
    print(f"Target gene: {original_target} (ENSG: {target_gene}, index {gene_idx})")

    # ----- 4. Load attention weights -----
    weight_path = os.path.join(weight_dir, f'sample_{sample_idx:06d}_gene2drug_8heads.npy')
    if not os.path.exists(weight_path):
        print(f"Error: Weight file not found: {weight_path}")
        return None
    weights = np.load(weight_path)
    print(f"Loaded weights: {weight_path}")
    print(f"  Shape: {weights.shape}")

    raw_weights = weights[head_idx, gene_idx, :]
    num_atoms_weight = len(raw_weights)
    print(f"Weight vector length: {num_atoms_weight}")

    # ----- 5. Process molecule -----
    mol_noH = Chem.MolFromSmiles(smiles_noH)
    if mol_noH is None:
        print(f"Error: Cannot parse non-hydrogenated SMILES")
        return None
    AllChem.Compute2DCoords(mol_noH)
    num_atoms_noH = mol_noH.GetNumAtoms()
    print(f"Heavy atom count: {num_atoms_noH}")

    # ----- 6. Draw bar chart -----
    bar_path = plot_atomic_weights_bar(
        raw_weights, num_atoms_weight, original_target,
        smiles_h if smiles_h else smiles_noH, head_idx, output_dir
    )

    # ----- 7. Draw molecular structure -----
    aligned_weights_noH = raw_weights[:num_atoms_noH] if len(raw_weights) >= num_atoms_noH else raw_weights
    structure_paths = plot_mol_images(
        mol_noH, output_dir, original_target, head_idx, aligned_weights_noH, dpi=dpi
    )

    return {'bar': bar_path, 'structure_jpg': structure_paths['jpg'], 'structure_pdf': structure_paths['pdf']}


def plot_all_heads(weight_dir, sample_idx, target_gene, smiles_h, smiles_noH, output_dir):
    results = []
    for head_idx in range(8):
        print(f"\nProcessing Head {head_idx + 1}/8...")
        result = plot_rdkit_micro_attention(
            weight_dir, sample_idx, target_gene,
            smiles_h, smiles_noH, output_dir, head_idx
        )
        if result:
            results.append(result)
    print(f"\nDone! Generated {len(results)} sets of files")
    return results


# Main function

def main():
    parser = argparse.ArgumentParser(description='Plot RDKit target-space aligned molecular diagrams')
    parser.add_argument('--weight_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--sample_idx', type=int, required=True)
    parser.add_argument('--target_gene', type=str, default=None)
    parser.add_argument('--drug_id', type=str, default=None)
    parser.add_argument('--smiles_h', type=str, default=None, help='Hydrogenated SMILES')
    parser.add_argument('--smiles_noH', type=str, default=None, help='Non-hydrogenated SMILES')
    parser.add_argument('--head_idx', type=int, default=0, choices=range(8))
    parser.add_argument('--all_heads', action='store_true')
    parser.add_argument('--top_n', type=int, default=10)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ----- Get SMILES -----
    smiles_h = args.smiles_h
    smiles_noH = args.smiles_noH

    if args.drug_id is not None:
        print(f"Looking up SMILES for drug_id={args.drug_id}...")
        meta_h, meta_noH = get_smiles_pair_from_metadata(args.weight_dir, args.sample_idx)
        if meta_h is not None and meta_noH is not None:
            print("  Successfully retrieved from sample_metadata.csv")
            smiles_h = meta_h if smiles_h is None else smiles_h
            smiles_noH = meta_noH if smiles_noH is None else smiles_noH
        else:
            data_h, data_noH = get_smiles_pair_from_drug_id(args.drug_id)
            if data_h is not None and data_noH is not None:
                print("  Successfully retrieved from data file")
                smiles_h = data_h if smiles_h is None else smiles_h
                smiles_noH = data_noH if smiles_noH is None else smiles_noH

    if smiles_h is None or smiles_noH is None:
        raise ValueError("Must provide both hydrogenated and non-hydrogenated SMILES, use --smiles_h and --smiles_noH")

    print(f"Hydrogenated SMILES: {smiles_h[:60]}...")
    print(f"Non-hydrogenated SMILES: {smiles_noH[:60]}...")

    # ----- Auto-select target gene -----
    if args.target_gene is None:
        print("No target gene specified, auto-selecting highest-attention gene...")
        top_genes = find_top_genes(args.weight_dir, args.sample_idx, args.top_n)
        print("\nHighest-attention genes (Drug->Gene):")
        for i, (symbol, ensg, attn) in enumerate(top_genes):
            print(f"  {i + 1}. {symbol} ({ensg}): {attn:.6f}")
        args.target_gene = top_genes[0][0]  # Use Symbol

    # ----- Plot -----
    if args.all_heads:
        plot_all_heads(args.weight_dir, args.sample_idx, args.target_gene,
                       smiles_h, smiles_noH, args.output_dir)
    else:
        plot_rdkit_micro_attention(args.weight_dir, args.sample_idx, args.target_gene,
                                   smiles_h, smiles_noH, args.output_dir,
                                   head_idx=args.head_idx)


if __name__ == '__main__':
    main()
