# bca_hfp/data/loader.py
import os
import torch
import pandas as pd
import numpy as np
from rdkit import Chem
import pickle
from rdkit.Chem import AllChem
from rdkit import RDLogger

# silence RDKit C++ level logs (e.g. "DEPRECATION WARNING: please use MorganGenerator").
# they are printed on every legacy fingerprint call, flooding the console and nohup
# logs, and Python's warnings module cannot intercept them; use the RDKit logger.
RDLogger.DisableLog('rdApp.*')


def standardize_smiles(smi):
    """Standardize SMILES string to canonical form using RDKit."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)


def load_drug_atom_features(h5_path):
    """
    Load precomputed drug atom features (n_atoms, 256) from HDF5 file.
    Returns dict {drug_id: atom_feat_array}.
    """
    with pd.HDFStore(h5_path, mode='r') as store:
        drug_ids = store['drug_smiles']['DRUG_ID'].tolist()
        atom_features_series = store['atom_features']['atom_features']
        atom_features_list = atom_features_series.tolist()
    return dict(zip(map(str, drug_ids), atom_features_list))


def filter_special_tokens(embeddings, gene_names):
    """Remove special tokens (<cls>, <eos>, <pad>) from GeneFormer output."""
    special_tokens = {'<cls>', '<eos>', '<pad>'}
    keep_idx = [i for i, name in enumerate(gene_names) if name not in special_tokens]
    filtered_emb = embeddings[keep_idx]  # (n_real_genes, dim)
    filtered_names = [gene_names[i] for i in keep_idx]
    return filtered_emb, filtered_names


def load_gene_embeddings(pt_dir, cell_id_col='GDSC_ID'):
    """
    Load gene embeddings from .pt files in directory.
    Returns (cell_dict, filtered_gene_names):
        cell_dict: {cell_id: filtered_embeddings_array}
        filtered_gene_names: list of str, global gene name list after filtering
    """
    cell_dict = {}
    filtered_gene_names = None
    filter_indices = None
    for fname in os.listdir(pt_dir):
        if not fname.endswith('.pt'):
            continue
        data = torch.load(os.path.join(pt_dir, fname), weights_only=False)
        meta = data['metadata']
        cell_id = str(meta.get(cell_id_col))
        if cell_id == 'None' or cell_id is None:
            continue
        emb = data['embeddings'].float().numpy()  # (n_genes, dim)
        if 'gene_names' in data:
            gene_names = data['gene_names']
            if filter_indices is None:
                special_tokens = {'<cls>', '<eos>', '<pad>'}
                filter_indices = [i for i, name in enumerate(gene_names) if name not in special_tokens]
                filtered_gene_names = [gene_names[i] for i in filter_indices]
            filtered_emb = emb[filter_indices]
        else:
            filtered_emb = emb
            if filtered_gene_names is None:
                filtered_gene_names = [f'gene_{i}' for i in range(emb.shape[0])]
        cell_dict[cell_id] = filtered_emb
    return cell_dict, filtered_gene_names


def load_response(h5_path, cell_id_col='GDSC_ID', drug_id_col='DRUG_ID'):
    """Load raw response data from HDF5. Returns DataFrame with drug ID, cell ID, and label columns."""
    with pd.HDFStore(h5_path, mode='r') as store:
        df = store['original_data']
    df[drug_id_col] = df[drug_id_col].astype(str)
    df[cell_id_col] = df[cell_id_col].astype(str)
    return df


def pool_drug_atom_features(atom_feat_dict):
    """Average atom features along atom dimension to get fixed-length vector for baseline models."""
    pooled = {}
    for smi, feat in atom_feat_dict.items():
        pooled[smi] = np.mean(feat, axis=0) if feat.shape[0] > 0 else np.zeros(256)
    return pooled


def pool_gene_embeddings(gene_emb_dict):
    """Average gene embeddings along gene dimension to get global expression vector for baseline models."""
    pooled = {}
    for cell_id, emb in gene_emb_dict.items():
        pooled[cell_id] = np.mean(emb, axis=0)
    return pooled


def load_drug_atom_features_and_fingerprint(h5_path, fp_bits=2048, fp_radius=2):
    """
    Load atom features and generate Morgan fingerprints.
    Returns (atom_feat_dict, fingerprint_dict):
        atom_feat_dict: {drug_id: atom_feat_matrix} (n_atoms, 256)
        fingerprint_dict: {drug_id: fingerprint_vector} (fp_bits,)
    """
    with pd.HDFStore(h5_path, mode='r') as store:
        drug_ids = store['drug_smiles']['DRUG_ID'].tolist()
        atom_features_series = store['atom_features']['atom_features']
        atom_features_list = atom_features_series.tolist()
        smiles_list = store['drug_smiles']['DRUG_SMILE'].tolist()

    atom_dict = {}
    fp_dict = {}
    for did, smi, atom_feat in zip(drug_ids, smiles_list, atom_features_list):
        atom_dict[str(did)] = atom_feat
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, fp_radius, nBits=fp_bits)
            fp_dict[str(did)] = np.array(fp, dtype=np.float32)
        else:
            fp_dict[str(did)] = np.zeros(fp_bits, dtype=np.float32)
    return atom_dict, fp_dict


def load_drug_global_embeddings(pkl_path):
    """Load precomputed drug global features from pickle file."""
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)  # dict {drug_id: np.array(256,)}


# drug physicochemical descriptors: tied to potency and fully available for unseen drugs (SMILES only)
PHYSCHEM_DESC_NAMES = ['MolWt', 'MolLogP', 'TPSA', 'NumHDonors', 'NumHAcceptors',
                       'NumAromaticRings', 'NumRotatableBonds', 'FractionCSP3']
PHYSCHEM_DESC_DIM = len(PHYSCHEM_DESC_NAMES)


def _physchem_descriptors(mol):
    """compute 8 physicochemical descriptors from an RDKit mol; None on failure."""
    try:
        from rdkit.Chem import Descriptors, rdMolDescriptors
        return np.array([
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.TPSA(mol),
            rdMolDescriptors.CalcNumHBD(mol),
            rdMolDescriptors.CalcNumHBA(mol),
            rdMolDescriptors.CalcNumAromaticRings(mol),
            rdMolDescriptors.CalcNumRotatableBonds(mol),
            rdMolDescriptors.CalcFractionCSP3(mol),
        ], dtype=np.float32)
    except Exception:
        return None


def pool_fgbert_global(atom_feat_dict, mode='meanmax', standardize=True):
    """pool atom-level FG-BERT embeddings (n_atoms, 256) into one drug-level vector.

    mode='mean'   : mean over atoms (overall chemical environment)
    mode='max'    : per-dimension max over atoms (most prominent functional group)
    mode='meanmax': concat of both (2*dim), keeps mean environment and extremes

    standardize=True applies a per-column z-score over drugs (features only, no
    labels), putting FG-BERT on the same scale as Morgan bits and the already
    z-scored descriptors, which keeps the Linear projection gradients stable.
    """
    out = {}
    for did, feat in atom_feat_dict.items():
        a = np.asarray(feat, dtype=np.float32)
        if a.ndim != 2 or a.shape[0] == 0:
            dim = a.shape[-1] if a.ndim == 2 else 256
            n = dim * 2 if mode == 'meanmax' else dim
            out[did] = np.zeros(n, dtype=np.float32)
            continue
        mean_v = a.mean(axis=0)
        if mode == 'meanmax':
            out[did] = np.concatenate([mean_v, a.max(axis=0)], axis=0).astype(np.float32)
        elif mode == 'max':
            out[did] = a.max(axis=0).astype(np.float32)
        else:
            out[did] = mean_v.astype(np.float32)

    if standardize and out:
        keys = sorted(out)
        mat = np.stack([out[k] for k in keys])
        mu = mat.mean(axis=0, keepdims=True)
        sd = mat.std(axis=0, keepdims=True)
        sd[sd < 1e-8] = 1.0
        mat = (mat - mu) / sd
        for i, k in enumerate(keys):
            out[k] = mat[i].astype(np.float32)
    return out


def load_morgan_fingerprints(h5_path, fp_bits=256, fp_radius=2, use_desc=False,
                             fgbert_dict=None):
    """
    Generate Morgan fingerprints from SMILES in HDF5 as drug global features.

    use_desc=True: append the 8 physicochemical descriptors (per-column z-score),
                   output dim = fp_bits + PHYSCHEM_DESC_DIM. Standardisation uses
                   SMILES only (no labels), so there is no label leakage.
    fgbert_dict:   optional {drug_id: pooled drug-level FG-BERT vector} (see
                   pool_fgbert_global), appended last. Concat order is fixed:
                   Morgan -> descriptors -> FG-BERT.
    """
    fp_dict = {}
    desc_dict = {}
    with pd.HDFStore(h5_path, mode='r') as store:
        drug_ids = store['drug_smiles']['DRUG_ID'].tolist()
        smiles_list = store['drug_smiles']['DRUG_SMILE'].tolist()

    for did, smi in zip(drug_ids, smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, fp_radius, nBits=fp_bits)
            fp_dict[str(did)] = np.array(fp, dtype=np.float32)
            desc_dict[str(did)] = _physchem_descriptors(mol)
        else:
            fp_dict[str(did)] = np.zeros(fp_bits, dtype=np.float32)
            desc_dict[str(did)] = None

    keys = list(fp_dict.keys())
    if fgbert_dict is not None:
        _dim = int(next(iter(fgbert_dict.values())).shape[0])
        for k in keys:
            v = fgbert_dict.get(k)
            if v is None:
                v = np.zeros(_dim, dtype=np.float32)
            fp_dict[k] = np.concatenate(
                [fp_dict[k], np.asarray(v, dtype=np.float32)], axis=0)

    if not use_desc:
        return fp_dict

    mat = np.stack([
        desc_dict[k] if desc_dict[k] is not None
        else np.full(PHYSCHEM_DESC_DIM, np.nan, dtype=np.float32)
        for k in keys
    ])
    # fill missing values with the column mean
    col_mean = np.nanmean(mat, axis=0)
    rows, cols = np.where(np.isnan(mat))
    if len(rows) > 0:
        mat[rows, cols] = np.take(col_mean, cols)
    # per-column z-score
    mu = mat.mean(axis=0, keepdims=True)
    sd = mat.std(axis=0, keepdims=True)
    sd[sd < 1e-8] = 1.0
    mat = (mat - mu) / sd

    for i, k in enumerate(keys):
        fp_dict[k] = np.concatenate([fp_dict[k], mat[i]], axis=0).astype(np.float32)
    return fp_dict
