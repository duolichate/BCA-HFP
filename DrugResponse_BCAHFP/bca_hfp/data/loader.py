# data/loader.py
"""
Load drug features, gene embeddings, and response data. Provides pooling functions for baseline models.
"""
import os
import torch
import pandas as pd
import numpy as np
from rdkit import Chem
import pickle
from rdkit.Chem import AllChem


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


def load_morgan_fingerprints(h5_path, fp_bits=256, fp_radius=2):
    """
    Generate Morgan fingerprints from SMILES in HDF5 as drug global features.
    Default 256-dim to match existing model architecture.
    """
    fp_dict = {}
    with pd.HDFStore(h5_path, mode='r') as store:
        drug_ids = store['drug_smiles']['DRUG_ID'].tolist()
        smiles_list = store['drug_smiles']['DRUG_SMILE'].tolist()

    for did, smi in zip(drug_ids, smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, fp_radius, nBits=fp_bits)
            fp_dict[str(did)] = np.array(fp, dtype=np.float32)
        else:
            fp_dict[str(did)] = np.zeros(fp_bits, dtype=np.float32)

    return fp_dict
