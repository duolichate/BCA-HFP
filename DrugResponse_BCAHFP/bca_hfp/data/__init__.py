# data/__init__.py
"""
Data loading, preprocessing, and splitting with cell line/drug stratification to prevent data leakage.
"""
from .loader import (
    load_drug_atom_features, load_gene_embeddings, load_response,
    pool_drug_atom_features, pool_gene_embeddings
)
from .dataset import DrugGeneDataset, BaselineDataset, collate_fn

__all__ = [
    'load_drug_atom_features', 'load_gene_embeddings', 'load_response',
    'pool_drug_atom_features', 'pool_gene_embeddings',
    'DrugGeneDataset', 'BaselineDataset', 'collate_fn'
]

