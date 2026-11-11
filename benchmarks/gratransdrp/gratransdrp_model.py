"""
GraTransDRP architecture implementation.
Uses GAT + GCN + Transformer to encode drug molecular graphs.
Cell line features projected via Linear layer (adapted for 256-dim GeneFormer embeddings).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.data import Data, Batch
from rdkit import Chem
import numpy as np


def atom_features(atom):
    """Atom feature encoding for GraTransDRP (78-dim)."""
    return np.array([
        one_of_k_encoding_unk(atom.GetSymbol(), [
            'C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na',
            'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb',
            'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge',
            'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg',
            'Pb', 'Unknown'
        ]) +
        one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
        one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
        one_of_k_encoding_unk(atom.GetValence(Chem.ValenceType.IMPLICIT), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
        [atom.GetIsAromatic()]
    ])


def one_of_k_encoding(x, allowable_set):
    return [1 if x == s else 0 for s in allowable_set]


def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return [1 if x == s else 0 for s in allowable_set]


def smile_to_graph(smile):
    """Convert SMILES to PyG graph format."""
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        return 0, [], []

    c_size = mol.GetNumAtoms()

    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        s = np.sum(feature)
        features.append(feature / s if s > 0 else feature)

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])

    directed_edges = []
    for e1, e2 in edges:
        directed_edges.append([e1, e2])
        directed_edges.append([e2, e1])

    if directed_edges:
        edge_index = torch.tensor(directed_edges, dtype=torch.long).t().contiguous()
    else:
        # bondless molecule (isolated ion/salt): a 1-d empty tensor makes Batch + GCN crash in the
        # scatter_add_ bounds check, so return a valid (2,0) empty edge index
        edge_index = torch.empty((2, 0), dtype=torch.long)

    return c_size, features, edge_index


class TransformerBlock(nn.Module):
    """Manual multi-head self-attention with flexible input dimensions."""
    def __init__(self, d_model, nhead=1, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        assert self.head_dim * nhead == d_model, "d_model must be divisible by nhead"

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Input x: (N, d_model), (N, 1, d_model) or (N, 1, 1, d_model).
        Output: (N, 1, d_model).
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)
        elif x.dim() == 4:
            x = x.squeeze(1).squeeze(1)
            if x.dim() == 2:
                x = x.unsqueeze(1)
        elif x.dim() != 3:
            raise ValueError(f"Unsupported input dimension: {x.dim()}")

        batch, seq_len, d_model = x.shape
        assert d_model == self.d_model, f"d_model mismatch: {d_model} vs {self.d_model}"

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        Q = Q.view(batch, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        K = K.view(batch, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        V = V.view(batch, seq_len, self.nhead, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_out = torch.matmul(attn_weights, V)
        attn_out = attn_out.transpose(1, 2).contiguous()
        attn_out = attn_out.view(batch, seq_len, self.d_model)
        attn_out = self.out_proj(attn_out)

        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        ffn_out = self.ffn(x)
        x = x + self.dropout(ffn_out)
        x = self.norm2(x)

        return x  # (batch, seq_len, d_model)


class GAT_GCN_Transformer_DRP(nn.Module):
    """GraTransDRP architecture (full version with Transformer)."""
    def __init__(self,
                 n_output=1,
                 num_features_xd=78,
                 output_dim=128,
                 gene_input_dim=256,
                 dropout=0.3,
                 n_filters=32,
                 config=None):
        super().__init__()

        self.config = config
        self.num_features_xd = num_features_xd
        self.output_dim = output_dim
        self.gene_input_dim = gene_input_dim

        self.transformer_block1 = TransformerBlock(
            d_model=num_features_xd,
            nhead=1,
            dropout=dropout
        )

        self.conv1 = GATConv(num_features_xd, num_features_xd, heads=10, dropout=dropout)

        self.transformer_block2 = TransformerBlock(
            d_model=num_features_xd * 10,
            nhead=1,
            dropout=dropout
        )

        self.conv2 = GCNConv(num_features_xd * 10, num_features_xd * 10)

        self.fc_g1 = nn.Linear(num_features_xd * 10 * 2, 1500)
        self.fc_g2 = nn.Linear(1500, output_dim)

        self.fc_gene = nn.Linear(gene_input_dim, output_dim)

        self.fc1 = nn.Linear(output_dim * 2, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.out = nn.Linear(128, n_output)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(128)

        print(f"[Model Init] fc_gene: in_features={self.fc_gene.in_features}, out_features={self.fc_gene.out_features}")

    def forward(self, gene_features, drug_graph, drug_mask=None, gene_mask=None,
                drug_global=None, return_attn=False):
        B = len(drug_graph)
        device = gene_features.device

        batch = Batch.from_data_list(drug_graph).to(device)
        x, edge_index = batch.x, batch.edge_index

        x = x.unsqueeze(1)
        x = self.transformer_block1(x)
        x = x.squeeze(1)

        x = self.conv1(x, edge_index)
        x = self.relu(x)

        x = x.unsqueeze(1)
        x = self.transformer_block2(x)
        x = x.squeeze(1)

        x = self.conv2(x, edge_index)
        x = self.relu(x)

        x_gap = gap(x, batch.batch)
        x_gmp = gmp(x, batch.batch)
        x = torch.cat([x_gap, x_gmp], dim=1)

        x = self.relu(self.fc_g1(x))
        x = self.dropout(x)
        drug_emb = self.fc_g2(x)

        if gene_features.dim() == 3:
            gene_pooled = gene_features.mean(dim=1)
        else:
            gene_pooled = gene_features

        gene_emb = self.fc_gene(gene_pooled)

        fused = torch.cat([drug_emb, gene_emb], dim=1)

        fused = self.fc1(fused)
        fused = self.bn1(fused)
        fused = self.relu(fused)
        fused = self.dropout(fused)

        fused = self.fc2(fused)
        fused = self.bn2(fused)
        fused = self.relu(fused)
        fused = self.dropout(fused)

        out = self.out(fused)
        out = out.squeeze(-1)

        if return_attn:
            return out, None

        return out


class RegressorGraTransDRP(GAT_GCN_Transformer_DRP):
    pass


def create_gra_trans_drp_model(config):
    return RegressorGraTransDRP(
        n_output=1,
        num_features_xd=78,
        output_dim=128,
        gene_input_dim=config['data']['gene_input_dim'],
        dropout=config['model']['dropout'],
        config=config
    )