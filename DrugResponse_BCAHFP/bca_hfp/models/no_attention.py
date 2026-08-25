# models/no_attention.py
"""
Baseline model: drug pooling + gene pooling + drug global features.
"""
import torch
import torch.nn as nn


class RegressorBaseline(nn.Module):
    def __init__(self, feature_dim, dropout, hidden_dims,
                 gene_input_dim, drug_global_dim, config=None):

        super().__init__()

        self.feature_dim = feature_dim
        self.config = config

        # Drug feature projection
        self.drug_proj = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Gene feature projection
        self.gene_proj = nn.Sequential(
            nn.Linear(gene_input_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Drug global feature projection
        self.drug_global_proj = nn.Sequential(
            nn.Linear(drug_global_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Regression head input dim = drug_emb + gene_emb + drug_global_emb
        input_dim = feature_dim * 2
        input_dim += feature_dim
        self.regression_head = self._build_regression_head(input_dim, hidden_dims, dropout)

    def _build_regression_head(self, input_dim, hidden_dims, dropout):
        layers = []
        curr = input_dim
        for hdim in hidden_dims:
            layers.extend([
                nn.Linear(curr, hdim),
                nn.BatchNorm1d(hdim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            curr = hdim
        layers.append(nn.Linear(curr, 1))
        return nn.Sequential(*layers)

    def forward(self, gene_features, drug_features, drug_mask=None, gene_mask=None,
                drug_global=None, **kwargs):
        """
        Args:
            gene_features: (B, n_genes, gene_input_dim)
            drug_features: (B, n_atoms, feature_dim)
            drug_mask: (B, n_atoms) bool, True for valid atoms
            gene_mask: (B, n_genes) bool, True for valid genes
            drug_global: (B, drug_global_dim) drug global features (Morgan fingerprint)
        """
        # Pool drug features
        if drug_mask is not None:
            mask_exp = drug_mask.unsqueeze(-1).float()
            drug_pooled = (drug_features * mask_exp).sum(dim=1) / (mask_exp.sum(dim=1) + 1e-9)
        else:
            drug_pooled = drug_features.mean(dim=1)
        drug_emb = self.drug_proj(drug_pooled)  # (B, feature_dim)

        # Pool gene features
        if gene_mask is not None:
            mask_exp_gene = gene_mask.unsqueeze(-1).float()
            gene_pooled = (gene_features * mask_exp_gene).sum(dim=1) / (mask_exp_gene.sum(dim=1) + 1e-9)
        else:
            gene_pooled = gene_features.mean(dim=1)
        gene_emb = self.gene_proj(gene_pooled)  # (B, feature_dim)

        # Drug global features
        drug_global_emb = None
        if self.drug_global_proj is not None and drug_global is not None:
            drug_global_emb = self.drug_global_proj(drug_global)  # (B, feature_dim)

        # Modality dropout
        if self.training and self.config is not None:
            drug_global_dropout = self.config['training'].get('drug_global_dropout')
            global_gene_dropout = self.config['training'].get('global_gene_dropout')

            if drug_global_emb is not None and torch.rand(1).item() < drug_global_dropout:
                drug_global_emb = torch.zeros_like(drug_global_emb)

            if torch.rand(1).item() < global_gene_dropout:
                gene_emb = torch.zeros_like(gene_emb)

        # Fusion
        to_cat = [drug_emb, gene_emb, drug_global_emb]
        fused = torch.cat(to_cat, dim=1)  # (B, input_dim)
        out = self.regression_head(fused).squeeze(-1)  # (B,)

        return out
