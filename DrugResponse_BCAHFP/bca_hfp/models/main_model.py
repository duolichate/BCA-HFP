# bca_hfp/models/main_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import CrossAttention
from .gated_fusion import GatedFusionLayer


class Regressor(nn.Module):
    def __init__(self, feature_dim, num_heads, dropout, hidden_dims,
                 gene_input_dim, drug_global_dim, num_fusion_layers, config=None):
        super().__init__()

        self.config = config
        self.feature_dim = feature_dim
        self.num_fusion_layers = num_fusion_layers

        # Bidirectional cross-attention encoder
        self.cross_attn_drug2gene = CrossAttention(
            d_model=feature_dim, num_heads=num_heads, dropout=dropout,
            gene_input_dim=gene_input_dim
        )
        self.cross_attn_gene2drug = CrossAttention(
            d_model=feature_dim, num_heads=num_heads, dropout=dropout,
            gene_input_dim=feature_dim
        )

        # Global feature projection
        self.gene_global_proj = nn.Linear(gene_input_dim,
                                          feature_dim) if gene_input_dim != feature_dim else nn.Identity()
        self.drug_global_proj = nn.Linear(drug_global_dim,
                                          feature_dim) if drug_global_dim != feature_dim else nn.Identity()

        # Multi-layer gated fusion decoder
        self.fusion_layers = nn.ModuleList([
            GatedFusionLayer(feature_dim, num_heads, dropout, num_modalities=3,
                             temperature=config['training']['gate_temperature'])
            for _ in range(num_fusion_layers)
        ])

        # Regression head
        self.regression_head = self._build_regression_head(feature_dim, hidden_dims, dropout)

        self._current_epoch = 0
        self._diagnosed_epoch = -1
        self._enable_diagnostic = True

    def set_epoch(self, epoch):
        """Called by trainer at the start of each epoch."""
        self._current_epoch = epoch

    def enable_diagnostic(self, enable=True):
        """Enable or disable diagnostic printing."""
        self._enable_diagnostic = enable

    def _build_regression_head(self, input_dim, hidden_dims, dropout):
        layers = [nn.Dropout(0.5)]
        curr_dim = input_dim

        for hdim in hidden_dims:
            layers.extend([
                nn.Linear(curr_dim, hdim),
                nn.BatchNorm1d(hdim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            curr_dim = hdim
        layers.append(nn.Linear(curr_dim, 1))
        return nn.Sequential(*layers)

    def forward(self, gene_features, drug_features, drug_mask=None, gene_mask=None,
                drug_global=None, return_attn=False):

        # Drug -> Gene
        drug_attended = self.cross_attn_drug2gene(drug_features, gene_features, drug_mask, gene_mask)
        drug2gene_attn = self.cross_attn_drug2gene.last_attention_weights_per_head  # (B, 8, atoms, genes)

        if drug_mask is not None:
            mask_exp = drug_mask.unsqueeze(-1).float()
            drug_pooled = (drug_attended * mask_exp).sum(dim=1) / (mask_exp.sum(dim=1) + 1e-9)
        else:
            drug_pooled = drug_attended.mean(dim=1)

        # Gene -> Drug
        gene_attended = self.cross_attn_gene2drug(gene_features, drug_features, gene_mask, drug_mask)
        gene2drug_attn = self.cross_attn_gene2drug.last_attention_weights_per_head  # (B, 8, genes, atoms)

        if gene_mask is not None:
            mask_exp_gene = gene_mask.unsqueeze(-1).float()
            gene_pooled = (gene_attended * mask_exp_gene).sum(dim=1) / (mask_exp_gene.sum(dim=1) + 1e-9)
        else:
            gene_pooled = gene_attended.mean(dim=1)

        # Global gene
        if gene_mask is not None:
            mask_exp_global = gene_mask.unsqueeze(-1).float()
            global_gene_raw = (gene_features * mask_exp_global).sum(dim=1) / (mask_exp_global.sum(dim=1) + 1e-9)
        else:
            global_gene_raw = gene_features.mean(dim=1)
        global_gene = self.gene_global_proj(global_gene_raw)

        # Drug global
        drug_global_proj = self.drug_global_proj(drug_global) if drug_global is not None else None

        # Modality dropout
        if self.training and self.config is not None:
            drug_global_dropout = self.config['training'].get('drug_global_dropout', 0.5)
            global_gene_dropout = self.config['training'].get('global_gene_dropout', 0.3)

            if drug_global_proj is not None and torch.rand(1).item() < drug_global_dropout:
                drug_global_proj = torch.zeros_like(drug_global_proj)
            if torch.rand(1).item() < global_gene_dropout:
                global_gene = torch.zeros_like(global_gene)

        # Build modality feature sequences
        norm_shape = (self.feature_dim,)
        gene_pooled_norm = F.layer_norm(gene_pooled, norm_shape)
        global_gene_norm = F.layer_norm(global_gene, norm_shape)
        drug_global_norm = F.layer_norm(drug_global_proj, norm_shape) if drug_global_proj is not None else None

        # Per-epoch diagnostic: print once at the first batch of each epoch
        if self.training and self._enable_diagnostic and self._diagnosed_epoch != self._current_epoch:
            print("\n" + "=" * 60)
            print(f"[Diagnostic] Epoch {self._current_epoch} - Feature distribution after LayerNorm")
            print("=" * 60)

            gp_mean, gp_std = gene_pooled_norm.mean().item(), gene_pooled_norm.std().item()
            gp_min, gp_max = gene_pooled_norm.min().item(), gene_pooled_norm.max().item()
            print(
                f"Gene Pooled    | Mean: {gp_mean:>7.4f} | Std: {gp_std:>7.4f} | Min: {gp_min:>7.4f} | Max: {gp_max:>7.4f}")

            gg_mean, gg_std = global_gene_norm.mean().item(), global_gene_norm.std().item()
            gg_min, gg_max = global_gene_norm.min().item(), global_gene_norm.max().item()
            print(
                f"Global Gene    | Mean: {gg_mean:>7.4f} | Std: {gg_std:>7.4f} | Min: {gg_min:>7.4f} | Max: {gg_max:>7.4f}")

            if drug_global_norm is not None:
                dg_mean, dg_std = drug_global_norm.mean().item(), drug_global_norm.std().item()
                dg_min, dg_max = drug_global_norm.min().item(), drug_global_norm.max().item()
                print(
                    f"Drug Global    | Mean: {dg_mean:>7.4f} | Std: {dg_std:>7.4f} | Min: {dg_min:>7.4f} | Max: {dg_max:>7.4f}")
            else:
                print(f"Drug Global    | (zeroed by dropout)")

            print("=" * 60 + "\n")

            self._diagnosed_epoch = self._current_epoch

        # Build modality sequences (each shape (B,1,d_model))
        gene_pooled_seq = gene_pooled_norm.unsqueeze(1)
        global_gene_seq = global_gene_norm.unsqueeze(1)
        modality_features = [gene_pooled_seq, global_gene_seq]
        if drug_global_norm is not None:
            modality_features.append(drug_global_norm.unsqueeze(1))

        # Initial query (drug pooled)
        query = drug_pooled.unsqueeze(1)  # (B,1,d_model)

        # Multi-layer gated fusion
        for layer in self.fusion_layers:
            query, _ = layer(query, modality_features)

        fused = query.squeeze(1)  # (B,d_model)
        out = self.regression_head(fused).squeeze(-1)

        # Apply sigmoid if label is AUC (0~1)
        if self.config and self.config['data']['label_col'] == 'AUC':
            out = torch.sigmoid(out)

        if return_attn:
            return out, drug2gene_attn, gene2drug_attn

        return out
