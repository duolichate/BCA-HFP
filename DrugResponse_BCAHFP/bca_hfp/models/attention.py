# bca_hfp/models/attention.py
import torch
import torch.nn as nn


class CrossAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout, gene_input_dim):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.drug_proj = nn.Linear(d_model, d_model) if d_model != 256 else nn.Identity()
        self.gene_proj = nn.Linear(gene_input_dim, d_model) if gene_input_dim != d_model else nn.Identity()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model)
        )
        self.dropout = nn.Dropout(dropout)
        self.input_norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.last_attention_weights_per_head = None
        self.last_attention_weights = None
        self.attention_history = {}

    def forward(self, drug_feat, gene_feat, drug_mask=None, gene_mask=None):
        drug = self.drug_proj(drug_feat)
        gene = self.gene_proj(gene_feat)
        drug = self.input_norm(drug)
        gene = self.input_norm(gene)
        key_padding_mask = ~gene_mask if gene_mask is not None else None
        attn_output, attn_weights = self.cross_attn(
            query=drug, key=gene, value=gene,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False
        )
        self.last_attention_weights = attn_weights.detach().mean(dim=1)
        self.last_attention_weights_per_head = attn_weights.detach()  # (B, 8, atoms, genes)
        drug = self.norm(drug + self.dropout(attn_output))
        drug = drug + self.dropout(self.ffn(drug))
        return drug

    def accumulate_attention_weights(self, epoch, sample_idx=0):
        if self.last_attention_weights is not None:
            weights = self.last_attention_weights[sample_idx].cpu().numpy()
            self.attention_history[(epoch, sample_idx)] = weights

    def save_accumulated_weights(self, fold_idx, output_dir, suffix=''):
        if not self.attention_history:
            return
        import os, numpy as np
        os.makedirs(output_dir, exist_ok=True)
        for (epoch, sample_idx), weights in self.attention_history.items():
            file_path = os.path.join(output_dir,
                                     f'attn_weights_fold{fold_idx}_epoch{epoch}_sample{sample_idx}{suffix}.npy')
            np.save(file_path, weights)
        self.attention_history.clear()
