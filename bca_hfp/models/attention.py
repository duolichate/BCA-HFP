# bca_hfp/models/attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout, gene_input_dim,
                 center_keys=False, center_query=False, keep_pooled=False):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        # key centering: subtract the cross-gene mean direction; identity under softmax
        self.center_keys = center_keys
        # query centering: subtract the mean query direction, keeping only each drug's
        # deviation from the mean; unlike key centering this changes the logits.
        # per-atom query norm is restored afterwards
        self.center_query = center_query
        if center_query:
            self.register_buffer('q_mean', torch.zeros(d_model))
            self.q_momentum = 0.9
        # per-sample pooled query of the current batch, for contrastive loss (not used at inference)
        self.keep_pooled = keep_pooled
        self.last_query_pooled = None
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

    @staticmethod
    def _film(x, cond):
        """FiLM: cond = [gamma | beta] with d dims each; returns x*(1+gamma) + beta."""
        gamma, beta = cond.chunk(2, dim=-1)
        return x * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)

    def forward(self, drug_feat, gene_feat, drug_mask=None, gene_mask=None,
                cond=None, cond_pos='post', key_cond=None):
        drug = self.drug_proj(drug_feat)
        gene = self.gene_proj(gene_feat)
        drug = self.input_norm(drug)
        gene = self.input_norm(gene)

        _cond_used = False
        if cond is not None and cond_pos == 'pre':
            drug = self._film(drug, cond)
            _cond_used = True

        # query centering: subtract mean query direction c. Estimated from the current
        # batch during training and written to a running buffer; the buffer is used at
        # inference so predictions do not depend on batch composition.
        _centered = self.center_query and drug_mask is not None
        if _centered:
            m = drug_mask.unsqueeze(-1).float()
            pooled = (drug * m).sum(dim=1) / (m.sum(dim=1) + 1e-9)      # (B, d) per-sample pooled query
            if self.training:
                c = pooled.mean(dim=0)                                    # (d,)
                with torch.no_grad():
                    self.q_mean.mul_(self.q_momentum).add_((1 - self.q_momentum) * c.detach())
            else:
                c = self.q_mean
            drug_c = drug - c.view(1, 1, -1)
            if cond is not None and cond_pos == 'mid':
                drug_c = self._film(drug_c, cond)
                _cond_used = True
            # restore each atom's original query norm; clamp at 50 in case a batch has
            # near-identical drugs and the ratio blows up
            n_orig = drug.norm(dim=-1, keepdim=True)
            n_new = drug_c.norm(dim=-1, keepdim=True)
            drug = drug_c * (n_orig / (n_new + 1e-6)).clamp(max=50.0)

        # 'post' position, or 'mid' fallback when center_query is off
        if cond is not None and not _cond_used:
            drug = self._film(drug, cond)

        if self.keep_pooled:
            if drug_mask is not None:
                mm = drug_mask.unsqueeze(-1).float()
                pooled = (drug * mm).sum(dim=1) / (mm.sum(dim=1) + 1e-9)
            else:
                pooled = drug.mean(dim=1)
            self.last_query_pooled = F.linear(
                pooled, self.cross_attn.in_proj_weight[:self.d_model])

        key = gene
        if key_cond is not None:
            # drug-conditioned key: multiplicative per-dim reweighting of gene keys.
            # multiplicative only; an additive offset is constant over genes and
            # cancels in softmax. zero-init gamma = 0 leaves the key unchanged
            key = key * (1.0 + key_cond.unsqueeze(1))
        if self.center_keys:
            # center key only; value keeps the original features
            if gene_mask is not None:
                m = gene_mask.unsqueeze(-1).float()
                key = key - (key * m).sum(dim=1, keepdim=True) / (m.sum(dim=1, keepdim=True) + 1e-9)
            else:
                key = key - key.mean(dim=1, keepdim=True)
        key_padding_mask = ~gene_mask if gene_mask is not None else None
        attn_output, attn_weights = self.cross_attn(
            query=drug, key=key, value=gene,
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
