import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedFusionLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout, temperature, num_modalities=3):
        super().__init__()
        self.d_model = d_model
        self.num_modalities = num_modalities
        self.temperature = temperature

        self.cross_attns = nn.ModuleList([
            nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
            for _ in range(num_modalities)
        ])

        self.gate_mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, num_modalities)
        )

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.last_gate_weights = None
        self.last_attended_features = None

    def forward(self, query, modality_features, modality_masks=None):
        attended = []
        for i, (attn, feat) in enumerate(zip(self.cross_attns, modality_features)):
            mask = modality_masks[i] if modality_masks is not None else None
            attn_out, _ = attn(query=query, key=feat, value=feat, key_padding_mask=mask)
            attended.append(attn_out)  # (B,1,d_model)

        query_global = query.squeeze(1)  # (B,d_model)
        gate_logits = self.gate_mlp(query_global)
        temperature = self.config.get('gate_temperature', 1.0) if hasattr(self, 'config') else 1.0
        gate_weights = F.softmax(gate_logits / temperature, dim=-1)  # (B,num_modalities)
        self.last_gate_weights = gate_weights.detach().cpu()

        stacked = torch.stack(attended, dim=1)  # (B,num_modalities,1,d_model)
        weights = gate_weights.unsqueeze(-1).unsqueeze(-1)  # (B,num_modalities,1,1)
        fused = (weights * stacked).sum(dim=1)  # (B,1,d_model)

        out = self.norm(query + self.dropout(fused))
        return out, gate_weights
