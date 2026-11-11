# bca_hfp/models/gated_fusion.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedFusionLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout, temperature, num_modalities=3,
                 use_residual=True, gate_content=False, gate_hidden=None):
        super().__init__()
        self.d_model = d_model
        self.num_modalities = num_modalities
        self.gate_content = gate_content
        self.temperature = temperature
        self.use_residual = use_residual
        # learnable temperature, log-parameterized to stay positive; initialized from config
        self.log_temperature = nn.Parameter(torch.tensor(math.log(float(temperature))))

        self.cross_attns = nn.ModuleList([
            nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
            for _ in range(num_modalities)
        ])

        _gate_in = d_model * (1 + num_modalities) if gate_content else d_model
        self.gate_hidden = int(gate_hidden) if gate_hidden else d_model // 2
        self.gate_mlp = nn.Sequential(
            nn.Linear(_gate_in, self.gate_hidden),
            nn.ReLU(),
            nn.Linear(self.gate_hidden, num_modalities)
        )
        if gate_content:
            # content columns zero-init; query columns and bias re-init with fan_in=d_model
            with torch.no_grad():
                _w = self.gate_mlp[0].weight
                nn.init.kaiming_uniform_(_w[:, :d_model], a=math.sqrt(5))
                _w[:, d_model:].zero_()
                _bound = 1.0 / math.sqrt(d_model)
                nn.init.uniform_(self.gate_mlp[0].bias, -_bound, _bound)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.last_gate_weights = None
        self.last_attended_features = None
        self.last_gate_weights_grad = None
        self.gate_override = None

    def forward(self, query, modality_features, modality_masks=None, modality_present=None):
        attended = []
        for i, (attn, feat) in enumerate(zip(self.cross_attns, modality_features)):
            mask = modality_masks[i] if modality_masks is not None else None
            attn_out, _ = attn(query=query, key=feat, value=feat, key_padding_mask=mask)
            attended.append(attn_out)  # (B,1,d_model)

        query_global = query.squeeze(1)  # (B,d_model)
        if self.gate_content:
            # gate input = query concatenated with each modality's attention output
            gate_ctx = torch.cat([query_global] + [a.squeeze(1) for a in attended], dim=-1)
        else:
            gate_ctx = query_global
        # softmax over present modalities only (2 when drug_global is missing)
        gate_logits = self.gate_mlp(gate_ctx)[:, :len(attended)]
        if modality_present is not None:
            gate_logits = gate_logits.masked_fill(~modality_present[:, :len(attended)], -1e9)
        # exp keeps temperature > 0; clamp avoids hard selection or fully uniform gating
        temperature = self.log_temperature.exp().clamp(min=0.05, max=20.0)
        gate_weights = F.softmax(gate_logits / temperature, dim=-1)  # (B,num_modalities)
        if self.gate_override is not None:
            if isinstance(self.gate_override, str) and self.gate_override == 'uniform':
                gate_weights = torch.full_like(gate_weights, 1.0 / max(len(attended), 1))
            else:
                ov = torch.as_tensor(self.gate_override, dtype=gate_weights.dtype,
                                     device=gate_weights.device).view(1, -1)
                gate_weights = ov.expand_as(gate_weights).contiguous()
        self.last_gate_weights = gate_weights.detach().cpu()
        # graph-attached copy for gate entropy regularization (--gate_entropy)
        self.last_gate_weights_grad = gate_weights

        stacked = torch.stack(attended, dim=1)  # (B,num_modalities,1,d_model)
        weights = gate_weights.unsqueeze(-1).unsqueeze(-1)  # (B,num_modalities,1,1)
        fused = (weights * stacked).sum(dim=1)  # (B,1,d_model)

        if self.use_residual:
            out = self.norm(query + self.dropout(fused))
        else:
            out = self.norm(self.dropout(fused))
        return out, gate_weights
