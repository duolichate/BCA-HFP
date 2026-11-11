# bca_hfp/models/main_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import CrossAttention
from .gated_fusion import GatedFusionLayer


class SafeBatchNorm1d(nn.BatchNorm1d):
    """BatchNorm1d that uses running stats when batch size is 1.

    BatchNorm1d in training mode requires more than one value per channel, so a
    trailing single-sample batch would raise ValueError. Falling back to running
    stats avoids drop_last. Identical to BatchNorm1d for batch > 1.
    """

    def forward(self, x):
        if self.training and x.size(0) == 1:
            return F.batch_norm(
                x, self.running_mean, self.running_var, self.weight, self.bias,
                training=False, momentum=0.0, eps=self.eps
            )
        return super().forward(x)


class Regressor(nn.Module):
    def __init__(self, feature_dim, num_heads, dropout, hidden_dims,
                 gene_input_dim, drug_global_dim, num_fusion_layers, config=None):
        super().__init__()

        self.config = config
        self.feature_dim = feature_dim
        self.num_fusion_layers = num_fusion_layers

        # Bidirectional cross-attention encoder
        _center_keys = bool((config or {}).get('model', {}).get('center_keys', False))
        # query centering applies only to the drug-as-query branch
        _center_query = bool((config or {}).get('model', {}).get('center_query', False))
        _num_drugs = int((config or {}).get('model', {}).get('num_drugs', 0) or 0)
        _use_drug_bias = bool((config or {}).get('model', {}).get('drug_bias', False)) and _num_drugs > 0
        self.drug_bias = nn.Embedding(_num_drugs, 1) if _use_drug_bias else None
        if self.drug_bias is not None:
            nn.init.zeros_(self.drug_bias.weight)
        _keep_pooled = _center_query or float(
            (config or {}).get('training', {}).get('query_contrast_weight', 0.0) or 0.0) > 0
        # drug global vector (Morgan + descriptors) predicts FiLM gamma/beta over the
        # atom sequence before attention; zero-init keeps training equal to the baseline
        _use_atom_cond = bool((config or {}).get('model', {}).get('atom_cond', False))
        self.atom_cond_proj = nn.Linear(drug_global_dim, 2 * feature_dim) if _use_atom_cond else None
        if self.atom_cond_proj is not None:
            nn.init.zeros_(self.atom_cond_proj.weight)
            nn.init.zeros_(self.atom_cond_proj.bias)
        self.atom_cond_pos = str((config or {}).get('model', {}).get('atom_cond_pos', 'post'))
        _use_key_cond = bool((config or {}).get('model', {}).get('key_cond', False))
        self.key_cond_proj = nn.Linear(drug_global_dim, feature_dim) if _use_key_cond else None
        if self.key_cond_proj is not None:
            nn.init.zeros_(self.key_cond_proj.weight)
            nn.init.zeros_(self.key_cond_proj.bias)
        g2d_pool_cfg = str((config or {}).get('model', {}).get('g2d_pool', 'mean') or 'mean')
        self.g2d_pool = g2d_pool_cfg
        self.g2d_pool_proj = None
        if g2d_pool_cfg == 'meanmax':
            self.g2d_pool_proj = nn.Linear(2 * feature_dim, feature_dim)
            with torch.no_grad():
                self.g2d_pool_proj.weight.zero_()
                self.g2d_pool_proj.weight[:, :feature_dim] = torch.eye(feature_dim)
                self.g2d_pool_proj.bias.zero_()
        self.cross_attn_drug2gene = CrossAttention(
            d_model=feature_dim, num_heads=num_heads, dropout=dropout,
            gene_input_dim=gene_input_dim, center_keys=_center_keys,
            center_query=_center_query, keep_pooled=_keep_pooled
        )
        self.cross_attn_gene2drug = CrossAttention(
            d_model=feature_dim, num_heads=num_heads, dropout=dropout,
            gene_input_dim=gene_input_dim, center_keys=_center_keys
        )

        # Global feature projection
        self.gene_global_proj = nn.Linear(gene_input_dim,
                                          feature_dim) if gene_input_dim != feature_dim else nn.Identity()
        self.drug_global_proj = nn.Linear(drug_global_dim,
                                          feature_dim) if drug_global_dim != feature_dim else nn.Identity()

        self.query_source = 'atom'          # 'atom'=pooled atom features, Eq.(6) | 'morgan'
                                            # | 'global_gene' | 'gene_pooled'
        self.add_atom_modality = False      # include pooled atom features as a fourth gated modality
        # with query=morgan the Morgan vector is both query and drug_global modality;
        # this switch drops the duplicate modality
        self.drop_drug_global_modality = False
        if config is not None:
            _mcfg = config.get('model', {})
            if _mcfg.get('decouple_atom', False):   # same as morgan query + atom modality
                self.query_source = 'morgan'
                self.add_atom_modality = True
            if _mcfg.get('query_source') is not None:
                self.query_source = _mcfg['query_source']
            if _mcfg.get('add_atom_modality') is not None:
                self.add_atom_modality = bool(_mcfg['add_atom_modality'])
            if _mcfg.get('drop_drug_global_modality') is not None:
                self.drop_drug_global_modality = bool(_mcfg['drop_drug_global_modality'])
        self.drop_global_gene_modality = (self.query_source == 'global_gene')
        self.drop_gene_pooled_modality = (self.query_source == 'gene_pooled')
        self.decouple_atom = (self.query_source == 'morgan' and self.add_atom_modality)
        n_modalities = ((0 if self.drop_gene_pooled_modality else 1)
                        + (0 if self.drop_global_gene_modality else 1)
                        + (0 if self.drop_drug_global_modality else 1)
                        + (1 if self.add_atom_modality else 0))

        self.use_residual = not bool(config is not None and
                                     config.get('model', {}).get('no_residual', False))

        _gate_content = bool((config or {}).get('model', {}).get('gate_content', False))
        _gate_hidden = (config or {}).get('model', {}).get('gate_hidden', None)
        # Multi-layer gated fusion decoder
        self.fusion_layers = nn.ModuleList([
            GatedFusionLayer(feature_dim, num_heads, dropout, num_modalities=n_modalities,
                             temperature=config['training']['gate_temperature'],
                             use_residual=self.use_residual, gate_content=_gate_content,
                             gate_hidden=_gate_hidden)
            for _ in range(num_fusion_layers)
        ])

        # Regression head
        self.regression_head = self._build_regression_head(feature_dim, hidden_dims, dropout)

        self.prior_dim = int((config or {}).get('model', {}).get('prior_dim', 0) or 0)
        if self.prior_dim > 0:
            self.prior_proj = nn.Linear(self.prior_dim, feature_dim)
            nn.init.zeros_(self.prior_proj.weight)
            nn.init.zeros_(self.prior_proj.bias)
        else:
            self.prior_proj = None

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

        norm_type = 'layernorm'
        if self.config is not None:
            norm_type = self.config.get('model', {}).get('reg_norm', 'layernorm')

        for hdim in hidden_dims:
            if norm_type == 'batchnorm':
                norm_layer = SafeBatchNorm1d(hdim)
            else:
                norm_layer = nn.LayerNorm(hdim)
            layers.extend([
                nn.Linear(curr_dim, hdim),
                norm_layer,
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            curr_dim = hdim
        layers.append(nn.Linear(curr_dim, 1))
        return nn.Sequential(*layers)

    def forward(self, gene_features, drug_features, drug_mask=None, gene_mask=None,
                drug_global=None, return_attn=False, prior=None, drop_modalities=None,
                drug_idx=None):

        # Drug -> Gene
        _atom_cond = None
        if self.atom_cond_proj is not None and drug_global is not None:
            _atom_cond = self.atom_cond_proj(drug_global)
        _kcond = None
        if self.key_cond_proj is not None and drug_global is not None:
            _kcond = self.key_cond_proj(drug_global)
        drug_attended = self.cross_attn_drug2gene(drug_features, gene_features, drug_mask, gene_mask,
                                                  cond=_atom_cond, cond_pos=self.atom_cond_pos,
                                                  key_cond=_kcond)
        drug2gene_attn = self.cross_attn_drug2gene.last_attention_weights_per_head  # (B, 8, atoms, genes)
        # for contrastive loss; None when disabled
        self.last_query_pooled = self.cross_attn_drug2gene.last_query_pooled

        if drug_mask is not None:
            mask_exp = drug_mask.unsqueeze(-1).float()
            drug_pooled = (drug_attended * mask_exp).sum(dim=1) / (mask_exp.sum(dim=1) + 1e-9)
        else:
            drug_pooled = drug_attended.mean(dim=1)

        # Gene -> Drug
        gene_attended = self.cross_attn_gene2drug(gene_features, drug_features, gene_mask, drug_mask)
        gene2drug_attn = self.cross_attn_gene2drug.last_attention_weights_per_head  # (B, 8, genes, atoms)

        if self.g2d_pool == 'meanmax':
            if gene_mask is not None:
                me = gene_mask.unsqueeze(-1).float()
                g_mean = (gene_attended * me).sum(dim=1) / (me.sum(dim=1) + 1e-9)
                g_max = gene_attended.masked_fill(~gene_mask.unsqueeze(-1), -1e9).max(dim=1).values
            else:
                g_mean = gene_attended.mean(dim=1)
                g_max = gene_attended.max(dim=1).values
            gene_pooled = self.g2d_pool_proj(torch.cat([g_mean, g_max], dim=-1))
        elif self.g2d_pool == 'd2g':
            # weight genes by drug->gene attention summed over heads and atoms
            w = drug2gene_attn.mean(dim=1).sum(dim=1)           # (B, A, G) -> sum over atoms -> (B, G)
            if gene_mask is not None:
                w = w.masked_fill(~gene_mask, 0.0)
            w = w / (w.sum(dim=-1, keepdim=True) + 1e-9)
            gene_pooled = (gene_attended * w.unsqueeze(-1)).sum(dim=1)
        elif gene_mask is not None:
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
        # the modality matching the query source is excluded so one tensor is not both
        B = gene_pooled_norm.shape[0]
        device = gene_pooled_norm.device
        ones = torch.ones(B, dtype=torch.bool, device=device)

        modality_features, present, modality_names = [], [], []
        if not self.drop_gene_pooled_modality:
            modality_features.append(gene_pooled_norm.unsqueeze(1))
            present.append(ones)
            modality_names.append('gene_pooled')
        if not self.drop_global_gene_modality:
            modality_features.append(global_gene_norm.unsqueeze(1))
            present.append(ones)
            modality_names.append('global_gene')
        if drug_global_norm is not None and not self.drop_drug_global_modality:
            modality_features.append(drug_global_norm.unsqueeze(1))
            present.append(ones)
            modality_names.append('drug_global')

        if self.training and self.config is not None:
            gg_p = self.config['training'].get('global_gene_dropout', 0.0) or 0.0
            dg_p = self.config['training'].get('drug_global_dropout', 0.0) or 0.0
            for idx, name in enumerate(modality_names):
                if name == 'global_gene' and gg_p > 0:
                    present[idx] = torch.rand(B, device=device) >= gg_p      # per sample
                elif name == 'drug_global' and dg_p > 0:
                    present[idx] = torch.rand(B, device=device) >= dg_p

        # pooled atom features as an extra gated modality
        if self.add_atom_modality and drug_global_norm is not None:
            drug_atom_norm = F.layer_norm(drug_pooled, norm_shape)
            modality_features.append(drug_atom_norm.unsqueeze(1))
            modality_names.append('drug_atom')
            atom_present = torch.ones(B, dtype=torch.bool, device=device)
            if self.training and self.config is not None:
                dg_p = self.config['training'].get('drug_global_dropout', 0.0) or 0.0
                if dg_p > 0:
                    atom_present = torch.rand(B, device=device) >= dg_p
            present.append(atom_present)

        if drop_modalities:
            for _i, _name in enumerate(modality_names):
                if _name in drop_modalities:
                    present[_i] = torch.zeros(B, dtype=torch.bool, device=device)
        self._last_modality_names = list(modality_names)

        modality_present = torch.stack(present, dim=1)  # (B, num_modalities)

        # fusion query source: 'atom' (Eq.(6)) | 'morgan' | 'global_gene' | 'gene_pooled'
        if self.query_source == 'morgan' and drug_global_norm is not None:
            query = drug_global_norm.unsqueeze(1)   # (B,1,d_model)
        elif self.query_source == 'global_gene':
            query = global_gene_norm.unsqueeze(1)
        elif self.query_source == 'gene_pooled':
            query = gene_pooled_norm.unsqueeze(1)
        elif self.query_source == 'atommean':
            if drug_mask is not None:
                _m = drug_mask.unsqueeze(-1).float()
                atom_mean = (drug_features * _m).sum(dim=1) / (_m.sum(dim=1) + 1e-9)
            else:
                atom_mean = drug_features.mean(dim=1)
            query = F.layer_norm(atom_mean, (self.feature_dim,)).unsqueeze(1)
        else:
            query = drug_pooled.unsqueeze(1)        # (B,1,d_model)

        # Multi-layer gated fusion
        for layer in self.fusion_layers:
            query, _ = layer(query, modality_features, modality_present=modality_present)

        fused = query.squeeze(1)  # (B,d_model)

        # interaction prior: zero-init projection adds a learnable bias direction
        if prior is not None and self.prior_proj is not None:
            fused = fused + self.prior_proj(prior)

        out = self.regression_head(fused).squeeze(-1)

        if self.drug_bias is not None and drug_idx is not None:
            out = out + self.drug_bias(drug_idx).squeeze(-1)

        # Apply sigmoid if label is AUC (0~1)
        if self.config and self.config['data']['label_col'] == 'AUC':
            out = torch.sigmoid(out)

        if return_attn:
            return out, drug2gene_attn, gene2drug_attn

        return out
