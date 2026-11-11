# bca_hfp/training/trainer.py
import torch
import torch.nn.functional as F
import os
import sys
import math
import numpy as np
import time
import pandas as pd

if sys.stdout.isatty():
    from tqdm import tqdm
else:
    tqdm = None


class BaseTrainer:
    def __init__(self, model, config, device):
        self.model = model
        self.config = config
        self.device = device
        # loss: smoothl1 (default, robust to outliers) | mse
        _loss_name = config['training'].get('loss', 'smoothl1')
        if _loss_name == 'mse':
            self.criterion = torch.nn.MSELoss()
        else:
            self.criterion = torch.nn.SmoothL1Loss(beta=1.0)
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['training']['lr'],
            weight_decay=config['training']['weight_decay']
        )
        self.scaler = None
        self.best_metrics = {'r2': -float('inf')}
        self.patience_counter = 0
        # drug mean auxiliary loss weight (0 = off); needs DrugGroupedSampler to group drugs per batch
        self.drug_mean_weight = float(config['training'].get('drug_mean_weight', 0.0) or 0.0)
        # within-drug loss weight: subtract each drug's batch mean, then MSE
        self.within_weight = float(config['training'].get('within_weight', 0.0) or 0.0)
        # interaction prior: prior_dim > 0 appends a (B, prior_dim) tensor to each batch
        self.use_prior = int((config.get('model', {}) or {}).get('prior_dim', 0) or 0) > 0
        # best_balance selection rule, see _balance_improved
        self.balance_rule = config['training'].get('balance_rule', 'gated')
        # --- censoring, all controlled by config['training'], off by default ---
        # trailing cens tensor: (B, 4) = [b_lo, b_hi, cens_flag, w]
        self.use_cens = bool(config['training'].get('use_cens', False))
        # tobit_sigma > 0 replaces MSE / SmoothL1 with the Tobit censored likelihood
        self.tobit_sigma = float(config['training'].get('tobit_sigma', 0.0) or 0.0)
        # reweight the regression loss by 1/(1+(RMSE/0.1)^2), normalized to mean 1
        self.rmse_weight = bool(config['training'].get('rmse_weight', False))
        # clamp predictions to [b_lo-2, b_hi+2] then [ln0.001, ln100] at inference
        self.pred_clamp = bool(config['training'].get('pred_clamp', False))
        # training-time soft penalty on predictions outside the per-drug window +/-2 (0 = off)
        self.win_penalty = float(config['training'].get('win_penalty', 0.0) or 0.0)
        self.last_nll = 0.0
        self.last_aux_loss = 0.0
        self.last_within_loss = 0.0
        self.last_query_contrast = 0.0
        self.gate_entropy = float(config['training'].get('gate_entropy', 0.0) or 0.0)
        self.last_gate_entropy = 0.0
        # query contrastive loss: pull same-drug queries together, push different drugs apart (in-batch InfoNCE)
        self.query_contrast_weight = float(
            config['training'].get('query_contrast_weight', 0.0) or 0.0)
        self.query_contrast_tau = float(config['training'].get('query_contrast_tau', 0.1) or 0.1)
        self.drug_bias_l2 = float(config['training'].get('drug_bias_l2', 0.0) or 0.0)
        self.last_preds = None
        self.last_targets = None
        self.best_balance_epoch = -1
        self.best_balance_score = float('inf')

    def _drug_mean_loss(self, pred, target, drug_idx):
        """Per-drug L1 penalty on (predicted mean - true mean) within a batch.

        Only drugs appearing >=2 times contribute. Returns 0 when none qualify.
        """
        if drug_idx is None or pred.numel() == 0:
            return pred.new_zeros(())
        uniq, inv = torch.unique(drug_idx, return_inverse=True)
        n = uniq.numel()
        cnt = torch.zeros(n, device=pred.device, dtype=pred.dtype).index_add(
            0, inv, torch.ones_like(pred))
        p_sum = torch.zeros(n, device=pred.device, dtype=pred.dtype).index_add(0, inv, pred)
        y_sum = torch.zeros(n, device=pred.device, dtype=pred.dtype).index_add(
            0, inv, target.detach())
        valid = cnt >= 2
        if not bool(valid.any()):
            return pred.new_zeros(())
        return (((p_sum - y_sum).abs() / cnt)[valid]).mean()

    def _within_drug_loss(self, pred, target, drug_idx):
        """Within-drug MSE after subtracting each drug's batch mean.

        Total MSE is dominated by between-drug potency differences; centering leaves
        only cell-line variation within a drug. Only drugs appearing >=2 times contribute.
        """
        if drug_idx is None or pred.numel() == 0:
            return pred.new_zeros(())
        uniq, inv = torch.unique(drug_idx, return_inverse=True)
        n = uniq.numel()
        cnt = torch.zeros(n, device=pred.device, dtype=pred.dtype).index_add(
            0, inv, torch.ones_like(pred))
        valid = cnt >= 2
        if not bool(valid.any()):
            return pred.new_zeros(())
        denom = cnt.clamp(min=1)
        p_mean = torch.zeros(n, device=pred.device, dtype=pred.dtype).index_add(0, inv, pred) / denom
        y_mean = torch.zeros(n, device=pred.device, dtype=pred.dtype).index_add(
            0, inv, target.detach()) / denom
        p_c = pred - p_mean[inv]
        y_c = target - y_mean[inv]
        mask = valid[inv]
        return (((p_c - y_c) ** 2) * mask).sum() / mask.sum().clamp(min=1)

    def _query_contrast_loss(self, q, drug_idx, tau):
        """In-batch InfoNCE over drug queries; same-drug samples are positives.

        Without this every drug query collapses onto one common direction; the loss
        pushes distinct drugs apart along more directions.
        """
        if q is None or drug_idx is None or q.shape[0] < 4:
            return torch.zeros((), device=self.device)
        z = F.normalize(q, dim=-1)
        sim = (z @ z.t()) / tau
        n = z.shape[0]
        eye = torch.eye(n, dtype=torch.bool, device=z.device)
        sim = sim.masked_fill(eye, -1e4)
        same = (drug_idx[:, None] == drug_idx[None, :]) & (~eye)
        pos_cnt = same.sum(dim=1)
        if not bool((pos_cnt > 0).any()):
            return torch.zeros((), device=self.device)
        log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
        per_sample = -log_prob.masked_fill(~same, 0.0).sum(dim=1) / pos_cnt.clamp(min=1)
        return per_sample[pos_cnt > 0].mean()

    def _tobit_nll(self, pred, target, cens):
        """Tobit censored likelihood merging right-censored, left-censored and in-window samples.

        cens columns: [b_lo, b_hi, flag, w]
          flag=+1 right-censored (true IC50 >= b_hi): P = Phi((mu - b_hi)/sigma)
          flag=-1 left-censored  (true IC50 <= b_lo): P = Phi((b_lo - mu)/sigma)
          flag=0  in-window observation: density N(y; mu, sigma)
        Returns -mean(w * log P). With sigma=1 the in-window term is 0.5*(y-mu)^2.
        """
        b_lo = cens[:, 0]
        b_hi = cens[:, 1]
        flag = cens[:, 2]
        w = cens[:, 3] if self.rmse_weight else torch.ones_like(b_lo)
        sigma = max(self.tobit_sigma, 1e-3)
        z = (target - pred) / sigma
        ll_obs = -0.5 * z * z - math.log(sigma)
        ll_right = torch.special.log_ndtr((pred - b_hi) / sigma)
        ll_left = torch.special.log_ndtr((b_lo - pred) / sigma)
        ll = torch.where(flag > 0, ll_right, torch.where(flag < 0, ll_left, ll_obs))
        return -(w * ll).mean()

    def _regression_loss(self, pred, target, cens):
        """Regression loss entry point: Tobit > weighted MSE > default criterion."""
        if cens is not None and self.tobit_sigma > 0:
            return self._tobit_nll(pred, target, cens)
        if cens is not None and self.rmse_weight:
            return (cens[:, 3] * (pred - target) ** 2).mean()
        return self.criterion(pred, target)

    def _clamp_pred(self, pred, cens):
        """Inference post-processing: clamp to the per-row concentration window +/-2,
        then to [ln0.001, ln100]."""
        if not self.pred_clamp or cens is None:
            return pred
        out = torch.max(torch.min(pred, cens[:, 1] + 2.0), cens[:, 0] - 2.0)
        return torch.clamp(out, min=math.log(0.001), max=math.log(100.0))

    def _balance_improved(self, train_r2, val_r2, val_loss):
        """best_balance checkpoint selection.

        - gap      : minimize |train_r2 - val_r2|
        - gated    : with val_r2 > 0, minimize signed gap (train_r2 - val_r2);
                     no candidate while val_r2 <= 0, caller falls back to final
        - val_loss : minimize val loss
        Returns (score, improved).
        """
        if self.balance_rule == 'val_loss':
            score = float(val_loss)
            return score, score < self.best_balance_score
        gap = float(train_r2) - float(val_r2)
        if self.balance_rule == 'gap':
            score = abs(gap)
            return score, score < self.best_balance_score
        # gated
        if val_r2 <= 0:
            return float('inf'), False
        return gap, gap < self.best_balance_score

    def _compute_metrics(self, y_true, y_pred):
        from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
        from scipy.stats import pearsonr
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        pearson = pearsonr(y_true, y_pred)[0] if len(y_true) > 1 else 0.0
        return {'r2': r2, 'mae': mae, 'rmse': rmse, 'pearson': pearson}

    def _learned_temperatures(self):
        """Current effective temperature of each GatedFusionLayer."""
        temps = []
        for m in self.model.modules():
            if hasattr(m, 'log_temperature'):
                temps.append(float(m.log_temperature.exp().clamp(min=0.05, max=20.0).item()))
        return temps

    def save_checkpoint(self, save_dir, name, fold_idx, epoch, suffix=''):
        os.makedirs(save_dir, exist_ok=True)
        if fold_idx is None:
            model_path = os.path.join(save_dir, f"{name}{suffix}.pth")
        else:
            model_path = os.path.join(save_dir, f"{name}_fold{fold_idx + 1}{suffix}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': self.best_metrics,
            'config': self.config
        }, model_path)
        if hasattr(self.model, 'cross_attn_drug2gene'):
            attn_dir = self.config['output']['attention_csv_dir']
            self.model.cross_attn_drug2gene.save_accumulated_weights(fold_idx, attn_dir, suffix=suffix)
        return model_path


class VariableLengthTrainer(BaseTrainer):
    def __init__(self, model, config, device):
        super().__init__(model, config, device)
        self.save_attention_epochs = list(range(10, config['training']['epochs'] + 1, 10))
        if config['training']['epochs'] not in self.save_attention_epochs:
            self.save_attention_epochs.append(config['training']['epochs'])

    def train_epoch(self, loader, epoch, total_epochs):
        if hasattr(self.model, 'set_epoch'):
            self.model.set_epoch(epoch)
        self.model.train()
        total_loss = 0
        total_aux = 0
        total_within = 0
        total_qc = 0
        preds, targets = [], []
        accumulation_steps = self.config['training']['gradient_accumulation_steps']
        self.optimizer.zero_grad()
        batch_idx = -1

        desc = f"Epoch {epoch}/{total_epochs} Training"
        if tqdm is not None:
            pbar = tqdm(loader, desc=desc, unit="batch", leave=False, mininterval=5)
        else:
            pbar = loader

        for batch_idx, batch in enumerate(pbar):
            cens = None
            if self.use_cens:
                # cens is appended last by collate; strip it so the length checks below hold
                batch = list(batch)
                cens = batch.pop().to(self.device)
            prior = None
            if self.use_prior and len(batch) >= 7:
                if len(batch) == 8:
                    (atom_feat, atom_mask, gene_feat, gene_mask, labels,
                     drug_global, drug_idx, prior) = batch
                    drug_idx = drug_idx.to(self.device)
                else:
                    (atom_feat, atom_mask, gene_feat, gene_mask, labels,
                     drug_global, prior) = batch
                    drug_idx = None
                prior = prior.to(self.device)
            elif len(batch) == 7:
                atom_feat, atom_mask, gene_feat, gene_mask, labels, drug_global, drug_idx = batch
                drug_idx = drug_idx.to(self.device)
            else:
                atom_feat, atom_mask, gene_feat, gene_mask, labels, drug_global = batch
                drug_idx = None
            atom_feat = atom_feat.to(self.device)
            atom_mask = atom_mask.to(self.device)
            gene_feat = gene_feat.to(self.device)
            gene_mask = gene_mask.to(self.device)
            labels = labels.to(self.device)
            if drug_global is not None:
                drug_global = drug_global.to(self.device)

            _has_bias = getattr(self.model, 'drug_bias', None) is not None
            outputs = self.model(gene_feat, atom_feat, atom_mask, gene_mask,
                                 drug_global=drug_global, prior=prior,
                                 drug_idx=(drug_idx if _has_bias else None))
            loss = self._regression_loss(outputs, labels, cens)
            if self.win_penalty > 0 and cens is not None:
                # penalty only on the part outside window +/-2; differentiable, unlike a hard clamp
                _lo = cens[:, 0] - 2.0
                _hi = cens[:, 1] + 2.0
                _pen = (torch.relu(outputs - _hi) + torch.relu(_lo - outputs)).mean()
                loss = loss + self.win_penalty * _pen
            if self.drug_bias_l2 > 0 and _has_bias:
                # L2 on per-drug bias: shrinkage of the drug-level random effect
                loss = loss + self.drug_bias_l2 * self.model.drug_bias.weight.pow(2).mean()
            if self.drug_mean_weight > 0:
                # drug mean auxiliary loss: counter systematic per-drug bias
                aux = self._drug_mean_loss(outputs, labels, drug_idx)
                loss = loss + self.drug_mean_weight * aux
                total_aux += float(aux.detach()) * atom_feat.size(0)
            if self.within_weight > 0:
                # within-drug loss: MSE after removing each drug's mean
                wl = self._within_drug_loss(outputs, labels, drug_idx)
                loss = loss + self.within_weight * wl
                total_within += float(wl.detach()) * atom_feat.size(0)
            if self.query_contrast_weight > 0:
                # query contrastive loss, see _query_contrast_loss
                ql = self._query_contrast_loss(
                    getattr(self.model, 'last_query_pooled', None),
                    drug_idx, self.query_contrast_tau)
                loss = loss + self.query_contrast_weight * ql
                total_qc += float(ql.detach()) * atom_feat.size(0)
            if self.gate_entropy > 0:
                _gw = [getattr(_l, 'last_gate_weights_grad', None)
                       for _l in getattr(self.model, 'fusion_layers', [])]
                _gw = [g for g in _gw if g is not None]
                if _gw:
                    _p = torch.cat([g.clamp_min(1e-9) for g in _gw], dim=0)
                    _ent = -(_p * _p.log()).sum(dim=-1).mean() / math.log(_p.size(-1))
                    loss = loss + self.gate_entropy * (-_ent)
                    self.last_gate_entropy = float(_ent.detach())
            loss = loss / accumulation_steps

            loss.backward()

            total_loss += loss.item() * atom_feat.size(0) * accumulation_steps
            preds.extend(outputs.detach().cpu().float().numpy())
            targets.extend(labels.detach().cpu().float().numpy())

            if (batch_idx + 1) % accumulation_steps == 0:
                if self.config['training']['gradient_clip'] > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['training']['gradient_clip'])
                self.optimizer.step()
                self.optimizer.zero_grad()

            if tqdm is not None:
                pbar.set_postfix(loss=loss.item() * accumulation_steps, refresh=False)

        # Handle remaining gradients
        if (batch_idx + 1) % accumulation_steps != 0:
            if self.config['training']['gradient_clip'] > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['training']['gradient_clip'])
            self.optimizer.step()
            self.optimizer.zero_grad()

        avg_loss = total_loss / len(loader.dataset)
        self.last_aux_loss = total_aux / len(loader.dataset)
        self.last_within_loss = total_within / len(loader.dataset)
        self.last_query_contrast = total_qc / len(loader.dataset)
        metrics = self._compute_metrics(np.array(targets), np.array(preds))
        return avg_loss, metrics

    def validate(self, loader, fold_idx=None, epoch=None):
        if epoch is not None and hasattr(self.model, 'set_epoch'):
            self.model.set_epoch(epoch)
        self.model.eval()
        total_loss = 0
        preds, targets = [], []
        saved = False

        desc = f"Epoch {epoch} Validation" if epoch is not None else "Validation"
        if tqdm is not None:
            pbar = tqdm(loader, desc=desc, unit="batch", leave=False, mininterval=5)
        else:
            pbar = loader

        with torch.no_grad():
            for batch_idx, batch in enumerate(pbar):
                cens = None
                if self.use_cens:
                    batch = list(batch)
                    cens = batch.pop().to(self.device)
                prior = None
                if self.use_prior and len(batch) >= 7:
                    if len(batch) == 8:
                        (atom_feat, atom_mask, gene_feat, gene_mask, labels,
                         drug_global, _, prior) = batch
                    else:
                        (atom_feat, atom_mask, gene_feat, gene_mask, labels,
                         drug_global, prior) = batch
                    prior = prior.to(self.device)
                elif len(batch) == 7:
                    atom_feat, atom_mask, gene_feat, gene_mask, labels, drug_global, _ = batch
                else:
                    atom_feat, atom_mask, gene_feat, gene_mask, labels, drug_global = batch
                atom_feat = atom_feat.to(self.device)
                atom_mask = atom_mask.to(self.device)
                gene_feat = gene_feat.to(self.device)
                gene_mask = gene_mask.to(self.device)
                labels = labels.to(self.device)
                if drug_global is not None:
                    drug_global = drug_global.to(self.device)

                outputs = self.model(gene_feat, atom_feat, atom_mask, gene_mask,
                                     drug_global=drug_global, prior=prior)
                loss = self._regression_loss(outputs, labels, cens)
                # metrics and saved output use post-processed predictions; _clamp_pred is a no-op when off
                pred_out = self._clamp_pred(outputs, cens)

                total_loss += loss.item() * atom_feat.size(0)
                preds.extend(pred_out.cpu().float().numpy())
                targets.extend(labels.cpu().float().numpy())

                if not saved and epoch is not None and epoch in self.save_attention_epochs:
                    if hasattr(self.model, 'cross_attn_drug2gene'):
                        self.model.cross_attn_drug2gene.accumulate_attention_weights(epoch, sample_idx=0)
                        saved = True
                if tqdm is not None:
                    pbar.set_postfix(loss=loss.item(), refresh=False)

        avg_loss = total_loss / len(loader.dataset)
        self.last_preds = np.asarray(preds, dtype=np.float32)
        self.last_targets = np.asarray(targets, dtype=np.float32)
        metrics = self._compute_metrics(np.array(targets), np.array(preds))
        return avg_loss, metrics

    def _apply_fixed_lr(self, epoch, num_epochs):
        """Fixed epoch milestones, independent of val (avoids implicit leakage from ReduceLROnPlateau).

        Halves at 50% and 75%: num_epochs=20 -> e1..10 base, e11..15 base*0.5, e16..20 base*0.25.
        """
        base = float(self.config['training'].get('lr', 2e-5))
        m1 = max(1, int(round(num_epochs * 0.5)))
        m2 = max(m1 + 1, int(round(num_epochs * 0.75)))
        if epoch <= m1:
            target = base
        elif epoch <= m2:
            target = base * 0.5
        else:
            target = base * 0.25
        old = self.optimizer.param_groups[0]['lr']
        if abs(target - old) > 1e-12:
            for g in self.optimizer.param_groups:
                g['lr'] = target
            print(f"  LR schedule(fixed): epoch {epoch} -> {target:.2e}")

    def train(self, train_loader, val_loader, num_epochs, save_dir, fold_idx, name,
              test_loader=None, eval_test_each_epoch=False):
        # eval_test_each_epoch: extra test pass per epoch, logged to history only;
        # never used for best_val / best_balance / early_stop selection
        os.makedirs(save_dir, exist_ok=True)
        history = []
        scheduler_type = self.config['training'].get('scheduler', 'cosine')
        # 'fixed' = epoch milestones, no val read
        self._fixed_lr_schedule = (scheduler_type == 'fixed')
        if scheduler_type == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=num_epochs, eta_min=1e-6
            )
        elif self._fixed_lr_schedule:
            self.scheduler = None
        else:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='max', factor=self.config['training'].get('reduce_factor', 0.5),
                patience=self.config['training'].get('reduce_patience', 5)
            )

        # SWA: average weights of the last K epochs; only float parameters are
        # accumulated, integer buffers such as num_batches_tracked keep the last value
        swa_last_k = int(self.config['training'].get('swa_last_k', 0) or 0)
        swa_start = (num_epochs - swa_last_k + 1) if swa_last_k > 0 else None
        swa_accum, swa_count = None, 0
        self.swa_model_path = None

        n_total = sum(p.numel() for p in self.model.parameters())
        n_train = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"  model params : total={n_total:,}  trainable={n_train:,}  "
              f"init_temperature={[f'{t:.4f}' for t in self._learned_temperatures()]}")

        best_val_r2 = -float('inf')
        best_val_epoch = -1
        best_balance_gap = float('inf')
        best_val_model_path = None
        best_balance_model_path = None

        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()
            if self._fixed_lr_schedule:
                self._apply_fixed_lr(epoch, num_epochs)
            train_loss, train_metrics = self.train_epoch(train_loader, epoch, num_epochs)
            val_loss, val_metrics = self.validate(val_loader, fold_idx, epoch)

            test_metrics = None
            if eval_test_each_epoch and test_loader is not None:
                _, test_metrics = self.validate(test_loader, fold_idx, None)

            if self.scheduler is not None:
                old_lr = self.optimizer.param_groups[0]['lr']
                if scheduler_type == 'cosine':
                    self.scheduler.step()
                else:
                    self.scheduler.step(val_metrics['r2'])
                new_lr = self.optimizer.param_groups[0]['lr']
                if new_lr != old_lr:
                    print(f"  Learning rate changed from {old_lr:.2e} to {new_lr:.2e}")

            epoch_time = time.time() - epoch_start

            if val_metrics['r2'] > best_val_r2:
                best_val_r2 = val_metrics['r2']
                best_val_epoch = epoch
                model_path = self.save_checkpoint(save_dir, name, fold_idx, epoch, suffix='_best_val')
                best_val_model_path = model_path
                print(f"  New best val model saved at epoch {epoch} (val R²={val_metrics['r2']:.4f})")

            balance_score, balance_improved = self._balance_improved(
                train_metrics['r2'], val_metrics['r2'], val_loss)
            if balance_improved:
                best_balance_gap = balance_score
                self.best_balance_score = balance_score
                self.best_balance_epoch = epoch
                model_path = self.save_checkpoint(save_dir, name, fold_idx, epoch, suffix='_best_balance')
                best_balance_model_path = model_path
                print(f"  New best balance model saved at epoch {epoch} "
                      f"(rule={self.balance_rule}, score={balance_score:.4f})")

            patience_limit = self.config['training'].get('early_stop_patience', 10) or 0
            if val_metrics['r2'] > self.best_metrics['r2']:
                self.best_metrics = val_metrics
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                # patience_limit <= 0 disables early stopping
                if patience_limit > 0 and self.patience_counter >= patience_limit:
                    print(f"Early stopping at epoch {epoch}")
                    break

            # SWA: accumulate the last K epochs
            if swa_start is not None and epoch >= swa_start:
                sd = self.model.state_dict()
                if swa_accum is None:
                    swa_accum = {k: v.detach().clone() for k, v in sd.items()}
                else:
                    for k, v in sd.items():
                        if swa_accum[k].is_floating_point():
                            swa_accum[k] += v.detach()
                swa_count += 1

            fold_display = fold_idx + 1 if fold_idx is not None else 0
            entry = {
                'fold': fold_display,
                'epoch': epoch,
                'lr': self.optimizer.param_groups[0]['lr'],
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_r2': train_metrics['r2'],
                'val_r2': val_metrics['r2'],
                'train_pearson': train_metrics['pearson'],
                'val_pearson': val_metrics['pearson'],
                'train_mae': train_metrics['mae'],
                'val_mae': val_metrics['mae'],
                'train_rmse': train_metrics['rmse'],
                'val_rmse': val_metrics['rmse'],
                'learned_temperature': (self._learned_temperatures() or [float('nan')])[0],
            }
            if eval_test_each_epoch:
                entry['test_r2'] = test_metrics['r2'] if test_metrics is not None else float('nan')
                entry['test_pearson'] = test_metrics['pearson'] if test_metrics is not None else float('nan')
                entry['test_mae'] = test_metrics['mae'] if test_metrics is not None else float('nan')
                entry['test_rmse'] = test_metrics['rmse'] if test_metrics is not None else float('nan')
            if self.drug_mean_weight > 0:
                entry['train_aux_loss'] = self.last_aux_loss
            if self.within_weight > 0:
                entry['train_within_loss'] = self.last_within_loss
            history.append(entry)
            temps = self._learned_temperatures()
            temp_str = f" | T {temps[0]:.3f}" if temps else ""
            aux_str = f" | Aux {self.last_aux_loss:.4f}" if self.drug_mean_weight > 0 else ""
            win_str = f" | Win {self.last_within_loss:.4f}" if self.within_weight > 0 else ""
            test_str = f" | TeR2 {test_metrics['r2']:+.4f}" if test_metrics is not None else ""
            print(f"Fold {fold_display} | Epoch {epoch:3d} | {epoch_time:6.0f}s | "
                  f"LR {self.optimizer.param_groups[0]['lr']:.2e} | "
                  f"TrLoss {train_loss:.4f} | TrR2 {train_metrics['r2']:+.4f} | "
                  f"VaLoss {val_loss:.4f} | VaR2 {val_metrics['r2']:+.4f} | "
                  f"VaMAE {val_metrics['mae']:.4f} | VaRMSE {val_metrics['rmse']:.4f} | "
                  f"VaPearson {val_metrics['pearson']:.4f}{test_str}{aux_str}{win_str} | "
                  f"patience {self.patience_counter}/{patience_limit if patience_limit > 0 else 'off'}{temp_str}")

        self.save_checkpoint(save_dir, name, fold_idx, num_epochs, suffix='_final')

        # SWA: average the last K epochs and save
        if swa_count > 0:
            for k, v in swa_accum.items():
                if v.is_floating_point():
                    v.div_(swa_count)
            if fold_idx is None:
                swa_path = os.path.join(save_dir, f"{name}_swa.pth")
            else:
                swa_path = os.path.join(save_dir, f"{name}_fold{fold_idx + 1}_swa.pth")
            torch.save({'model_state_dict': swa_accum, 'epoch': num_epochs,
                        'swa_count': swa_count}, swa_path)
            self.swa_model_path = swa_path
            print(f"SWA model saved (last {swa_count} epochs averaged) -> {swa_path}")
        history_df = pd.DataFrame(history)
        result_dir = self.config['output']['result_save_dir']
        os.makedirs(result_dir, exist_ok=True)
        history_path = os.path.join(result_dir, f"{name}_training_history.csv")
        history_df.to_csv(history_path, index=False)
        print(f"Training history saved to {history_path}")
        print(f"  best epoch : {best_val_epoch} (val R2={best_val_r2:+.4f}) | "
              f"best balance: epoch={self.best_balance_epoch} "
              f"rule={self.balance_rule} score={best_balance_gap:.4f} | "
              f"final learned temperature={[f'{t:.4f}' for t in self._learned_temperatures()]}")
        return history, self.best_metrics, best_val_model_path, best_balance_model_path


class FixedLengthTrainer(BaseTrainer):
    def __init__(self, model, config, device):
        super().__init__(model, config, device)

    def train_epoch(self, loader, epoch, total_epochs):
        if hasattr(self.model, 'set_epoch'):
            self.model.set_epoch(epoch)
        self.model.train()
        total_loss = 0
        preds, targets = [], []
        accumulation_steps = self.config['training']['gradient_accumulation_steps']
        self.optimizer.zero_grad()
        batch_idx = -1

        desc = f"Epoch {epoch}/{total_epochs} Training"
        if tqdm is not None:
            pbar = tqdm(loader, desc=desc, unit="batch", leave=False, mininterval=5)
        else:
            pbar = loader

        for batch_idx, (drug, gene, labels) in enumerate(pbar):
            drug = drug.to(self.device)
            gene = gene.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(gene, drug)
            loss = self.criterion(outputs, labels)
            loss = loss / accumulation_steps

            loss.backward()

            total_loss += loss.item() * drug.size(0) * accumulation_steps
            preds.extend(outputs.detach().cpu().float().numpy())
            targets.extend(labels.detach().cpu().float().numpy())

            if (batch_idx + 1) % accumulation_steps == 0:
                if self.config['training']['gradient_clip'] > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['training']['gradient_clip'])
                self.optimizer.step()
                self.optimizer.zero_grad()

            if tqdm is not None:
                pbar.set_postfix(loss=loss.item() * accumulation_steps, refresh=False)

        if (batch_idx + 1) % accumulation_steps != 0:
            if self.config['training']['gradient_clip'] > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['training']['gradient_clip'])
            self.optimizer.step()
            self.optimizer.zero_grad()

        avg_loss = total_loss / len(loader.dataset)
        metrics = self._compute_metrics(np.array(targets), np.array(preds))
        return avg_loss, metrics

    def validate(self, loader):
        self.model.eval()
        total_loss = 0
        preds, targets = [], []

        desc = "Validation"
        if tqdm is not None:
            pbar = tqdm(loader, desc=desc, unit="batch", leave=False, mininterval=5)
        else:
            pbar = loader
        with torch.no_grad():
            for batch_idx, (drug, gene, labels) in enumerate(pbar):
                drug = drug.to(self.device)
                gene = gene.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(gene, drug)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item() * drug.size(0)
                preds.extend(outputs.cpu().float().numpy())
                targets.extend(labels.cpu().float().numpy())
            if tqdm is not None:
                pbar.set_postfix(loss=loss.item(), refresh=False)

        avg_loss = total_loss / len(loader.dataset)
        metrics = self._compute_metrics(np.array(targets), np.array(preds))
        return avg_loss, metrics

    def train(self, train_loader, val_loader, num_epochs, save_dir, fold_idx, name):
        os.makedirs(save_dir, exist_ok=True)
        history = []
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=num_epochs, eta_min=1e-6
        )
        n_total = sum(p.numel() for p in self.model.parameters())
        n_train = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"  model params : total={n_total:,}  trainable={n_train:,}  "
              f"init_temperature={[f'{t:.4f}' for t in self._learned_temperatures()]}")

        best_val_r2 = -float('inf')
        best_val_epoch = -1
        best_balance_gap = float('inf')
        best_val_model_path = None
        best_balance_model_path = None

        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()
            train_loss, train_metrics = self.train_epoch(train_loader, epoch, num_epochs)
            val_loss, val_metrics = self.validate(val_loader)

            self.scheduler.step()
            epoch_time = time.time() - epoch_start

            if val_metrics['r2'] > best_val_r2:
                best_val_r2 = val_metrics['r2']
                best_val_epoch = epoch
                model_path = self.save_checkpoint(save_dir, name, fold_idx, epoch, suffix='_best_val')
                best_val_model_path = model_path
                print(f"  New best val model saved at epoch {epoch} (val R²={val_metrics['r2']:.4f})")

            balance_score, balance_improved = self._balance_improved(
                train_metrics['r2'], val_metrics['r2'], val_loss)
            if balance_improved:
                best_balance_gap = balance_score
                self.best_balance_score = balance_score
                self.best_balance_epoch = epoch
                model_path = self.save_checkpoint(save_dir, name, fold_idx, epoch, suffix='_best_balance')
                best_balance_model_path = model_path
                print(f"  New best balance model saved at epoch {epoch} "
                      f"(rule={self.balance_rule}, score={balance_score:.4f})")

            patience_limit = self.config['training'].get('early_stop_patience', 10) or 0
            if val_metrics['r2'] > self.best_metrics['r2']:
                self.best_metrics = val_metrics
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                # patience_limit <= 0 disables early stopping
                if patience_limit > 0 and self.patience_counter >= patience_limit:
                    print(f"Early stopping at epoch {epoch}")
                    break

            fold_display = fold_idx + 1 if fold_idx is not None else 0
            history.append({
                'fold': fold_display,
                'epoch': epoch,
                'lr': self.optimizer.param_groups[0]['lr'],
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_r2': train_metrics['r2'],
                'val_r2': val_metrics['r2'],
                'train_pearson': train_metrics['pearson'],
                'val_pearson': val_metrics['pearson'],
                'train_mae': train_metrics['mae'],
                'val_mae': val_metrics['mae'],
                'train_rmse': train_metrics['rmse'],
                'val_rmse': val_metrics['rmse'],
                'learned_temperature': (self._learned_temperatures() or [float('nan')])[0],
            })
            temps = self._learned_temperatures()
            temp_str = f" | T {temps[0]:.3f}" if temps else ""
            print(f"Fold {fold_display} | Epoch {epoch:3d} | {epoch_time:6.0f}s | "
                  f"LR {self.optimizer.param_groups[0]['lr']:.2e} | "
                  f"TrLoss {train_loss:.4f} | TrR2 {train_metrics['r2']:+.4f} | "
                  f"VaLoss {val_loss:.4f} | VaR2 {val_metrics['r2']:+.4f} | "
                  f"VaMAE {val_metrics['mae']:.4f} | VaRMSE {val_metrics['rmse']:.4f} | "
                  f"VaPearson {val_metrics['pearson']:.4f} | "
                  f"patience {self.patience_counter}/{patience_limit if patience_limit > 0 else 'off'}{temp_str}")

        self.save_checkpoint(save_dir, name, fold_idx, num_epochs, suffix='_final')
        history_df = pd.DataFrame(history)
        result_dir = self.config['output']['result_save_dir']
        os.makedirs(result_dir, exist_ok=True)
        history_path = os.path.join(result_dir, f"{name}_training_history.csv")
        history_df.to_csv(history_path, index=False)
        print(f"Training history saved to {history_path}")
        print(f"  best epoch : {best_val_epoch} (val R2={best_val_r2:+.4f}) | "
              f"best balance: epoch={self.best_balance_epoch} "
              f"rule={self.balance_rule} score={best_balance_gap:.4f} | "
              f"final learned temperature={[f'{t:.4f}' for t in self._learned_temperatures()]}")
        return history, self.best_metrics, best_val_model_path, best_balance_model_path
