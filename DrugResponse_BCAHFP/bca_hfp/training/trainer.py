# bca_hfp/training/trainer.py
import torch
import os
import sys
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
        self.criterion = torch.nn.SmoothL1Loss(beta=1.0)
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['training']['lr'],
            weight_decay=config['training']['weight_decay']
        )
        self.scaler = None
        self.best_metrics = {'r2': -float('inf')}
        self.patience_counter = 0

    def _compute_metrics(self, y_true, y_pred):
        from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
        from scipy.stats import pearsonr
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        pearson = pearsonr(y_true, y_pred)[0] if len(y_true) > 1 else 0.0
        return {'r2': r2, 'mae': mae, 'rmse': rmse, 'pearson': pearson}

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
        preds, targets = [], []
        accumulation_steps = self.config['training']['gradient_accumulation_steps']
        self.optimizer.zero_grad()
        batch_idx = -1

        desc = f"Epoch {epoch}/{total_epochs} Training"
        if tqdm is not None:
            pbar = tqdm(loader, desc=desc, unit="batch", leave=False)
        else:
            pbar = loader

        for batch_idx, (atom_feat, atom_mask, gene_feat, gene_mask, labels, drug_global) in enumerate(pbar):
            atom_feat = atom_feat.to(self.device)
            atom_mask = atom_mask.to(self.device)
            gene_feat = gene_feat.to(self.device)
            gene_mask = gene_mask.to(self.device)
            labels = labels.to(self.device)
            if drug_global is not None:
                drug_global = drug_global.to(self.device)

            outputs = self.model(gene_feat, atom_feat, atom_mask, gene_mask, drug_global=drug_global)
            loss = self.criterion(outputs, labels)
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
                pbar.set_postfix(loss=loss.item() * accumulation_steps)

        # Handle remaining gradients
        if (batch_idx + 1) % accumulation_steps != 0:
            if self.config['training']['gradient_clip'] > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['training']['gradient_clip'])
            self.optimizer.step()
            self.optimizer.zero_grad()

        avg_loss = total_loss / len(loader.dataset)
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
            pbar = tqdm(loader, desc=desc, unit="batch", leave=False)
        else:
            pbar = loader

        with torch.no_grad():
            for batch_idx, (atom_feat, atom_mask, gene_feat, gene_mask, labels, drug_global) in enumerate(pbar):
                atom_feat = atom_feat.to(self.device)
                atom_mask = atom_mask.to(self.device)
                gene_feat = gene_feat.to(self.device)
                gene_mask = gene_mask.to(self.device)
                labels = labels.to(self.device)
                if drug_global is not None:
                    drug_global = drug_global.to(self.device)

                outputs = self.model(gene_feat, atom_feat, atom_mask, gene_mask, drug_global=drug_global)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item() * atom_feat.size(0)
                preds.extend(outputs.cpu().float().numpy())
                targets.extend(labels.cpu().float().numpy())

                if not saved and epoch is not None and epoch in self.save_attention_epochs:
                    if hasattr(self.model, 'cross_attn_drug2gene'):
                        self.model.cross_attn_drug2gene.accumulate_attention_weights(epoch, sample_idx=0)
                        saved = True
                if tqdm is not None:
                    pbar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(loader.dataset)
        metrics = self._compute_metrics(np.array(targets), np.array(preds))
        return avg_loss, metrics

    def train(self, train_loader, val_loader, num_epochs, save_dir, fold_idx, name):
        os.makedirs(save_dir, exist_ok=True)
        history = []
        scheduler_type = self.config['training'].get('scheduler', 'cosine')
        if scheduler_type == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=num_epochs, eta_min=1e-6
            )
        else:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='max', factor=self.config['training'].get('reduce_factor', 0.5),
                patience=self.config['training'].get('reduce_patience', 5)
            )

        best_val_r2 = -float('inf')
        best_balance_gap = float('inf')
        best_val_model_path = None
        best_balance_model_path = None

        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()
            train_loss, train_metrics = self.train_epoch(train_loader, epoch, num_epochs)
            val_loss, val_metrics = self.validate(val_loader, fold_idx, epoch)

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
                model_path = self.save_checkpoint(save_dir, name, fold_idx, epoch, suffix='_best_val')
                best_val_model_path = model_path
                print(f"  New best val model saved at epoch {epoch} (val R²={val_metrics['r2']:.4f})")

            gap = abs(train_metrics['r2'] - val_metrics['r2'])
            if gap < best_balance_gap:
                best_balance_gap = gap
                model_path = self.save_checkpoint(save_dir, name, fold_idx, epoch, suffix='_best_balance')
                best_balance_model_path = model_path
                print(f"  New best balance model saved at epoch {epoch} (gap={gap:.4f})")

            if val_metrics['r2'] > self.best_metrics['r2']:
                self.best_metrics = val_metrics
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config['training']['early_stop_patience']:
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
                'val_rmse': val_metrics['rmse']
            })
            print(f"Fold {fold_display} | Epoch {epoch:3d} | Time {epoch_time:.2f}s | LR {self.optimizer.param_groups[0]['lr']:.2e} | Train R² {train_metrics['r2']:.4f} | Val R² {val_metrics['r2']:.4f} | Val RMSE {val_metrics['rmse']:.4f} | Val Pearson {val_metrics['pearson']:.4f}")

        self.save_checkpoint(save_dir, name, fold_idx, num_epochs, suffix='_final')
        history_df = pd.DataFrame(history)
        result_dir = self.config['output']['result_save_dir']
        os.makedirs(result_dir, exist_ok=True)
        history_path = os.path.join(result_dir, f"{name}_training_history.csv")
        history_df.to_csv(history_path, index=False)
        print(f"Training history saved to {history_path}")
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
            pbar = tqdm(loader, desc=desc, unit="batch", leave=False)
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
                pbar.set_postfix(loss=loss.item() * accumulation_steps)

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
            pbar = tqdm(loader, desc=desc, unit="batch", leave=False)
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
                pbar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(loader.dataset)
        metrics = self._compute_metrics(np.array(targets), np.array(preds))
        return avg_loss, metrics

    def train(self, train_loader, val_loader, num_epochs, save_dir, fold_idx, name):
        os.makedirs(save_dir, exist_ok=True)
        history = []
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=num_epochs, eta_min=1e-6
        )
        best_val_r2 = -float('inf')
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
                model_path = self.save_checkpoint(save_dir, name, fold_idx, epoch, suffix='_best_val')
                best_val_model_path = model_path
                print(f"  New best val model saved at epoch {epoch} (val R²={val_metrics['r2']:.4f})")

            gap = abs(train_metrics['r2'] - val_metrics['r2'])
            if gap < best_balance_gap:
                best_balance_gap = gap
                model_path = self.save_checkpoint(save_dir, name, fold_idx, epoch, suffix='_best_balance')
                best_balance_model_path = model_path
                print(f"  New best balance model saved at epoch {epoch} (gap={gap:.4f})")

            if val_metrics['r2'] > self.best_metrics['r2']:
                self.best_metrics = val_metrics
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config['training']['early_stop_patience']:
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
                'val_rmse': val_metrics['rmse']
            })
            print(f"Fold {fold_display} | Epoch {epoch:3d} | Time {epoch_time:.2f}s | LR {self.optimizer.param_groups[0]['lr']:.2e} | Train R² {train_metrics['r2']:.4f} | Val R² {val_metrics['r2']:.4f} | Val RMSE {val_metrics['rmse']:.4f} | Val Pearson {val_metrics['pearson']:.4f}")

        self.save_checkpoint(save_dir, name, fold_idx, num_epochs, suffix='_final')
        history_df = pd.DataFrame(history)
        result_dir = self.config['output']['result_save_dir']
        os.makedirs(result_dir, exist_ok=True)
        history_path = os.path.join(result_dir, f"{name}_training_history.csv")
        history_df.to_csv(history_path, index=False)
        print(f"Training history saved to {history_path}")
        return history, self.best_metrics, best_val_model_path, best_balance_model_path
