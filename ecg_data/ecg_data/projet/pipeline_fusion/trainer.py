#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Entraîneur multi-label pour la fusion ECG."""

import os
import json
import logging
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from tqdm import tqdm

from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.metrics import average_precision_score

from common.utils import CLASS_NAMES, NUM_CLASSES, CLASS_TO_IDX
from common.losses import MultiLabelLogitAdjustedBCE, FocalLoss


class FusionTrainer:
    """Entraîneur multi-label avec fusion croisée et métadonnées."""

    def __init__(self, model: nn.Module, config: Dict, device: torch.device,
                 logger: logging.Logger):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.logger = logger
        self.use_amp = bool(config.get('use_amp', False)) and self.device.type == 'cuda'
        self.grad_scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        # [ANCIEN] self.aux_alpha = 0.3
        self.aux_alpha = 0.1
        # [ANCIEN] self.cd_beta = 0.2
        self.cd_beta = 0.05
        # [ANCIEN] self.mi_beta = 0.3
        self.mi_beta = 0.15  # V4: Augmenté pour meilleure détection MI
        self.cd_idx = CLASS_TO_IDX['CD']
        self.mi_idx = CLASS_TO_IDX['MI']

        self.optimizer = AdamW(
            model.parameters(),
            # [ANCIEN] lr=config.get('learning_rate', 3e-4),
            lr=config.get('learning_rate', 1e-4),
            # [V1] weight_decay=config.get('weight_decay', 1e-4)
            weight_decay=config.get('weight_decay', 5e-4)
        )

        # [ANCIEN]
        # self.scheduler = CosineAnnealingWarmRestarts(
        #     self.optimizer,
        #     T_0=10,
        #     T_mult=2,
        #     eta_min=1e-6,
        # )
        self.scheduler = None
        self.scheduler_interval = 'batch'
        self.total_steps = 0
        self.global_step = 0

        self.best_f1 = 0.0
        self.patience_counter = 0
        self.criterion = None
        self.cd_pos_weight = None
        self.mi_pos_weight = None
        self.loss_type = 'focal'  # V4: Focal Loss pour meilleure convergence

    def setup_loss(self, class_counts: np.ndarray, loss_type: str = 'logit_adj',
                   tau: float = 1.0):
        self.loss_type = loss_type
        total = class_counts.sum()
        pos_counts = np.maximum(class_counts.astype(float), 1.0)
        pos_weight = np.sqrt((total - pos_counts) / pos_counts)
        pos_weight = np.clip(pos_weight, 1.0, 8.0)
        pos_weight_t = torch.tensor(pos_weight, dtype=torch.float32).to(self.device)

        if loss_type == 'logit_adj':
            self.criterion = MultiLabelLogitAdjustedBCE(
                class_counts, tau=tau, pos_weight=pos_weight_t).to(self.device)
            self.logger.info(f"Loss: MultiLabelLogitAdjustedBCE (τ={tau})")
        else:
            # [ANCIEN] self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_t).to(self.device)
            self.criterion = FocalLoss(gamma=2.0, alpha=0.25, reduction='mean').to(self.device)
            self.logger.info("Loss: FocalLoss(gamma=2.0, alpha=0.25)")

        self.logger.info(f"Criterion: {self.criterion}")

        self.cd_pos_weight = torch.tensor(pos_weight[self.cd_idx], dtype=torch.float32).to(self.device)
        self.mi_pos_weight = torch.tensor(pos_weight[self.mi_idx], dtype=torch.float32).to(self.device)

        self.logger.info(f"  Poids par classe: { {CLASS_NAMES[i]: f'{pos_weight[i]:.3f}' for i in range(NUM_CLASSES)} }")
        self.logger.info(f"  α (aux 1D+2D): {self.aux_alpha}")
        self.logger.info(f"  β_cd: {self.cd_beta}")
        self.logger.info(f"  β_mi: {self.mi_beta}")
        self.logger.info(f"Poids multi-task: aux={self.aux_alpha}, cd={self.cd_beta}, mi={self.mi_beta}")

    def _binary_loss(self, logits: torch.Tensor, targets: torch.Tensor,
                     pos_weight: torch.Tensor) -> torch.Tensor:
        logits_sq = logits.squeeze(-1)
        if self.loss_type == 'focal':
            # [ANCIEN] return F.binary_cross_entropy_with_logits(
            # [ANCIEN]     logits.squeeze(-1), targets,
            # [ANCIEN]     pos_weight=pos_weight)
            # Focal Loss pour cohérence avec la fusion head
            bce = F.binary_cross_entropy_with_logits(
                logits_sq, targets, reduction='none'
            )
            p_t = torch.exp(-bce)
            focal = 0.25 * (1.0 - p_t) ** 2.0 * bce
            return focal.mean()
        else:
            # BCE pondérée (logit_adj)
            return F.binary_cross_entropy_with_logits(
                logits_sq, targets,
                pos_weight=pos_weight)

    def _build_threshold_array(self, thresholds: Dict | None) -> np.ndarray | None:
        if thresholds is None:
            return None
        return np.array([
            float(thresholds.get(class_name, 0.5))
            for class_name in CLASS_NAMES
        ], dtype=np.float32)

    def find_optimal_thresholds(self, loader: DataLoader, step: float = 0.01,
                                min_thr: float = 0.1, max_thr: float = 0.9) -> Dict[str, float]:
        self.model.eval()
        all_probs, all_labels = [], []

        with torch.no_grad():
            for signals, images, labels, masks, metadata in loader:
                signals = signals.to(self.device)
                images = images.to(self.device)
                labels = labels.to(self.device)
                masks = masks.to(self.device)
                metadata = metadata.to(self.device)

                logits = self.model(signals, images, masks, metadata=metadata)
                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.append(probs)
                all_labels.append(labels.cpu().numpy())

        all_probs = np.concatenate(all_probs, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        best_thresholds: Dict[str, float] = {}
        for i, cls in enumerate(CLASS_NAMES):
            best_f1, best_thr = 0.0, 0.5
            for thr in np.arange(min_thr, max_thr, step):
                preds = (all_probs[:, i] >= thr).astype(np.int32)
                f1 = f1_score(all_labels[:, i], preds, zero_division=0)
                if f1 > best_f1:
                    best_f1, best_thr = f1, float(thr)
            best_thresholds[cls] = round(best_thr, 2)
            self.logger.info(
                f"  {cls}: seuil optimal = {best_thr:.2f} (F1={best_f1:.4f})")

        return best_thresholds

    def train_epoch(self, loader: DataLoader, max_batches: int = None) -> Dict:
        self.model.train()
        total_loss = 0.0
        loss_comp = {'fusion': 0.0, 'aux_1d': 0.0, 'aux_2d': 0.0, 'cd': 0.0, 'mi': 0.0}
        all_preds, all_labels = [], []

        pbar = tqdm(loader, desc="Train", leave=False)
        for batch_idx, (signals, images, labels, masks, metadata) in enumerate(pbar):
            if max_batches is not None and batch_idx >= max_batches:
                break
            signals = signals.to(self.device)
            images = images.to(self.device)
            labels = labels.to(self.device)
            masks = masks.to(self.device)
            metadata = metadata.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                out_f, out_1d, out_2d, out_cd, out_mi, fused = self.model(
                    signals, images, masks, metadata=metadata)

                loss_f = self.criterion(out_f, labels)
                loss_1d = self.criterion(out_1d, labels)
                loss_2d = self.criterion(out_2d, labels)
                loss_cd = self._binary_loss(out_cd, labels[:, self.cd_idx], self.cd_pos_weight)
                loss_mi = self._binary_loss(out_mi, labels[:, self.mi_idx], self.mi_pos_weight)

                loss = (loss_f
                        + self.aux_alpha * loss_1d
                        + self.aux_alpha * loss_2d
                        + self.cd_beta * loss_cd
                        + self.mi_beta * loss_mi)

            self.grad_scaler.scale(loss).backward()

            total_norm = sum(
                p.grad.data.norm(2).item() ** 2
                for p in self.model.parameters() if p.grad is not None
            ) ** 0.5
            if self.global_step < 10:
                self.logger.debug(f"Grad norm before clip: {total_norm:.4f}")

            if self.use_amp:
                self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
            if self.scheduler is not None and self.scheduler_interval == 'batch':
                self.scheduler.step()
            self.global_step += 1

            total_loss += loss.item()
            loss_comp['fusion'] += loss_f.item()
            loss_comp['aux_1d'] += loss_1d.item()
            loss_comp['aux_2d'] += loss_2d.item()
            loss_comp['cd'] += loss_cd.item()
            loss_comp['mi'] += loss_mi.item()

            preds = (torch.sigmoid(out_f) >= 0.5).float().cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy())

            lr = self.optimizer.param_groups[0]['lr']
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{lr:.2e}'})

        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        n_batches = max(1, min(len(loader), max_batches or len(loader)))

        return {
            'loss': total_loss / n_batches,
            'loss_fusion': loss_comp['fusion'] / n_batches,
            'loss_aux_1d': loss_comp['aux_1d'] / n_batches,
            'loss_aux_2d': loss_comp['aux_2d'] / n_batches,
            'loss_cd': loss_comp['cd'] / n_batches,
            'loss_mi': loss_comp['mi'] / n_batches,
            'accuracy': accuracy_score(all_labels, all_preds),
            'f1_macro': f1_score(all_labels, all_preds, average='macro', zero_division=0),
        }

    @torch.no_grad()
    def validate(self, loader: DataLoader, max_batches: int = None,
                 thresholds: Dict | None = None) -> Tuple[Dict, np.ndarray, np.ndarray]:
        self.model.eval()
        total_loss = 0.0
        all_preds, all_labels = [], []
        all_probs = []

        threshold_array = self._build_threshold_array(thresholds)

        for batch_idx, (signals, images, labels, masks, metadata) in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            signals = signals.to(self.device)
            images = images.to(self.device)
            labels = labels.to(self.device)
            masks = masks.to(self.device)
            metadata = metadata.to(self.device)

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                logits = self.model(signals, images, masks, metadata=metadata)
                loss = self.criterion(logits, labels)

            total_loss += loss.item()
            probs = torch.sigmoid(logits).cpu().numpy()
            if threshold_array is None:
                preds = (probs >= 0.5).astype(np.float32)
            else:
                preds = (probs >= threshold_array[None, :]).astype(np.float32)
            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs)

        all_probs = np.concatenate(all_probs, axis=0)
        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        f1_cls = f1_score(all_labels, all_preds, average=None, zero_division=0)
        rec_cls = recall_score(all_labels, all_preds, average=None, zero_division=0)

        try:
            pr_auc = average_precision_score(all_labels, all_probs, average='macro')
        except Exception:
            pr_auc = 0.0

        n_batches = max(1, min(len(loader), max_batches or len(loader)))

        return {
            'loss': total_loss / n_batches,
            'accuracy': accuracy_score(all_labels, all_preds),
            'f1_macro': f1_score(all_labels, all_preds, average='macro', zero_division=0),
            'pr_auc_macro': round(float(pr_auc), 4),
            'f1_per_class': {CLASS_NAMES[i]: round(float(f1_cls[i]), 4) for i in range(NUM_CLASSES)},
            'recall_per_class': {CLASS_NAMES[i]: round(float(rec_cls[i]), 4) for i in range(NUM_CLASSES)},
        }, all_preds, all_labels

    def train_loop(self, train_loader, val_loader, epochs, ckpt_dir,
                   start_epoch: int = 1, max_batches: int = None,
                   scheduler_state_dict=None) -> Dict:
        os.makedirs(ckpt_dir, exist_ok=True)
        patience = self.config.get('patience', 10)
        history = {'train_loss': [], 'val_loss': [], 'val_f1': []}

        self.total_steps = max(int(epochs * len(train_loader)), 2)
        warmup_steps = min(200, self.total_steps - 1)
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_steps
        )
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=max(1, self.total_steps - warmup_steps),
            eta_min=1e-6
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps]
        )
        if scheduler_state_dict is not None:
            try:
                self.scheduler.load_state_dict(scheduler_state_dict)
                self.logger.info("Scheduler state restauré depuis checkpoint")
            except Exception as e:
                self.logger.warning(f"Impossible de restaurer scheduler: {e}")
        self.logger.info(
            f"Scheduler: LinearWarmup({warmup_steps} steps) -> CosineAnnealingLR (total_steps={self.total_steps})")
        self.logger.info(f"LR initial: {self.optimizer.param_groups[0]['lr']:.2e}")

        self.logger.info(
            f"Entraînement: epochs {start_epoch}→{epochs}, patience={patience}")
        if start_epoch > 1:
            self.logger.info(
                f"Reprise depuis epoch {start_epoch} (best_f1={self.best_f1:.4f})")
        self.logger.info(
            f"Multi-task: L_fus + {self.aux_alpha}*L_1d + {self.aux_alpha}*L_2d + {self.cd_beta}*L_cd + {self.mi_beta}*L_mi")

        metrics_path = os.path.join(ckpt_dir, 'training_metrics.jsonl')

        for epoch in range(start_epoch, epochs + 1):
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Epoch {epoch}/{epochs}")

            train_m = self.train_epoch(train_loader, max_batches=max_batches)
            val_m, _, _ = self.validate(val_loader, max_batches=max_batches)

            lr = self.optimizer.param_groups[0]['lr']
            self.logger.info(
                f"Train — Loss: {train_m['loss']:.4f} (fus={train_m['loss_fusion']:.3f} 1d={train_m['loss_aux_1d']:.3f} 2d={train_m['loss_aux_2d']:.3f} cd={train_m['loss_cd']:.3f} mi={train_m['loss_mi']:.3f}), Acc: {train_m['accuracy']:.4f}, F1: {train_m['f1_macro']:.4f}")
            self.logger.info(
                f"Val   — Loss: {val_m['loss']:.4f}, Acc: {val_m['accuracy']:.4f}, F1: {val_m['f1_macro']:.4f}, PR-AUC: {val_m['pr_auc_macro']:.4f}, LR: {lr:.2e}")
            self.logger.info(f"F1/classe:    {val_m['f1_per_class']}")
            self.logger.info(f"Sensibilité:  {val_m['recall_per_class']}")

            history['train_loss'].append(train_m['loss'])
            history['val_loss'].append(val_m['loss'])
            history['val_f1'].append(val_m['f1_macro'])

            epoch_metrics = {
                'epoch': epoch,
                'train_loss': train_m['loss'],
                'train_f1': train_m['f1_macro'],
                'val_loss': val_m['loss'],
                'val_f1': val_m['f1_macro'],
                'val_pr_auc': val_m['pr_auc_macro'],
                'val_f1_per_class': val_m['f1_per_class'],
                'val_recall_per_class': val_m['recall_per_class'],
                'lr': lr,
                'loss_components': {
                    'fusion': train_m['loss_fusion'],
                    'aux_1d': train_m['loss_aux_1d'],
                    'aux_2d': train_m['loss_aux_2d'],
                    'cd': train_m['loss_cd'],
                    'mi': train_m['loss_mi'],
                },
            }
            with open(metrics_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(epoch_metrics, default=str) + '\n')

            if val_m['f1_macro'] > self.best_f1:
                self.best_f1 = val_m['f1_macro']
                self.patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'best_f1': self.best_f1,
                    'patience_counter': self.patience_counter,
                    'config': self.config,
                    'loss_type': self.loss_type,
                }, os.path.join(ckpt_dir, 'best_fusion_model.pth'))
                self.logger.info(
                    f"★ Meilleur modèle sauvegardé (F1: {self.best_f1:.4f})")
            else:
                self.patience_counter += 1
                self.logger.info(
                    f"  Patience: {self.patience_counter}/{patience}")
                if self.patience_counter >= patience:
                    self.logger.info(f"Early stopping epoch {epoch}")
                    break

        return {'best_f1': self.best_f1, 'history': history}
