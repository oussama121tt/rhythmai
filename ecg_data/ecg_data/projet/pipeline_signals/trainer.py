#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Entraîneur pour classification ECG par signaux 1D (multi-label)."""

import os
import logging
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm

from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.metrics import average_precision_score

from common.utils import CLASS_NAMES
from common.losses import MultiLabelLogitAdjustedBCE


class SignalTrainer:
    """Entraîneur pour classification de signaux ECG 1D."""

    def __init__(self, model: nn.Module, config: Dict, device: torch.device,
                 logger: logging.Logger):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.logger = logger

        self.optimizer = AdamW(
            model.parameters(),
            lr=config.get('learning_rate', 3e-4),
            weight_decay=config.get('weight_decay', 1e-4)
        )

        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,
            T_mult=2,
            eta_min=1e-6,
        )

        self.best_f1 = 0.0
        self.patience_counter = 0
        self.criterion = None

    def setup_loss(self, class_counts: np.ndarray, loss_type: str = 'logit_adj',
                   tau: float = 1.0):
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
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_t).to(self.device)
            self.logger.info("Loss: BCEWithLogitsLoss pondérée")

        weight_dict = {CLASS_NAMES[i]: f'{pos_weight[i]:.3f}'
                       for i in range(len(CLASS_NAMES))}
        self.logger.info(f"  Poids par classe: {weight_dict}")

    def train_epoch(self, loader: DataLoader) -> Dict:
        self.model.train()
        total_loss = 0.0
        all_preds, all_labels = [], []

        pbar = tqdm(loader, desc="Training", leave=False)
        for signals, labels in pbar:
            signals = signals.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(signals)
            loss = self.criterion(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            preds = (torch.sigmoid(logits) >= 0.5).float().cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy())

            current_lr = self.optimizer.param_groups[0]['lr']
            pbar.set_postfix({'loss': f'{loss.item():.4f}',
                              'lr': f'{current_lr:.2e}'})

        all_preds_arr = np.concatenate(all_preds, axis=0)
        all_labels_arr = np.concatenate(all_labels, axis=0)

        return {
            'loss': total_loss / len(loader),
            'accuracy': accuracy_score(all_labels_arr, all_preds_arr),
            'f1_macro': f1_score(all_labels_arr, all_preds_arr, average='macro',
                                 zero_division=0)
        }

    @torch.no_grad()
    def validate(self, loader: DataLoader) -> Tuple[Dict, np.ndarray,
                                                     np.ndarray]:
        self.model.eval()
        total_loss = 0.0
        all_preds, all_labels = [], []
        all_probs = []

        for signals, labels in loader:
            signals = signals.to(self.device)
            labels = labels.to(self.device)

            logits = self.model(signals)
            loss = self.criterion(logits, labels)

            total_loss += loss.item()
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs >= 0.5).astype(np.float32)
            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs)

        all_probs = np.concatenate(all_probs, axis=0)
        all_preds_arr = np.concatenate(all_preds, axis=0)
        all_labels_arr = np.concatenate(all_labels, axis=0)

        f1_per_class = f1_score(all_labels_arr, all_preds_arr, average=None,
                                zero_division=0)
        recall_per_class = recall_score(all_labels_arr, all_preds_arr, average=None,
                                        zero_division=0)

        try:
            pr_auc = average_precision_score(all_labels_arr, all_probs, average='macro')
        except Exception:
            pr_auc = 0.0

        return {
            'loss': total_loss / len(loader),
            'accuracy': accuracy_score(all_labels_arr, all_preds_arr),
            'f1_macro': f1_score(all_labels_arr, all_preds_arr, average='macro',
                                 zero_division=0),
            'pr_auc_macro': round(float(pr_auc), 4),
            'f1_per_class': {CLASS_NAMES[i]: round(float(f1_per_class[i]), 4)
                             for i in range(len(CLASS_NAMES))},
            'recall_per_class': {CLASS_NAMES[i]: round(float(recall_per_class[i]), 4)
                                 for i in range(len(CLASS_NAMES))},
        }, all_preds_arr, all_labels_arr

    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              epochs: int, checkpoint_dir: str) -> Dict:
        os.makedirs(checkpoint_dir, exist_ok=True)

        patience = self.config.get('patience', 12)
        history = {'train_loss': [], 'val_loss': [], 'val_f1': []}

        self.logger.info(
            f"Entraînement: {epochs} epochs, patience={patience}")

        for epoch in range(1, epochs + 1):
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Epoch {epoch}/{epochs}")

            train_metrics = self.train_epoch(train_loader)
            val_metrics, _, _ = self.validate(val_loader)

            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            self.logger.info(
                f"Train - Loss: {train_metrics['loss']:.4f}, "
                f"Acc: {train_metrics['accuracy']:.4f}, "
                f"F1: {train_metrics['f1_macro']:.4f}"
            )
            self.logger.info(
                f"Val   - Loss: {val_metrics['loss']:.4f}, "
                f"Acc: {val_metrics['accuracy']:.4f}, "
                f"F1: {val_metrics['f1_macro']:.4f}, "
                f"PR-AUC: {val_metrics['pr_auc_macro']:.4f}, "
                f"LR: {current_lr:.2e}"
            )
            self.logger.info(f"F1/classe: {val_metrics['f1_per_class']}")
            self.logger.info(
                f"Sensibilité/classe: {val_metrics['recall_per_class']}")

            history['train_loss'].append(train_metrics['loss'])
            history['val_loss'].append(val_metrics['loss'])
            history['val_f1'].append(val_metrics['f1_macro'])

            if val_metrics['f1_macro'] > self.best_f1:
                self.best_f1 = val_metrics['f1_macro']
                self.patience_counter = 0

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'best_f1': self.best_f1,
                    'config': self.config,
                }, os.path.join(checkpoint_dir, 'best_signal_model.pth'))

                self.logger.info(
                    f"★ Nouveau meilleur modèle (F1: {self.best_f1:.4f})")
            else:
                self.patience_counter += 1
                self.logger.info(
                    f"  Patience: {self.patience_counter}/{patience}")
                if self.patience_counter >= patience:
                    self.logger.info(
                        f"Early stopping à l'epoch {epoch}")
                    break

        return {'best_f1': self.best_f1, 'history': history}
