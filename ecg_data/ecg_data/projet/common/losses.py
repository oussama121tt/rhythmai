#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fonctions de loss personnalisées.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .utils import CLASS_NAMES, NUM_CLASSES


class LogitAdjustedCE(nn.Module):
    """
    Logit-Adjusted Cross-Entropy (Menon et al., 2021).
      Training  : CE(logits - τ·log(π), targets) — pénalise classes fréquentes
      Inference : argmax(logits + τ·log(π))       — restaure calibration
    """

    def __init__(self, class_counts: np.ndarray, tau: float = 1.0):
        super().__init__()
        prior = class_counts / class_counts.sum()
        log_prior = np.log(prior + 1e-8)
        self.register_buffer('log_prior',
                             torch.tensor(log_prior, dtype=torch.float32))
        self.tau = tau

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        adjusted = logits - self.tau * self.log_prior
        return F.cross_entropy(adjusted, targets)

    def adjust_for_inference(self, logits: torch.Tensor) -> torch.Tensor:
        """Appliquer à l'inférence pour des prédictions calibrées."""
        return logits + self.tau * self.log_prior


class MultiLabelLogitAdjustedBCE(nn.Module):
    """
    BCEWithLogits multi-label avec ajustement des logits par prior.

    Training : BCEWithLogits(logits - tau * log_prior, targets)
    Inference : logits + tau * log_prior pour la calibration.
    """

    def __init__(self, class_counts: np.ndarray, tau: float = 1.0,
                 pos_weight: Optional[torch.Tensor] = None):
        super().__init__()
        counts = np.asarray(class_counts, dtype=np.float32)
        prior = counts / max(float(counts.sum()), 1.0)
        log_prior = np.log(prior + 1e-8)
        self.register_buffer('log_prior',
                             torch.tensor(log_prior, dtype=torch.float32))
        self.tau = tau
        if pos_weight is not None:
            self.register_buffer('pos_weight', pos_weight.float())
        else:
            self.pos_weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        adjusted = logits - self.tau * self.log_prior
        return F.binary_cross_entropy_with_logits(
            adjusted, targets, pos_weight=self.pos_weight)

    def adjust_for_inference(self, logits: torch.Tensor) -> torch.Tensor:
        return logits + self.tau * self.log_prior


class FocalLoss(nn.Module):
    """Focal Loss multi-label sur logits (Lin et al., RetinaNet)."""

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25,
                 reduction: str = 'mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )
        p_t = torch.exp(-bce)
        focal = self.alpha * (1.0 - p_t) ** self.gamma * bce
        if self.reduction == 'mean':
            return focal.mean()
        return focal.sum()
