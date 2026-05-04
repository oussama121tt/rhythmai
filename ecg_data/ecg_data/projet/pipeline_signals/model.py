#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classifieur signal-only : ResNet1D pour classification ECG 12-lead.
"""

import torch.nn as nn
import torch.nn.functional as F

from common.utils import NUM_CLASSES
from common.signal_backbone import ECGResNet1D


class ECGSignalClassifier(nn.Module):
    """
    Classifieur de signaux ECG basé sur ResNet1D + SE + Attention Pooling.

    Architecture :
      Signal (12, 5000) → ECGResNet1D → embedding (256)
      → FC(128) → ReLU → Dropout → FC(5) → logits

    Paramètres : ~17.4M (backbone) + ~33K (head) ≈ 17.5M
    """

    def __init__(self, num_classes: int = NUM_CLASSES, embed_dim: int = 256,
                 dropout: float = 0.3):
        super().__init__()

        self.backbone = ECGResNet1D(embed_dim=embed_dim)

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

        self.num_classes = num_classes

    def forward(self, x):
        """x: (B, 12, 5000) → (B, num_classes)"""
        emb = self.backbone(x)   # (B, 256)
        return self.classifier(emb)

    def predict_proba(self, x):
        """Retourne les probabilités."""
        logits = self.forward(x)
        return F.softmax(logits, dim=1)
