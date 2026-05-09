#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Branche image 2D : EfficientNet-B3 avec head renforcé (1280→512→256).
Partagée entre pipeline_images et pipeline_fusion.
"""

import torch.nn as nn
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights


class ECGImageBranch(nn.Module):
    """
    EfficientNet-B3 avec head renforcé pour images ECG.
    Utilisé comme branche image dans le modèle de fusion.
    """

    def __init__(self, embed_dim: int = 256, pretrained: bool = True,
                 input_channels: int = 3):
        super().__init__()
        weights = EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = efficientnet_b3(weights=weights)
        if input_channels != 3:
            first_conv = self.backbone.features[0][0]
            new_conv = nn.Conv2d(input_channels, first_conv.out_channels,
                                 kernel_size=first_conv.kernel_size,
                                 stride=first_conv.stride,
                                 padding=first_conv.padding,
                                 bias=False)
            self.backbone.features[0][0] = new_conv

        in_features = self.backbone.classifier[1].in_features  # 1536

        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 768),
            nn.BatchNorm1d(768),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(768, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
        )

    def forward_features(self, x):
        """Retourne la carte de features EfficientNet avant pooling."""
        return self.backbone.features(x)

    def forward(self, x):
        """x: (B, 3, 224, 224) → (B, embed_dim)"""
        return self.backbone(x)
