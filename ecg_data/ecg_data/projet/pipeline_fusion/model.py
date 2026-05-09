#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Modèle de fusion multimodale : co-attention croisée + fusion hiérarchique."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from common.utils import NUM_CLASSES
from common.signal_backbone import ECGResNet1D
from common.image_backbone import ECGImageBranch
from common.blocks import CrossModalCoAttention


class GatedFusion(CrossModalCoAttention):
    """Alias de compatibilité pour les anciens imports."""
    pass


class ECGFusionModel(nn.Module):
    """Fusion multimodale à deux étages avec métadonnées."""

    def __init__(self, num_classes: int = NUM_CLASSES, embed_dim: int = 256,
                 pretrained: bool = True, dropout: float = 0.3,
                 modality_drop_p: float = 0.15, metadata_dim: int = 3):
        super().__init__()
        self.embed_dim = embed_dim
        self.modality_drop_p = modality_drop_p
        self.num_classes = num_classes
        self.metadata_dim = metadata_dim

        self.signal_branch = ECGResNet1D(embed_dim=embed_dim)
        self.image_branch = ECGImageBranch(embed_dim=embed_dim,
                                           pretrained=pretrained)

        self.co_attention = CrossModalCoAttention(
            signal_dim=512,
            image_dim=1536,
            embed_dim=embed_dim,
            num_heads=8,
            dropout=dropout,
        )

        self.signal_context_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )
        self.image_context_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

        self.metadata_encoder = nn.Sequential(
            nn.Linear(metadata_dim, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 16),
            nn.ReLU(inplace=True),
        )

        self.intermediate_fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        final_in = embed_dim * 3 + 16
        self.final_fusion = nn.Sequential(
            nn.Linear(final_in, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

        self.fusion_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

        self.aux_head_1d = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )
        self.aux_head_2d = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

        self.cd_head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )
        self.mi_head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def _apply_modality_dropout(self, signal_mask: torch.Tensor,
                                image_mask: torch.Tensor):
        if not self.training or self.modality_drop_p <= 0:
            return signal_mask, image_mask
        batch_size = signal_mask.size(0)
        both_present = (signal_mask * image_mask).bool()
        rand = torch.rand(batch_size, 1, device=signal_mask.device)
        drop_signal = both_present & (rand < self.modality_drop_p)
        drop_image = both_present & (~drop_signal) & (rand < 2 * self.modality_drop_p)
        signal_mask = signal_mask * (~drop_signal).float()
        image_mask = image_mask * (~drop_image).float()
        return signal_mask, image_mask

    def forward(self, signal: torch.Tensor, image: torch.Tensor,
                mask: torch.Tensor, metadata: Optional[torch.Tensor] = None):
        if metadata is None:
            metadata = torch.zeros(signal.size(0), self.metadata_dim,
                                   device=signal.device, dtype=signal.dtype)

        signal_mask = mask[:, 0:1]
        image_mask = mask[:, 1:2]
        signal_mask, image_mask = self._apply_modality_dropout(signal_mask, image_mask)

        signal_feature_map = self.signal_branch.extract_feature_map(signal) * signal_mask.unsqueeze(-1)
        image_feature_map = self.image_branch.forward_features(image) * image_mask.unsqueeze(-1).unsqueeze(-1)

        signal_tokens = self.signal_branch.encode_tokens(signal_feature_map)
        image_tokens = image_feature_map.flatten(2).transpose(1, 2)

        signal_ctx, image_ctx = self.co_attention(signal_tokens, image_tokens)
        signal_ctx_emb = self.signal_context_proj(signal_ctx.mean(dim=1))
        image_ctx_emb = self.image_context_proj(image_ctx.mean(dim=1))

        intermediate = self.intermediate_fusion(torch.cat([signal_ctx_emb, image_ctx_emb], dim=-1))

        signal_global = self.signal_branch(signal) * signal_mask
        image_global = self.image_branch(image) * image_mask
        metadata_emb = self.metadata_encoder(metadata)

        fused = self.final_fusion(torch.cat([
            intermediate,
            signal_global,
            image_global,
            metadata_emb,
        ], dim=-1))

        logits = self.fusion_head(fused)

        if self.training:
            aux_1d = self.aux_head_1d(signal_global)
            aux_2d = self.aux_head_2d(image_global)
            out_cd = self.cd_head(fused)
            out_mi = self.mi_head(signal_global)
            return logits, aux_1d, aux_2d, out_cd, out_mi, fused

        return logits

    def predict_proba(self, signal, image, mask, metadata=None):
        """Pour inférence : probabilités sigmoid multi-label."""
        logits = self.forward(signal, image, mask, metadata=metadata)
        return torch.sigmoid(logits)
