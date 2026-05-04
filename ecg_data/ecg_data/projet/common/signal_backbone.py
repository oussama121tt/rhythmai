#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Branche signal 1D : ResNet1D + SE-blocks + Attention Pooling.
Partagée entre pipeline_signals et pipeline_fusion.
"""

import torch
import torch.nn as nn

from .utils import NUM_LEADS
from .blocks import ResBlock1D, AttentionPool1D, LeadWiseAttention, TemporalTransformerEncoder


class ECGResNet1D(nn.Module):
    """
    ResNet1D pour signaux ECG 12-lead.
      ▸ SE-blocks pour attention par lead
      ▸ Progressive kernel sizes (7→7→11→15) pour champ réceptif
      ▸ Dual pooling : attention + global average
    """

    def __init__(self, embed_dim: int = 256, transformer_layers: int = 1,
                 transformer_heads: int = 8, max_tokens: int = 256):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(NUM_LEADS, 64, kernel_size=15, stride=2, padding=7,
                      bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(3, stride=2, padding=1),
        )

        self.lead_attention = LeadWiseAttention(64)

        self.layer1 = self._make_layer(64, 64, n_blocks=2, stride=1,
                                       kernel_size=7)
        self.layer2 = self._make_layer(64, 128, n_blocks=2, stride=2,
                                       kernel_size=7)
        self.layer3 = self._make_layer(128, 256, n_blocks=2, stride=2,
                                       kernel_size=11)
        self.layer4 = self._make_layer(256, 512, n_blocks=2, stride=2,
                                       kernel_size=15)

        self.positional_encoding = nn.Parameter(
            torch.zeros(1, max_tokens, 512))
        self.temporal_encoder = TemporalTransformerEncoder(
            d_model=512,
            num_layers=transformer_layers,
            nhead=transformer_heads,
            dim_feedforward=1024,
            dropout=0.1,
        )

        self.attn_pool = AttentionPool1D(512)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Sequential(
            nn.Linear(512 * 2, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
        )

    def _make_layer(self, in_ch, out_ch, n_blocks, stride, kernel_size=7):
        downsample = None
        if stride != 1 or in_ch != out_ch:
            downsample = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )
        layers = [ResBlock1D(in_ch, out_ch, kernel_size=kernel_size,
                             stride=stride, downsample=downsample)]
        for _ in range(1, n_blocks):
            layers.append(ResBlock1D(out_ch, out_ch, kernel_size=kernel_size))
        return nn.Sequential(*layers)

    def extract_feature_map(self, x):
        """Retourne la carte de features 1D avant l'encodeur temporel."""
        x = self.stem(x)
        x = self.lead_attention(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def encode_tokens(self, feature_map):
        """Projette la carte 1D en tokens temporels contextualisés."""
        tokens = feature_map.permute(0, 2, 1)
        seq_len = tokens.size(1)
        if seq_len <= self.positional_encoding.size(1):
            tokens = tokens + self.positional_encoding[:, :seq_len, :]
        else:
            pos = self.positional_encoding
            repeat_factor = seq_len // pos.size(1) + 1
            tokens = tokens + pos.repeat(1, repeat_factor, 1)[:, :seq_len, :]
        return self.temporal_encoder(tokens)

    def forward(self, x):
        """x: (B, 12, 5000) → (B, embed_dim)"""
        feature_map = self.extract_feature_map(x)
        tokens = self.encode_tokens(feature_map)

        token_features = tokens.permute(0, 2, 1)
        attn_out = self.attn_pool(token_features)
        avg_out = self.avg_pool(token_features).squeeze(-1)
        x = torch.cat([attn_out, avg_out], dim=1)

        return self.fc(x)
