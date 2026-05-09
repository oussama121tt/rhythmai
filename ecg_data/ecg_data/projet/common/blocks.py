#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Blocs architecturaux réutilisables : SE-Block 1D, Attention Pooling, ResBlock 1D.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock1D(nn.Module):
    """Squeeze-and-Excitation block pour pondération des canaux (leads)."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        reduced = max(channels // reduction, 4)
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _ = x.shape
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1)
        return x * y


class AttentionPool1D(nn.Module):
    """Self-attention weighted pooling — pondère les segments informatifs."""

    def __init__(self, channels: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(channels, channels // 4),
            nn.Tanh(),
            nn.Linear(channels // 4, 1, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_t = x.permute(0, 2, 1)           # (B, T, C)
        attn = self.attention(x_t)          # (B, T, 1)
        attn = F.softmax(attn, dim=1)       # (B, T, 1)
        out = (x_t * attn).sum(dim=1)       # (B, C)
        return out


class ResBlock1D(nn.Module):
    """Bloc résiduel 1D avec Squeeze-and-Excitation."""

    def __init__(self, in_ch, out_ch, kernel_size=7, stride=1,
                 downsample=None):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, stride, pad,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, 1, pad,
                               bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dropout = nn.Dropout(0.1)
        self.se = SEBlock1D(out_ch)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class LeadWiseAttention(nn.Module):
    """Pondère dynamiquement les leads via une attention de type canal."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.attention(x).unsqueeze(-1)
        return x * weights


class TemporalTransformerEncoder(nn.Module):
    """Petit encodeur Transformer pour les dépendances temporelles longues."""

    def __init__(self, d_model: int, num_layers: int = 1,
                 nhead: int = 8, dim_feedforward: int = 1024,
                 dropout: float = 0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class CrossModalCoAttention(nn.Module):
    """Co-attention signal-image avec attention bidirectionnelle."""

    def __init__(self, signal_dim: int, image_dim: int, embed_dim: int = 256,
                 num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.signal_q = nn.Linear(signal_dim, embed_dim)
        self.signal_k = nn.Linear(signal_dim, embed_dim)
        self.signal_v = nn.Linear(signal_dim, embed_dim)
        self.image_q = nn.Linear(image_dim, embed_dim)
        self.image_k = nn.Linear(image_dim, embed_dim)
        self.image_v = nn.Linear(image_dim, embed_dim)

        self.signal_to_image = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True)
        self.image_to_signal = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True)

        self.signal_out = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
        )
        self.image_out = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
        )

    def forward(self, signal_tokens: torch.Tensor,
                image_tokens: torch.Tensor):
        sig_q = self.signal_q(signal_tokens)
        sig_k = self.signal_k(signal_tokens)
        sig_v = self.signal_v(signal_tokens)
        img_q = self.image_q(image_tokens)
        img_k = self.image_k(image_tokens)
        img_v = self.image_v(image_tokens)

        sig_ctx, _ = self.signal_to_image(sig_q, img_k, img_v)
        img_ctx, _ = self.image_to_signal(img_q, sig_k, sig_v)

        sig_ctx = self.signal_out(sig_ctx + sig_q)
        img_ctx = self.image_out(img_ctx + img_q)
        return sig_ctx, img_ctx
