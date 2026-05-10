#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Explicabilité pour ECG: Grad-CAM sur image et Integrated Gradients sur signal."""

from __future__ import annotations

from typing import Callable, Tuple, Optional

import torch
import torch.nn.functional as F


def generate_gradcam(model: torch.nn.Module,
                     feature_maps: torch.Tensor,
                     logits: torch.Tensor,
                     target_class: int) -> torch.Tensor:
    """Calcule une carte Grad-CAM à partir d'une carte de features 2D.

    `feature_maps` doit être la dernière carte convolutionnelle avant pooling.
    """
    score = logits[:, target_class].sum()
    gradients = torch.autograd.grad(score, feature_maps, retain_graph=True, create_graph=False)[0]
    weights = gradients.mean(dim=(2, 3), keepdim=True)
    cam = (weights * feature_maps).sum(dim=1, keepdim=True)
    cam = F.relu(cam)
    cam = F.interpolate(cam, size=feature_maps.shape[-2:], mode='bilinear', align_corners=False)
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    return cam.squeeze(1)


def _interpolate_baseline(inputs: torch.Tensor, baseline: torch.Tensor, steps: int):
    for alpha in torch.linspace(0, 1, steps, device=inputs.device):
        yield baseline + alpha * (inputs - baseline)


def integrated_gradients_signal(model: torch.nn.Module,
                                inputs: torch.Tensor,
                                target_class: int,
                                baseline: Optional[torch.Tensor] = None,
                                steps: int = 32) -> torch.Tensor:
    """Integrated Gradients pour un signal ECG 1D.

    Retourne une carte de saillance temporelle de forme (B, T).
    """
    if baseline is None:
        baseline = torch.zeros_like(inputs)

    inputs = inputs.requires_grad_(True)
    total_gradients = torch.zeros_like(inputs)

    for interpolated in _interpolate_baseline(inputs, baseline, steps):
        interpolated = interpolated.requires_grad_(True)
        logits = model(interpolated)
        score = logits[:, target_class].sum()
        gradients = torch.autograd.grad(score, interpolated, retain_graph=False, create_graph=False)[0]
        total_gradients += gradients

    avg_gradients = total_gradients / float(steps)
    attributions = (inputs - baseline) * avg_gradients
    return attributions.sum(dim=1)
