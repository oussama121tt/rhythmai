#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Outils temps-fréquence pour scalogrammes CWT ECG."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .utils import NUM_LEADS, SIGNAL_LENGTH, SAMPLE_RATE
from .preprocessing import preprocess_ecg_signal


def _safe_group_mean(signal: np.ndarray, lead_indices: list[int]) -> np.ndarray:
    indices = [idx for idx in lead_indices if 0 <= idx < signal.shape[0]]
    if not indices:
        return signal.mean(axis=0)
    return signal[indices].mean(axis=0)


def _morlet_kernel(kernel_size: int, width: float, w: float = 6.0) -> np.ndarray:
    half = kernel_size // 2
    x = np.linspace(-half, half, kernel_size, dtype=np.float32)
    sigma = max(float(width), 1.0)
    kernel = np.exp(1j * w * x / sigma) * np.exp(-(x ** 2) / (2.0 * sigma ** 2))
    kernel = kernel / (np.sqrt(np.sum(np.abs(kernel) ** 2)) + 1e-8)
    return kernel.astype(np.complex64)


def _ricker_kernel(kernel_size: int, width: float) -> np.ndarray:
    half = kernel_size // 2
    x = np.linspace(-half, half, kernel_size, dtype=np.float32)
    sigma = max(float(width), 1.0)
    xsq = (x / sigma) ** 2
    kernel = (1.0 - xsq) * np.exp(-xsq / 2.0)
    kernel = kernel / (np.sqrt(np.sum(kernel ** 2)) + 1e-8)
    return kernel.astype(np.float32)


def _cwt_map_1d(signal_1d: np.ndarray, wavelet: str, widths: np.ndarray) -> np.ndarray:
    coeffs = []
    for width in widths:
        kernel_size = int(max(8 * width, 16))
        if kernel_size % 2 == 0:
            kernel_size += 1
        if wavelet == "ricker":
            kernel = _ricker_kernel(kernel_size, width)
        else:
            kernel = _morlet_kernel(kernel_size, width, w=6.0)
        conv = np.convolve(signal_1d, np.conjugate(kernel[::-1]), mode="same")
        coeffs.append(np.abs(conv))
    coeffs = np.asarray(coeffs, dtype=np.float32)
    scalogram = np.log1p(np.abs(coeffs))
    scalogram = scalogram - scalogram.min()
    denom = scalogram.max() - scalogram.min() + 1e-8
    scalogram = scalogram / denom
    return scalogram.astype(np.float32)


def build_ecg_scalogram(signal: np.ndarray,
                        fs: int = SAMPLE_RATE,
                        output_size: int = 300,
                        # [ANCIEN] use_ricker: bool = True,
                        wavelet: str = "morlet") -> torch.Tensor:
    """Construit un pseudo-image CWT à 3 canaux à partir d'un ECG 12-lead.

    Canaux:
      1. Morlet sur dérivations précordiales (V1-V6)
      2. Morlet sur dérivations des membres (I, II, III, aVR, aVL, aVF)
      3. Ricker sur le signal global moyen (optionnel)
    """
    if signal.shape[0] != NUM_LEADS:
        raise ValueError(f"Expected {NUM_LEADS} leads, got {signal.shape[0]}")

    signal = preprocess_ecg_signal(signal, fs=fs)
    signal = signal[:, :SIGNAL_LENGTH]

    precordial = _safe_group_mean(signal, [6, 7, 8, 9, 10, 11])
    limb = _safe_group_mean(signal, [0, 1, 2, 3, 4, 5])
    global_mean = signal.mean(axis=0)

    # [ANCIEN] widths = np.arange(1, 65)
    scales = np.arange(1, 65)
    selected_wavelet = "morlet" if wavelet.lower() in {"morlet", "morl"} else "morlet"
    morlet_precordial = _cwt_map_1d(precordial, selected_wavelet, scales)
    morlet_limb = _cwt_map_1d(limb, selected_wavelet, scales)
    # [ANCIEN] ricker_global = _cwt_map_1d(global_mean, "ricker", widths) if use_ricker else morlet_limb
    morlet_global = _cwt_map_1d(global_mean, selected_wavelet, scales)

    maps = [morlet_precordial, morlet_limb, morlet_global]
    resized = []
    for scalogram in maps:
        tensor = torch.tensor(scalogram, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        tensor = F.interpolate(tensor, size=(output_size, output_size), mode="bilinear", align_corners=False)
        resized.append(tensor.squeeze(0))

    return torch.cat(resized, dim=0)
