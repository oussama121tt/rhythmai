#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prétraitement du signal ECG : filtre passe-haut 1Hz + filtre notch 50Hz.
"""

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch

HAS_SCIPY = True  # kept for backward compatibility; scipy is now mandatory

from .utils import SAMPLE_RATE


def preprocess_ecg_signal(signal: np.ndarray, fs: int = SAMPLE_RATE) -> np.ndarray:
    """
    Prétraitement robuste du signal ECG :
      1. Filtre passe-haut 1Hz (supprime dérive de ligne de base)
      2. Filtre notch 50Hz (supprime interférence secteur)
    """
    b_hp, a_hp = butter(4, 1.0, btype='high', fs=fs)
    b_notch, a_notch = iirnotch(50.0, 30.0, fs=fs)

    filtered = np.zeros_like(signal)
    for i in range(signal.shape[0]):
        lead = signal[i].copy()
        if np.std(lead) < 1e-6:
            filtered[i] = lead
            continue
        try:
            lead = filtfilt(b_hp, a_hp, lead)
            lead = filtfilt(b_notch, a_notch, lead)
        except Exception:
            pass
        filtered[i] = lead

    return filtered
