#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset signal-only pour classification ECG (signaux 1D — 12 leads).
"""

import os
import random
import logging

import numpy as np
import pandas as pd
import wfdb
from tqdm import tqdm

import torch
from torch.utils.data import Dataset

from common.utils import (
    CLASS_TO_IDX, SIGNAL_LENGTH, NUM_LEADS, SAMPLE_RATE,
    parse_scp, extract_label, extract_multilabel, build_metadata_vector,
)
from common.preprocessing import preprocess_ecg_signal


# =========================================================================
# Chargement des métadonnées (CSV → DataFrame filtré)
# =========================================================================

def load_signal_dataset(csv_path: str, signals_dir: str,
                        logger: logging.Logger) -> pd.DataFrame:
    """
    Charge le dataset depuis le CSV et vérifie les fichiers signal.

    Returns:
        DataFrame avec colonnes: ecg_id, patient_id, label, target,
                                 strat_fold, signal_path, metadata
    """
    logger.info(f"Chargement du CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info(f"  {len(df)} enregistrements dans le CSV")

    records = []
    missing_signals = 0
    no_label = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        ecg_id = int(row['ecg_id'])

        # Chemin du signal 500Hz
        sig_path = os.path.join(signals_dir, row['filename_hr'])
        sig_hea = sig_path + '.hea'

        if not os.path.exists(sig_hea):
            missing_signals += 1
            continue

        scp_codes = parse_scp(row['scp_codes'])
        label = extract_label(scp_codes)
        target = extract_multilabel(scp_codes)

        if label is None:
            no_label += 1
            continue

        metadata = build_metadata_vector(row.get('age'), row.get('sex'))

        records.append({
            'ecg_id': ecg_id,
            'patient_id': row['patient_id'],
            'label': label,
            'target': target,
            'strat_fold': int(row['strat_fold']),
            'signal_path': sig_path,
            'metadata': metadata,
        })

    result_df = pd.DataFrame(records)

    logger.info(f"  Signaux manquants: {missing_signals}")
    logger.info(f"  Sans label valide: {no_label}")
    logger.info(f"  Enregistrements valides: {len(result_df)}")

    return result_df


# =========================================================================
# Dataset PyTorch — Signaux uniquement
# =========================================================================

class ECGSignalDataset(Dataset):
    """Dataset pour signaux ECG 12-lead avec augmentation."""

    def __init__(self, df: pd.DataFrame, augment: bool = False):
        self.df = df.reset_index(drop=True)
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def _load_signal(self, path: str) -> torch.Tensor:
        """Charge, prétraite et normalise un signal ECG 12-lead."""
        try:
            signal, _ = wfdb.rdsamp(path)  # (5000, 12)
            signal = signal.T  # (12, 5000)

            # Pad/truncate
            if signal.shape[1] < SIGNAL_LENGTH:
                pad = np.zeros((NUM_LEADS, SIGNAL_LENGTH - signal.shape[1]))
                signal = np.concatenate([signal, pad], axis=1)
            else:
                signal = signal[:, :SIGNAL_LENGTH]

            # Prétraitement (HPF 1Hz + Notch 50Hz)
            signal = preprocess_ecg_signal(signal, fs=SAMPLE_RATE)

            # Z-score normalization per lead
            for i in range(NUM_LEADS):
                m = signal[i].mean()
                s = signal[i].std() + 1e-8
                signal[i] = (signal[i] - m) / s

            # Augmentation
            if self.augment:
                signal = self._augment_signal(signal)

            return torch.tensor(signal, dtype=torch.float32)
        except Exception:
            return torch.zeros(NUM_LEADS, SIGNAL_LENGTH, dtype=torch.float32)

    def _augment_signal(self, signal: np.ndarray) -> np.ndarray:
        """Augmentations safe pour signaux ECG."""
        # Gaussian noise
        if random.random() < 0.5:
            noise = np.random.normal(0, 0.05, signal.shape)
            signal = signal + noise

        # Random scaling
        if random.random() < 0.3:
            scale = np.random.uniform(0.9, 1.1)
            signal = signal * scale

        # Baseline wander (sinusoïde basse fréquence)
        if random.random() < 0.3:
            freq = np.random.uniform(0.1, 0.5)
            t = np.linspace(0, 10, SIGNAL_LENGTH)
            wander = 0.1 * np.sin(2 * np.pi * freq * t)
            signal = signal + wander[np.newaxis, :]

        # Lead masking (masquer 1-2 leads aléatoires)
        if random.random() < 0.2:
            n_mask = random.randint(1, 2)
            leads_to_mask = random.sample(range(NUM_LEADS), n_mask)
            for lead_idx in leads_to_mask:
                signal[lead_idx] = 0.0

        # Time masking (masquer un segment temporel)
        if random.random() < 0.2:
            mask_len = random.randint(50, 250)
            start = random.randint(0, SIGNAL_LENGTH - mask_len)
            signal[:, start:start + mask_len] = 0.0

        return signal

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = torch.tensor(row['target'], dtype=torch.float32)
        signal = self._load_signal(row['signal_path'])
        return signal, label
