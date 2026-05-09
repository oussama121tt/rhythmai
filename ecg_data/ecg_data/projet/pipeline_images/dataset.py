#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dataset image-only pour classification ECG via scalogrammes CWT."""

import os
import logging

import numpy as np
import pandas as pd
import wfdb
from tqdm import tqdm

import torch
from torch.utils.data import Dataset

from common.utils import parse_scp, extract_label, extract_multilabel, build_metadata_vector, SIGNAL_LENGTH
from common.time_frequency import build_ecg_scalogram


# =========================================================================
# Chargement des métadonnées (CSV → DataFrame filtré)
# =========================================================================

def load_image_dataset(csv_path: str, signals_dir: str,
                       logger: logging.Logger) -> pd.DataFrame:
    """
    Charge le dataset depuis le CSV et prépare les signaux pour scalogrammes.

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

        signal_path = os.path.join(signals_dir, row['filename_hr'])
        signal_header = signal_path + '.hea'
        if not os.path.exists(signal_header):
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
            'signal_path': signal_path,
            'metadata': metadata,
        })

    result_df = pd.DataFrame(records)

    logger.info(f"  Signaux manquants: {missing_signals}")
    logger.info(f"  Sans label valide: {no_label}")
    logger.info(f"  Enregistrements valides: {len(result_df)}")

    return result_df


# =========================================================================
# Dataset PyTorch — images scalogrammes
# =========================================================================

class ECGImageDataset(Dataset):
    """Dataset pour scalogrammes ECG CWT avec augmentation légère."""

    def __init__(self, df: pd.DataFrame, augment: bool = False):
        self.df = df.reset_index(drop=True)
        self.augment = augment
        self.output_size = 300

    def __len__(self):
        return len(self.df)

    def _augment_tensor(self, image: torch.Tensor) -> torch.Tensor:
        if not self.augment:
            return image
        if torch.rand(1).item() < 0.5:
            image = torch.flip(image, dims=[2])
        if torch.rand(1).item() < 0.4:
            noise = torch.randn_like(image) * 0.03
            image = torch.clamp(image + noise, 0.0, 1.0)
        return image

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        signal, _ = wfdb.rdsamp(row['signal_path'])
        signal = signal.T
        if signal.shape[1] < SIGNAL_LENGTH:
            pad = np.zeros((signal.shape[0], SIGNAL_LENGTH - signal.shape[1]))
            signal = np.concatenate([signal, pad], axis=1)
        else:
            signal = signal[:, :SIGNAL_LENGTH]

        image = build_ecg_scalogram(signal, output_size=self.output_size)
        image = self._augment_tensor(image)

        label = torch.tensor(row['target'], dtype=torch.float32)
        return image, label
