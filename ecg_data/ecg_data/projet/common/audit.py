#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit de split et de normalisation pour ECG."""

from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd


def _patient_overlap(left: pd.DataFrame, right: pd.DataFrame) -> int:
    if 'patient_id' not in left.columns or 'patient_id' not in right.columns:
        return 0
    return len(set(left['patient_id']).intersection(set(right['patient_id'])))


def audit_split_and_normalization(train_df: pd.DataFrame,
                                  val_df: pd.DataFrame,
                                  test_df: pd.DataFrame,
                                  normalization_mode: str = 'per_sample',
                                  train_stats: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, object]:
    """Retourne un rapport simple de risque de fuite de données.

    Le mode idéal pour ce projet est `per_sample`, car il n'exploite aucune
    statistique globale. Si une normalisation globale est utilisée, `train_stats`
    doit être fourni et ne provenir que du split train.
    """
    report: Dict[str, object] = {
        'normalization_mode': normalization_mode,
        'patient_overlap_train_val': _patient_overlap(train_df, val_df),
        'patient_overlap_train_test': _patient_overlap(train_df, test_df),
        'patient_overlap_val_test': _patient_overlap(val_df, test_df),
        'status': 'ok',
        'warnings': [],
    }

    if normalization_mode != 'per_sample' and train_stats is None:
        report['status'] = 'warning'
        report['warnings'].append(
            'Normalisation globale détectée sans statistiques train-only.')

    for key in ('patient_overlap_train_val', 'patient_overlap_train_test', 'patient_overlap_val_test'):
        if int(report[key]) > 0:
            report['status'] = 'error'
            report['warnings'].append(f"Fuite potentielle: {key}={report[key]}")

    if normalization_mode == 'per_sample':
        report['warnings'].append(
            'Normalisation per-sample: aucune statistique globale n\'est partagée entre splits.')

    return report
