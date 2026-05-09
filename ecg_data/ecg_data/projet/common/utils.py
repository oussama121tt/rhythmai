#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Constantes et utilitaires partagés entre les 3 pipelines.
"""

import os
import random
import yaml
import logging
import ast
import warnings
from typing import Dict, Optional, Iterable

import numpy as np
import torch

warnings.filterwarnings('ignore')

# =============================================================================
# CONSTANTES — 5 CLASSES CIBLES
# =============================================================================

CLASS_NAMES = ['NORM', 'MI', 'STTC', 'CD', 'ARR']
NUM_CLASSES = 5
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {i: c for i, c in enumerate(CLASS_NAMES)}

# Priorité clinique (multi-label → single-label)
CLASS_PRIORITY = ['MI', 'ARR', 'CD', 'STTC', 'NORM']

# Mapping des codes SCP vers les 5 classes
SCP_TO_CLASS = {
    # NORM
    "NORM": "NORM", "SR": "NORM",
    # MI
    "IMI": "MI", "ASMI": "MI", "AMI": "MI", "ALMI": "MI", "LMI": "MI",
    "ILMI": "MI", "IPLMI": "MI", "IPMI": "MI", "PMI": "MI",
    "INJAS": "MI", "INJAL": "MI", "INJIN": "MI", "INJLA": "MI", "INJIL": "MI",
    "QWAVE": "MI",
    # STTC
    "NDT": "STTC", "NST_": "STTC", "DIG": "STTC", "LNGQT": "STTC",
    "ISC_": "STTC", "ISCAL": "STTC", "ISCIN": "STTC", "ISCIL": "STTC",
    "ISCAS": "STTC", "ISCLA": "STTC", "APTS": "STTC", "STD_": "STTC",
    "STE_": "STTC", "STTC": "STTC", "INVT": "STTC",
    # CD
    "LAFB": "CD", "LPFB": "CD", "IRBBB": "CD", "CRBBB": "CD", "CLBBB": "CD",
    "ILBBB": "CD", "WPW": "CD", "1AVB": "CD", "2AVB": "CD", "3AVB": "CD",
    "IVCD": "CD", "AVB": "CD", "LBBB": "CD", "RBBB": "CD", "CD": "CD",
    # ARR
    "AFIB": "ARR", "AFLT": "ARR", "STACH": "ARR", "SBRAD": "ARR",
    "SARRH": "ARR", "SVTAC": "ARR", "PSVT": "ARR", "BIGU": "ARR",
    "TRIGU": "ARR", "PAC": "ARR", "PVC": "ARR", "PACE": "ARR",
    "SVARR": "ARR", "ARR": "ARR",
}

# Signal config
SIGNAL_LENGTH = 5000   # 500Hz × 10s
NUM_LEADS = 12
SAMPLE_RATE = 500


# =============================================================================
# REPRODUCTIBILITÉ
# =============================================================================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# LOGGING
# =============================================================================

def setup_logging(log_file: str, logger_name: str = "ECG") -> logging.Logger:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.handlers = []
    fmt = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def load_config(path: str) -> Dict:
    with open(path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Resolve relative paths from the config file's directory
    config_dir = os.path.dirname(os.path.abspath(path))
    if 'paths' in config:
        for key, val in config['paths'].items():
            if isinstance(val, str) and not os.path.isabs(val):
                config['paths'][key] = os.path.join(config_dir, val)

    return config


# =============================================================================
# LABEL EXTRACTION
# =============================================================================

def parse_scp(scp_str: str) -> Dict[str, float]:
    try:
        return ast.literal_eval(scp_str)
    except Exception:
        return {}


def extract_multilabel(scp_codes: Dict[str, float],
                       threshold: float = 50.0) -> np.ndarray:
    """Retourne une cible multi-label binaire de taille 5."""
    target = np.zeros(NUM_CLASSES, dtype=np.float32)
    for code, likelihood in scp_codes.items():
        if likelihood < threshold:
            continue
        mapped_class = SCP_TO_CLASS.get(code)
        if mapped_class is None:
            continue
        target[CLASS_TO_IDX[mapped_class]] = 1.0
    return target


def extract_label(scp_codes: Dict[str, float]) -> Optional[str]:
    """
    Extrait le label (1 des 5 classes) depuis les codes SCP.
    Utilise la priorité clinique si multi-label.
    Seuil de confiance : 50%.
    """
    scores = {c: 0.0 for c in CLASS_NAMES}
    for code, likelihood in scp_codes.items():
        if code in SCP_TO_CLASS:
            cls = SCP_TO_CLASS[code]
            scores[cls] = max(scores[cls], likelihood)
    best_cls, best_score = None, 0.0
    for cls in CLASS_PRIORITY:
        if scores[cls] > best_score:
            best_score = scores[cls]
            best_cls = cls
    return best_cls if best_score >= 50.0 else None


def normalize_age(age: Optional[float]) -> float:
    """Normalise l'âge en [0, 1] sur une plage clinique usuelle."""
    if age is None:
        return 0.0
    try:
        age_val = float(age)
    except Exception:
        return 0.0
    age_val = max(0.0, min(age_val, 100.0))
    return age_val / 100.0


def encode_sex(sex: Optional[object]) -> np.ndarray:
    """Encode le sexe en vecteur binaire [female, male]."""
    if sex is None:
        return np.array([0.0, 0.0], dtype=np.float32)
    sex_str = str(sex).strip().lower()
    if sex_str in {'1', 'm', 'male', 'man'}:
        return np.array([0.0, 1.0], dtype=np.float32)
    if sex_str in {'0', 'f', 'female', 'woman'}:
        return np.array([1.0, 0.0], dtype=np.float32)
    return np.array([0.0, 0.0], dtype=np.float32)


def build_metadata_vector(age: Optional[float], sex: Optional[object]) -> np.ndarray:
    """Construit le vecteur métadonnées [age_norm, sex_f, sex_m]."""
    return np.concatenate([
        np.array([normalize_age(age)], dtype=np.float32),
        encode_sex(sex),
    ])


def multilabel_counts(targets: Iterable[np.ndarray]) -> np.ndarray:
    """Compte le nombre d'occurrences positives par classe."""
    stacked = np.asarray(list(targets), dtype=np.float32)
    if stacked.size == 0:
        return np.zeros(NUM_CLASSES, dtype=np.int64)
    return stacked.sum(axis=0).astype(np.int64)
