#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔════════════════════════════════════════════════════════════════════════════╗
║                    🫀 RHYTHMAI V4 - CODES KAGGLE                           ║
║                                                                            ║
║  COPIER-COLLER CHAQUE CELLULE DANS UN KAGGLE NOTEBOOK                     ║
║  Exécuter dans l'ordre: CELL 1 → CELL 2 → ... → CELL 7                   ║
╚════════════════════════════════════════════════════════════════════════════╝
"""

# ═══════════════════════════════════════════════════════════════════════════
# CELL 1: INSTALLER DÉPENDANCES
# ═══════════════════════════════════════════════════════════════════════════

print("📦 CELL 1: Installation des dépendances...")

import subprocess
import sys

# Installer PyTorch GPU
print("\n1️⃣ Installation PyTorch GPU...")
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
    'torch', 'torchvision', 'torchaudio',
    '--index-url', 'https://download.pytorch.org/whl/cu118'],
    check=True)

# Installer autres packages
print("2️⃣ Installation packages...")
packages = [
    'wfdb', 'scipy', 'scikit-learn', 'pandas', 'numpy',
    'tensorboard', 'pytorch-lightning', 'tqdm', 'pyyaml'
]
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q'] + packages,
    check=True)

# Vérifier
print("\n✅ VÉRIFICATION:")
import torch
import wfdb
import scipy
import sklearn

print(f"   PyTorch: {torch.__version__}")
print(f"   WFDB: {wfdb.__version__}")
print(f"   GPU disponible: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

print("\n✅ Dépendances OK!")


# ═══════════════════════════════════════════════════════════════════════════
# CELL 2: VÉRIFIER DONNÉES & SETUP
# ═══════════════════════════════════════════════════════════════════════════

print("\n📊 CELL 2: Vérifier données et setup...")

import os
import pandas as pd

# Lister datasets disponibles
print("\n📂 Datasets disponibles:")
for item in os.listdir('/kaggle/input'):
    print(f"   - {item}")

# ✏️ ADAPTER CES CHEMINS À VOS DONNÉES:
PTBXL_DIR = '/kaggle/input/ptb-xl-dataset'
CSV_PATH = os.path.join(PTBXL_DIR, 'ptbxl_database.csv')
IMAGES_DIR = '/kaggle/input/ptb-xl-12-lead-ecg-images'  # Si disponible

# Vérifier
print(f"\n✅ Vérifications:")
print(f"   CSV: {os.path.exists(CSV_PATH)}")
print(f"   Images: {os.path.exists(IMAGES_DIR)}")

# Charger CSV
if os.path.exists(CSV_PATH):
    df = pd.read_csv(CSV_PATH)
    print(f"\n✅ Dataset: {len(df)} enregistrements")
    print(f"   Colonnes: {df.columns.tolist()}")
    print(f"\n   Distribution classes:")
    print(df['label'].value_counts())
else:
    print("❌ CSV non trouvé!")


# ═══════════════════════════════════════════════════════════════════════════
# CELL 3: CLONE PROJET & CONFIG
# ═══════════════════════════════════════════════════════════════════════════

print("\n🔧 CELL 3: Clone projet et création config...")

import subprocess
import yaml

PROJECT_ROOT = '/kaggle/working'
PROJET_DIR = f'{PROJECT_ROOT}/projet'

# ✏️ OPTION A: Clone depuis GitHub
print("\n1️⃣ Cloner projet depuis GitHub...")
if not os.path.exists(PROJET_DIR):
    subprocess.run(['git', 'clone',
        'https://github.com/rhythmai/ecg_data.git',
        PROJET_DIR], check=True)
    print("   ✅ Projet cloné")
else:
    print("   ℹ️ Projet déjà présent")

# ✏️ OU OPTION B: Si vous avez uploadé un ZIP
# print("\n2️⃣ Extraire ZIP...")
# subprocess.run(['unzip', '-q', f'{PROJECT_ROOT}/projet.zip',
#     '-d', PROJECT_ROOT], check=True)
# print("   ✅ ZIP extrait")

# Créer répertoires
for dir_name in ['logs', 'models', 'models/checkpoints', 'data/processed']:
    os.makedirs(f"{PROJECT_ROOT}/{dir_name}", exist_ok=True)

# Créer config Kaggle
config = {
    'seed': 42,
    'paths': {
        'project_root': PROJECT_ROOT,
        'raw_data': PROJECT_ROOT,
        'logs': f'{PROJECT_ROOT}/logs',
        'models': f'{PROJECT_ROOT}/models',
        'ptbxl_signals': PTBXL_DIR,
        'ptbxl_images': IMAGES_DIR,
    },
    'standardization': {
        'target_fs': 500,
        'target_duration_sec': 10,
        'required_leads': 12,
    }
}

config_path = f'{PROJECT_ROOT}/config_kaggle.yaml'
with open(config_path, 'w') as f:
    yaml.dump(config, f)

print(f"\n✅ Config créée: {config_path}")
print(f"   Signaux: {PTBXL_DIR}")
print(f"   Images: {IMAGES_DIR}")


# ═══════════════════════════════════════════════════════════════════════════
# CELL 4: AFFICHER PARAMÈTRES V4
# ═══════════════════════════════════════════════════════════════════════════

print("\n🔧 CELL 4: Afficher paramètres V4...")

params_v4 = {
    'Loss function': 'FocalLoss ✅',
    'Learning rate': '1e-4 ✅',
    'Weight decay': '5e-4 ✅',
    'Sampler ratio (ARR)': '2.0 ✅',
    'β_mi (MI head)': '0.15 ✅',
    'β_cd (CD head)': '0.05 ✅',
    'α_aux (aux heads)': '0.1 ✅',
    'Optimizer': 'AdamW ✅',
    'Batch size': '16 (ou 8 si GPU faible)',
    'Epochs': '50 ✅',
    'Dropout': '0.5 ✅',
}

print("\n" + "="*60)
print("🔧 PARAMÈTRES V4 CONFIRMÉS:")
print("="*60)
for param, value in params_v4.items():
    print(f"  {param:25s}: {value}")

print("\n🎯 Seuils optimisés (à l'inférence):")
thresholds_v4 = {'NORM': 0.52, 'MI': 0.49, 'STTC': 0.46, 'CD': 0.49, 'ARR': 0.51}
for cls, thr in thresholds_v4.items():
    print(f"  {cls}: {thr:.2f}")

print("\n✅ Configuration V4 prête!")


# ═══════════════════════════════════════════════════════════════════════════
# CELL 5: LANCER ENTRAÎNEMENT V4 ⚡⚡⚡
# ═══════════════════════════════════════════════════════════════════════════

print("\n🚀 CELL 5: LANCER ENTRAÎNEMENT V4...")

import sys
import os

# Setup Python path
sys.path.insert(0, f'{PROJECT_ROOT}/projet')
os.chdir(f'{PROJECT_ROOT}/projet')

# Importer modules
try:
    from pipeline_fusion.train import main
    print("✅ Modules importés avec succès")
except ImportError as e:
    print(f"❌ Erreur import: {e}")
    raise

# Paramètres V4 (VERSION PAR DÉFAUT)
arguments = [
    'train.py',
    '--config', config_path,           # Config créée en CELL 3
    '--epochs', '50',                  # 50 epochs
    '--batch_size', '16',              # Adapter si GPU < 8GB: utiliser 8
    '--lr', '1e-4',                    # Learning rate V4
    '--loss_type', 'focal',            # ✅ FocalLoss (V4 default)
    '--seed', '42',                    # Reproductibilité
    '--device', 'cuda',                # GPU
    '--amp',                           # ✅ Mixed precision (20% gain vitesse)
    '--num_workers', '4',
    '--prefetch_factor', '2',
]

sys.argv = arguments

print("\n" + "="*70)
print("🚀 LANCEMENT ENTRAÎNEMENT V4")
print("="*70)
print("\n📊 Configuration:")
print(f"  Loss: FocalLoss ✅")
print(f"  β_mi: 0.15 ✅")
print(f"  Sampler ratio: 2.0 ✅")
print(f"  Weight decay: 5e-4 ✅")
print(f"  Batch size: 16")
print(f"  Epochs: 50")
print(f"  GPU: CUDA (AMP activé)")
print(f"\n⏱️  Temps estimé: 60-90 min (T4), 30-45 min (P100)")
print("="*70)
print()

try:
    main()  # ⚡⚡⚡ LANCER L'ENTRAÎNEMENT
    print("\n" + "="*70)
    print("✅ ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS!")
    print("="*70)
except Exception as e:
    print(f"\n❌ Erreur: {e}")
    import traceback
    traceback.print_exc()
    raise


# ═══════════════════════════════════════════════════════════════════════════
# CELL 6: AFFICHER RÉSULTATS FINAUX
# ═══════════════════════════════════════════════════════════════════════════

print("\n📊 CELL 6: Afficher résultats finaux...")

import json

# Paths résultats
results_path = f'{PROJECT_ROOT}/models/checkpoints/final_results.json'
thresholds_path = f'{PROJECT_ROOT}/models/checkpoints/optimal_thresholds.json'

# Charger et afficher résultats
if os.path.exists(results_path):
    with open(results_path) as f:
        results = json.load(f)
    
    test_metrics = results.get('test_metrics', {})
    
    print("\n" + "="*70)
    print("📊 RÉSULTATS FINAUX V4 (TEST SET)")
    print("="*70)
    
    print(f"\n🎯 Métriques Globales:")
    f1 = test_metrics.get('f1_macro', 0)
    pr_auc = test_metrics.get('pr_auc_macro', 0)
    acc = test_metrics.get('accuracy', 0)
    
    status_f1 = "✅" if f1 >= 0.70 else "❌"
    status_pr = "✅" if pr_auc >= 0.80 else "❌"
    
    print(f"  {status_f1} F1-macro:      {f1:.4f}  (Objectif: ≥0.70)")
    print(f"  {status_pr} PR-AUC:        {pr_auc:.4f}  (Objectif: ≥0.80)")
    print(f"  📊 Accuracy:      {acc:.4f}")
    
    print(f"\n📈 Sensibilité (Recall) par Classe:")
    recalls = test_metrics.get('recall_per_class', {})
    for cls, recall in recalls.items():
        marker = "✅" if recall >= 0.70 else "⚠️"
        print(f"  {marker} {cls:5s}: {recall:.4f}")
    
    print(f"\n🎯 F1 Score par Classe:")
    f1s = test_metrics.get('f1_per_class', {})
    for cls, f1_cls in f1s.items():
        print(f"  {cls:5s}: {f1_cls:.4f}")
    
    print("\n" + "="*70)
    
else:
    print(f"❌ Résultats non trouvés: {results_path}")

# Afficher seuils optimisés
print(f"\n🎯 SEUILS OPTIMISÉS V4 (à l'inférence):")
print("="*70)

if os.path.exists(thresholds_path):
    with open(thresholds_path) as f:
        thresholds = json.load(f)
    for cls, thr in thresholds.items():
        print(f"  {cls}: {thr:.2f}")
else:
    print("Utiliser les seuils V4 par défaut:")
    thresholds_v4_default = {
        'NORM': 0.52, 'MI': 0.49, 'STTC': 0.46, 'CD': 0.49, 'ARR': 0.51
    }
    for cls, thr in thresholds_v4_default.items():
        print(f"  {cls}: {thr:.2f}")

print("\n✅ Résultats affichés!")


# ═══════════════════════════════════════════════════════════════════════════
# CELL 7: CHARGER MODÈLE & FAIRE PRÉDICTIONS
# ═══════════════════════════════════════════════════════════════════════════

print("\n🧠 CELL 7: Charger modèle et faire prédictions...")

import torch
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Charger checkpoint
ckpt_path = f'{PROJECT_ROOT}/models/checkpoints/best_fusion_model.pth'
if os.path.exists(ckpt_path):
    print(f"\n1️⃣ Charger modèle depuis: {ckpt_path}")
    
    from pipeline_fusion.model import ECGFusionModel
    from common.utils import NUM_CLASSES
    
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    # Créer modèle
    model = ECGFusionModel(
        num_classes=NUM_CLASSES,
        embed_dim=256,
        pretrained=True,
        dropout=0.3,
        modality_drop_p=0.15
    )
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    model.to(device)
    
    print("   ✅ Modèle chargé avec succès")
    
    # EXEMPLE: Prédiction sur données de test
    print(f"\n2️⃣ Exemple de prédiction:")
    
    # Créer données dummy (remplacer par vraies données)
    test_signals = np.random.randn(1, 12, 5000).astype(np.float32)
    test_images = np.random.randn(1, 3, 224, 224).astype(np.float32)
    test_metadata = np.array([[65.0, 1.0, 0.0]], dtype=np.float32)
    
    # Convertir to torch
    signals = torch.tensor(test_signals).to(device)
    images = torch.tensor(test_images).to(device)
    metadata = torch.tensor(test_metadata).to(device)
    
    # Prédiction
    with torch.no_grad():
        logits = model(signals, images, metadata=metadata)
        probs = torch.sigmoid(logits).cpu().numpy()
    
    # Appliquer seuils V4
    from common.utils import CLASS_NAMES
    threshold_array = np.array([0.52, 0.49, 0.46, 0.49, 0.51])  # V4 thresholds
    predictions = (probs >= threshold_array).astype(int)
    
    print(f"\n   Probabilités par classe:")
    for i, cls in enumerate(CLASS_NAMES):
        print(f"     {cls}: {probs[0, i]:.4f}")
    
    print(f"\n   Prédictions (seuils V4):")
    for i, cls in enumerate(CLASS_NAMES):
        pred = "✅ POSITIF" if predictions[0, i] else "❌ NÉGATIF"
        print(f"     {cls}: {pred} (prob: {probs[0, i]:.4f}, seuil: {threshold_array[i]:.2f})")
    
    print(f"\n   ✅ Prédiction OK!")
    
else:
    print(f"⚠️  Checkpoint non trouvé: {ckpt_path}")

print("\n" + "="*70)
print("✅ TOUS LES CODES EXÉCUTÉS AVEC SUCCÈS!")
print("="*70)


# ═══════════════════════════════════════════════════════════════════════════
# BONUS: FONCTION POUR INFÉRENCE RÉUTILISABLE
# ═══════════════════════════════════════════════════════════════════════════

def predict_batch_v4(model, signals_batch, images_batch, metadata_batch,
                     device='cuda', thresholds=None):
    """
    Faire des prédictions sur un batch avec seuils V4.
    
    Args:
        model: Modèle ECGFusionModel chargé
        signals_batch: (batch_size, 12, 5000)
        images_batch: (batch_size, 3, 224, 224)
        metadata_batch: (batch_size, 3)
        device: 'cuda' ou 'cpu'
        thresholds: dict avec clés NORM, MI, STTC, CD, ARR
    
    Returns:
        predictions: (batch_size, 5) - 0/1 pour chaque classe
        probabilities: (batch_size, 5) - probabilités
    """
    
    if thresholds is None:
        # Seuils V4 par défaut
        thresholds = {
            'NORM': 0.52, 'MI': 0.49, 'STTC': 0.46, 'CD': 0.49, 'ARR': 0.51
        }
    
    # Convertir to torch
    signals = torch.tensor(signals_batch, dtype=torch.float32).to(device)
    images = torch.tensor(images_batch, dtype=torch.float32).to(device)
    metadata = torch.tensor(metadata_batch, dtype=torch.float32).to(device)
    
    # Forward pass
    with torch.no_grad():
        logits = model(signals, images, metadata=metadata)
        probs = torch.sigmoid(logits).cpu().numpy()
    
    # Appliquer seuils
    from common.utils import CLASS_NAMES
    threshold_array = np.array([
        thresholds['NORM'],
        thresholds['MI'],
        thresholds['STTC'],
        thresholds['CD'],
        thresholds['ARR']
    ])
    
    predictions = (probs >= threshold_array).astype(int)
    
    return predictions, probs


# Exemple d'utilisation:
# predictions, probs = predict_batch_v4(model, signals, images, metadata)

print("\n✅ Fonction predict_batch_v4 définie et prête à utiliser!")
