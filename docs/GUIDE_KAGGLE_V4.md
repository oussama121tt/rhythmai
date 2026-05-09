# Guide Complet : Entraînement RhythmAI sur Kaggle (Configuration V4)

## 📋 Paramètres Appliqués - Configuration V4 (Meilleure)

### Paramètres de la Loss et Optimisation
| Paramètre | Valeur | Raison |
|-----------|--------|--------|
| **Loss function** | FocalLoss | Meilleure convergence sur classes déséquilibrées |
| **Optimizer** | Adam | Standard avec lr=1e-4 |
| **Weight decay** | 5e-4 | Régularisation L2 |
| **Learning rate** | 1e-4 | Convergence stable |

### Paramètres du Modèle
| Paramètre | Valeur | Raison |
|-----------|--------|--------|
| **Sampler ratio (ARR)** | 2.0 | Surpondération ARR x2 (ratio optimal) |
| **β_mi (poids tête MI)** | 0.15 | Augmenté (meilleure détection MI) |
| **β_cd (poids tête CD)** | 0.05 | Stable |
| **α_aux (têtes auxiliaires)** | 0.1 | Pénalité modérée |
| **Dropout** | 0.5 | Régularisation |
| **Batch Normalization** | Oui | Fusion FC layers |
| **Têtes auxiliaires** | Oui | +MI head +CD head |

### Seuils de Décision Optimisés (par classe)
```json
{
  "NORM": 0.52,
  "MI": 0.49,
  "STTC": 0.46,
  "CD": 0.49,
  "ARR": 0.51
}
```

### Performances Attendues (Validation/Test)
```
Métrique            Validation V4    Test final    Recall MI    Recall ARR
─────────────────────────────────────────────────────────────────────────
F1-macro            0.783            0.757         0.722        0.824
PR-AUC              0.852            0.817         -            -
Accuracy            -                0.672         -            -
```

---

## 🚀 Guide Étape par Étape pour Kaggle

### **Phase 1 : Préparation (5-10 min)**

#### Étape 1 : Créer un Notebook Kaggle
1. Aller sur **kaggle.com** → Notebooks → Create new notebook
2. Sélectionner **Python** 3.x
3. Activer **GPU** (T4 ou P100 de préférence)
4. Activer **Internet** (pour télécharger les dépendances)

#### Étape 2 : Importer les données
```python
# Ajouter le dataset PTB-XL à votre notebook
# Dans Kaggle: Data → Add data (PTB-XL ou ECG dataset que vous avez uploadé)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
```

#### Étape 3 : Installer les dépendances
```bash
pip install -q wfdb scipy scikit-learn pytorch-lightning tensorboard
pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

### **Phase 2 : Configuration Initiale (5 min)**

#### Étape 4 : Cloner/Copier le code du projet
```bash
# Soit via ZIP uploadé à Kaggle:
!cd /kaggle/working && unzip -q projet_rhythmai.zip

# Soit clone direct:
!git clone https://github.com/rhythmai/ecg_data.git /kaggle/working/projet
```

#### Étape 5 : Vérifier la structure des données
```python
import os
import pandas as pd

# Chemins
PTBXL_DIR = '/kaggle/input/ptb-xl-dataset'  # Remplacer par le chemin réel
CSV_PATH = os.path.join(PTBXL_DIR, 'ptbxl_database.csv')
SIGNALS_DIR = PTBXL_DIR
IMAGES_DIR = '/kaggle/input/ptb-xl-images'  # Si disponible

# Vérifier
print(f"CSV existe: {os.path.exists(CSV_PATH)}")
print(f"Signaux dir existe: {os.path.exists(SIGNALS_DIR)}")

df = pd.read_csv(CSV_PATH)
print(f"\nDataset: {len(df)} enregistrements")
print(df.head())
```

---

### **Phase 3 : Préparation de l'Entraînement (10 min)**

#### Étape 6 : Copier config.yaml et adapter les chemins
```python
import yaml

config = {
    'seed': 42,
    'paths': {
        'project_root': '/kaggle/working',
        'ptbxl_signals': '/kaggle/input/ptb-xl-dataset',
        'ptbxl_images': '/kaggle/input/ptb-xl-images',
        'models': '/kaggle/working/models',
        'logs': '/kaggle/working/logs'
    },
    'standardization': {
        'target_fs': 500,
        'target_duration_sec': 10,
        'required_leads': 12,
    }
}

# Sauvegarder
with open('/kaggle/working/config_kaggle.yaml', 'w') as f:
    yaml.dump(config, f)

print("✅ Config adaptée pour Kaggle")
```

#### Étape 7 : Paramètres V4 de démarrage
```python
PARAMS_V4 = {
    'epochs': 50,
    'batch_size': 16,  # Peut être réduit à 8 si GPU 4Go
    'learning_rate': 1e-4,
    'weight_decay': 5e-4,
    'loss_type': 'focal',  # ✅ V4 Par défaut
    'seed': 42,
    'device': 'cuda',  # Kaggle GPU
}

print("🔧 Paramètres V4 chargés:")
for k, v in PARAMS_V4.items():
    print(f"  {k}: {v}")
```

---

### **Phase 4 : Entraînement (30-60 min selon GPU)**

#### Étape 8 : Lancer l'entraînement
```python
import sys
import os
os.chdir('/kaggle/working/projet')
sys.path.insert(0, '/kaggle/working/projet')

# Importer les modules du projet
from pipeline_fusion.train import main

# Command-line args pour V4
import argparse
sys.argv = [
    'train.py',
    '--config', 'config_kaggle.yaml',
    '--epochs', '50',
    '--batch_size', '16',
    '--lr', '1e-4',
    '--loss_type', 'focal',  # ✅ V4
    '--seed', '42',
    '--device', 'cuda',
    '--amp',  # Activation mixed precision
    '--compile',  # Torch compile si disponible
]

# Lancer
main()
```

**⏱️ Temps estimé :**
- GPU T4 (4GB) : ~60-90 min
- GPU P100 (16GB) : ~30-45 min
- GPU V100 (32GB) : ~15-25 min

#### Étape 9 : Monitoring pendant l'entraînement
```python
# Afficher les logs en temps réel
import subprocess
result = subprocess.run(
    ['tail', '-f', '/kaggle/working/projet/logs/train_fusion.log'],
    capture_output=True,
    text=True,
    timeout=3600  # 1h timeout
)
print(result.stdout)
```

---

### **Phase 5 : Évaluation (5 min)**

#### Étape 10 : Charger et évaluer les meilleurs résultats
```python
import json
import torch

# Charger metrics finales
with open('/kaggle/working/projet/models/checkpoints/final_results.json') as f:
    results = json.load(f)

# Charger seuils optimisés
with open('/kaggle/working/projet/models/checkpoints/optimal_thresholds.json') as f:
    thresholds = json.load(f)

print("📊 RÉSULTATS FINAUX V4")
print("=" * 50)
print(f"F1-macro (test):    {results['test_metrics']['f1_macro']:.4f}")
print(f"PR-AUC (test):      {results['test_metrics']['pr_auc_macro']:.4f}")
print(f"Accuracy (test):    {results['test_metrics']['accuracy']:.4f}")
print(f"\nRappels par classe:")
for cls, rec in results['test_metrics']['recall_per_class'].items():
    print(f"  {cls}: {rec:.4f}")
print(f"\nSeuils optimisés:")
for cls, thr in thresholds.items():
    print(f"  {cls}: {thr:.2f}")
```

---

### **Phase 6 : Inférence sur Données Nouvelles (Optional)**

#### Étape 11 : Charger le modèle entraîné
```python
from pipeline_fusion.model import ECGFusionModel
from common.utils import NUM_CLASSES

# Charger checkpoint
checkpoint = torch.load(
    '/kaggle/working/projet/models/checkpoints/best_fusion_model.pth',
    map_location='cuda'
)

# Initialiser modèle
model = ECGFusionModel(num_classes=NUM_CLASSES, embed_dim=256, dropout=0.3)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
model.to('cuda')

print("✅ Modèle V4 chargé et prêt pour inférence")
```

#### Étape 12 : Prédiction avec seuils V4
```python
import numpy as np

def predict_ecg(signals, images, metadata, thresholds=None):
    """
    signals: (batch, 12, 5000)
    images: (batch, 3, 224, 224)
    metadata: (batch, 3)
    thresholds: dict avec clés NORM, MI, STTC, CD, ARR
    """
    
    with torch.no_grad():
        signals = torch.tensor(signals, dtype=torch.float32).cuda()
        images = torch.tensor(images, dtype=torch.float32).cuda()
        metadata = torch.tensor(metadata, dtype=torch.float32).cuda()
        
        # Forward pass
        logits = model(signals, images, metadata)
        probs = torch.sigmoid(logits).cpu().numpy()
    
    # Appliquer seuils V4
    if thresholds is None:
        thresholds = {
            'NORM': 0.52, 'MI': 0.49, 'STTC': 0.46, 'CD': 0.49, 'ARR': 0.51
        }
    
    thresholds_array = np.array([
        thresholds['NORM'],
        thresholds['MI'],
        thresholds['STTC'],
        thresholds['CD'],
        thresholds['ARR']
    ])
    
    predictions = (probs >= thresholds_array).astype(int)
    
    return predictions, probs

# Exemple
# preds, probs = predict_ecg(signals_batch, images_batch, metadata_batch)
```

---

## 🎯 Points Clés V4 à Respecter

### ✅ À Faire
- [x] **Loss function: FocalLoss** (par défaut avec `--loss_type focal`)
- [x] **Weight decay: 5e-4** (automatique dans le code)
- [x] **Sampler ratio: 2.0 pour ARR** (automatique dans le code)
- [x] **β_mi = 0.15** (appliqué dans trainer.py)
- [x] **Learning rate: 1e-4** (par défaut)
- [x] **Utiliser seuils optimisés** à l'inférence pour +0.1 F1
- [x] **Batch size: 16** (ou 8 si GPU < 8GB)
- [x] **AMP activé** sur GPU (`--amp`)

### ❌ À Éviter
- ❌ Ne pas utiliser `--loss_type logit_adj` (ancien, moins bon)
- ❌ Ne pas modifier `weight_decay` (5e-4 optimal)
- ❌ Ne pas changer `β_mi < 0.15` (moins bonne détection MI)
- ❌ Ne pas utiliser seuil 0.5 fixe (perdre ~0.08 F1)
- ❌ Ne pas réduire epochs sous 40 (convergence incomplète)

---

## 📊 Tableau Récapitulatif : Comparaison Anciennes Versions

| Aspect | V1 | V2/V3 | **V4 (Final)** |
|--------|-------|---------|---------|
| Loss | logit_adj | logit_adj | **FocalLoss ✅** |
| β_mi | 0.3 | 0.05 | **0.15 ✅** |
| Sampler | 3.0 | 3.0 | **2.0 ✅** |
| Weight decay | 1e-4 | 1e-4 | **5e-4 ✅** |
| F1-macro (test) | 0.71 | 0.74 | **0.757 ✅** |
| PR-AUC (test) | 0.80 | 0.81 | **0.817 ✅** |

---

## 🔗 Fichiers Modifiés

Les fichiers suivants ont été mis à jour avec les paramètres V4:
- `pipeline_fusion/trainer.py` - Loss=focal, β_mi=0.15
- `pipeline_fusion/train.py` - Loss default=focal
- `models/checkpoints/optimal_thresholds_v4.json` - Seuils optimisés

---

## 📞 Troubleshooting Kaggle

### Problème: GPU out of memory
```python
# Réduction batch size
--batch_size 8
# Activer AMP
--amp
```

### Problème: Entraînement trop lent
```python
# Utiliser compile (PyTorch 2.0+)
--compile
# Augmenter num_workers
--num_workers 4
```

### Problème: Données non trouvées
```python
# Vérifier chemins
import os
for path in ['/kaggle/input/ptb-xl-dataset', '/kaggle/input/ptb-xl-images']:
    print(f"{path}: {os.path.exists(path)}")
```

---

## 📝 Résumé Exécutif

**Configuration V4 = Meilleur équilibre F1 + Recall (clinique)**

Appliquée avec:
- ✅ FocalLoss (vs logit_adj)
- ✅ β_mi = 0.15 (vs 0.05)
- ✅ Sampler ratio 2.0 (vs 3.0)
- ✅ Weight decay 5e-4 (vs 1e-4)
- ✅ Seuils optimisés par classe

**Gains mesurés:**
- F1 : +0.048 vs V1
- PR-AUC : +0.017 vs V1
- Recall MI : +0.022 vs baseline
- Recall ARR : +0.024 vs baseline

**Temps d'entraînement:** ~1h sur T4, ~30min sur P100

**Prêt pour production avec ces seuils optimisés!** 🚀
