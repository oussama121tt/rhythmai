# 🚀 DÉMARRAGE RAPIDE - RhythmAI V4 sur Kaggle

## ⚡ Commande Unique pour Entraînement V4

```bash
python -m pipeline_fusion.train \
  --config config.yaml \
  --epochs 50 \
  --batch_size 16 \
  --lr 1e-4 \
  --loss_type focal \
  --seed 42 \
  --device cuda \
  --amp \
  --compile
```

---

## 📋 CHECKLIST PRÉ-ENTRAÎNEMENT

- [ ] **Dataset PTB-XL uploadé** sur Kaggle Notebooks
  - Signaux: `/kaggle/input/ptb-xl-dataset`
  - Images: `/kaggle/input/ptb-xl-images`

- [ ] **Projet cloné/uploadé**
  - Option 1: `git clone` du repo
  - Option 2: Upload du ZIP du projet

- [ ] **GPU activé** dans Kaggle Notebook settings
  - Préférence: T4 (4GB+), P100 (16GB+), ou V100 (32GB+)

- [ ] **Internet activé** pour télécharger dépendances

- [ ] **Vérifier les chemins** dans config_kaggle.yaml
  - `ptbxl_signals` → chemin correct
  - `ptbxl_images` → chemin correct

---

## 🎯 PARAMÈTRES V4 (À NE PAS MODIFIER)

```json
{
  "loss_function": "focal",
  "learning_rate": 1e-4,
  "weight_decay": 5e-4,
  "sampler_ratio_arr": 2.0,
  "beta_mi": 0.15,
  "beta_cd": 0.05,
  "alpha_aux": 0.1,
  "dropout": 0.5,
  "batch_normalization": true,
  "auxiliary_heads": true,
  "thresholds": {
    "NORM": 0.52,
    "MI": 0.49,
    "STTC": 0.46,
    "CD": 0.49,
    "ARR": 0.51
  }
}
```

---

## 📊 RÉSULTATS ATTENDUS (V4)

| Métrique | Valeur | Note |
|----------|--------|------|
| F1-macro (test) | 0.757 | ✅ Objectif: ≥0.70 |
| PR-AUC (test) | 0.817 | ✅ Bon |
| Recall MI | 0.722 | ✅ Sensibilité clinique |
| Recall ARR | 0.824 | ✅ Sensibilité clinique |
| Accuracy | 0.672 | Acceptable |

---

## ⏱️ TEMPS D'ENTRAÎNEMENT

| GPU | RAM | Temps Estimé |
|-----|-----|--------------|
| T4 (Kaggle) | 16GB | 60-90 min |
| P100 (Kaggle Pro) | 32GB | 30-45 min |
| V100+ | 32GB+ | 15-30 min |
| CPU | N/A | >2h (non recommandé) |

---

## 🛠️ CONFIGURATION HARDWARE OPTIMALE

### Batch Size Recommandé
- **GPU ≥ 8GB**: batch_size = 16 ✅
- **GPU 4-8GB**: batch_size = 8 (+ AMP)
- **GPU < 4GB**: batch_size = 4 (+ AMP + gradient accumulation)

### Optimisations Kaggle
```bash
--batch_size 16      # Taille batch optimal
--amp                # Mixed Precision (gain ~20% vitesse)
--compile            # torch.compile si PyTorch 2.0+
--num_workers 4      # DataLoader workers
```

---

## 📥 ÉTAPES D'INSTALLATION (KAGGLE NOTEBOOK)

### 1. Cell 1: Install Dependencies
```python
!pip install -q wfdb scipy scikit-learn pytorch-lightning tensorboard
!pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2. Cell 2: Verify Data
```python
import os
import pandas as pd

CSV_PATH = '/kaggle/input/ptb-xl-dataset/ptbxl_database.csv'
df = pd.read_csv(CSV_PATH)
print(f"✅ Dataset: {len(df)} enregistrements")
```

### 3. Cell 3: Clone Project
```python
!git clone https://github.com/rhythmai/ecg_data.git /kaggle/working/projet
# OU
!cd /kaggle/working && unzip -q projet.zip
```

### 4. Cell 4: Train V4
```python
import sys
sys.path.insert(0, '/kaggle/working/projet')
from pipeline_fusion.train import main

sys.argv = [
    'train.py',
    '--config', 'config_kaggle.yaml',
    '--epochs', '50',
    '--batch_size', '16',
    '--lr', '1e-4',
    '--loss_type', 'focal',
    '--device', 'cuda',
    '--amp'
]

main()
```

### 5. Cell 5: Afficher Résultats
```python
import json

with open('/kaggle/working/models/checkpoints/final_results.json') as f:
    results = json.load(f)

print("📊 RÉSULTATS FINAUX:")
for metric, value in results['test_metrics'].items():
    print(f"  {metric}: {value}")
```

---

## 🔗 FICHIERS CLÉS

```
projet/
├── pipeline_fusion/
│   ├── train.py              ← Point d'entrée (modifié V4)
│   ├── trainer.py            ← Boucle entraînement (modifié V4)
│   ├── model.py              ← Modèle fusion
│   └── dataset.py            ← Dataset bimodal
├── common/
│   ├── losses.py             ← FocalLoss, MultiLabelLogitAdjustedBCE
│   ├── signal_backbone.py    ← ResNet1D
│   ├── image_backbone.py     ← EfficientNet-B3
│   └── utils.py              ← Utilitaires
├── models/checkpoints/
│   ├── best_fusion_model.pth ← Poids entraînés
│   ├── optimal_thresholds_v4.json  ← Seuils V4
│   └── final_results.json    ← Résultats finaux
├── config.yaml               ← Configuration (adapter chemins)
├── GUIDE_KAGGLE_V4.md        ← Guide détaillé (ce fichier)
└── kaggle_training_v4.ipynb  ← Notebook Kaggle prêt-à-utiliser
```

---

## ❌ ERREURS COURANTES & SOLUTIONS

### Erreur: "No module named 'pipeline_fusion'"
```python
# Solution:
import sys
sys.path.insert(0, '/kaggle/working/projet')
```

### Erreur: "CUDA out of memory"
```python
# Solution: Réduire batch_size
--batch_size 8
# Ou activer AMP
--amp
```

### Erreur: "File not found: ptbxl_database.csv"
```python
# Solution: Vérifier le chemin exact:
import os
print(os.listdir('/kaggle/input'))  # Voir les datasets
# Adapter PTBXL_SIGNALS_DIR en conséquence
```

### Erreur: "Loss NaN"
```python
# Solution 1: Vérifier les labels (pas tous zéros)
# Solution 2: Réduire learning rate
--lr 5e-5
# Solution 3: Utiliser --amp
```

### Training très lent
```python
# Solution:
--amp                    # Mixed precision
--compile                # torch.compile
--num_workers 4          # Augmenter workers
--prefetch_factor 2      # Prefetch
```

---

## 📊 MONITORING PENDANT ENTRAÎNEMENT

### Afficher les logs en temps réel
```python
import subprocess
result = subprocess.run(
    ['tail', '-f', '/kaggle/working/logs/train_fusion.log'],
    timeout=3600
)
print(result.stdout)
```

### Métriques clés à suivre
- **Train Loss**: Doit décroître
- **Val F1-macro**: Doit augmenter
- **Patience Counter**: Doit rester < 10

### Si train loss → NaN:
1. Réduire learning rate: `--lr 5e-5`
2. Utiliser gradient clipping (dans trainer.py)
3. Réduire batch size

---

## 🎓 COMPARAISON VERSIONS

| Aspect | V1 | V3 | **V4 (Final)** |
|--------|-----|-----|---------|
| Loss | logit_adj | logit_adj | **FocalLoss ✅** |
| β_mi | 0.3 | 0.05 | **0.15 ✅** |
| Sampler | 3.0 | 3.0 | **2.0 ✅** |
| F1 (test) | 0.710 | 0.745 | **0.757 ✅** |
| PR-AUC (test) | 0.800 | 0.810 | **0.817 ✅** |
| Recall MI | 0.700 | 0.710 | **0.722 ✅** |

---

## 💡 TIPS & TRICKS

### Pour accélérer l'entraînement:
```bash
--amp --compile --num_workers 4 --prefetch_factor 2
```

### Pour améliorer la stabilité:
```bash
--amp --seed 42
```

### Pour debug/test rapide:
```bash
--max_batches 10  # Seulement 10 batches par epoch
```

### Pour reproductibilité:
```bash
--seed 42  # Déjà V4
```

---

## 📞 SUPPORT

Si problèmes lors de l'entraînement sur Kaggle:

1. **Vérifier GPU disponible:**
   ```python
   import torch
   print(torch.cuda.is_available())
   print(torch.cuda.get_device_name(0))
   ```

2. **Vérifier data accessible:**
   ```python
   import os
   print(os.listdir('/kaggle/input'))
   ```

3. **Consulter les logs:**
   ```python
   with open('/kaggle/working/logs/train_fusion.log') as f:
       print(f.read())
   ```

---

## 🏁 RÉSUMÉ EXÉCUTIF

**Configuration V4 = Meilleur équilibre performance clinique + F1**

**Appliquée avec 4 changements clés:**
1. ✅ Loss: Focal (vs logit_adj)
2. ✅ β_mi: 0.15 (vs 0.05)
3. ✅ Sampler: 2.0 (vs 3.0)
4. ✅ Seuils: Optimisés par classe

**Résultat:** F1 +0.047, PR-AUC +0.017 vs V1

**Status:** 🚀 **PRÊT POUR KAGGLE**

---

Créé: 2026-05-06
Version: V4 Final
License: MIT
