# Projet — Classification ECG avec PTB-XL

## Vue d'ensemble

Système de classification d'ECG sur PTB-XL, centré sur 5 classes: NORM, MI, STTC, CD, ARR.

Le dépôt contient maintenant trois chemins d'entraînement:
- pipeline signaux 1D: ResNet1D avec attention par leads et petit encodeur Transformer temporel
- pipeline images 2D: scalogrammes CWT produits depuis les signaux bruts, puis EfficientNet-B3
- pipeline fusion: co-attention croisée signal-image, fusion hiérarchique et injection de métadonnées

Le projet est configuré en multi-label Binary Relevance. Chaque classe est prédite indépendamment, avec BCEWithLogits ou Logit-Adjusted BCE selon la configuration.

## Les 5 classes

| Classe | C'est quoi | En gros |
|--------|-----------|---------|
| NORM | Normal | ECG sans anomalie majeure |
| MI | Myocardial Infarction | Signes compatibles infarctus |
| STTC | ST/T Changes | Anomalies du segment ST ou de l'onde T |
| CD | Conduction Disturbance | Troubles de conduction |
| ARR | Arrhythmia | Arythmies |

Les labels proviennent des codes SCP du PTB-XL. Le projet garde aussi une version compacte de l'ancien label unique pour compatibilité, mais la logique actuelle d'entraînement est multi-label.

## Données et prétraitement

- PTB-XL: environ 21 800 ECG, 12 leads, 10 secondes, 500 Hz
- Split patient-wise via `strat_fold` pour limiter le leakage
- Prétraitement signal: passe-haut 1 Hz, notch 50 Hz, puis z-score par lead
- Les signaux sont lissés puis transformés en scalogrammes CWT Morlet, avec un canal Ricker optionnel
- Les métadonnées âge et sexe sont injectées dans la fusion

## Architecture actuelle

### `common/`

- `utils.py`: constantes, mapping SCP, gestion des labels, métadonnées
- `preprocessing.py`: filtrage ECG
- `losses.py`: Logit-Adjusted CE et MultiLabelLogitAdjustedBCE
- `blocks.py`: SEBlock1D, AttentionPool1D, ResBlock1D, LeadWiseAttention, TemporalTransformerEncoder, CrossModalCoAttention
- `signal_backbone.py`: backbone 1D avec attention par leads et encodeur temporel
- `image_backbone.py`: wrapper EfficientNet-B3
- `time_frequency.py`: génération de scalogrammes CWT
- `audit.py`: audit de split et de normalisation
- `xai.py`: Grad-CAM et Integrated Gradients

### `pipeline_signals/`

- Dataset: signaux WFDB, preprocessing, augmentation signal, cible multi-label et métadonnées
- Modèle: classification 1D multi-label sur embeddings ECGResNet1D
- Entraînement: BCE pondérée ou logit-adjusted, métriques multi-label

### `pipeline_images/`

- Dataset: génération de scalogrammes depuis les signaux bruts
- Modèle: EfficientNet-B3 + tête de classification
- Entraînement: même logique multi-label que la branche signal

### `pipeline_fusion/`

- Dataset: signal 1D + scalogramme 2D + masque de modalité + métadonnées
- Modèle: co-attention signal-image, fusion intermédiaire puis fusion finale avec métadonnées
- Entraînement: perte principale multi-label plus pertes auxiliaires sur les branches et têtes spécialisées

## Résultats de référence

Les résultats ci-dessous viennent du checkpoint référent sauvegardé dans `models/checkpoints/final_results.json`.

| Métrique | Valeur |
|----------|--------|
| Accuracy test | 0.794 |
| F1-macro test | 0.752 |
| PR-AUC macro | 0.820 |

| Classe | F1 | Rappel |
|--------|-----|--------|
| NORM | 0.8845 | 0.9042 |
| MI | 0.7194 | 0.7163 |
| STTC | 0.7316 | 0.7394 |
| CD | 0.7219 | 0.6612 |
| ARR | 0.7008 | 0.7542 |

La matrice de confusion annotée se trouve dans `models/checkpoints/confusion_matrix.png`.

## Organisation du dépôt

```
projet/
├── common/
├── pipeline_images/
├── pipeline_signals/
├── pipeline_fusion/
├── models/checkpoints/
├── logs/
├── PTB-XL ECG dataset/
├── PTB-XL ECG image (GMC2024)/
├── config.yaml
├── requirements.txt
└── PROJECT_DESCRIPTION.md
```

## À retenir

- Le dépôt n’est plus basé sur un simple GatedFusion avec images PNG.
- Le pipeline image utilise maintenant des scalogrammes dérivés du signal brut.
- La fusion utilise de la co-attention et des métadonnées, pas seulement un gate MLP.
- La supervision est multi-label, ce qui est plus fidèle aux cooccurrences cliniques du PTB-XL.
