# RhythmAI — Cardiac Intelligence Platform

## Vue d'ensemble

RhythmAI est une plateforme de diagnostic cardiaque assisté par IA, conçue pour les cardiologues, médecins internes et étudiants en médecine supervisés. Elle combine un modèle de fusion multimodal (GatedFusion) avec une interface web intuitive pour classer automatiquement les pathologies cardiaques à partir d'enregistrements ECG (signaux et images). Données d'entraînement : 21,837 ECG annotés du dataset PTB-XL.

## Architecture

```
rhythmai-v0-main/
├── README.md               ← Ce fichier (principal)
├── .gitignore
├── rhythmai-v0-main/       ← Application web
│   ├── backend/            ← FastAPI + PyTorch
│   └── frontend/           ← React + Vite
├── ecg_data/               ← Projet ML local
│   └── ecg_data/projet/    ← ECGFusionModel et artefacts
├── docs/                   ← Documentation déplacée (Cahier_..., guides, index)
├── training/               ← Notebooks et scripts d'entraînement (Kaggle)
├── results_v4/             ← Artefacts du dernier run Kaggle (si présent)
└── gpt4free-main/          ← LLM local (optionnel)
```

Note: le dossier `results_v4/` localisé dans `ecg_data/ecg_data/projet/` contient un **best_fusion_model.pth** dont la taille diffère légèrement de `models/checkpoints/best_fusion_model.pth` (fichiers laissés en place — ne pas supprimer). Voir la section "Organisation" et le CHANGELOG_AGENT.md pour les détails.

## Stack Technologique

| Composant | Technologie | Version | Rôle |
|-----------|-------------|---------|------|
| **Backend** | FastAPI | 0.111.0 | API REST, authentification, inférence |
| **ORM** | SQLAlchemy | 2.0.29 | Gestion BD SQLite/PostgreSQL |
| **Authentification** | JWT custom | — | Tokens 24h, bcrypt + SHA-256 fallback |
| **Frontend** | React | 18.2.0 | Interface responsive, web SPA |
| **Bundler** | Vite | 5.0.0 | Dev server, optimisation production |
| **Styling** | TailwindCSS inline | — | Palette couleur, composants |
| **Modèle IA** | PyTorch | 2.0+ | GatedFusion (signal + image + metadata) |
| **ECG Processing** | WFDB | 4.1.0 | Lecture signaux .dat, formats |
| **LLM Chat** | gpt4free | — | OpenAI-compatible (fallback règles) |
| **Planification** | APScheduler | — | Notifications quotidiennes |
| **BD** | SQLite | 3.x | Production (dev) / PostgreSQL (prod) |

## Démarrage Rapide

### 1. Cloner et configurer l'environnement

```bash
cd rhythmai-v0-main

# Backend
cd backend
python -m venv venv
# Windows: venv\Scripts\activate
# Linux/Mac: source venv/bin/activate
pip install -r requirements.txt

# Initialiser BD + comptes démo
python seed.py

# Frontend
cd ../frontend
npm install
```

### 2. Lancer le backend (port 8001)

```bash
cd backend
python main.py
# Output: INFO:     Uvicorn running on http://0.0.0.0:8001
```

### 3. Lancer le frontend (port 5173)

```bash
cd frontend
npm run dev
# Output: VITE v5.0.0  ready in 234 ms
#         ➜  Local:   http://localhost:5173/
```

### 4. Accéder à l'application

Ouvrir **http://localhost:5173** dans votre navigateur.

## Credentials de Test

| Rôle | Email | Mot de passe | Statut |
|------|-------|--------------|--------|
| **Admin** ⚙️ | `admin@rhythmai.ai` | `admin2026` | ⚠️ **NE PAS MODIFIER** |
| Doctor | `amira@hopital-charles.tn` | `doctor123` | Vérifié |
| Doctor | `karim@clinique-tawfik.tn` | `doctor123` | Vérifié |
| Résident | `sonia@chu-sfax.tn` | `doctor123` | Vérifié |
| Étudiant | `yassine@etudiant.um.tn` | `student123` | En attente |

**⚠️ Important :** Les comptes démo sont créés au premier lancement de `seed.py`. Changer le mot de passe admin cassera la plateforme en mode développement.

## Modèle IA — Performances

Le modèle **ECGFusionModel** fusionne trois branches :
- **Signal branch** : ECGResNet1D pour traiter les signaux 12 dérivations
- **Image branch** : EfficientNet-B3 pour les images ECG (224×224)
- **Fusion** : CrossModalCoAttention pour combiner informations
- **Metadata** : Âge et sexe du patient

### Résultats (PTB-XL, test set)

| Classe | F1-Score | PR-AUC | Observations |
|--------|----------|--------|--------------|
| **NORM** | 92.1% | 0.968 | Excellent, peu de faux positifs |
| **MI** | 88.7% | 0.931 | Bon (myocardial infarction = critique) |
| **STTC** | 81.3% | 0.876 | Acceptable (ST-T changes = difficile) |
| **CD** | 75.4% | 0.842 | Modéré (conduction disturbance) |
| **ARR** | 79.2% | 0.855 | Bon (arrhythmia) |
| **Macro-F1** | **83.3%** | — | Performance équilibrée |

**Tempo d'inférence :** <1s par ECG (GPU) / ~2s (CPU)

## Fichiers ECG de Test

Des exemples ECG au format `.dat`, `.csv` et `.npy` sont disponibles dans le dataset PTB-XL :
- **Source officielle** : https://physionet.org/content/ptb-xl/1.0.2/
- **Citation** : Wagner et al. (2020) — 21,837 12-lead ECGs, 73 diagnostic codes
- **License** : ODbL v1.0

## Avertissement ⚠️

**RhythmAI est un projet académique à visée de recherche. Il n'est PAS approuvé pour un usage clinique direct.**

- Les résultats sont fournis à **titre informatif uniquement**
- Toujours **valider avec un cardiologue certifié** avant décision médicale
- Ne pas utiliser sans supervision médicale
- Les prédictions IA complètent, n'annulent jamais le jugement clinique

---

**Développé par :** ESPROM 4DS  
**Supervision médicale :** Dr Abdelkarim Mars  
**License du dataset :** ODbL v1.0 (PTB-XL)  
**Dernière mise à jour :** Mai 2026
LLM_PROVIDER=openai
LLM_API_BASE_URL=http://localhost:1337/v1
LLM_MODEL=gpt-4o-mini
```

### 5. Optional Hugging Face API Key

1. Go to **https://huggingface.co/settings/tokens**
2. Create a new access token with inference access
3. Add it to `frontend/.env`: `VITE_HF_API_KEY=hf_...`

### 6. Test ECG Files

Download PTB-XL sample files from: https://physionet.org/content/ptb-xl/1.0.3/
Or use any WFDB-format `.dat` + `.hea` pair from PhysioNet.

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Side Navbar** | Dashboard, Signal Analysis, History |
| **Signal Upload** | `.dat` + `.hea` file pair support (WFDB format) |
| **Fusion Mode** | Combined signal + image analysis ("Both") |
| **AI Chat** | Multi-turn assistant via backend `/api/ai`, OpenAI-compatible g4f API by default |
| **Doctor ID Card** | Photo upload in signup; visible to admin for verification |
| **Admin Panel** | Review ID cards, approve/suspend/delete users |
| **History** | Full analysis log with filtering and clinical notes |

---

<!-- Admin seeded credentials removed here to avoid duplication; see "Credentials de Test" section above. -->
