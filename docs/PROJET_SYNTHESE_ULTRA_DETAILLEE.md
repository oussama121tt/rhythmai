# Projet RhythmAI - Synthese ultra detaillee

## 1. Objet du depot

Ce depot contient deux blocs qui doivent etre lus ensemble :

- `rhythmai-v0-main/` : l'application web RhythmAI elle-meme, avec le backend FastAPI et le frontend React.
- `ecg_data/ecg_data/projet/` : le projet ECG local qui fournit le vrai modele entraine, les checkpoints, les seuils, les scripts de training et la documentation technique.

L'objectif fonctionnel actuel est le suivant :

- permettre a un medecin verifie de se connecter;
- uploader un ECG sous forme de signal, d'image, ou des deux;
- executer une inference sur le vrai modele local ECG;
- stocker le resultat dans la base;
- afficher l'historique et les details d'analyses;
- fournir un chat d'assistance clinique via le backend `/api/ai`.

Le point important apres correction est que le backend ne pointe plus vers un projet externe absent : il utilise le projet ECG local present dans le workspace.

---

## 2. Structure globale du code

### 2.1 Racine `rhythmai-v0-main/`

Fichiers principaux utiles :

- `README.md` : guide de demarrage et vue fonctionnelle du produit.
- `backend/` : API FastAPI et logique serveur.
- `frontend/` : SPA React/Vite.
- `PROJET_SYNTHESE_ULTRA_DETAILLEE.md` : ce fichier de synthese.

### 2.2 Backend `rhythmai-v0-main/backend/`

Fichiers actifs :

- `main.py` : routes API, inference, auth, admin, chat IA.
- `auth.py` : hash password, verification, JWT minimal.
- `database.py` : modele SQLAlchemy et session DB.
- `preprocessing.py` : nettoyage et normalisation des entrees ECG.
- `model_loader.py` : chargement du vrai modele local et inference.
- `seed.py` : creation du compte admin initial.
- `reset_admin.py` : utilitaire de remise a zero de l'admin si besoin.
- `requirements.txt` : dependances Python du backend.

Fichiers supprimes pendant le nettoyage :

- `check_login.py`
- `fix_user.py`

### 2.3 Frontend `rhythmai-v0-main/frontend/`

Fichiers principaux :

- `src/App.jsx` : SPA monofichier avec la quasi-totalite des pages et composants.
- `src/main.jsx` : point d'entree React.
- `package.json` : scripts Vite.
- `vite.config.js` : configuration Vite.

### 2.4 Projet ECG local `ecg_data/ecg_data/projet/`

Elements importants :

- `pipeline_fusion/model.py` : vraie architecture du modele fusion.
- `pipeline_fusion/test_interface.py` : reference d'inference locale.
- `models/checkpoints/best_fusion_model.pth` : poids entraines.
- `models/checkpoints/optimal_thresholds.json` : seuils optimaux par classe.
- `models/checkpoints/final_results.json` : resume des performances.
- `PROJECT_DESCRIPTION.md` : documentation technique detaillee.
- `README.md` : vue rapide du projet ECG.
- `common/` : briques partagées, preprocessing, pertes, backbones, XAI.
- `pipeline_signals/`, `pipeline_images/`, `pipeline_fusion/` : pipelines d'entrainement et d'evaluation.

---

## 3. Backend : logique et routes

### 3.1 `backend/main.py`

Ce fichier est le coeur de l'API.

#### Initialisation

- cree l'application FastAPI `app`;
- active CORS avec origines explicites (`http://localhost:5173`, `http://127.0.0.1:5173`, etc.) et `allow_credentials=True`;
- initialise les tables a `startup()` via `create_tables()`;
- expose un endpoint de sante `/api/health`.

#### Schemas Pydantic

- `LoginBody` : email + password.
- `NotesBody` : note texte pour une analyse.
- `UpdateUserBody` : statut, role, flag admin.
- `AIRequest` : message du chat IA.

#### Helpers d'authentification

- `_current_user()` : extrait le token Bearer, charge l'utilisateur, bloque les comptes suspendus.
- `_admin_user()` : ajoute la verif `is_admin`.
- `_audit()` : ecrit un audit log.
- `_user_dict()` : normalise la forme JSON d'un utilisateur.
- `_analysis_dict()` : normalise la forme JSON d'une analyse et ses scores.

#### Routes auth

- `POST /api/auth/register` : inscription avec upload optionnel de photo de carte pro.
- `POST /api/auth/login` : connexion et generation du JWT.
- `GET /api/auth/me` : profil de l'utilisateur courant.

#### Routes analyses

- `GET /api/analyses` : liste des analyses du medecin connecte.
- `POST /api/analyses/predict` : point d'entree principal de l'inference.
- `GET /api/analyses/{analysis_id}` : detail d'une analyse.
- `PATCH /api/analyses/{analysis_id}/notes` : mise a jour des notes.
- `DELETE /api/analyses/{analysis_id}` : suppression d'une analyse.

#### Routes admin

- `GET /api/admin/stats` : statistiques globales.
- `GET /api/admin/users` : liste des utilisateurs.
- `PATCH /api/admin/users/{id}` : changement de statut / role / admin.
- `DELETE /api/admin/users/{id}` : suppression d'un utilisateur.
- `GET /api/admin/analyses` : liste globale des analyses.

#### Chat IA

- `POST /api/ai` : assistant textuel clinique avec deux strategies.

Cette route n'est pas le modele ECG de diagnostic. C'est un service annexe de support clinique.

Strategie actuelle :
1. Essai du LLM distant (Pollinations, OpenAI, ou custom) avec timeout court (6s).
2. Si timeout ou erreur, fallback sur un assistant local deterministe base sur une base de connaissances medicales.
3. L'assistant local peut generer du texte coherent meme quand les services distants sont indisponibles (mode offline-friendly).

Le fallback local utilise `_local_ai_response()` qui infere la pathologie (ARR, MI, CD, STTC, NORM) et genere une reponse contextuelle avec points cliniques, directives et dernieres recherches.

### 3.2 `backend/auth.py`

Ce module contient une auth minimaliste mais fonctionnelle :

- hash mot de passe via `passlib` (bcrypt) si disponible;
- fallback SHA-256 salte si `passlib` manque ou si `FORCE_SIMPLE_HASH=1` est defini (mode dev);
- generation JWT maison via HMAC SHA-256;
- extraction du user id depuis un header `Authorization: Bearer ...`;
- verif du flag admin.

Point important : cette couche est autonome et ne depend pas du modele ECG.

Note pour le dev local : en environnement Windows/dev, les erreurs de version bcrypt peuvent survenir. Utiliser `FORCE_SIMPLE_HASH=1` force le fallback SHA-256, qui reste securise pour le dev.

### 3.3 `backend/database.py`

Le schema persiste :

- `User`
- `Analysis`
- `AnalysisScore`
- `AuditLog`

Relations importantes :

- un `User` a plusieurs `Analysis`;
- une `Analysis` a plusieurs `AnalysisScore`;
- un `User` a plusieurs `AuditLog`.

La table `Analysis` stocke :

- `ecg_id`
- `doctor_id`
- `patient_ref`
- `patient_age`
- `patient_sex`
- `input_mode`
- `result`
- `confidence`
- `inference_time`
- `notes`
- `created_at`

### 3.4 `backend/preprocessing.py`

Ce module prepare les entrees pour le modele local.

#### Signal

- `bandpass_filter()` : filtre passe-bande.
- `notch_filter()` : suppression du 50 Hz.
- `zscore_normalize()` : normalisation par derivee.
- `preprocess_ecg_signal()` : pipeline complet pour obtenir du `(12, 5000)`.
- `load_signal_csv()` : charge un CSV ECG.
- `load_signal_npy()` : charge un NPY ECG.

#### Image

- `preprocess_ecg_image()` : redimensionne et normalise une image ECG pour le backbone image.

Le contrat important ici est que le backend doit fournir des tenseurs compatibles avec le modele fusion local.

### 3.5 `backend/model_loader.py`

C'est la piece qui a ete corrigee pour brancher le vrai modele local.

Ce que fait le module maintenant :

- localise automatiquement `ecg_data/ecg_data/projet`;
- ajoute ce projet au `sys.path`;
- charge `ECGFusionModel` depuis `pipeline_fusion.model`;
- charge `models/checkpoints/best_fusion_model.pth`;
- lit `optimal_thresholds.json` ou `final_results.json`;
- produit une sortie multi-label via `torch.sigmoid()`;
- conserve une reponse compatible avec le frontend via :
  - `predicted`
  - `scores`
  - `confidence`
  - `inference_time`

Comportement important :

- si le vrai projet ou le checkpoint manque, un `MockFusionModel` reste disponible comme filet de secours;
- mais dans ce workspace, le vrai projet existe, donc l'objectif est bien de passer sur le vrai poids local.

#### Strategie de decision

Le modele local est multi-label. Pour rester compatible avec la base existante qui stocke un `result` unique :

- les probabilites de classes sont calculees;
- les seuils optimaux sont appliques par classe;
- `predicted` prend la premiere classe active la plus probable, ou sinon la classe a plus forte probabilite;
- `scores` contient les probabilites de toutes les classes au format pourcentage.

#### Metadonnees

Le modele accepte un vecteur de metadonnees. Le backend envoie :

- age normalise;
- sexe encode;
- troisieme valeur placeholder a `0.0`.

Cela permet deja de nourrir le contrat du vrai modele au lieu de laisser un vecteur vide.

---

## 4. Frontend : pages, composants et flux

### 4.1 `frontend/src/App.jsx`

Le frontend est une SPA monofichier tres dense. Il contient :

- les constantes API et couleurs;
- les helpers de fetch;
- les composants UI de base;
- la navigation interne par etat local `page`;
- les pages metiers.

#### Composants globaux

- `Avatar`
- `ECGLine`
- `DonutChart`
- `Sparkline`
- `Sidebar`
- `Topbar`
- `StatusBadge`
- composants de toast / modal / cartes utilitaires

#### Pages metiers

- `Landing`
- `AuthPage`
- `AdminLogin`
- `Dashboard`
- `AnalysisPage`
- `HistoryPage`
- `AdminDashboard`
- `AIChatPanel`

#### Flux de navigation

L'app utilise un `page` interne, pas React Router :

- `landing`
- `login`
- `signup`
- `admin-login`
- `admin`
- `dashboard`
- `analysis`
- `history`

Cela simplifie la logique, mais garde tout dans un seul gros fichier.

### 4.2 Pages principales

#### `Landing`

- page d'accueil marketing;
- dirige vers login, signup et admin login.

#### `AuthPage`

- login / signup dans le meme composant;
- gestion du formulaire medecin;
- upload de la carte professionnelle lors de l'inscription.

#### `AdminLogin`

- porte d'entree admin.

#### `Dashboard`

- resume de l'activite;
- stats rapides;
- dernieres analyses;
- raccourcis vers analyse et historique.

#### `AnalysisPage`

- upload de signal `.dat`, `.npy`, `.csv` et header `.hea`;
- upload image ECG;
- modes `signal`, `image`, `fusion`;
- collecte `patient_ref`, `age`, `sex`;
- appel `POST /api/analyses/predict`;
- affichage du resultat et de l'assistant IA.

#### `HistoryPage`

- liste et filtre des analyses;
- consultation des notes;
- historique clinique.

#### `AdminDashboard`

- gestion utilisateurs;
- verification des comptes;
- consultation globale des analyses.

#### `AIChatPanel`

- conversation clinique contextuelle;
- repose sur le backend `/api/ai`.

### 4.3 `frontend/src/main.jsx`

- bootstrap React standard;
- monte `App` dans `#root`.

### 4.4 Configuration frontend

`frontend/.env.example` definit :
- `VITE_API_URL=http://localhost:8001` : point d'entree API (port dev courant).

Note : en dev local, le backend s'execute souvent sur le port 8001 (conflit possible sur 8000). Verifier que `VITE_API_URL` correspond au port backend actif.

### 4.5 Observations de maintenance frontend

- il n'y a pas de doublon manifeste de pages fonctionnelles a supprimer tout de suite;
- le vrai probleme de maintenabilite est la taille de `App.jsx`;
- un refactor en composants/fichiers peut venir apres, mais il n'etait pas requis pour faire fonctionner le modele local.

---

## 5. Projet ECG local : logique du vrai modele

### 5.1 Architecture `pipeline_fusion/model.py`

Le vrai modele local est `ECGFusionModel`.

Il combine :

- une branche signal `ECGResNet1D`;
- une branche image `ECGImageBranch`;
- une co-attention `CrossModalCoAttention`;
- un encodeur de metadonnees;
- une fusion hierarchique finale;
- une tete de classification multi-label.

Les sorties d'entrainement incluent aussi des taches auxiliaires.

### 5.2 Contrat d'inference local

La reference locale attend :

- `signal` : tenseur `(B, 12, 5000)`;
- `image` : tenseur image;
- `mask` : presence des modalites;
- `metadata` : vecteur de metadonnees.

La methode `predict_proba()` retourne des probabilites sigmoide multi-label.

C'est ce contrat qui a guide la correction du backend.

### 5.3 Artefacts de run

Dans `models/checkpoints/` :

- `best_fusion_model.pth` : checkpoint principal;
- `optimal_thresholds.json` : seuils par classe;
- `final_results.json` : metriques et seuils;
- `run_manifest.json` : traces du run;
- `training_metrics.jsonl` : historique d'entrainement.

### 5.4 Documentation locale utile

- `PROJECT_DESCRIPTION.md` donne la vue la plus complete sur l'architecture, les pertes et la logique multi-label.
- `README.md` du sous-projet decrit le setup et la structure.

---

## 6. Nettoyage effectue

### 6.1 Supprimes

Les scripts suivants ont ete supprimes car ils ne participaient pas au flux principal de l'application :

- `backend/check_login.py`
- `backend/fix_user.py`

### 6.2 Conserves

Conserves car encore utiles :

- `backend/seed.py`
- `backend/reset_admin.py`
- tous les fichiers de route et de modele.

### 6.3 Documentation corrigee

Le `README.md` racine a ete aligne sur l'etat reel du projet :

- le chemin du modele local est maintenant explicite;
- la variable d'environnement du chat a ete corrigee vers `VITE_HF_API_KEY`;
- la section chat ne parle plus de Gemini mais du backend Hugging Face.

---

## 7. Flux complet de donnees

### 7.1 Inscription medecin

1. L'utilisateur remplit le formulaire dans `AuthPage`.
2. Le frontend envoie `POST /api/auth/register`.
3. Le backend cree un `User` avec statut `pending`.
4. L'admin verifie le compte dans `AdminDashboard`.

### 7.2 Connexion

1. L'utilisateur se connecte avec `POST /api/auth/login`.
2. Le backend renvoie un JWT et le profil utilisateur.
3. Le frontend stocke le token.
4. L'app navigue vers `dashboard`.

### 7.3 Inference ECG

1. `AnalysisPage` collecte les fichiers et les metadonnees.
2. Le frontend envoie `POST /api/analyses/predict`.
3. `backend/main.py` preprocess les entrees.
4. `backend/model_loader.py` charge le vrai modele local.
5. Les probabilites sont calculees avec sigmoid.
6. Les seuils par classe sont appliques.
7. Le resultat est persisté dans `Analysis` et `AnalysisScore`.
8. Le frontend affiche le diagnostic et l'historique se met a jour.

### 7.4 Chat clinique

1. L'utilisateur saisit une question dans le panneau IA.
2. Le frontend appelle `POST /api/ai` avec message, pathologie, contexte patient.
3. Le backend essaie le LLM distant (Pollinations / OpenAI / custom) avec timeout 6s.
4. Si le distant echoue ou timeout, l'assistant local deterministe genere une reponse basee sur la pathologie et la KB.
5. La reponse est renvoyee dans l'interface.

Avantage : l'application reste utilisable meme sans service distant (mode offline).

---

## 8. Points importants de logique

- Le diagnostic ECG n'est pas un simple mock : il utilise maintenant les vrais poids locaux.
- Le modele local est multi-label, donc la sortie n'est plus une simple softmax mono-classe.
- L'UI existante reste compatible grace a la conservation de `predicted`, `scores` et `confidence`.
- Le chat clinique est un service annexe, pas le modele ECG de diagnostic.
- La base stocke encore un `result` principal unique pour compatibilite, mais les probabilites completes sont aussi disponibles.

---

## 9. Etat de maintenance apres nettoyage et correction

### 9.1 Ce qui est plus propre

- plus de reference vers un projet ECG externe absent;
- plus de scripts backend redondants;
- documentation racine alignee avec le code;
- infererence plus fidele au vrai run local;
- chat IA robust avec fallback local et async timeout;
- CORS configure explicitement pour dev;
- authentification compatible avec fallback SHA-256 en dev.

### 9.2 Ce qui reste perfectible

- `frontend/src/App.jsx` reste tres volumineux;
- le chat IA pourrait etre extrait en service dedie;
- la navigation pourrait etre modernisee avec un routeur si le projet continue de grandir;
- le stockage d'un seul `result` principal est une simplification par rapport au vrai multi-label;
- le port 8001 est actuellement utilise en dev (port 8000 reserve); confirmer le port final pour la production.

Ces points ne bloquent pas le fonctionnement actuel, mais ils sont a envisager pour la suite.

### 9.3 Dernieres corrections apportees

- **Chat IA avec timeout**: l'endpoint `/api/ai` utilise maintenant un timeout asynchrone de 6s sur le LLM distant, puis bascule sur le fallback local deterministe.
- **Authentification**: support de `FORCE_SIMPLE_HASH=1` pour dev local (bcrypt issues sur Windows).
- **CORS robuste**: origines explicites et `allow_credentials=True` pour eviter les blocages navigateur.
- **Port 8001**: le backend s'execute en dev sur le port 8001 pour eviter les conflits. Le frontend est configure sur cette valeur.

---

## 10. Resume court

Le projet RhythmAI est maintenant compris comme un produit a deux couches :

- une application web de medecin / admin pour l'upload et le suivi d'analyses ECG;
- un vrai projet ML local pour l'inference fusion signal + image.

Le nettoyage a retire des scripts inutiles, la documentation a ete alignee, et le backend est branche sur le modele ECG local avec ses seuils et son contrat multi-label.
