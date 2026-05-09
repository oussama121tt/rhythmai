# ECG Classification — Signal + Image Fusion

Automatic ECG classification into 5 classes (NORM, MI, STTC, CD, ARR) using the PTB-XL dataset.

Three independent pipelines:
- **Signals** — ResNet1D + SE-blocks + Attention Pooling on raw 12-lead ECG
- **Images** — EfficientNet-B0 on ECG plot images
- **Fusion** — GatedFusion combining both modalities with multi-task learning

## Quick Start

```bash
# 1. Clone
git clone https://github.com/oussama121tt/ecg-analysis-project.git
cd ecg-analysis-project

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download datasets
python setup_data.py

# 5. Train
python -m pipeline_signals.train --config config.yaml     # Signal only
python -m pipeline_images.train --config config.yaml      # Image only
python -m pipeline_fusion.train --config config.yaml      # Fusion (both)
```

## Project Structure

```
common/              Shared code (utils, preprocessing, losses, backbones)
pipeline_signals/    Signal pipeline (ECGResNet1D, ~17.5M params)
pipeline_images/     Image pipeline (EfficientNet-B0, ~4M params)
pipeline_fusion/     Fusion pipeline (GatedFusion + multi-task, ~22.5M params)
setup_data.py        Dataset download & setup
config.yaml          All hyperparameters and paths
requirements.txt     Python dependencies
```

## Dataset

**PTB-XL** — 21,799 12-lead ECG recordings (10s, 500Hz) from 18,885 patients.

- Signals: downloaded automatically via `setup_data.py` from PhysioNet (or [Kaggle](https://www.kaggle.com/datasets/khyeh0719/ptb-xl-dataset))
- Images: manual download from [Kaggle](https://www.kaggle.com/datasets/bjoernjostein/ptb-xl-ecg-image-gmc2024) (the script tells you where to put them)

| Class | Description | Approx % |
|-------|-------------|----------|
| NORM  | Normal ECG  | 46% |
| MI    | Myocardial Infarction | 27% |
| STTC  | ST/T Changes | 25% |
| CD    | Conduction Disturbance | 24% |
| ARR   | Arrhythmia | 12% |

## Requirements

- Python 3.8+
- PyTorch 2.0+
- scipy (mandatory)
- ~7 GB disk space for datasets
- GPU optional (works on CPU)

See [requirements.txt](requirements.txt) for full list.

## Documentation

See [PROJECT_DESCRIPTION.md](PROJECT_DESCRIPTION.md) for full technical documentation (architecture, losses, training details, etc.)
