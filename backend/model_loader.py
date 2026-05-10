"""
RhythmAI — Model Loader
Loads the local ECG fusion checkpoint bundled in `ecg_data/ecg_data/projet`.
Falls back to a deterministic mock only if the real project or weights are
missing, so the API still starts in partial environments.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger("rhythmai.model")

CLASSES = ["NORM", "MI", "STTC", "CD", "ARR"]

BACKEND_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = BACKEND_DIR.parent.parent
LOCAL_PROJECT_DIR = WORKSPACE_ROOT / "ecg_data" / "ecg_data" / "projet"
LEGACY_PROJECT_DIR = WORKSPACE_ROOT / "ecg-analysis-project"

PROJECT_DIR = LOCAL_PROJECT_DIR if (LOCAL_PROJECT_DIR / "pipeline_fusion" / "model.py").exists() else LEGACY_PROJECT_DIR
CHECKPOINT = PROJECT_DIR / "models" / "checkpoints" / "best_fusion_model.pth"
THRESHOLDS_PATH = PROJECT_DIR / "models" / "checkpoints" / "optimal_thresholds.json"
FINAL_RESULTS_PATH = PROJECT_DIR / "models" / "checkpoints" / "final_results.json"


class MockFusionModel:
    """Deterministic mock used only when the real checkpoint cannot be loaded."""

    def __init__(self):
        self.device = "cpu"
        logger.warning(
            "Real model checkpoint not found. Using MockFusionModel instead. Expected checkpoint: %s",
            CHECKPOINT,
        )

    def __call__(self, signal_tensor, image_tensor, mask_tensor, metadata_tensor):
        import torch

        batch = signal_tensor.shape[0]
        energy = float(signal_tensor.abs().mean())
        rng = np.random.default_rng(int(energy * 1e4) % (2**31))
        logits = rng.random((batch, len(CLASSES))).astype(np.float32)
        return torch.tensor(logits)

    def eval(self):
        return self

    def to(self, device):
        return self


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else {}


def _load_thresholds() -> dict[str, float]:
    thresholds = _load_json(THRESHOLDS_PATH)
    if not thresholds:
        thresholds = _load_json(FINAL_RESULTS_PATH).get("optimal_thresholds", {})
    if not isinstance(thresholds, dict):
        return {label: 0.5 for label in CLASSES}
    return {label: float(thresholds.get(label, 0.5)) for label in CLASSES}


def _load_checkpoint(path: Path, device: str):
    import torch

    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _try_load_real_model():
    if not PROJECT_DIR.exists():
        logger.warning("ECG project directory not found at %s", PROJECT_DIR)
        return None

    if str(PROJECT_DIR) not in sys.path:
        sys.path.insert(0, str(PROJECT_DIR))

    try:
        import torch
        from pipeline_fusion.model import ECGFusionModel

        if not CHECKPOINT.exists():
            logger.warning("Checkpoint not found at %s", CHECKPOINT)
            return None

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = ECGFusionModel(num_classes=len(CLASSES))
        checkpoint = _load_checkpoint(CHECKPOINT, device)
        state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.eval()

        logger.info("Loaded ECGFusionModel from %s on %s", CHECKPOINT, device)
        return model
    except Exception as exc:
        logger.warning("Could not load local ECG model: %s", exc)
        return None


_model = None


def get_model():
    global _model
    if _model is None:
        _model = _try_load_real_model() or MockFusionModel()
    return _model


def _build_metadata_tensor(torch_mod, patient_age: int | None, patient_sex: str | None):
    age_value = 0.0 if patient_age is None else max(0.0, min(float(patient_age) / 100.0, 1.5))
    sex_value = 0.5
    if isinstance(patient_sex, str):
        sex_normalized = patient_sex.strip().lower()
        if sex_normalized in {"m", "male", "1"}:
            sex_value = 1.0
        elif sex_normalized in {"f", "female", "0"}:
            sex_value = 0.0
    return torch_mod.tensor([[age_value, sex_value, 0.0]], dtype=torch_mod.float32)


def run_inference(
    signal_np: np.ndarray | None,
    image_tensor=None,
    mode: str = "fusion",
    patient_age: int | None = None,
    patient_sex: str | None = None,
) -> dict:
    """Run the ECG fusion model and return probabilities in the legacy response shape."""
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_model()
    thresholds = _load_thresholds()

    if signal_np is not None:
        sig_t = torch.tensor(signal_np, dtype=torch.float32).unsqueeze(0).to(device)
    else:
        sig_t = torch.zeros(1, 12, 5000, device=device)

    if image_tensor is not None:
        img_t = image_tensor.unsqueeze(0).to(device)
    else:
        img_t = torch.zeros(1, 3, 224, 224, device=device)

    mask_s = 1.0 if signal_np is not None else 0.0
    mask_i = 1.0 if image_tensor is not None else 0.0
    mask_t = torch.tensor([[mask_s, mask_i]], dtype=torch.float32, device=device)
    meta_t = _build_metadata_tensor(torch, patient_age, patient_sex).to(device)

    started = time.perf_counter()
    with torch.no_grad():
        logits = model(sig_t, img_t, mask_t, meta_t)
        probs = torch.sigmoid(logits).detach().cpu().numpy()[0]
    elapsed = round(time.perf_counter() - started, 3)

    label_probs = {label: float(prob) for label, prob in zip(CLASSES, probs)}
    ranked_labels = sorted(label_probs.items(), key=lambda item: item[1], reverse=True)
    active_labels = [label for label, prob in ranked_labels if prob >= thresholds.get(label, 0.5)]
    predicted = active_labels[0] if active_labels else ranked_labels[0][0]
    confidence = round(label_probs[predicted] * 100, 1)
    scores = [round(label_probs[label] * 100, 1) for label in CLASSES]

    return {
        "predicted": predicted,
        "predicted_labels": active_labels,
        "scores": scores,
        "probabilities": label_probs,
        "thresholds": thresholds,
        "confidence": confidence,
        "inference_time": elapsed,
        "mode": mode,
    }
