"""
CardioScan AI — Preprocessing Utilities
Mirrors the interface expected by pipeline_fusion from ecg-analysis-project.
Used by main.py when the real model is loaded.
"""

import numpy as np
from scipy import signal as scipy_signal


# ── Signal preprocessing ──────────────────────────────────────────────────────
def bandpass_filter(sig: np.ndarray, fs: float = 500,
                    lowcut: float = 0.5, highcut: float = 40.0,
                    order: int = 4) -> np.ndarray:
    """Butterworth bandpass filter applied per lead."""
    nyq = fs / 2.0
    b, a = scipy_signal.butter(order,
                                [lowcut / nyq, highcut / nyq],
                                btype="band")
    return scipy_signal.filtfilt(b, a, sig, axis=-1)


def notch_filter(sig: np.ndarray, fs: float = 500,
                 freq: float = 50.0, q: float = 30.0) -> np.ndarray:
    """Notch filter to remove power-line interference (50 Hz)."""
    b, a = scipy_signal.iirnotch(freq / (fs / 2), q)
    return scipy_signal.filtfilt(b, a, sig, axis=-1)


def zscore_normalize(sig: np.ndarray) -> np.ndarray:
    """Z-score normalise each lead independently."""
    mean = sig.mean(axis=-1, keepdims=True)
    std  = sig.std(axis=-1, keepdims=True) + 1e-8
    return (sig - mean) / std


def preprocess_ecg_signal(signal: np.ndarray, fs: float = 500) -> np.ndarray:
    """
    Full preprocessing pipeline matching common/preprocessing.py.
    Input : (12, N) float32
    Output: (12, 5000) float32 — bandpass → notch → zscore
    """
    if signal.ndim == 2 and signal.shape[0] != 12:
        signal = signal.T                          # ensure (12, N)

    # Pad / truncate to exactly 5000 samples
    target = 5000
    if signal.shape[1] < target:
        pad = np.zeros((12, target - signal.shape[1]))
        signal = np.concatenate([signal, pad], axis=1)
    else:
        signal = signal[:, :target]

    signal = bandpass_filter(signal, fs)
    signal = notch_filter(signal, fs)
    signal = zscore_normalize(signal)
    return signal.astype(np.float32)


# ── Image preprocessing ───────────────────────────────────────────────────────
def preprocess_ecg_image(img_array: np.ndarray,
                          target_size: int = 224) -> np.ndarray:
    """
    Resize and normalise an ECG image for EfficientNet-B0.
    Input : HxWx3 uint8
    Output: 3xHxW float32 (ImageNet normalised)
    """
    from PIL import Image as PilImage
    import torchvision.transforms as T

    img = PilImage.fromarray(img_array).convert("RGB")
    transform = T.Compose([
        T.Resize((target_size, target_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std =[0.229, 0.224, 0.225]),
    ])
    return transform(img)        # (3, H, W) torch.Tensor


# ── Numpy signal from CSV ─────────────────────────────────────────────────────
def load_signal_csv(path: str, fs: float = 500) -> np.ndarray:
    """
    Load a 12-lead ECG from a CSV file.
    Accepts: rows=samples, cols=12 leads  OR  rows=12, cols=samples.
    Returns: (12, 5000) float32 preprocessed.
    """
    data = np.genfromtxt(path, delimiter=",", skip_header=1)
    if data.shape[0] == 12:
        signal = data
    elif data.shape[1] == 12:
        signal = data.T
    else:
        raise ValueError(f"Unexpected CSV shape {data.shape}. Expected 12 leads.")
    return preprocess_ecg_signal(signal, fs)


def load_signal_npy(path: str, fs: float = 500) -> np.ndarray:
    """Load a .npy ECG file. Returns (12, 5000) float32."""
    data = np.load(path)
    return preprocess_ecg_signal(data, fs)
