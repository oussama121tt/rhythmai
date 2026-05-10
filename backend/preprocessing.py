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
    """Apply Butterworth bandpass filter per lead.
    
    Args:
        sig: Input signal, shape (channels, samples)
        fs: Sampling frequency in Hz
        lowcut: Low cutoff frequency (Hz)
        highcut: High cutoff frequency (Hz)
        order: Filter order
    Returns:
        Filtered signal, same shape as input
    """
    nyq = fs / 2.0
    b, a = scipy_signal.butter(order,
                                [lowcut / nyq, highcut / nyq],
                                btype="band")
    return scipy_signal.filtfilt(b, a, sig, axis=-1)


def notch_filter(sig: np.ndarray, fs: float = 500,
                 freq: float = 50.0, q: float = 30.0) -> np.ndarray:
    """Apply notch filter to remove power-line interference (50/60 Hz).
    
    Args:
        sig: Input signal, shape (channels, samples)
        fs: Sampling frequency in Hz
        freq: Frequency to remove (Hz), typically 50 or 60
        q: Quality factor
    Returns:
        Filtered signal, same shape as input
    """
    b, a = scipy_signal.iirnotch(freq / (fs / 2), q)
    return scipy_signal.filtfilt(b, a, sig, axis=-1)


def zscore_normalize(sig: np.ndarray) -> np.ndarray:
    """Normalize signal using z-score per lead independently.
    
    Args:
        sig: Input signal, shape (channels, samples)
    Returns:
        Normalized signal (mean=0, std=1 per channel)
    """
    mean = sig.mean(axis=-1, keepdims=True)
    std  = sig.std(axis=-1, keepdims=True) + 1e-8
    return (sig - mean) / std


def preprocess_ecg_signal(signal: np.ndarray, fs: float = 500) -> np.ndarray:
    """Complete ECG preprocessing pipeline.
    
    Steps: bandpass (0.5-40 Hz) → notch (50 Hz) → z-score normalize → pad/truncate to 5000 samples
    
    Args:
        signal: Input ECG signal, shape (12, N) or (N, 12) float32
        fs: Sampling frequency in Hz
    Returns:
        Preprocessed signal, shape (12, 5000) float32
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
    """Resize and normalize ECG image for EfficientNet-B3 inference.
    
    Input : HxWx3 uint8 (any resolution)
    Output: 3x224x224 float32 (ImageNet normalized)
    
    Args:
        img_array: Input image array
        target_size: Target resolution (pixels)
    Returns:
        Preprocessed image as torch.Tensor (3, target_size, target_size) float32
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
    """Load 12-lead ECG from CSV file and preprocess.
    
    Accepts both formats:
    - rows=samples, cols=12 leads (N×12 CSV)
    - rows=12, cols=samples (12×N CSV)
    
    Args:
        path: Path to CSV file
        fs: Sampling frequency in Hz
    Returns:
        Preprocessed signal, shape (12, 5000) float32
    Raises:
        ValueError: If CSV shape is not compatible with 12 leads
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
    """Load 12-lead ECG from NumPy .npy file and preprocess.
    
    Args:
        path: Path to .npy file (shape should be (12, N) or (N, 12))
        fs: Sampling frequency in Hz
    Returns:
        Preprocessed signal, shape (12, 5000) float32
    """
    data = np.load(path)
    return preprocess_ecg_signal(data, fs)
