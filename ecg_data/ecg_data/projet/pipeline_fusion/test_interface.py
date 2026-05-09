#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Streamlit inference UI for clinicians: upload signal/image and predict."""

import os
import sys
import tempfile
import json
from pathlib import Path

# Add project root to PYTHONPATH for local module imports.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import streamlit as st
import torch
import wfdb
from PIL import Image

from common.preprocessing import preprocess_ecg_signal
from common.time_frequency import build_ecg_scalogram
from common.utils import (
        CLASS_NAMES,
        IDX_TO_CLASS,
        NUM_LEADS,
        SAMPLE_RATE,
        SIGNAL_LENGTH,
)
from pipeline_fusion.model import ECGFusionModel


CLASS_DESCRIPTIONS = {
        "NORM": "Normal ECG",
        "MI": "Myocardial Infarction",
        "STTC": "ST/T Changes",
        "CD": "Conduction Disturbance",
        "ARR": "Arrhythmia",
}


def inject_css() -> None:
    """Inject a clinical blue/white theme for a professional look."""
    st.markdown(
        """
        <style>
            :root {
                --primary: #0d47a1;
                --primary-light: #1e88e5;
                --bg: #f5f8ff;
                --card: #ffffff;
                --card-strong: #eef3ff;
                --text: #0b1c38;
                --muted: #5f6b7a;
            }
            .stApp { background: radial-gradient(circle at 20% 20%, #e9f0ff 0, #f5f8ff 35%, #f5f8ff 100%); color: var(--text); }
            header[data-testid="stHeader"] { background: transparent; }
            /* Layout */
            .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
            .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 { color: var(--primary); }
            /* Hero card */
            .hero-card {
                background: linear-gradient(135deg, #ffffff 0%, #eef4ff 60%);
                border-radius: 16px;
                padding: 18px 20px;
                box-shadow: 0 14px 35px rgba(13,71,161,0.12);
                border: 1px solid #dce6ff;
            }
            /* Cards */
            .stMetric { background: var(--card); border-radius: 12px; padding: 12px; box-shadow: 0 8px 24px rgba(0,0,0,0.06); }
            /* Buttons */
            button[kind="primary"], .stButton>button {
                background: linear-gradient(90deg, var(--primary), var(--primary-light));
                color: #fff; border: none; border-radius: 10px; padding: 0.6rem 1.2rem;
                box-shadow: 0 10px 20px rgba(13,71,161,0.25);
            }
            button[kind="primary"]:hover, .stButton>button:hover { filter: brightness(1.05); }
            /* Sidebar */
            section[data-testid="stSidebar"] {
                background: linear-gradient(180deg, #e9f0ff 0%, #eef3ff 60%, #f5f8ff 100%);
                border-right: 1px solid #d6ddf2;
                box-shadow: 6px 0 18px rgba(13,71,161,0.08);
            }
            section[data-testid="stSidebar"] .sidebar-content { padding-top: 1rem; }
            /* Tables */
            .dataframe { background: var(--card); border-radius: 12px; overflow: hidden; }
            .stMarkdown code { background: #e9eefb; color: var(--primary); }
            /* Inputs */
            .stTextInput>div>div>input, .stTextArea textarea, .stFileUploader label {
                border-radius: 10px; border: 1px solid #d6ddf2; background: #f9fbff;
            }
            /* Charts */
            .stAltairChart, .stPlotlyChart, .stVegaLiteChart { background: var(--card); border-radius: 12px; padding: 12px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_built() and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@st.cache_resource
def load_model(ckpt_path: str, device: torch.device) -> ECGFusionModel:
    # V4 architecture settings used during training.
    model = ECGFusionModel(
        num_classes=len(CLASS_NAMES),
        embed_dim=256,
        pretrained=True,
        dropout=0.3,
        modality_drop_p=0.15,
    )
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


@st.cache_data
def load_thresholds(thresholds_path: str, final_results_path: str) -> dict[str, float]:
    if os.path.exists(thresholds_path):
        with open(thresholds_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return {k: float(v) for k, v in data.items() if k in CLASS_NAMES}

    if os.path.exists(final_results_path):
        with open(final_results_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        opt = data.get("optimal_thresholds", {}) if isinstance(data, dict) else {}
        if isinstance(opt, dict) and opt:
            return {k: float(v) for k, v in opt.items() if k in CLASS_NAMES}

    return {cls: 0.5 for cls in CLASS_NAMES}


@st.cache_data
def load_run_info(final_results_path: str) -> dict:
    if not os.path.exists(final_results_path):
        return {}
    with open(final_results_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _pick_existing_path(candidates: list[Path]) -> Path:
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def normalize_signal_shape(signal: np.ndarray) -> np.ndarray:
    """Normalize user-provided signal into shape (12, 5000)."""
    if signal.ndim == 1:
        signal = signal[np.newaxis, :]

    if signal.ndim != 2:
        raise ValueError("Signal format invalide. Utiliser 1D ou 2D.")

    # Accept (5000,12) or (12,5000)
    if signal.shape[0] == SIGNAL_LENGTH:
        signal = signal.T

    leads, length = signal.shape

    if leads < NUM_LEADS:
        padded = np.zeros((NUM_LEADS, length), dtype=np.float32)
        padded[:leads, :] = signal
        signal = padded
    elif leads > NUM_LEADS:
        signal = signal[:NUM_LEADS, :]

    if signal.shape[1] < SIGNAL_LENGTH:
        pad = np.zeros((NUM_LEADS, SIGNAL_LENGTH - signal.shape[1]), dtype=np.float32)
        signal = np.concatenate([signal, pad], axis=1)
    else:
        signal = signal[:, :SIGNAL_LENGTH]

    return signal.astype(np.float32)


def _signal_to_tensor(signal: np.ndarray) -> torch.Tensor:
    signal = normalize_signal_shape(signal)
    signal = preprocess_ecg_signal(signal, fs=SAMPLE_RATE)

    for i in range(NUM_LEADS):
        m = signal[i].mean()
        s = signal[i].std() + 1e-8
        signal[i] = (signal[i] - m) / s

    return torch.tensor(signal, dtype=torch.float32)


def parse_uploaded_signal(uploaded_files, signals_root: str | None = None) -> torch.Tensor:
    if not uploaded_files:
        raise ValueError("Aucun fichier signal fourni")

    if len(uploaded_files) == 1:
        uploaded_file = uploaded_files[0]
        suffix = Path(uploaded_file.name).suffix.lower()

        if suffix == ".npy":
            signal = np.load(uploaded_file)
            return _signal_to_tensor(signal)

        if suffix in {".csv", ".txt"}:
            uploaded_file.seek(0)
            signal = np.genfromtxt(uploaded_file, delimiter=",", dtype=np.float32)
            if np.isnan(signal).all():
                uploaded_file.seek(0)
                signal = np.genfromtxt(uploaded_file, delimiter=None, dtype=np.float32)
            return _signal_to_tensor(signal)

    # WFDB mode: upload .hea/.dat, and auto-resolve missing pair from local root.
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        for uf in uploaded_files:
            original = Path(uf.name).name
            # Keep stem, normalize extension casing for robust matching.
            p = Path(original)
            normalized_name = f"{p.stem}{p.suffix.lower()}"
            dest = tmpdir_path / normalized_name
            with open(dest, "wb") as f:
                f.write(uf.getbuffer())

        all_files = sorted([p for p in tmpdir_path.iterdir() if p.is_file()])
        hea_files = [p for p in all_files if p.suffix.lower() == ".hea"]
        dat_files = [p for p in all_files if p.suffix.lower() == ".dat"]

        # If user provided only one side, try to resolve counterpart from local dataset.
        if signals_root and Path(signals_root).exists():
            root = Path(signals_root)
            if not hea_files and dat_files:
                stem = dat_files[0].stem
                candidates = list(root.rglob(f"{stem}.hea"))
                if candidates:
                    src = candidates[0]
                    dst = tmpdir_path / src.name
                    dst.write_bytes(src.read_bytes())
                    all_files = sorted([p for p in tmpdir_path.iterdir() if p.is_file()])
                    hea_files = [p for p in all_files if p.suffix.lower() == ".hea"]

            if hea_files and not dat_files:
                stem = hea_files[0].stem
                candidates = list(root.rglob(f"{stem}.dat"))
                if candidates:
                    src = candidates[0]
                    dst = tmpdir_path / src.name
                    dst.write_bytes(src.read_bytes())
                    all_files = sorted([p for p in tmpdir_path.iterdir() if p.is_file()])
                    dat_files = [p for p in all_files if p.suffix.lower() == ".dat"]

        if not hea_files:
            names = ", ".join(p.name for p in all_files)
            hint = ""
            if names.endswith("_lr.dat") or "_lr.dat" in names:
                hint = " (hint: _lr. ressemble aux fichiers image ECG, pas aux signaux WFDB)"
            raise ValueError(
                "WFDB requiert un fichier .hea + .dat. "
                f"Fichiers recus: [{names}]{hint}"
            )

        hea_path = hea_files[0]
        dat_path = next(
            (
                p for p in all_files
                if p.suffix.lower() == ".dat"
                and p.stem.lower() == hea_path.stem.lower()
            ),
            None,
        )
        if dat_path is None:
            names = ", ".join(p.name for p in all_files)
            raise ValueError(
                f"Fichier .dat manquant pour {hea_path.name}. "
                f"Fichiers recus: [{names}]"
            )

        try:
            signal, _ = wfdb.rdsamp(str(hea_path.with_suffix("")))
        except FileNotFoundError as err:
            raise ValueError(
                "Fichier WFDB incomplet ou non apparié (.hea/.dat). "
                "Vérifie que les deux fichiers existent et partagent le même nom de base."
            ) from err

        signal = signal.T
        return _signal_to_tensor(signal)


def parse_uploaded_image(uploaded_file) -> tuple[torch.Tensor, Image.Image]:
    pil_img = Image.open(uploaded_file).convert("RGB")
    # Keep preprocessing close to training distribution (0..1 tensor, 300x300).
    img = pil_img.resize((300, 300), Image.Resampling.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    tensor_img = torch.from_numpy(arr)
    return tensor_img, pil_img


def build_image_from_signal_tensor(signal_tensor: torch.Tensor) -> torch.Tensor:
    signal_np = signal_tensor.detach().cpu().numpy()
    return build_ecg_scalogram(signal_np, output_size=300)


def predict_one(model: ECGFusionModel, device: torch.device,
                signal_tensor: torch.Tensor, image_tensor: torch.Tensor,
                has_signal: float, has_image: float) -> np.ndarray:
    sig = signal_tensor.unsqueeze(0).to(device)
    img = image_tensor.unsqueeze(0).to(device)
    mask = torch.tensor([[has_signal, has_image]], dtype=torch.float32, device=device)

    with torch.no_grad():
        probs = model.predict_proba(sig, img, mask).squeeze(0).cpu().numpy()
    return probs


def main():
    st.set_page_config(page_title="ECG Clinical Inference", layout="wide")
    inject_css()
    st.markdown(
        """
        <div class="hero-card">
            <h1 style="margin-bottom:4px;">ECG Clinical Inference – Système d'aide à la décision</h1>
            <p style="margin:0; color:#30415a;">Upload signal and/or image, then run prediction from the trained fusion model.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    project_root = Path(__file__).resolve().parents[1]
    default_ckpt = _pick_existing_path([
        project_root / "logs" / "best_fusion_model.pth",
        project_root / "fusion_run_artifacts" / "best_fusion_model.pth",
        project_root / "models" / "checkpoints" / "best_fusion_model.pth",
    ])
    default_thresholds = _pick_existing_path([
        project_root / "logs" / "optimal_thresholds.json",
        project_root / "fusion_run_artifacts" / "optimal_thresholds.json",
        project_root / "models" / "checkpoints" / "optimal_thresholds.json",
    ])
    default_results = _pick_existing_path([
        project_root / "logs" / "final_results.json",
        project_root / "fusion_run_artifacts" / "final_results.json",
        project_root / "models" / "checkpoints" / "final_results.json",
    ])
    default_signals_root = project_root / "PTB-XL ECG dataset" / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1"

    with st.sidebar:
        st.header("Configuration")
        ckpt_path = st.text_input("Checkpoint", str(default_ckpt))
        thresholds_path = st.text_input("Thresholds (V4)", str(default_thresholds))
        final_results_path = st.text_input("Final results (V4)", str(default_results))
        signals_root = st.text_input("Local signals root (auto-find .hea/.dat)", str(default_signals_root))
        use_signal_scalogram = st.checkbox(
            "Use V4 scalogram from signal when signal is provided",
            value=True,
        )
        st.markdown("---")
        st.markdown("Inference mode: **V4 multilabel thresholds**")
        st.markdown("Signal accepted: .hea + .dat, .csv, .txt, .npy")
        st.markdown("Expected shape: 12x5000 or 5000x12")
        st.markdown("Image accepted: .png .jpg .jpeg")

    device = resolve_device()

    if not os.path.exists(ckpt_path):
        st.error("Checkpoint not found.")
        return

    model = load_model(ckpt_path, device)
    thresholds = load_thresholds(thresholds_path, final_results_path)
    run_info = load_run_info(final_results_path)

    if run_info:
        with st.expander("V4 Run Metadata", expanded=False):
            st.write({
                "loss_type": run_info.get("loss_type", "unknown"),
                "best_val_f1": run_info.get("best_val_f1", "unknown"),
                "epochs_trained": run_info.get("epochs_trained", "unknown"),
            })
            st.write("thresholds", thresholds)

    st.subheader("Patient Input")
    up_col1, up_col2 = st.columns(2)
    with up_col1:
        uploaded_signal_files = st.file_uploader(
            "Upload ECG signal",
            type=["hea", "dat", "csv", "txt", "npy"],
            accept_multiple_files=True,
            help="WFDB (.hea + .dat) or matrix (.csv/.txt/.npy)",
        )
    with up_col2:
        uploaded_image = st.file_uploader(
            "Upload ECG image",
            type=["png", "jpg", "jpeg"],
        )

    run_btn = st.button("Run Prediction", type="primary")

    if not run_btn:
        st.stop()

    has_signal = bool(uploaded_signal_files)
    has_image = uploaded_image is not None

    if not has_signal and not has_image:
        st.error("Please upload at least one input: signal or image.")
        return

    try:
        if has_signal:
            signal_tensor = parse_uploaded_signal(uploaded_signal_files, signals_root=signals_root)
        else:
            signal_tensor = torch.zeros(NUM_LEADS, SIGNAL_LENGTH, dtype=torch.float32)

        if has_signal and use_signal_scalogram:
            image_tensor = build_image_from_signal_tensor(signal_tensor)
            pil_img = None
        elif has_image:
            image_tensor, pil_img = parse_uploaded_image(uploaded_image)
        else:
            image_tensor = torch.zeros(3, 300, 300, dtype=torch.float32)
            pil_img = None
    except Exception as exc:
        st.error(f"Input parsing failed: {exc}")
        return

    probs = predict_one(
        model,
        device,
        signal_tensor,
        image_tensor,
        1.0 if has_signal else 0.0,
        1.0 if has_image else 0.0,
    )

    thr_arr = np.array([thresholds.get(cls, 0.5) for cls in CLASS_NAMES], dtype=np.float32)
    pred_multi = (probs >= thr_arr).astype(np.int32)
    active_labels = [CLASS_NAMES[i] for i, v in enumerate(pred_multi) if v == 1]

    pred_idx = int(np.argmax(probs))
    top1_label = IDX_TO_CLASS[pred_idx]
    top1_desc = CLASS_DESCRIPTIONS[top1_label]

    left, right = st.columns(2)
    with left:
        st.subheader("Prediction")
        st.metric("Top-1 Class", top1_label)
        st.write(top1_desc)
        st.write("Predicted labels (V4 thresholds):", active_labels if active_labels else ["None"])
        proba_df = pd.DataFrame({
            "Class": CLASS_NAMES,
            "Probability": probs,
            "Threshold": thr_arr,
            "Predicted": pred_multi,
        }).sort_values("Probability", ascending=False)
        proba_df["Clinical Meaning"] = proba_df["Class"].map(CLASS_DESCRIPTIONS)
        proba_df["Probability"] = proba_df["Probability"].map(lambda x: round(float(x), 4))
        proba_df["Threshold"] = proba_df["Threshold"].map(lambda x: round(float(x), 2))
        st.dataframe(proba_df, use_container_width=True)
        st.bar_chart(proba_df.set_index("Class")["Probability"])

    with right:
        st.subheader("Input Preview")
        st.write(f"Signal provided: {has_signal}")
        st.write(f"Image provided: {has_image}")

        if has_signal:
            st.line_chart(signal_tensor[0].numpy())
        if pil_img is not None:
            st.image(pil_img, caption="Uploaded ECG image", use_container_width=True)

    st.markdown("---")
    st.warning(
        "Clinical use note: this tool is for decision support and research demo. "
        "Final diagnosis must be confirmed by a qualified cardiologist."
    )


if __name__ == "__main__":
    main()