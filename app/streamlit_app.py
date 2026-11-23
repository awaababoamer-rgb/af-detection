"""Streamlit demo: upload or pick an ECG, see filtered trace, R-peaks, prediction.

Run locally:
    streamlit run app/streamlit_app.py
"""

import json
import sys
import tempfile
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import wfdb

ROOT = Path(__file__).resolve().parents[1]
# Lets the app import the `af` package on hosts where it isn't pip-installed.
sys.path.insert(0, str(ROOT / "src"))

from af.config import FS  # noqa: E402
from af.data import load_csv_signal  # noqa: E402
from af.features import analyze  # noqa: E402

FEATURE_LABELS = {
    "n_beats": "Beats detected",
    "mean_rr": "Mean RR (s)",
    "heart_rate": "Heart rate (bpm)",
    "sdnn": "SDNN (s)",
    "rmssd": "RMSSD (s)",
    "norm_rmssd": "RMSSD / mean RR",
    "cv_rr": "RR coefficient of variation",
    "pnn50": "pNN50",
    "sd1": "Poincare SD1 (s)",
    "sd2": "Poincare SD2 (s)",
    "sd1_sd2": "Poincare SD1/SD2",
    "sample_entropy": "Sample entropy of RR",
    "duration_s": "Duration (s)",
    "amp_range": "Amplitude range (mV)",
    "hf_noise_ratio": "Power > 40 Hz (fraction)",
    "baseline_ratio": "Power < 0.5 Hz (fraction)",
    "template_corr": "Beat shape consistency",
    "detector_agreement": "R-peak detector agreement",
}

st.set_page_config(page_title="AF detection demo", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load(ROOT / "models" / "model.joblib")


@st.cache_data
def load_samples() -> dict:
    return json.loads((ROOT / "app" / "samples.json").read_text())


@st.cache_data(show_spinner="Fetching recording from PhysioNet ...")
def fetch_physionet_record(record: str) -> tuple[np.ndarray, int]:
    # Sample recordings are streamed from PhysioNet instead of being stored in
    # the repo, so no raw ECG data is ever committed.
    folder = record[:3]  # A00001 lives in training/A00/
    rec = wfdb.rdrecord(record, pn_dir=f"challenge-2017/1.0.0/training/{folder}")
    return rec.p_signal[:, 0].astype(float), int(rec.fs)


def read_upload(files) -> tuple[np.ndarray, int] | None:
    names = {Path(f.name).suffix.lower(): f for f in files}
    if ".hea" in names and ".mat" in names:
        with tempfile.TemporaryDirectory() as tmp:
            for f in files:
                (Path(tmp) / f.name).write_bytes(f.getvalue())
            stem = Path(names[".hea"].name).stem
            rec = wfdb.rdrecord(str(Path(tmp) / stem))
        return rec.p_signal[:, 0].astype(float), int(rec.fs)
    for ext in (".csv", ".txt"):
        if ext in names:
            return load_csv_signal(names[ext], fs=FS)
    st.error("Upload either a .hea + .mat pair, or a one-column .csv/.txt sampled at 300 Hz in mV.")
    return None


def plot_trace(filtered: np.ndarray, peaks: np.ndarray, fs: int, seconds: float):
    n = min(len(filtered), int(seconds * fs))
    t = np.arange(n) / fs
    p = peaks[peaks < n]
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(t, filtered[:n], color="black", lw=0.7)
    ax.plot(p / fs, filtered[p], "o", ms=6, mfc="none", color="tab:red", label="R-peaks")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("mV")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


# --- Page ------------------------------------------------------------------

st.title("Atrial fibrillation detection from single-lead ECG")
st.warning(
    "**Not a medical device. Research and education demo only.** "
    "Do not use this to make any decision about anyone's health."
)

artifact = load_model()
samples = load_samples()

with st.sidebar:
    st.header("Input")
    source = st.radio("ECG source", ["Sample recording", "Upload a file"])
    signal = fs = None
    if source == "Sample recording":
        choice = st.selectbox("Recording", list(samples), format_func=lambda k: f"{k}  ({samples[k]})")
        try:
            signal, fs = fetch_physionet_record(choice)
        except Exception as e:  # network problems on the host
            st.error(f"Could not fetch from PhysioNet: {e}")
    else:
        uploaded = st.file_uploader(
            "WFDB (.hea + .mat) or one-column CSV in mV at 300 Hz",
            type=["hea", "mat", "csv", "txt"],
            accept_multiple_files=True,
        )
        if uploaded:
            loaded = read_upload(uploaded)
            if loaded:
                signal, fs = loaded
    st.caption(
        f"Model: {artifact['model_name']}, trained on the PhysioNet/CinC Challenge 2017 "
        f"dataset. Held-out test macro F1: {artifact['test_macro_f1']:.2f}."
    )

if signal is None:
    st.info("Pick a sample recording or upload a file to begin.")
    st.stop()

if len(signal) < 5 * fs:
    st.error("Recording is shorter than 5 seconds; too short to analyse.")
    st.stop()

result = analyze(signal, fs)
feats = result["features"]
X = pd.DataFrame([feats])[artifact["feature_columns"]]
proba = artifact["model"].predict_proba(X)[0]
classes = artifact["classes"]
pred = classes[int(np.argmax(proba))]

col1, col2 = st.columns([1, 2])
with col1:
    st.subheader("Prediction")
    st.metric("Predicted rhythm class", pred, f"model score {proba.max():.0%}", delta_color="off")
    st.bar_chart(pd.Series(proba, index=classes, name="score"), horizontal=True)
    st.caption(
        "Scores are the model's class probabilities. They are not calibrated "
        "clinical confidence, and 'Other' covers many different rhythms."
    )
with col2:
    st.subheader("Filtered ECG (0.5-40 Hz) with detected R-peaks")
    seconds = st.slider("Seconds shown", 5, int(len(signal) / fs), min(15, int(len(signal) / fs)))
    st.pyplot(plot_trace(result["filtered"], result["peaks"], fs, seconds))

st.subheader("Features behind the prediction")
st.caption(
    "Features ranked by how much the model relies on them overall (permutation "
    "importance on validation data), with this recording's value next to the "
    "typical (median) value for each class in the training data."
)
medians = artifact["class_medians"]
top = artifact["importance"].head(8).index
table = pd.DataFrame(
    {
        "Feature": [FEATURE_LABELS.get(f, f) for f in top],
        "This recording": [feats[f] for f in top],
        **{f"Typical {c}": [medians.loc[c, f] for f in top] for c in ["Normal", "AF", "Other", "Noisy"]},
    }
)
st.dataframe(table.style.format(precision=3), hide_index=True, use_container_width=True)

with st.expander("All features for this recording"):
    st.dataframe(
        pd.DataFrame({"Feature": [FEATURE_LABELS.get(k, k) for k in feats], "Value": list(feats.values())}),
        hide_index=True,
    )

st.divider()
st.caption(
    "Data: Clifford GD et al., AF Classification from a Short Single Lead ECG "
    "Recording: the PhysioNet/Computing in Cardiology Challenge 2017. Goldberger AL "
    "et al., PhysioBank, PhysioToolkit, and PhysioNet, Circulation 2000."
)
