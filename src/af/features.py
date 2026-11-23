"""RR-interval / HRV features plus signal-quality features.

AF's hallmark on the ECG is an "irregularly irregular" ventricular rhythm
(the AV node is bombarded by chaotic atrial activity and conducts at random)
and absent P-waves. Most features below quantify RR irregularity; the quality
features help the model separate real arrhythmia from noise.
"""

import numpy as np
from scipy.signal import welch

from af.filtering import bandpass_filter
from af.peaks import correct_polarity, detect_r_peaks_neurokit, detect_r_peaks_simple, match_peaks

# Order matters: this list defines the model's input columns.
RR_FEATURES = [
    "n_beats",
    "mean_rr",
    "heart_rate",
    "sdnn",
    "rmssd",
    "norm_rmssd",
    "cv_rr",
    "pnn50",
    "sd1",
    "sd2",
    "sd1_sd2",
    "sample_entropy",
]
QUALITY_FEATURES = [
    "duration_s",
    "amp_range",
    "hf_noise_ratio",
    "baseline_ratio",
    "template_corr",
    "detector_agreement",
]
FEATURE_COLUMNS = RR_FEATURES + QUALITY_FEATURES


# --- RR-interval features -------------------------------------------------

def rr_intervals(peaks: np.ndarray, fs: int) -> np.ndarray:
    """Seconds between consecutive R-peaks."""
    return np.diff(np.asarray(peaks)) / fs


def rmssd(rr: np.ndarray) -> float:
    """Root mean square of successive RR differences (seconds).

    Measures beat-to-beat change. In sinus rhythm neighbouring beats are
    similar; in AF each interval is unrelated to the last, so RMSSD jumps.
    """
    if len(rr) < 2:
        return float("nan")
    return float(np.sqrt(np.mean(np.diff(rr) ** 2)))


def pnn50(rr: np.ndarray) -> float:
    """Fraction of successive RR differences larger than 50 ms."""
    if len(rr) < 2:
        return float("nan")
    return float(np.mean(np.abs(np.diff(rr)) > 0.05))


def cv_rr(rr: np.ndarray) -> float:
    """Coefficient of variation: SD of RR / mean RR.

    Dividing by the mean makes it rate-independent: 50 ms of variation is a
    lot at 150 bpm but little at 50 bpm.
    """
    if len(rr) < 2:
        return float("nan")
    return float(np.std(rr, ddof=1) / np.mean(rr))


def sample_entropy(x: np.ndarray, m: int = 2, r_factor: float = 0.2) -> float:
    """Sample entropy (Richman & Moorman 2000).

    Asks: if two stretches of m beats look alike (within tolerance r), how
    often do they still look alike when extended to m+1 beats?
    Predictable rhythm (sinus) -> low entropy; random rhythm (AF) -> high.
    Returns NaN when undefined (too few beats or no matches), which is common
    for short recordings; the model's imputer handles it.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < m + 2:
        return float("nan")
    r = r_factor * np.std(x)
    if r == 0:
        return float("nan")

    def count_matches(length: int) -> int:
        # Use the same n - m starting points for both lengths so B and A are comparable.
        templates = np.array([x[i : i + length] for i in range(n - m)])
        count = 0
        for i in range(len(templates) - 1):
            # Chebyshev distance: the largest element-wise difference.
            dist = np.max(np.abs(templates[i + 1 :] - templates[i]), axis=1)
            count += int(np.sum(dist <= r))
        return count

    b = count_matches(m)
    a = count_matches(m + 1)
    if a == 0 or b == 0:
        return float("nan")
    return float(-np.log(a / b))


def poincare_sd1_sd2(rr: np.ndarray) -> tuple[float, float]:
    """Poincare plot widths (plot of RR[n+1] vs RR[n]).

    SD1 = spread perpendicular to the identity line (short-term variability).
    SD2 = spread along it (longer-term variability). Sinus rhythm gives a thin
    cigar (small SD1/SD2); AF gives a round cloud (SD1/SD2 near 1).
    """
    if len(rr) < 3:
        return float("nan"), float("nan")
    diff_var = np.var(np.diff(rr), ddof=1)
    sd1 = np.sqrt(0.5 * diff_var)
    sd2_sq = 2 * np.var(rr, ddof=1) - 0.5 * diff_var
    sd2 = np.sqrt(sd2_sq) if sd2_sq > 0 else 0.0
    return float(sd1), float(sd2)


def rr_features(rr: np.ndarray) -> dict:
    out = dict.fromkeys(RR_FEATURES, float("nan"))
    out["n_beats"] = len(rr) + 1 if len(rr) else 0
    # Below 3 intervals almost every statistic is meaningless.
    if len(rr) < 3:
        return out
    mean_rr = float(np.mean(rr))
    sd1, sd2 = poincare_sd1_sd2(rr)
    out.update(
        mean_rr=mean_rr,
        heart_rate=60.0 / float(np.median(rr)),
        sdnn=float(np.std(rr, ddof=1)),
        rmssd=rmssd(rr),
        norm_rmssd=rmssd(rr) / mean_rr,
        cv_rr=cv_rr(rr),
        pnn50=pnn50(rr),
        sd1=sd1,
        sd2=sd2,
        sd1_sd2=sd1 / sd2 if sd2 > 0 else float("nan"),
        sample_entropy=sample_entropy(rr),
    )
    return out


# --- Signal-quality features ---------------------------------------------

def band_power_ratios(raw: np.ndarray, fs: int) -> tuple[float, float]:
    """Fraction of raw-signal power above 40 Hz (muscle/mains noise) and below
    0.5 Hz (baseline wander). High values suggest a noisy recording."""
    freqs, psd = welch(raw - np.mean(raw), fs=fs, nperseg=min(len(raw), 4 * fs))
    total = np.sum(psd)
    if total == 0:
        return float("nan"), float("nan")
    return float(np.sum(psd[freqs > 40]) / total), float(np.sum(psd[freqs < 0.5]) / total)


def template_correlation(filtered: np.ndarray, peaks: np.ndarray, fs: int) -> float:
    """Mean correlation of each beat with the median beat shape.

    Clean recordings have near-identical QRS complexes (close to 1). Noise or
    frequent ectopic beats with different morphology push this down.
    """
    before, after = int(0.2 * fs), int(0.4 * fs)
    beats = [filtered[p - before : p + after] for p in peaks if p - before >= 0 and p + after <= len(filtered)]
    if len(beats) < 3:
        return float("nan")
    beats = np.array(beats)
    template = np.median(beats, axis=0)
    corrs = [np.corrcoef(b, template)[0, 1] for b in beats]
    return float(np.nanmean(corrs))


# --- Full pipeline ---------------------------------------------------------

def analyze(raw: np.ndarray, fs: int) -> dict:
    """Filter, detect peaks and compute all features for one recording.

    Returns a dict with the filtered signal, the R-peaks and the feature dict,
    so the web demo can plot exactly what the model saw.
    """
    raw = np.asarray(raw, dtype=float)
    filtered = correct_polarity(bandpass_filter(raw, fs), fs)
    peaks = detect_r_peaks_neurokit(filtered, fs)
    simple_peaks = detect_r_peaks_simple(filtered, fs)

    feats = rr_features(rr_intervals(peaks, fs))
    hf, baseline = band_power_ratios(raw, fs)
    agreement = match_peaks(peaks, simple_peaks, fs)
    # F1 of the two detectors against each other: two independent detectors
    # agree on clean signals and disagree on noise (the "bSQI" idea).
    denom = 2 * agreement["tp"] + agreement["fn"] + agreement["fp"]
    feats.update(
        duration_s=len(raw) / fs,
        amp_range=float(np.percentile(filtered, 99) - np.percentile(filtered, 1)),
        hf_noise_ratio=hf,
        baseline_ratio=baseline,
        template_corr=template_correlation(filtered, peaks, fs),
        detector_agreement=2 * agreement["tp"] / denom if denom else float("nan"),
    )
    return {"filtered": filtered, "peaks": peaks, "features": feats}


def extract_features(raw: np.ndarray, fs: int) -> dict:
    return analyze(raw, fs)["features"]
