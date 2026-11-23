"""R-peak detection: neurokit2 (reference) and a simple Pan-Tompkins-style detector.

Everything downstream (RR intervals, HRV, the model) depends on R-peaks, so we
keep two detectors and measure how well they agree.
"""

import neurokit2 as nk
import numpy as np
from scipy.signal import find_peaks

# 250 ms refractory period = max ~240 bpm. Two real R-peaks can't be closer.
REFRACTORY_S = 0.25


def correct_polarity(filtered: np.ndarray, fs: int = 300) -> np.ndarray:
    """Flip the signal if the QRS points downward.

    Handheld AliveCor recordings are often inverted (device held the other way).
    Each 2 s window almost always contains a beat, so we take the max and min
    of every window and compare their medians. One big artifact only affects
    one window, unlike the global max/min (an earlier global-percentile version
    was fooled by single spikes on noisy recordings).
    """
    x = np.asarray(filtered, dtype=float)
    win = 2 * fs
    n_windows = len(x) // win
    if n_windows < 2:
        top, bottom = np.max(x), np.min(x)
    else:
        windows = x[: n_windows * win].reshape(n_windows, win)
        top = np.median(windows.max(axis=1))
        bottom = np.median(windows.min(axis=1))
    return -x if abs(bottom) > abs(top) else x


def detect_r_peaks_neurokit(filtered: np.ndarray, fs: int) -> np.ndarray:
    """Sample indices of R-peaks from neurokit2's default detector."""
    x = correct_polarity(filtered, fs)
    try:
        _, info = nk.ecg_peaks(x, sampling_rate=fs, method="neurokit")
    except Exception:
        # neurokit can raise on flat or extremely noisy signals; downstream
        # code treats "no peaks" as a low-quality recording.
        return np.array([], dtype=int)
    return np.asarray(info["ECG_R_Peaks"], dtype=int)


def detect_r_peaks_simple(filtered: np.ndarray, fs: int) -> np.ndarray:
    """A minimal Pan-Tompkins-style detector.

    1. Derivative: QRS is the steepest part of the beat, so slope highlights it.
    2. Square: makes everything positive and boosts big slopes over small ones.
    3. Moving-window integration (150 ms, about one QRS wide): merges the
       QRS's several slope spikes into one bump per beat.
    4. Threshold + refractory period: keep bumps above a fraction of a robust max.
    5. Refine: the integrated bump lags the R-peak, so look back at the filtered
       signal near each bump and take its maximum.
    """
    x = correct_polarity(filtered, fs)

    slope = np.diff(x, prepend=x[0])
    energy = slope**2
    window = max(1, int(0.15 * fs))
    integrated = np.convolve(energy, np.ones(window) / window, mode="same")

    # 98th percentile instead of max, so one artifact spike can't set a
    # threshold so high that every real beat is missed.
    threshold = 0.3 * np.percentile(integrated, 98)
    candidates, _ = find_peaks(integrated, height=threshold, distance=int(REFRACTORY_S * fs))

    search = int(0.1 * fs)
    refined = []
    for c in candidates:
        lo, hi = max(0, c - search), min(len(x), c + search)
        refined.append(lo + int(np.argmax(x[lo:hi])))

    # Refinement can pull two candidates onto (nearly) the same beat; drop the
    # later one if it falls inside the refractory period.
    peaks = []
    for p in sorted(set(refined)):
        if not peaks or p - peaks[-1] >= REFRACTORY_S * fs:
            peaks.append(p)
    return np.asarray(peaks, dtype=int)


def match_peaks(reference: np.ndarray, test: np.ndarray, fs: int, tolerance_s: float = 0.05) -> dict:
    """Compare two peak lists. A test peak within +/-50 ms of a reference peak is a match.

    Returns true positives, false negatives, false positives, sensitivity
    (fraction of reference beats found) and PPV (fraction of test beats that are real).
    """
    tol = int(tolerance_s * fs)
    unmatched = list(test)
    tp = 0
    for r in reference:
        # Greedy nearest match; each test peak can be used once.
        if not unmatched:
            break
        dists = np.abs(np.asarray(unmatched) - r)
        j = int(np.argmin(dists))
        if dists[j] <= tol:
            tp += 1
            unmatched.pop(j)
    fn = len(reference) - tp
    fp = len(test) - tp
    sensitivity = tp / len(reference) if len(reference) else float("nan")
    ppv = tp / len(test) if len(test) else float("nan")
    return {"tp": tp, "fn": fn, "fp": fp, "sensitivity": sensitivity, "ppv": ppv}
