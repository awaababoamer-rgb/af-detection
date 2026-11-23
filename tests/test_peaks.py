import neurokit2 as nk
import numpy as np
import pytest

from af.filtering import bandpass_filter
from af.peaks import correct_polarity, detect_r_peaks_neurokit, detect_r_peaks_simple, match_peaks

FS = 300


@pytest.fixture(scope="module")
def synthetic_ecg():
    # 30 s at 60 bpm -> about 30 beats. Synthetic so the test needs no dataset download.
    ecg = nk.ecg_simulate(duration=30, sampling_rate=FS, heart_rate=60, noise=0.02, random_state=0)
    return bandpass_filter(ecg, FS)


@pytest.mark.parametrize("detector", [detect_r_peaks_neurokit, detect_r_peaks_simple])
def test_detects_expected_number_of_beats(synthetic_ecg, detector):
    peaks = detector(synthetic_ecg, FS)
    assert 27 <= len(peaks) <= 31


@pytest.mark.parametrize("detector", [detect_r_peaks_neurokit, detect_r_peaks_simple])
def test_handles_inverted_ecg(synthetic_ecg, detector):
    upright = detector(synthetic_ecg, FS)
    inverted = detector(-synthetic_ecg, FS)
    assert match_peaks(upright, inverted, FS)["sensitivity"] > 0.95


def test_polarity_not_fooled_by_single_artifact(synthetic_ecg):
    # One huge downward spike used to make the whole recording look inverted.
    x = synthetic_ecg.copy()
    x[4500] = -20 * np.max(x)
    assert np.allclose(correct_polarity(x, FS), x)
    assert np.allclose(correct_polarity(-synthetic_ecg, FS), synthetic_ecg)


def test_simple_detector_agrees_with_neurokit(synthetic_ecg):
    ref = detect_r_peaks_neurokit(synthetic_ecg, FS)
    simple = detect_r_peaks_simple(synthetic_ecg, FS)
    result = match_peaks(ref, simple, FS)
    assert result["sensitivity"] > 0.95
    assert result["ppv"] > 0.95


def test_simple_detector_respects_refractory_period(synthetic_ecg):
    peaks = detect_r_peaks_simple(synthetic_ecg, FS)
    assert np.all(np.diff(peaks) >= 0.25 * FS)


def test_match_peaks_counts():
    ref = np.array([100, 400, 700, 1000])
    test = np.array([105, 400, 900])  # 2 matches (within 15 samples), 1 false positive
    r = match_peaks(ref, test, FS)
    assert (r["tp"], r["fn"], r["fp"]) == (2, 2, 1)
    assert r["sensitivity"] == pytest.approx(0.5)
    assert r["ppv"] == pytest.approx(2 / 3)
