import math

import neurokit2 as nk
import numpy as np
import pytest

from af.features import (
    FEATURE_COLUMNS,
    analyze,
    cv_rr,
    pnn50,
    poincare_sd1_sd2,
    rmssd,
    rr_features,
    rr_intervals,
    sample_entropy,
)

FS = 300


def test_rr_intervals_in_seconds():
    assert np.allclose(rr_intervals(np.array([0, 300, 750]), FS), [1.0, 1.5])


def test_rmssd_known_values():
    assert rmssd(np.array([1.0, 1.0, 1.0])) == 0
    # Successive differences are all +/-0.2 s, so RMSSD = 0.2.
    assert rmssd(np.array([0.8, 1.0, 0.8, 1.0])) == pytest.approx(0.2)


def test_pnn50_known_values():
    # Diffs: 0.03, 0.10, 0.06 -> two of three exceed 50 ms.
    assert pnn50(np.array([1.00, 1.03, 0.93, 0.99])) == pytest.approx(2 / 3)


def test_cv_rr_is_rate_independent():
    rr = np.array([0.8, 0.9, 1.0, 0.85])
    assert cv_rr(rr) == pytest.approx(cv_rr(rr * 2))


def test_sample_entropy_regular_lower_than_random():
    rng = np.random.default_rng(0)
    regular = np.tile([0.8, 0.85, 0.9, 0.85], 15)
    irregular = rng.uniform(0.4, 1.2, 60)
    assert sample_entropy(regular) < sample_entropy(irregular)


def test_sample_entropy_undefined_cases_return_nan():
    assert math.isnan(sample_entropy(np.ones(20)))   # zero variance -> r = 0
    assert math.isnan(sample_entropy(np.array([1.0, 2.0])))  # too short


def test_poincare_constant_rhythm_is_zero():
    sd1, sd2 = poincare_sd1_sd2(np.full(10, 0.8))
    assert sd1 == pytest.approx(0) and sd2 == pytest.approx(0)


def test_rr_features_too_few_beats_gives_nan_not_crash():
    out = rr_features(np.array([0.8]))
    assert out["n_beats"] == 2
    assert math.isnan(out["rmssd"])


def test_af_like_rhythm_more_irregular_than_sinus():
    rng = np.random.default_rng(1)
    sinus = 0.8 + 0.01 * rng.standard_normal(40)
    af = rng.uniform(0.4, 1.1, 40)
    s, a = rr_features(sinus), rr_features(af)
    for key in ["rmssd", "cv_rr", "pnn50", "sd1_sd2"]:
        assert a[key] > s[key], key


def test_analyze_on_synthetic_ecg():
    ecg = nk.ecg_simulate(duration=30, sampling_rate=FS, heart_rate=75, noise=0.02, random_state=0)
    result = analyze(ecg, FS)
    feats = result["features"]
    assert set(FEATURE_COLUMNS) <= set(feats)
    assert feats["heart_rate"] == pytest.approx(75, abs=5)
    assert feats["duration_s"] == pytest.approx(30)
    assert feats["template_corr"] > 0.9
    assert feats["detector_agreement"] > 0.9


def test_analyze_survives_pure_noise():
    rng = np.random.default_rng(0)
    feats = analyze(rng.standard_normal(10 * FS), FS)["features"]
    assert set(FEATURE_COLUMNS) <= set(feats)
