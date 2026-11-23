import numpy as np

from af.filtering import bandpass_filter

FS = 300


def sine(freq, seconds=10, fs=FS):
    t = np.arange(seconds * fs) / fs
    return np.sin(2 * np.pi * freq * t)


def middle(x):
    # Ignore the edges, where any filter has start-up transients.
    n = len(x)
    return x[n // 4 : 3 * n // 4]


def test_output_same_length():
    x = sine(10)
    assert len(bandpass_filter(x, FS)) == len(x)


def test_passes_in_band_frequency():
    # 10 Hz is mid-QRS band: amplitude should survive almost unchanged.
    y = middle(bandpass_filter(sine(10), FS))
    assert 0.95 < np.max(np.abs(y)) < 1.05


def test_removes_baseline_wander():
    y = middle(bandpass_filter(sine(0.05, seconds=60), FS))
    assert np.max(np.abs(y)) < 0.05


def test_attenuates_mains_interference():
    # 60 Hz is only 1.5x the 40 Hz cutoff, so a 4th-order filter reduces it by
    # about 23 dB (to ~7% amplitude) rather than removing it. 20 dB = 10%.
    y = middle(bandpass_filter(sine(60), FS))
    assert np.max(np.abs(y)) < 0.1


def test_zero_phase_keeps_peak_position():
    # A Gaussian bump (QRS-like width) must not shift in time after filtering,
    # otherwise R-peak indices from the filtered signal would be off.
    t = np.arange(10 * FS)
    x = np.exp(-0.5 * ((t - 1500) / 6) ** 2)
    assert abs(int(np.argmax(bandpass_filter(x, FS))) - 1500) <= 1
