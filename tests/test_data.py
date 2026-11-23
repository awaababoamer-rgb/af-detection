import io

import numpy as np

from af.data import load_csv_signal


def test_load_csv_signal_with_header():
    text = "ecg_mV\n0.1\n0.2\n-0.3\n"
    signal, fs = load_csv_signal(io.StringIO(text), fs=300)
    assert fs == 300
    assert np.allclose(signal, [0.1, 0.2, -0.3])


def test_load_csv_signal_without_header():
    signal, _ = load_csv_signal(io.StringIO("1\n2\n3\n"))
    assert np.allclose(signal, [1, 2, 3])
