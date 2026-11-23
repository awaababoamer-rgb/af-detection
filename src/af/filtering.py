"""Bandpass filtering of the raw ECG.

Why 0.5-40 Hz:
- Below 0.5 Hz is mostly baseline wander (breathing, electrode movement).
- Most QRS energy sits around 5-20 Hz, P/T waves lower; 40 Hz keeps all of that.
- Above 40 Hz is mostly muscle noise and 50/60 Hz mains interference.
"""

import numpy as np
from scipy.signal import butter, sosfiltfilt

LOW_CUTOFF_HZ = 0.5
HIGH_CUTOFF_HZ = 40.0


def bandpass_filter(
    signal: np.ndarray,
    fs: int,
    low: float = LOW_CUTOFF_HZ,
    high: float = HIGH_CUTOFF_HZ,
    order: int = 4,
) -> np.ndarray:
    """Zero-phase Butterworth bandpass filter.

    - Butterworth: flat passband, so in-band amplitudes are not distorted.
    - Second-order sections (sos): numerically stable, unlike the (b, a) form
      which can blow up for low cutoffs relative to fs.
    - filtfilt (forward + backward): cancels phase delay, so R-peaks stay at
      the same sample index as in the raw signal.
    """
    sos = butter(order, [low, high], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, np.asarray(signal, dtype=float))
