"""Loading ECG records and labels.

The challenge stores each recording as a WFDB pair: a .hea header (text:
sampling rate, gain, units) and a .mat file of raw integer ADC counts.
wfdb applies the header's gain for us, so `p_signal` comes back in millivolts.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

from af.config import FS, LABEL_NAMES, LABELS_CSV, RECORDS_DIR


def load_labels(labels_csv: Path = LABELS_CSV) -> pd.DataFrame:
    """Return a DataFrame with columns: record, code, label."""
    df = pd.read_csv(labels_csv, header=None, names=["record", "code"])
    df["label"] = df["code"].map(LABEL_NAMES)
    return df


def load_record(record_name: str, records_dir: Path = RECORDS_DIR) -> tuple[np.ndarray, int]:
    """Load one recording. Returns (signal in mV as 1D float array, sampling rate)."""
    return load_wfdb(records_dir / record_name)


def load_wfdb(path_without_ext: Path | str) -> tuple[np.ndarray, int]:
    """Load any single-lead WFDB record given its path without extension."""
    record = wfdb.rdrecord(str(path_without_ext))
    signal = record.p_signal[:, 0].astype(float)
    return signal, int(record.fs)


def load_csv_signal(file_like, fs: int = FS) -> tuple[np.ndarray, int]:
    """Load a one-column CSV/TXT of samples in mV (used by the web demo).

    A header row is tolerated; any non-numeric rows are dropped.
    """
    values = pd.read_csv(file_like, header=None).iloc[:, 0]
    signal = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    return signal, fs
