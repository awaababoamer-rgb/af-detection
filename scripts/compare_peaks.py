"""Phase 2: filter + R-peak detection, and compare the simple detector to neurokit2.

Usage:
    python scripts/compare_peaks.py
Outputs:
    reports/figures/peaks_examples.png   filtered signal with both detectors' peaks
    reports/peak_agreement.csv           per-class sensitivity / PPV of simple vs neurokit
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from af.config import CLASS_ORDER, FIGURES_DIR, RANDOM_STATE, REPORTS_DIR
from af.data import load_labels, load_record
from af.filtering import bandpass_filter
from af.peaks import correct_polarity, detect_r_peaks_neurokit, detect_r_peaks_simple, match_peaks

N_PER_CLASS = 100


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    labels = load_labels()
    sample = labels.groupby("label").sample(n=N_PER_CLASS, random_state=RANDOM_STATE)

    rows = []
    for name, cls in zip(sample["record"], sample["label"]):
        raw, fs = load_record(name)
        filtered = correct_polarity(bandpass_filter(raw, fs), fs)
        ref = detect_r_peaks_neurokit(filtered, fs)
        simple = detect_r_peaks_simple(filtered, fs)
        rows.append({"record": name, "label": cls, **match_peaks(ref, simple, fs)})
    df = pd.DataFrame(rows)
    summary = df.groupby("label")[["sensitivity", "ppv"]].median().reindex(CLASS_ORDER).round(3)
    summary.to_csv(REPORTS_DIR / "peak_agreement.csv")
    print("Median agreement of simple detector vs neurokit2 (neurokit as reference):")
    print(summary)

    # Plot raw vs filtered with both detectors for one record per class.
    fig, axes = plt.subplots(len(CLASS_ORDER), 1, figsize=(12, 10), sharex=True)
    for ax, cls in zip(axes, CLASS_ORDER):
        name = sample.loc[sample["label"] == cls, "record"].iloc[0]
        raw, fs = load_record(name)
        filtered = correct_polarity(bandpass_filter(raw, fs), fs)
        ref = detect_r_peaks_neurokit(filtered, fs)
        simple = detect_r_peaks_simple(filtered, fs)
        n = min(len(raw), 10 * fs)
        t = np.arange(n) / fs
        ax.plot(t, raw[:n] - np.median(raw[:n]), lw=0.5, color="0.7", label="raw (centred)")
        ax.plot(t, filtered[:n], lw=0.8, color="black", label="filtered 0.5-40 Hz")
        r = ref[ref < n]
        s = simple[simple < n]
        ax.plot(r / fs, filtered[r], "o", ms=7, mfc="none", color="tab:blue", label="neurokit2")
        ax.plot(s / fs, filtered[s], "x", ms=6, color="tab:red", label="simple")
        ax.set_title(f"{cls} ({name})", loc="left", fontsize=10)
        ax.set_ylabel("mV")
    axes[0].legend(loc="upper right", fontsize=8, ncol=4)
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    out = FIGURES_DIR / "peaks_examples.png"
    fig.savefig(out, dpi=120)
    print("Saved", out)


if __name__ == "__main__":
    main()
