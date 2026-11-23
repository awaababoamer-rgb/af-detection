"""Phase 1: class balance, recording lengths, and a raw ECG example per class.

Usage:
    python scripts/explore_data.py
Outputs:
    reports/class_balance.csv
    reports/figures/raw_examples.png
"""

import matplotlib

matplotlib.use("Agg")  # write files without needing a display
import matplotlib.pyplot as plt
import pandas as pd

from af.config import CLASS_ORDER, FIGURES_DIR, REPORTS_DIR
from af.data import load_labels, load_record


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    labels = load_labels()

    counts = labels["label"].value_counts().reindex(CLASS_ORDER)
    balance = pd.DataFrame({"count": counts, "percent": (100 * counts / counts.sum()).round(1)})
    balance.to_csv(REPORTS_DIR / "class_balance.csv")
    print("Class balance:\n", balance, "\nTotal:", counts.sum())

    # Durations: 9-61 s per record. Reading every header is quick enough.
    first = labels.groupby("label").head(200)
    durations = []
    for name in first["record"]:
        sig, fs = load_record(name)
        durations.append(len(sig) / fs)
    print(f"Durations (first 200/class): min {min(durations):.1f}s, "
          f"median {pd.Series(durations).median():.1f}s, max {max(durations):.1f}s")

    # One example per class, first 10 s, in mV, so units and scale are visible.
    fig, axes = plt.subplots(len(CLASS_ORDER), 1, figsize=(12, 9), sharex=True)
    for ax, cls in zip(axes, CLASS_ORDER):
        name = labels.loc[labels["label"] == cls, "record"].iloc[0]
        sig, fs = load_record(name)
        n = min(len(sig), 10 * fs)
        t = [i / fs for i in range(n)]
        ax.plot(t, sig[:n], lw=0.7, color="black")
        ax.set_title(f"{cls} ({name})", loc="left", fontsize=10)
        ax.set_ylabel("mV")
    axes[-1].set_xlabel("Time (s)  -  sampled at 300 Hz")
    fig.suptitle("Raw single-lead ECG, one recording per class")
    fig.tight_layout()
    out = FIGURES_DIR / "raw_examples.png"
    fig.savefig(out, dpi=120)
    print("Saved", out)


if __name__ == "__main__":
    main()
