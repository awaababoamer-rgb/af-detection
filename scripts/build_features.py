"""Phase 3: compute the feature table for every recording.

Usage:
    python scripts/build_features.py
Output:
    data/processed/features.csv  (one row per recording: record, label, features)
"""

import time

import pandas as pd
from joblib import Parallel, delayed

from af.config import FEATURES_CSV, PROCESSED_DIR
from af.data import load_labels, load_record
from af.features import FEATURE_COLUMNS, extract_features


def one(record: str) -> dict:
    raw, fs = load_record(record)
    return {"record": record, **extract_features(raw, fs)}


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    labels = load_labels()
    start = time.time()
    # Each recording is independent, so use all CPU cores.
    rows = Parallel(n_jobs=-1, verbose=5)(delayed(one)(r) for r in labels["record"])
    feats = pd.DataFrame(rows)
    table = labels[["record", "label"]].merge(feats, on="record")
    table = table[["record", "label", *FEATURE_COLUMNS]]
    table.to_csv(FEATURES_CSV, index=False)
    print(f"Saved {len(table)} rows to {FEATURES_CSV} in {time.time() - start:.0f}s")
    print("Missing values per feature:\n", table[FEATURE_COLUMNS].isna().sum()[lambda s: s > 0])


if __name__ == "__main__":
    main()
