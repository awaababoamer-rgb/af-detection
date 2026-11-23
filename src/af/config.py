"""Shared constants and paths, so every script agrees on them."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
RECORDS_DIR = RAW_DIR / "training2017"
LABELS_CSV = RAW_DIR / "REFERENCE-v3.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
FEATURES_CSV = PROCESSED_DIR / "features.csv"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# AliveCor devices in the 2017 challenge record at 300 Hz.
FS = 300

# Label codes used in REFERENCE-v3.csv -> readable names.
LABEL_NAMES = {"N": "Normal", "A": "AF", "O": "Other", "~": "Noisy"}
CLASS_ORDER = ["Normal", "AF", "Other", "Noisy"]

RANDOM_STATE = 42
