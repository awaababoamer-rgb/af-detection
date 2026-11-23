"""Download the PhysioNet/CinC Challenge 2017 training set into data/raw/.

Only the public challenge data is used. Raw data is gitignored, so anyone who
clones the repo runs this script once to reproduce the dataset.

Usage:
    python scripts/download_data.py
"""

import shutil
import urllib.request
import zipfile
from pathlib import Path

BASE_URL = "https://physionet.org/files/challenge-2017/1.0.0/"
ZIP_NAME = "training2017.zip"
# REFERENCE-v3 holds the final, re-reviewed labels. The REFERENCE.csv inside the
# zip is an older version, so we fetch v3 separately and use it everywhere.
LABELS_NAME = "REFERENCE-v3.csv"

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def download(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"Already have {dest.name}, skipping download.")
        return
    print(f"Downloading {url} ...")
    tmp = dest.with_suffix(dest.suffix + ".part")
    # Write to a .part file first so an interrupted download is never
    # mistaken for a complete one on the next run.
    with urllib.request.urlopen(url) as response, open(tmp, "wb") as f:
        shutil.copyfileobj(response, f)
    tmp.rename(dest)


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    zip_path = RAW_DIR / ZIP_NAME
    download(BASE_URL + ZIP_NAME, zip_path)
    download(BASE_URL + LABELS_NAME, RAW_DIR / LABELS_NAME)

    records_dir = RAW_DIR / "training2017"
    if not records_dir.exists():
        print("Extracting ...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(RAW_DIR)

    n_records = len(list(records_dir.glob("*.hea")))
    print(f"Done. {n_records} records in {records_dir}")


if __name__ == "__main__":
    main()
