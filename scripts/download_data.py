"""Download and extract the Dog Breed dataset (Section 2, Day 1)."""
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

# From the assignment PDF:
# https://drive.google.com/file/d/1PmfiGy1D1ssa8NFnwH8UDJJYCCj8NkHt/view?usp=sharing
FILE_ID = "1PmfiGy1D1ssa8NFnwH8UDJJYCCj8NkHt"


def main():
    parser = argparse.ArgumentParser(description="Download the Dog Breed dataset")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--file-id", default=FILE_ID)
    parser.add_argument("--keep-zip", action="store_true")
    args = parser.parse_args()

    try:
        import gdown
    except ImportError:
        sys.exit("gdown missing. Run: pip install -r requirements.txt")

    root = Path(args.data_root)
    root.mkdir(parents=True, exist_ok=True)
    zip_path = root / "dog_breed.zip"
    if zip_path.exists():
        print(f"{zip_path} exists, skipping download")
    else:
        print("downloading from Google Drive (may take a few minutes)...")
        gdown.download(id=args.file_id, output=str(zip_path), quiet=False)

    print("extracting...")
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(root)
    if not args.keep_zip:
        zip_path.unlink()
    print("done. Next: python scripts/inspect_data.py")


if __name__ == "__main__":
    main()
