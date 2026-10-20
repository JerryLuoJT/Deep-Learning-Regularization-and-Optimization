"""Summarize the extracted dataset and verify it against the assignment (Day 1)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

EXPECTED = {"train": 7155, "val": 1533, "test": 1534}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def main():
    parser = argparse.ArgumentParser(description="Inspect the extracted dataset")
    parser.add_argument("--data-root", default="data")
    args = parser.parse_args()
    root = Path(args.data_root)
    if not root.is_dir():
        sys.exit(f"{root} not found. Run scripts/download_data.py first.")

    print(f"top level of {root}:")
    for entry in sorted(root.iterdir()):
        print(f"  {entry.name}")

    all_ok = True
    for split, expected in EXPECTED.items():
        split_dir = None
        for base in [root, *[p for p in root.iterdir() if p.is_dir()]]:
            if (base / split).is_dir():
                split_dir = base / split
                break
        if split_dir is None:
            print(f"\n{split}: NOT FOUND (expected {expected} images)")
            all_ok = False
            continue
        images = [p for p in split_dir.rglob("*") if is_image(p)]
        subdirs = [d for d in split_dir.iterdir() if d.is_dir()]
        status = "ok" if len(images) == expected else "MISMATCH"
        if len(images) != expected:
            all_ok = False
        print(f"\n{split}: {len(images)} images, {len(subdirs)} subfolders (expected {expected}) [{status}]")
        sizes = set()
        for img in images[:100]:
            try:
                with Image.open(img) as image:
                    sizes.add(image.size)
            except Exception as exc:  # noqa: BLE001
                print(f"  unreadable: {img} ({exc})")
        print(f"  sample image sizes: {sorted(sizes)[:8]}")
        if subdirs:
            print(f"  first classes: {sorted(d.name for d in subdirs)[:5]}")

    csv_files = sorted(p.name for p in root.glob("*.csv"))
    print(f"\nlabel CSVs present: {csv_files if csv_files else 'none'}")
    print("layout verdict:", "looks fine, dataset.py should load it" if all_ok else "check the mismatches above")


if __name__ == "__main__":
    main()
