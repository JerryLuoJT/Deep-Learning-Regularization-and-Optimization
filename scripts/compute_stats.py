"""Compute per-channel mean/std on the training set (Section 2, 10%)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import compute_channel_stats


def main():
    parser = argparse.ArgumentParser(description="Compute normalization statistics")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--split", default="train")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    print(f"computing mean/std on the {args.split} split (resized to 100x100)...")
    mean, std = compute_channel_stats(args.data_root, split=args.split, workers=args.workers)
    stats = {"mean": mean, "std": std}
    print(json.dumps(stats, indent=2))

    output = Path(args.output) if args.output else Path(args.data_root) / "stats.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)
    print(f"saved to {output}")


if __name__ == "__main__":
    main()
