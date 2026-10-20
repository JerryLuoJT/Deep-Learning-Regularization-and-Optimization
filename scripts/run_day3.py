"""Run the three Section-3 learning-rate experiments sequentially (Day 3)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

EXPERIMENTS = ["lr_0.5", "lr_0.05", "lr_0.001"]


def main():
    root = Path(__file__).resolve().parents[1]
    for name in EXPERIMENTS:
        history = root / "runs" / name / "history.csv"
        if history.is_file():
            print(f"[skip] {name}: already finished", flush=True)
            continue
        config = root / "experiments" / f"{name}.json"
        print(f"\n=== running {name} ===", flush=True)
        code = subprocess.run(
            [sys.executable, "-m", "src.train", "--config", str(config)],
            cwd=root,
        ).returncode
        if code != 0:
            print(f"[fail] {name} exited with code {code}", flush=True)
            sys.exit(code)
    print("\nall three learning-rate experiments done", flush=True)


if __name__ == "__main__":
    main()
