"""Compare Section-5 runs: cosine baseline vs weight decay 5e-4 and 1e-4."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(history_path: Path):
    with open(history_path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main():
    root = Path(__file__).resolve().parents[1]
    runs = {"baseline (no wd)": "sched_cosine", "wd=5e-4": "wd_5e-4", "wd=1e-4": "wd_1e-4"}
    curves = {name: load(root / "runs" / folder / "history.csv") for name, folder in runs.items()}

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = {"baseline (no wd)": "tab:purple", "wd=5e-4": "tab:red", "wd=1e-4": "tab:blue"}
    for name, history in curves.items():
        epochs = [int(row["epoch"]) for row in history]
        color = colors[name]
        axes[0].plot(epochs, [float(r["train_loss"]) for r in history], color=color, label=f"{name} train")
        axes[0].plot(epochs, [float(r["val_loss"]) for r in history], linestyle=(0, (5, 3)), linewidth=1.8, color=color, label=f"{name} val")
        axes[1].plot(epochs, [float(r["train_acc"]) for r in history], color=color, label=f"{name} train")
        axes[1].plot(epochs, [float(r["val_acc"]) for r in history], linestyle=(0, (5, 3)), linewidth=1.8, color=color, label=f"{name} val")

    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].set_title("Loss curves (Section 5)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8, handlelength=3)
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    axes[1].set_title("Accuracy curves (Section 5)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8, handlelength=3)
    fig.tight_layout()
    out = root / "runs" / "wd_comparison.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
