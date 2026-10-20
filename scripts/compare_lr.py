"""Plot the three Section-3 learning-rate experiments on shared axes."""
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
    runs = {"lr=0.5": "lr_0.5", "lr=0.05": "lr_0.05", "lr=0.001": "lr_0.001"}
    curves = {name: load(root / "runs" / folder / "history.csv") for name, folder in runs.items()}

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = {"lr=0.5": "tab:red", "lr=0.05": "tab:blue", "lr=0.001": "tab:green"}
    for name, history in curves.items():
        epochs = [int(row["epoch"]) for row in history]
        color = colors[name]
        axes[0].plot(epochs, [float(r["train_loss"]) for r in history], marker="o", ms=3, color=color, label=f"{name} train")
        axes[0].plot(epochs, [float(r["val_loss"]) for r in history], marker="s", ms=3, linestyle=(0, (5, 3)), linewidth=1.8, color=color, label=f"{name} val")
        axes[1].plot(epochs, [float(r["train_acc"]) for r in history], marker="o", ms=3, color=color, label=f"{name} train")
        axes[1].plot(epochs, [float(r["val_acc"]) for r in history], marker="s", ms=3, linestyle=(0, (5, 3)), linewidth=1.8, color=color, label=f"{name} val")

    inset = axes[0].inset_axes([0.55, 0.13, 0.44, 0.42])
    for name, history in curves.items():
        epochs = [int(row["epoch"]) for row in history]
        inset.plot(epochs, [float(r["val_loss"]) for r in history], linestyle=(0, (5, 3)), linewidth=1.8, color=colors[name])
    inset.set_ylim(4, 6)
    inset.set_title("val loss (4-6, zoomed)", fontsize=7)
    inset.tick_params(labelsize=6)
    inset.grid(True, alpha=0.3)

    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].set_title("Loss curves (Section 3)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8, handlelength=3)
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    axes[1].set_title("Accuracy curves (Section 3)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8, handlelength=3)
    fig.tight_layout()
    out = root / "runs" / "lr_comparison.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
