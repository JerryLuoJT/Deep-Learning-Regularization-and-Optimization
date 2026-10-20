"""Compare Section-4 runs: constant lr vs cosine annealing (300 epochs)."""
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
    runs = {"constant": "sched_const", "cosine": "sched_cosine"}
    curves = {name: load(root / "runs" / folder / "history.csv") for name, folder in runs.items()}

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    colors = {"constant": "tab:orange", "cosine": "tab:purple"}
    for name, history in curves.items():
        epochs = [int(row["epoch"]) for row in history]
        color = colors[name]
        axes[0].plot(epochs, [float(r["train_loss"]) for r in history], color=color, label=f"{name} train")
        axes[0].plot(epochs, [float(r["val_loss"]) for r in history], linestyle=(0, (5, 3)), linewidth=1.8, color=color, label=f"{name} val")
        axes[1].plot(epochs, [float(r["train_acc"]) for r in history], color=color, label=f"{name} train")
        axes[1].plot(epochs, [float(r["val_acc"]) for r in history], linestyle=(0, (5, 3)), linewidth=1.8, color=color, label=f"{name} val")
        axes[2].plot(epochs, [float(r["lr"]) for r in history], color=color, label=f"{name} lr")

    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].set_title("Loss curves (Section 4)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8, handlelength=3)
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    axes[1].set_title("Accuracy curves (Section 4)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8, handlelength=3)
    axes[2].set_xlabel("epoch")
    axes[2].set_ylabel("lr")
    axes[2].set_title("Learning rate schedule")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(fontsize=8, handlelength=3)
    fig.tight_layout()
    out = root / "runs" / "schedule_comparison.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)

    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, history in curves.items():
        epochs = [int(row["epoch"]) for row in history]
        color = colors[name]
        axes2[0].plot(epochs, [float(r["train_loss"]) for r in history], color=color, label=f"{name} train")
        axes2[0].plot(epochs, [float(r["val_loss"]) for r in history], linestyle=(0, (5, 3)), linewidth=1.8, color=color, label=f"{name} val")
        axes2[1].plot(epochs, [float(r["train_acc"]) for r in history], color=color, label=f"{name} train")
        axes2[1].plot(epochs, [float(r["val_acc"]) for r in history], linestyle=(0, (5, 3)), linewidth=1.8, color=color, label=f"{name} val")
    axes2[0].set_xlabel("epoch")
    axes2[0].set_ylabel("loss")
    axes2[0].set_title("Loss curves (Section 4)")
    axes2[0].grid(True, alpha=0.3)
    axes2[0].legend(fontsize=8, handlelength=3)
    axes2[1].set_xlabel("epoch")
    axes2[1].set_ylabel("accuracy")
    axes2[1].set_title("Accuracy curves (Section 4)")
    axes2[1].grid(True, alpha=0.3)
    axes2[1].legend(fontsize=8, handlelength=3)
    fig2.tight_layout()
    out2 = root / "runs" / "schedule_curves.png"
    fig2.savefig(out2, dpi=150)
    plt.close(fig2)

    fig3, ax3 = plt.subplots(figsize=(6, 3.2))
    for name, history in curves.items():
        epochs = [int(row["epoch"]) for row in history]
        ax3.plot(epochs, [float(r["lr"]) for r in history], color=colors[name], label=name)
    ax3.set_xlabel("epoch")
    ax3.set_ylabel("learning rate")
    ax3.set_title("Learning rate trajectory")
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=9)
    fig3.tight_layout()
    out3 = root / "runs" / "schedule_lr.png"
    fig3.savefig(out3, dpi=150)
    plt.close(fig3)
    print(f"saved {out}")
    print(f"saved {out2}")
    print(f"saved {out3}")


if __name__ == "__main__":
    main()
