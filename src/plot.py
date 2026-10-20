"""Loss/accuracy curve plotting for the report."""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_curves(history: list[dict], out_dir: str) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    epochs = [h["epoch"] for h in history]
    outputs = []

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, [h["train_loss"] for h in history], marker="o", label="training loss")
    ax.plot(epochs, [h["val_loss"] for h in history], marker="s", label="validation loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("Loss curves")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = os.path.join(out_dir, "loss_curves.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    outputs.append(path)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, [h["train_acc"] for h in history], marker="o", label="training accuracy")
    ax.plot(epochs, [h["val_acc"] for h in history], marker="s", label="validation accuracy")
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy")
    ax.set_title("Accuracy curves")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = os.path.join(out_dir, "acc_curves.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    outputs.append(path)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, [h["train_loss"] for h in history], marker="o", label="train")
    axes[0].plot(epochs, [h["val_loss"] for h in history], marker="s", label="validation")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(epochs, [h["train_acc"] for h in history], marker="o", label="train")
    axes[1].plot(epochs, [h["val_acc"] for h in history], marker="s", label="validation")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, "training_curves.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    outputs.append(path)

    return outputs
