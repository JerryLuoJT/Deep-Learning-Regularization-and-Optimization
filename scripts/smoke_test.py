"""End-to-end smoke test for the Day 1-2 pipeline (few steps, tiny batches)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import build_dataloaders
from src.model import build_model
from src.train import mixup_sample
from src.utils import get_device, set_seed


def main():
    parser = argparse.ArgumentParser(description="Smoke test the pipeline")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if not Path(args.data_root).is_dir():
        sys.exit(f"{args.data_root} missing. Run scripts/download_data.py first.")

    set_seed(0)
    device = get_device(args.device)
    print(f"device={device}")

    cfg = argparse.Namespace(
        data_root=args.data_root, batch_size=8, num_workers=0,
        image_size=100, pad_size=12, flip_prob=0.5, mixup_alpha=0.2,
    )
    # Placeholder stats: fine for checking shapes and gradient flow.
    mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
    train_loader, val_loader = build_dataloaders(cfg, mean, std)
    print(f"train={len(train_loader.dataset)} val={len(val_loader.dataset)} classes={len(train_loader.dataset.classes)}")

    model = build_model().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    iterator = iter(train_loader)
    for step in range(args.steps):
        try:
            images, labels = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            images, labels = next(iterator)
        images, labels = images.to(device), labels.to(device)
        images, target_a, target_b, lam = mixup_sample(images, labels, cfg.mixup_alpha, device)
        optimizer.zero_grad()
        logits = model(images)
        loss = lam * criterion(logits, target_a) + (1.0 - lam) * criterion(logits, target_b)
        loss.backward()
        optimizer.step()
        print(f"step {step + 1}/{args.steps} loss={loss.item():.4f}")

    val_iterator = iter(val_loader)
    val_total = 0
    val_correct = 0
    model.eval()
    with torch.no_grad():
        for _ in range(args.steps):
            try:
                images, labels = next(val_iterator)
            except StopIteration:
                break
            images, labels = images.to(device), labels.to(device)
            val_total += labels.size(0)
            val_correct += (model(images).argmax(1) == labels).sum().item()
    print(f"validation sample: acc={val_correct / val_total:.4f} ({val_total} images)")
    print("smoke test passed: data loading, augmentation, mixup, forward/backward all work")


if __name__ == "__main__":
    main()
