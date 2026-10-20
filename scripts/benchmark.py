"""Measure real training throughput to estimate experiment runtimes."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Config
from src.dataset import build_dataloaders
from src.model import build_model
from src.utils import format_duration, get_device


def main():
    parser = argparse.ArgumentParser(description="Benchmark training throughput")
    parser.add_argument("--batches", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = get_device(args.device)
    print(f"device: {device}")
    cfg = Config(
        data_root=args.data_root, batch_size=args.batch_size, num_workers=args.workers,
        mixup_alpha=0.0,
    )
    mean, std = [0.4772, 0.4528, 0.3928], [0.2534, 0.2480, 0.2523]
    train_loader, _ = build_dataloaders(cfg, mean, std)
    model = build_model().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    times = []
    for step, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        start = time.time()
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        elapsed = time.time() - start
        if step >= 2:
            times.append(elapsed)
        print(f"batch {step + 1}: {elapsed:.2f}s", flush=True)
        if step + 1 >= args.batches:
            break

    avg = sum(times) / len(times)
    epoch = avg * len(train_loader)
    print(f"\navg {avg:.2f}s/batch, {len(train_loader)} batches/epoch")
    print(f"per epoch: {format_duration(epoch)}")
    print(f"15 epochs: {format_duration(epoch * 15)}")
    print(f"three 15-epoch runs: {format_duration(epoch * 45)}")


if __name__ == "__main__":
    main()
