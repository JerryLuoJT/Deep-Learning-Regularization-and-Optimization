"""Training entry point.

Fixed by the assignment: EfficientNetV2-S from scratch, SGD with momentum 0.9,
batch size 128. Tunable per experiment: lr, weight decay, cosine schedule, mixup.

Run from the project root:
  python -m src.train --name lr_0.05 --lr 0.05 --epochs 15
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Config
from src.dataset import build_dataloaders, load_or_compute_stats
from src.model import NUM_CLASSES, build_model, count_parameters
from src.plot import save_curves
from src.utils import format_duration, get_device, save_json, set_seed


def mixup_sample(images, labels, alpha, device):
    """Standard mixup [Zhang et al., ICLR 2018]: x' = lam*x_i + (1-lam)*x_j."""
    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    index = torch.randperm(images.size(0), device=device)
    mixed = lam * images + (1.0 - lam) * images[index]
    return mixed, labels, labels[index], lam


def train_epoch(model, loader, optimizer, criterion, device, cfg, scaler):
    model.train()
    total_loss = 0.0
    total_correct = 0.0
    seen = 0
    for images, labels in tqdm(loader, desc="train", leave=False):
        images, labels = images.to(device), labels.to(device)
        if cfg.mixup_alpha > 0:
            images, target_a, target_b, lam = mixup_sample(images, labels, cfg.mixup_alpha, device)
        optimizer.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=scaler is not None):
            logits = model(images)
        if cfg.mixup_alpha > 0:
            loss = lam * criterion(logits, target_a) + (1.0 - lam) * criterion(logits, target_b)
        else:
            loss = criterion(logits, labels)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            preds = logits.argmax(dim=1)
            if cfg.mixup_alpha > 0:
                correct = lam * (preds == target_a).float() + (1.0 - lam) * (preds == target_b).float()
            else:
                correct = (preds == labels).float()
        total_loss += loss.item() * images.size(0)
        total_correct += correct.sum().item()
        seen += images.size(0)
    return total_loss / seen, total_correct / seen


@torch.no_grad()
def validate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0.0
    seen = 0
    for images, labels in tqdm(loader, desc="val", leave=False):
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        preds = logits.argmax(dim=1)
        total_loss += loss.item() * images.size(0)
        total_correct += (preds == labels).sum().item()
        seen += images.size(0)
    return total_loss / seen, total_correct / seen


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="AI6103 EfficientNetV2-S training")
    parser.add_argument("--config", default=None, help="optional JSON config; CLI flags override it")
    parser.add_argument("--name", default=argparse.SUPPRESS)
    parser.add_argument("--lr", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--epochs", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--batch-size", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--weight-decay", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--schedule", choices=["none", "cosine"], default=argparse.SUPPRESS)
    parser.add_argument("--mixup-alpha", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--data-root", default=argparse.SUPPRESS)
    parser.add_argument("--output-dir", default=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--device", default=argparse.SUPPRESS)
    parser.add_argument("--num-workers", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--no-amp", action="store_true", default=argparse.SUPPRESS)
    args = parser.parse_args()

    cfg = Config.from_json(args.config) if args.config else Config()
    overrides = vars(args)
    if overrides.pop("no_amp", False):
        cfg.amp = False
    for key, value in overrides.items():
        if key != "config":
            setattr(cfg, key, value)
    return cfg


def main():
    cfg = parse_args()
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    os.makedirs(cfg.run_dir, exist_ok=True)
    cfg.save(os.path.join(cfg.run_dir, "config.json"))
    print(f"[{cfg.name}] device={device} seed={cfg.seed}")

    mean, std = load_or_compute_stats(cfg.data_root, workers=cfg.num_workers)
    print(f"normalization: mean={[round(v, 4) for v in mean]} std={[round(v, 4) for v in std]}")

    train_loader, val_loader = build_dataloaders(cfg, mean, std)
    classes = train_loader.dataset.classes
    print(f"train={len(train_loader.dataset)} val={len(val_loader.dataset)} classes={len(classes)}")
    if len(classes) != NUM_CLASSES:
        print(f"warning: expected {NUM_CLASSES} breeds, found {len(classes)}")

    model = build_model().to(device)
    print(f"EfficientNetV2-S parameters: {count_parameters(model):,}")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay
    )
    scheduler = None
    if cfg.lr_schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=0.0)
    scaler = torch.amp.GradScaler("cuda") if (cfg.amp and device.type == "cuda") else None

    history = []
    best_val_acc = 0.0
    start = time.time()
    for epoch in range(1, cfg.epochs + 1):
        epoch_start = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, cfg, scaler)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        if scheduler is not None:
            scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "train_acc": round(train_acc, 6),
                "val_loss": round(val_loss, 6),
                "val_acc": round(val_acc, 6),
                "lr": lr_now,
            }
        )
        print(
            f"epoch {epoch:3d}/{cfg.epochs} lr={lr_now:.2e} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
            f"({format_duration(time.time() - epoch_start)})"
        )

        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "history": history,
            "config": asdict(cfg),
        }
        torch.save(checkpoint, os.path.join(cfg.run_dir, "last.pt"))
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(checkpoint, os.path.join(cfg.run_dir, "best.pt"))

    with open(os.path.join(cfg.run_dir, "history.csv"), "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"]
        )
        writer.writeheader()
        writer.writerows(history)

    curves = save_curves(history, cfg.run_dir)
    save_json(
        os.path.join(cfg.run_dir, "summary.json"),
        {
            "best_val_acc": best_val_acc,
            "final": history[-1],
            "normalization": {"mean": mean, "std": std},
            "params": count_parameters(model),
        },
    )
    print(f"done in {format_duration(time.time() - start)}; best val acc = {best_val_acc:.4f}")
    for path in curves:
        print(f"saved {path}")


if __name__ == "__main__":
    main()
