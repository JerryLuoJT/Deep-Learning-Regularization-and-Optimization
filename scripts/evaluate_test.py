"""Evaluate a saved checkpoint on the hold-out test set (Section 6, 5%)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Config
from src.dataset import DogBreedDataset, get_transforms, load_or_compute_stats
from src.model import build_model
from src.utils import get_device, set_seed


def main():
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on the test split")
    parser.add_argument("checkpoint", help="path to best.pt (or last.pt)")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    set_seed(0)
    device = get_device(args.device)
    print(f"device={device}")

    checkpoint_path = Path(args.checkpoint)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = ckpt.get("config", {})
    epoch = ckpt.get("epoch", "?")
    history = ckpt.get("history", [])
    print(f"checkpoint {checkpoint_path.name}: epoch={epoch}")

    mean, std = load_or_compute_stats(args.data_root)
    test_set = DogBreedDataset(
        args.data_root,
        split="test",
        transform=get_transforms("val", mean, std, config.get("image_size", 100),
                                 config.get("pad_size", 12), config.get("flip_prob", 0.5)),
    )
    loader = torch.utils.data.DataLoader(
        test_set, batch_size=128, shuffle=False, num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )
    print(f"test samples={len(test_set)} classes={len(test_set.classes)}")

    model = build_model().to(device)
    model.load_state_dict(ckpt["model_state"])
    criterion = nn.CrossEntropyLoss()

    model.eval()
    total_loss, total_correct, seen = 0.0, 0.0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item() * images.size(0)
            total_correct += (logits.argmax(1) == labels).float().sum().item()
            seen += images.size(0)
    test_loss = total_loss / seen
    test_acc = total_correct / seen
    print(f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}")
    if history:
        print(f"best val acc in training history: {max(float(r['val_acc']) for r in history):.4f}")


if __name__ == "__main__":
    main()
