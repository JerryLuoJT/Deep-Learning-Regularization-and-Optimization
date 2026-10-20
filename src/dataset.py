"""Dog Breed dataset, data augmentation, and channel statistics.

The dataset is expected in one of three layouts after extraction:
  A. <root>/{train,val,test}/{breed}/xxx.jpg      (class folders per split)
  B. <root>/{train,val,test}/xxx.jpg + labels.csv (flat folder per split)
  C. <root>/images/xxx.jpg + labels.csv           (one folder + split column)

Run scripts/inspect_data.py to confirm which layout was extracted.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def _locate_split(data_root: str, split: str) -> tuple[Path | None, Path | None]:
    """Return (split_dir, labels_csv) for a split, supporting layouts A/B/C."""
    root = Path(data_root)
    split_dir = None
    if (root / split).is_dir():
        split_dir = root / split
    else:
        for child in sorted(root.iterdir()) if root.is_dir() else []:
            if child.is_dir() and (child / split).is_dir():
                split_dir = child / split
                break

    csv_path = None
    candidates = [root / "labels.csv", root / f"{split}.csv"]
    if split_dir is not None:
        candidates += [split_dir.parent / f"{split}_labels.csv", split_dir / "labels.csv"]
    for candidate in candidates:
        if candidate.is_file():
            csv_path = candidate
            break
    if split_dir is None and csv_path is None:
        raise FileNotFoundError(
            f"cannot locate '{split}' under {root}. "
            "Run scripts/download_data.py first, then scripts/inspect_data.py."
        )
    return split_dir, csv_path


class DogBreedDataset(Dataset):
    """Image classification dataset supporting the three layouts above."""

    def __init__(self, data_root: str, split: str = "train", transform=None):
        self.split = split
        self.transform = transform
        split_dir, csv_path = _locate_split(data_root, split)
        self.samples: list[tuple[str, int]] = []
        self.classes: list[str] = []

        if split_dir is not None and self._has_class_folders(split_dir):
            self._load_class_folders(split_dir)
        else:
            images_dir = self._resolve_images_dir(split_dir, Path(data_root))
            self._load_flat_folder(images_dir, csv_path)

        if not self.samples:
            raise FileNotFoundError(
                f"no images found for split '{split}'. Check scripts/inspect_data.py output."
            )

    def _has_class_folders(self, split_dir: Path) -> bool:
        subdirs = [p for p in split_dir.iterdir() if p.is_dir()]
        image_dirs = [d for d in subdirs if any(is_image(f) for f in d.iterdir())]
        if not image_dirs:
            return False
        generic = {
            d for d in image_dirs if d.name.lower() in {"images", "image", "imgs", "photos"}
        }
        breed_dirs = [d for d in image_dirs if d not in generic]
        return len(breed_dirs) >= 2 or (len(breed_dirs) == 1 and len(image_dirs) == 1)

    def _load_class_folders(self, split_dir: Path) -> None:
        self.classes = sorted(d.name for d in split_dir.iterdir() if d.is_dir())
        class_to_idx = {name: i for i, name in enumerate(self.classes)}
        for name in self.classes:
            for img in sorted((split_dir / name).iterdir()):
                if is_image(img):
                    self.samples.append((str(img), class_to_idx[name]))

    @staticmethod
    def _resolve_images_dir(split_dir: Path | None, root: Path) -> Path:
        if split_dir is not None:
            images_dir = split_dir / "images"
            return images_dir if images_dir.is_dir() else split_dir
        images_dir = root / "images"
        return images_dir if images_dir.is_dir() else root

    def _load_flat_folder(self, images_dir: Path, csv_path: Path | None) -> None:
        rows = []
        if csv_path is not None:
            with open(csv_path, newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

        split_column = "split" if rows and "split" in rows[0] else None
        rows = [
            row
            for row in rows
            if split_column is None or str(row.get(split_column, "")).strip() == self.split
        ]

        if rows:
            id_key = "id" if "id" in rows[0] else ("image" if "image" in rows[0] else None)
            if id_key is None:
                raise ValueError("labels CSV must contain an 'id' or 'image' column")
            if "breed" not in rows[0]:
                raise ValueError("labels CSV must contain a 'breed' column")
            self.classes = sorted({row["breed"] for row in rows})
            class_to_idx = {name: i for i, name in enumerate(self.classes)}
            for row in rows:
                stem = str(row[id_key])
                path = images_dir / stem if stem.lower().endswith(tuple(IMAGE_EXTENSIONS)) else self._find_image(images_dir, stem)
                if path is not None:
                    self.samples.append((str(path), class_to_idx[row["breed"]]))
            return

        # No CSV: label by parent folder even in a flat split (fallback).
        self.classes = sorted(
            {p.parent.name for p in images_dir.rglob("*") if is_image(p) and p.parent != images_dir}
        )
        class_to_idx = {name: i for i, name in enumerate(self.classes)}
        for img in sorted(images_dir.iterdir()):
            if is_image(img):
                self.samples.append((str(img), class_to_idx.get(img.parent.name, 0)))

    @staticmethod
    def _find_image(images_dir: Path, stem: str) -> Path | None:
        for ext in IMAGE_EXTENSIONS:
            candidate = images_dir / f"{stem}{ext}"
            if candidate.is_file():
                return candidate
        return None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def get_transforms(split: str, mean, std, image_size=100, pad_size=12, flip_prob=0.5):
    """Train split uses the assignment's augmentation; val/test only normalize."""
    pipeline = [transforms.Resize((image_size, image_size), antialias=True)]
    if split == "train":
        pipeline += [
            transforms.Pad(pad_size, padding_mode="constant", fill=0),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=flip_prob),
        ]
    pipeline += [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
    return transforms.Compose(pipeline)


def compute_channel_stats(data_root: str, split: str = "train", image_size=100, workers=0):
    """Mean/std per color channel over all pixels of a split (float64 accumulation)."""
    dataset = DogBreedDataset(
        data_root,
        split=split,
        transform=transforms.Compose(
            [transforms.Resize((image_size, image_size), antialias=True), transforms.ToTensor()]
        ),
    )
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=workers)
    total = torch.zeros(3, dtype=torch.float64)
    total_sq = torch.zeros(3, dtype=torch.float64)
    count = 0
    for images, _ in loader:
        pixels_per_channel = images.shape[2] * images.shape[3]
        count += images.shape[0] * pixels_per_channel
        total += images.sum(dim=(0, 2, 3), dtype=torch.float64)
        total_sq += (images**2).sum(dim=(0, 2, 3), dtype=torch.float64)
    mean = total / count
    var = total_sq / count - mean**2
    std = torch.sqrt(var.clamp(min=0.0))
    return mean.tolist(), std.tolist()


def load_or_compute_stats(data_root: str, split: str = "train", workers=0):
    """Load data/stats.json if present, otherwise compute and cache it."""
    stats_path = Path(data_root) / "stats.json"
    if stats_path.is_file():
        with open(stats_path, encoding="utf-8") as handle:
            stats = json.load(handle)
        return stats["mean"], stats["std"]
    mean, std = compute_channel_stats(data_root, split=split, workers=workers)
    with open(stats_path, "w", encoding="utf-8") as handle:
        json.dump({"mean": mean, "std": std}, handle, indent=2)
    return mean, std


def build_dataloaders(cfg, mean, std):
    train_set = DogBreedDataset(
        cfg.data_root,
        split="train",
        transform=get_transforms("train", mean, std, cfg.image_size, cfg.pad_size, cfg.flip_prob),
    )
    val_set = DogBreedDataset(
        cfg.data_root,
        split="val",
        transform=get_transforms("val", mean, std, cfg.image_size, cfg.pad_size, cfg.flip_prob),
    )
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_set, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_set, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=pin_memory
    )
    return train_loader, val_loader
