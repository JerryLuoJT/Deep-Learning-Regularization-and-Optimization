"""Model builder: EfficientNetV2-S trained from scratch."""
from __future__ import annotations

from torchvision.models import efficientnet_v2_s

NUM_CLASSES = 120  # Dog Breed Identification has 120 breeds


def build_model(num_classes: int = NUM_CLASSES):
    # weights=None -> random initialization, i.e. training from scratch.
    return efficientnet_v2_s(weights=None, num_classes=num_classes)


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())
