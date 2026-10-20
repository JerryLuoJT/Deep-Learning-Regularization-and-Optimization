"""Experiment configuration for the AI6103 homework."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass


@dataclass
class Config:
    # Run
    name: str = "experiment"
    seed: int = 0
    data_root: str = "data"
    output_dir: str = "runs"
    device: str = "auto"  # auto | cuda | mps | cpu
    num_workers: int = 4
    amp: bool = True  # automatic mixed precision, CUDA only

    # Data
    image_size: int = 100
    pad_size: int = 12
    flip_prob: float = 0.5

    # Optimization (fixed by the assignment unless noted)
    batch_size: int = 128
    epochs: int = 15
    lr: float = 0.05
    momentum: float = 0.9
    weight_decay: float = 0.0
    lr_schedule: str = "none"  # none | cosine
    mixup_alpha: float = 0.0  # 0.0 disables mixup

    @property
    def run_dir(self) -> str:
        return f"{self.output_dir}/{self.name}"

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(asdict(self), handle, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "Config":
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)
