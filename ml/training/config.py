"""Конфигурация обучения модели инпейнтинга."""

from dataclasses import dataclass


@dataclass
class TrainingConfig:
    data_dir: str = "data/raw"
    image_size: int = 256
    batch_size: int = 8
    epochs: int = 20
    lr: float = 1e-4
    base_channels: int = 32
    hole_weight: float = 6.0
    valid_weight: float = 1.0
    checkpoint_dir: str = "ml/checkpoints"
    sample_every: int = 200
    device: str = "auto"
