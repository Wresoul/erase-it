import statistics

import numpy as np
import torch
from PIL import Image, ImageDraw

from ml.training.config import TrainingConfig
from ml.training.train import train


def _make_structured_image(rng: np.random.Generator, size: int = 32) -> Image.Image:
    # Случайный шум CNN память из-за weight sharing по факту не переобучивает —
    # для честного sanity-check нужны изображения с локальной структурой, как у фото.
    image = Image.new("RGB", (size, size), tuple(rng.integers(0, 255, 3).tolist()))
    draw = ImageDraw.Draw(image)
    for _ in range(3):
        x0, y0 = rng.integers(0, size - 8, 2)
        x1, y1 = x0 + rng.integers(4, 8), y0 + rng.integers(4, 8)
        draw.rectangle([x0, y0, x1, y1], fill=tuple(rng.integers(0, 255, 3).tolist()))
    return image


def test_train_overfits_tiny_dataset(tmp_path):
    """Sanity-check: на паре картинок и множестве шагов loss должен заметно упасть.

    Не проверяет качество генерации — только то, что градиенты текут и модель
    вообще способна обучаться (типичный баг-детектор для новой архитектуры).
    """
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    for i in range(2):
        _make_structured_image(rng).save(tmp_path / f"img_{i}.png")

    config = TrainingConfig(
        data_dir=str(tmp_path),
        image_size=32,
        batch_size=2,
        epochs=2000,
        lr=3e-3,
        base_channels=8,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        sample_every=10000,
        device="cpu",
    )

    train(config)

    losses = [
        float(line.split(",")[1])
        for line in (tmp_path / "checkpoints" / "train_log.csv").read_text().splitlines()[1:]
    ]

    # Медиана устойчивее к редким скачкам loss, чем среднее последних N шагов.
    first_losses_avg = sum(losses[:10]) / 10
    last_losses_median = statistics.median(losses[-50:])
    assert last_losses_median < first_losses_avg * 0.5
