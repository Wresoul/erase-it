"""Визуализирует, какую маску SAM возвращает по клику пользователя.

Запуск: python -m app.backend.services.segmentation_demo --image photo.jpg --x 120 --y 80
Требует скачанный чекпоинт SAM (см. README, раздел "Локализация объекта").
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from app.backend.services.segmentation_service import (
    DEFAULT_CHECKPOINT,
    DEFAULT_MODEL_TYPE,
    SegmentationService,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--x", type=int, required=True)
    parser.add_argument("--y", type=int, required=True)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--model-type", default=DEFAULT_MODEL_TYPE)
    parser.add_argument("--output", default="ml/checkpoints/samples/segmentation_demo.png")
    args = parser.parse_args()

    image = np.array(Image.open(args.image).convert("RGB"))
    service = SegmentationService(checkpoint_path=args.checkpoint, model_type=args.model_type)
    mask = service.predict_mask(image, (args.x, args.y))

    overlay = image.copy()
    overlay[mask] = (overlay[mask] * 0.4 + np.array([255, 0, 0]) * 0.6).astype(np.uint8)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay).save(output_path)
    print(f"Клик ({args.x}, {args.y}) -> маска {mask.sum()} пикселей. Сохранено в {output_path}")


if __name__ == "__main__":
    main()
