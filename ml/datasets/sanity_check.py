"""Визуализирует несколько примеров (image, mask, masked_image) из InpaintingDataset.

Запуск: python -m ml.datasets.sanity_check --image-dir data/raw
"""

import argparse
from pathlib import Path

import torch
from torchvision.utils import make_grid, save_image

from ml.datasets.inpainting_dataset import InpaintingDataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", default="data/raw")
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--output", default="ml/checkpoints/samples/sanity_check.png")
    args = parser.parse_args()

    dataset = InpaintingDataset(args.image_dir)

    rows = []
    for i in range(min(args.num_samples, len(dataset))):
        sample = dataset[i]
        mask_rgb = sample["mask"].repeat(3, 1, 1)
        rows.extend([sample["image"], mask_rgb, sample["masked_image"]])

    grid = make_grid(torch.stack(rows), nrow=3)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, output_path)
    print(f"Saved sanity check grid ({len(dataset)} images available) to {output_path}")


if __name__ == "__main__":
    main()
