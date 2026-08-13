"""Тренировочный цикл baseline-модели инпейнтинга (без adversarial loss, см. шаг 3)."""

import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from ml.datasets.inpainting_dataset import InpaintingDataset
from ml.models.generator import InpaintingGenerator, composite
from ml.training.config import TrainingConfig
from ml.training.losses import reconstruction_loss


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train(config: TrainingConfig) -> InpaintingGenerator:
    device = resolve_device(config.device)

    dataset = InpaintingDataset(config.data_dir, image_size=config.image_size)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    model = InpaintingGenerator(base_channels=config.base_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    checkpoint_dir = Path(config.checkpoint_dir)
    samples_dir = checkpoint_dir / "samples"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)

    log_path = checkpoint_dir / "train_log.csv"
    with open(log_path, "w", newline="") as log_file:
        writer = csv.writer(log_file)
        writer.writerow(["step", "loss"])

        step = 0
        for epoch in range(config.epochs):
            for batch in dataloader:
                image = batch["image"].to(device)
                mask = batch["mask"].to(device)
                masked_image = batch["masked_image"].to(device)

                pred = model(masked_image, mask)
                loss = reconstruction_loss(
                    pred, image, mask, config.hole_weight, config.valid_weight
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                writer.writerow([step, loss.item()])
                if step % config.sample_every == 0:
                    log_file.flush()
                    print(f"epoch {epoch} step {step} loss {loss.item():.4f}")
                    with torch.no_grad():
                        result = composite(image, pred, mask)
                        save_image(result[:4], samples_dir / f"step_{step:06d}.png")

                step += 1

            torch.save(model.state_dict(), checkpoint_dir / "generator.pth")

    return model


def build_config_from_args() -> TrainingConfig:
    parser = argparse.ArgumentParser()
    defaults = TrainingConfig()
    for field_name, field_value in vars(defaults).items():
        arg_type = type(field_value) if not isinstance(field_value, bool) else str
        parser.add_argument(f"--{field_name.replace('_', '-')}", type=arg_type, default=field_value)
    args = parser.parse_args()
    return TrainingConfig(**vars(args))


if __name__ == "__main__":
    train(build_config_from_args())
