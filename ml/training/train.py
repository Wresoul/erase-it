"""Тренировочный цикл модели инпейнтинга: reconstruction loss + опциональный GAN."""

import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from ml.datasets.inpainting_dataset import InpaintingDataset
from ml.device import resolve_device
from ml.models.discriminator import PatchDiscriminator
from ml.models.generator import InpaintingGenerator, composite
from ml.training.config import TrainingConfig
from ml.training.losses import (
    discriminator_hinge_loss,
    generator_hinge_loss,
    reconstruction_loss,
)
from ml.training.metrics import psnr, ssim


def _generator_step(
    generator: InpaintingGenerator,
    discriminator: PatchDiscriminator | None,
    optimizer: torch.optim.Optimizer,
    image: torch.Tensor,
    mask: torch.Tensor,
    masked_image: torch.Tensor,
    config: TrainingConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    pred = generator(masked_image, mask)
    result = composite(image, pred, mask)
    recon_loss = reconstruction_loss(pred, image, mask, config.hole_weight, config.valid_weight)

    total_loss = recon_loss
    if discriminator is not None:
        discriminator.requires_grad_(False)
        adv_loss = generator_hinge_loss(discriminator(result, mask))
        discriminator.requires_grad_(True)
        total_loss = recon_loss + config.adv_weight * adv_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    return result.detach(), recon_loss.detach()


def _discriminator_step(
    discriminator: PatchDiscriminator,
    optimizer: torch.optim.Optimizer,
    image: torch.Tensor,
    mask: torch.Tensor,
    result: torch.Tensor,
) -> torch.Tensor:
    d_real = discriminator(image, mask)
    d_fake = discriminator(result, mask)
    d_loss = discriminator_hinge_loss(d_real, d_fake)

    optimizer.zero_grad()
    d_loss.backward()
    optimizer.step()

    return d_loss.detach()


def train(config: TrainingConfig) -> InpaintingGenerator:
    device = resolve_device(config.device)

    dataset = InpaintingDataset(config.data_dir, image_size=config.image_size)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    generator = InpaintingGenerator(base_channels=config.base_channels).to(device)
    optimizer = torch.optim.Adam(generator.parameters(), lr=config.lr)

    discriminator = None
    disc_optimizer = None
    if config.adv_weight > 0:
        discriminator = PatchDiscriminator(base_channels=config.disc_base_channels).to(device)
        disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=config.disc_lr, betas=(0.5, 0.999))

    checkpoint_dir = Path(config.checkpoint_dir)
    samples_dir = checkpoint_dir / "samples"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)

    log_path = checkpoint_dir / "train_log.csv"
    with open(log_path, "w", newline="") as log_file:
        writer = csv.writer(log_file)
        writer.writerow(["step", "recon_loss", "disc_loss", "psnr", "ssim"])

        step = 0
        for epoch in range(config.epochs):
            for batch in dataloader:
                image = batch["image"].to(device)
                mask = batch["mask"].to(device)
                masked_image = batch["masked_image"].to(device)

                result, recon_loss = _generator_step(
                    generator, discriminator, optimizer, image, mask, masked_image, config
                )

                disc_loss = None
                if discriminator is not None:
                    disc_loss = _discriminator_step(discriminator, disc_optimizer, image, mask, result)

                if step % config.sample_every == 0:
                    with torch.no_grad():
                        step_psnr = psnr(result, image)
                        step_ssim = ssim(result, image)
                    writer.writerow(
                        [step, recon_loss.item(), disc_loss.item() if disc_loss is not None else "", step_psnr, step_ssim]
                    )
                    log_file.flush()
                    print(
                        f"epoch {epoch} step {step} recon_loss {recon_loss.item():.4f} "
                        f"psnr {step_psnr:.2f} ssim {step_ssim:.3f}"
                    )
                    save_image(result[:4], samples_dir / f"step_{step:06d}.png")
                else:
                    writer.writerow([step, recon_loss.item(), disc_loss.item() if disc_loss is not None else "", "", ""])

                step += 1

            torch.save(generator.state_dict(), checkpoint_dir / "generator.pth")
            if discriminator is not None:
                torch.save(discriminator.state_dict(), checkpoint_dir / "discriminator.pth")

    return generator


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
