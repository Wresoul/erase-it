"""PyTorch Dataset: (image, mask, masked_image) для обучения инпейнтинга."""

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from ml.datasets.mask_generator import generate_random_mask

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


class InpaintingDataset(Dataset):
    """Отдаёт словарь {image, mask, masked_image} для одного примера обучения.

    Маска генерируется процедурно на лету (см. mask_generator.py) — датасет отвечает
    только за загрузку и подготовку изображений.
    """

    def __init__(self, image_dir: str, image_size: int = 256):
        self.image_paths = sorted(
            p for p in Path(image_dir).rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not self.image_paths:
            raise ValueError(f"No images found in {image_dir}")

        self.image_size = image_size
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict:
        image = Image.open(self.image_paths[idx]).convert("RGB")
        image = self.transform(image)

        mask = generate_random_mask(self.image_size, self.image_size)
        mask = torch.from_numpy(mask).unsqueeze(0)

        masked_image = image * (1 - mask)

        return {"image": image, "mask": mask, "masked_image": masked_image}
