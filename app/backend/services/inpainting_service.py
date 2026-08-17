"""Обёртка над моделью инпейнтинга: изображение + маска -> результат без объекта.

Если чекпоинт (ml/checkpoints/generator.pth) ещё не существует — используются
случайно инициализированные веса. Так сервис можно поднять и проверить пайплайн
целиком уже сейчас, до того как реальное обучение на Kaggle (шаг 3) будет прогнано
до конца — просто качество закрашивания будет случайным, а не осмысленным.
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from ml.models.generator import InpaintingGenerator, composite

DEFAULT_CHECKPOINT = Path("ml/checkpoints/generator.pth")


class InpaintingService:
    def __init__(
        self,
        checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
        image_size: int = 256,
        base_channels: int = 32,
        device: str = "cpu",
    ):
        self.image_size = image_size
        self.device = torch.device(device)
        self.model = InpaintingGenerator(base_channels=base_channels).to(self.device)

        checkpoint_path = Path(checkpoint_path)
        if checkpoint_path.exists():
            # weights_only=True запрещает unpickle произвольных объектов (только тензоры) —
            # torch.load по умолчанию использует pickle, который может исполнить код,
            # если чекпоинт когда-нибудь станет настраиваемым пользователем.
            state_dict = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state_dict)
        self.model.eval()

        self._to_tensor = transforms.ToTensor()

    def predict(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """image: HWC uint8 RGB. mask: HW uint8/bool, ненулевое = закрасить. Возвращает HWC uint8 RGB."""
        original_size = (image.shape[1], image.shape[0])  # PIL ожидает (width, height)

        image_pil = Image.fromarray(image).resize((self.image_size, self.image_size), Image.BILINEAR)
        mask_pil = Image.fromarray((mask > 0).astype(np.uint8) * 255).resize(
            (self.image_size, self.image_size), Image.NEAREST
        )

        image_tensor = self._to_tensor(image_pil).unsqueeze(0).to(self.device)
        mask_tensor = self._to_tensor(mask_pil).unsqueeze(0).to(self.device)
        masked_image = image_tensor * (1 - mask_tensor)

        with torch.no_grad():
            pred = self.model(masked_image, mask_tensor)
            result = composite(image_tensor, pred, mask_tensor)[0]

        result_image = transforms.ToPILImage()(result.clamp(0, 1).cpu())
        result_image = result_image.resize(original_size, Image.BILINEAR)
        return np.array(result_image)
