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

from ml.device import resolve_device
from ml.models.generator import InpaintingGenerator

DEFAULT_CHECKPOINT = Path("ml/checkpoints/generator.pth")


class InpaintingService:
    def __init__(
        self,
        checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
        image_size: int = 256,
        base_channels: int = 32,
        device: str = "auto",
    ):
        self.image_size = image_size
        self.device = resolve_device(device)
        self.model = InpaintingGenerator(base_channels=base_channels).to(self.device)

        checkpoint_path = Path(checkpoint_path)
        self.is_trained = checkpoint_path.exists()
        if self.is_trained:
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

        image_pil = Image.fromarray(image)
        small_image_pil = image_pil.resize((self.image_size, self.image_size), Image.BILINEAR)
        mask_pil = Image.fromarray((mask > 0).astype(np.uint8) * 255)
        small_mask_pil = mask_pil.resize((self.image_size, self.image_size), Image.NEAREST)

        small_image_tensor = self._to_tensor(small_image_pil).unsqueeze(0).to(self.device)
        small_mask_tensor = self._to_tensor(small_mask_pil).unsqueeze(0).to(self.device)
        masked_image = small_image_tensor * (1 - small_mask_tensor)

        with torch.no_grad():
            pred = self.model(masked_image, small_mask_tensor)

        # Модель работает только на фиксированных 256x256, но композитим на полном
        # разрешении: апскейлим только предсказание, а фон остаётся исходными
        # пикселями — иначе весь кадр терял бы чёткость от лишнего downscale+upscale,
        # даже там, где ничего не закрашивалось.
        pred_image = transforms.ToPILImage()(pred[0].clamp(0, 1).cpu())
        pred_image = pred_image.resize(original_size, Image.BILINEAR)

        # Для композита используем маску, апскейленную обратно из той же 256x256,
        # что видела модель, а не исходную полноразрешающую: если тонкий объект
        # (провод, столб) исчезает при NEAREST-downscale до 256, модель никогда не
        # получает его как дыру и не удаляет — а полноразмерная маска всё равно
        # пометила бы этот регион, наложив поверх него размытый неизменённый патч
        # вместо честного "не смогли распознать, оставили как есть". BILINEAR-
        # апскейл к тому же даёт мягкий край вместо шва между чётким фоном и
        # размытым патчем. Композитим через PIL (uint8), а не через float32-тензоры
        # на полном разрешении — на больших фото это на порядок меньше памяти.
        reveal_mask = small_mask_pil.resize(original_size, Image.BILINEAR)
        result_image = Image.composite(pred_image, image_pil, reveal_mask)
        return np.array(result_image.convert("RGB"))
