"""Обёртка над предобученной SAM: точка клика -> маска объекта.

Модель не обучается заново — как обсуждали в плане проекта, локализация объекта
использует готовую предобученную сеть, а собственное обучение (ml/) сосредоточено
на инпейнтинге.
"""

from pathlib import Path

import numpy as np
from segment_anything import SamPredictor, sam_model_registry

DEFAULT_CHECKPOINT = Path("ml/checkpoints/sam_vit_b.pth")
DEFAULT_MODEL_TYPE = "vit_b"


def select_best_mask(masks: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """SAM по одной точке возвращает несколько масок-кандидатов (объект/часть/деталь)
    с оценкой уверенности для каждой — берём кандидата с максимальной оценкой."""
    return masks[int(np.argmax(scores))]


class SegmentationService:
    """По клику пользователя (x, y) на изображении возвращает бинарную маску объекта."""

    def __init__(
        self,
        checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
        model_type: str = DEFAULT_MODEL_TYPE,
        device: str = "cpu",
    ):
        sam = sam_model_registry[model_type](checkpoint=str(checkpoint_path))
        sam.to(device)
        self._predictor = SamPredictor(sam)

    def predict_mask(self, image: np.ndarray, point: tuple[int, int]) -> np.ndarray:
        """image: HWC uint8 RGB. point: (x, y) в пикселях. Возвращает маску (H, W) bool."""
        self._predictor.set_image(image)
        point_coords = np.array([point])
        point_labels = np.array([1])  # 1 = точка внутри объекта, который нужно выделить

        masks, scores, _ = self._predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True,
        )
        return select_best_mask(masks, scores)
