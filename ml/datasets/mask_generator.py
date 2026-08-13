"""Процедурная генерация irregular free-form масок для обучения инпейнтинга.

Датасета с размеченными "нежелательными объектами" не существует, поэтому модель
обучается закрашивать произвольные формы (случайные мазки), а не конкретные классы
объектов. Это даёт обобщение на любую маску, которую в проде вернёт SAM по клику
пользователя — сама модель никогда не видит "объекты", только дыры произвольной формы.
"""

import math
import random

import numpy as np
import cv2


def generate_random_mask(
    height: int,
    width: int,
    min_strokes: int = 4,
    max_strokes: int = 10,
    max_angle: float = 4.0,
    max_length: int = 60,
    max_brush_width: int = 20,
) -> np.ndarray:
    """Возвращает маску (height, width) со значениями {0.0, 1.0}, где 1.0 — закрашенная область."""
    mask = np.zeros((height, width), dtype=np.float32)
    num_strokes = random.randint(min_strokes, max_strokes)

    for _ in range(num_strokes):
        start_x = random.randint(0, width - 1)
        start_y = random.randint(0, height - 1)
        num_segments = random.randint(1, 5)

        for segment in range(num_segments):
            angle = random.uniform(0.01, max_angle)
            if segment % 2 == 0:
                angle = 2 * math.pi - angle

            length = random.randint(10, max_length)
            brush_width = random.randint(5, max_brush_width)
            end_x = int(start_x + length * math.sin(angle))
            end_y = int(start_y + length * math.cos(angle))

            cv2.line(mask, (start_x, start_y), (end_x, end_y), 1.0, brush_width)
            cv2.circle(mask, (start_x, start_y), brush_width // 2, 1.0, -1)
            start_x, start_y = end_x, end_y

    return np.clip(mask, 0.0, 1.0)
