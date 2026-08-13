"""Функции потерь для обучения инпейнтинга."""

import torch
import torch.nn.functional as F


def reconstruction_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    hole_weight: float = 6.0,
    valid_weight: float = 1.0,
) -> torch.Tensor:
    """L1 отдельно внутри дыры и снаружи, с разными весами (Liu et al., 2018).

    Дыра обычно занимает меньшую площадь изображения, поэтому её ошибку взвешивают
    сильнее — иначе градиент от простого копирования валидных пикселей доминирует
    и модель не учится закрашивать сам объект.
    """
    hole_loss = F.l1_loss(pred * mask, target * mask)
    valid_loss = F.l1_loss(pred * (1 - mask), target * (1 - mask))
    return hole_weight * hole_loss + valid_weight * valid_loss
