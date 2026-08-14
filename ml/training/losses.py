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


def discriminator_hinge_loss(real_scores: torch.Tensor, fake_scores: torch.Tensor) -> torch.Tensor:
    """Hinge loss для дискриминатора: push реальные оценки выше 1, поддельные ниже -1."""
    return torch.mean(F.relu(1.0 - real_scores)) + torch.mean(F.relu(1.0 + fake_scores))


def generator_hinge_loss(fake_scores: torch.Tensor) -> torch.Tensor:
    """Hinge loss для генератора: заставляет дискриминатор оценивать подделку выше."""
    return -torch.mean(fake_scores)
