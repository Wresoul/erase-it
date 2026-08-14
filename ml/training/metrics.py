"""PSNR и SSIM — стандартные метрики качества восстановленного изображения.

Loss (reconstruction/adversarial) нужен только для обучения — он абстрактное число,
неудобное для человека. PSNR/SSIM выражают то же самое ("насколько предсказание
похоже на оригинал") в шкалах, которые сравнимы между разными моделями и статьями.
"""

import math

import torch
import torch.nn.functional as F


def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    """Peak Signal-to-Noise Ratio в дБ. Выше — лучше; >30 дБ обычно "хорошо"."""
    mse = F.mse_loss(pred, target).item()
    if mse == 0:
        return float("inf")
    return 10 * math.log10(max_val**2 / mse)


def ssim(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11, max_val: float = 1.0) -> float:
    """Structural Similarity Index (упрощённая версия на box-фильтре, без Гаусса).

    В отличие от PSNR, который сравнивает пиксели независимо, SSIM учитывает
    локальную структуру (яркость/контраст/корреляцию в окне) — ближе к тому,
    как разницу воспринимает человеческий глаз.
    """
    c1 = (0.01 * max_val) ** 2
    c2 = (0.03 * max_val) ** 2
    pad = window_size // 2

    def _blur(x: torch.Tensor) -> torch.Tensor:
        return F.avg_pool2d(x, window_size, stride=1, padding=pad)

    mu_pred = _blur(pred)
    mu_target = _blur(target)

    var_pred = _blur(pred * pred) - mu_pred**2
    var_target = _blur(target * target) - mu_target**2
    covar = _blur(pred * target) - mu_pred * mu_target

    numerator = (2 * mu_pred * mu_target + c1) * (2 * covar + c2)
    denominator = (mu_pred**2 + mu_target**2 + c1) * (var_pred + var_target + c2)
    return (numerator / denominator).mean().item()
