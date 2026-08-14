"""SN-PatchGAN дискриминатор (Yu et al., 2019) для adversarial-обучения инпейнтинга.

В отличие от обычного GAN-дискриминатора, который выдаёт одно число "фото
настоящее/поддельное" на всю картинку, PatchGAN выдаёт сетку оценок — по одной
на каждый локальный участок изображения. Так дискриминатор заставляет генератор
следить за реалистичностью текстуры в каждой точке дыры, а не только в среднем
по картинке. Spectral normalization (torch.nn.utils.spectral_norm) ограничивает
"силу" каждого слоя и стабилизирует обучение GAN, которое иначе легко расходится.
"""

import torch
import torch.nn as nn


def _sn_conv(in_channels: int, out_channels: int) -> nn.Module:
    conv = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=2, padding=2)
    return nn.utils.parametrizations.spectral_norm(conv)


class PatchDiscriminator(nn.Module):
    """Вход: изображение (3ch) + маска (1ch). Выход: сетка "реалистичности" без sigmoid."""

    def __init__(self, base_channels: int = 64):
        super().__init__()
        c = base_channels
        self.net = nn.Sequential(
            _sn_conv(4, c),
            nn.LeakyReLU(0.2),
            _sn_conv(c, c * 2),
            nn.LeakyReLU(0.2),
            _sn_conv(c * 2, c * 4),
            nn.LeakyReLU(0.2),
            _sn_conv(c * 4, c * 4),
            nn.LeakyReLU(0.2),
        )

    def forward(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = torch.cat([image, mask], dim=1)
        return self.net(x)
