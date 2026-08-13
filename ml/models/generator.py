"""Генератор инпейнтинга: encoder-decoder на gated convolutions (Yu et al., 2019).

Обычная свёртка одинаково доверяет пикселям внутри дыры и снаружи неё, хотя пиксели
внутри дыры — это невалидные (замаскированные) данные. Gated convolution обучает
дополнительный "gate", который на каждом слое сам решает, каким пространственным
позициям и каналам доверять — это стандартный приём для inpainting-моделей.
"""

import torch
import torch.nn as nn


class GatedConv2d(nn.Module):
    """Свёртка, выход которой поэлементно взвешен обучаемым sigmoid-гейтом."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        activate: bool = True,
    ):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * 2,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.activation = nn.LeakyReLU(0.2, inplace=False) if activate else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features, gate = self.conv(x).chunk(2, dim=1)
        if self.activation is not None:
            features = self.activation(features)
        return features * torch.sigmoid(gate)


class GatedDeconv2d(nn.Module):
    """Апсемплинг nearest x2 + GatedConv2d — без checkerboard-артефактов transposed conv."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = GatedConv2d(in_channels, out_channels, kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.upsample(x))


class InpaintingGenerator(nn.Module):
    """Вход: masked_image (3ch) + mask (1ch). Выход: предсказанный RGB в [0, 1]."""

    def __init__(self, base_channels: int = 32):
        super().__init__()
        c = base_channels
        self.encoder = nn.Sequential(
            GatedConv2d(4, c, kernel_size=5),
            GatedConv2d(c, c * 2, stride=2),
            GatedConv2d(c * 2, c * 2),
            GatedConv2d(c * 2, c * 4, stride=2),
            GatedConv2d(c * 4, c * 4),
        )
        self.dilated = nn.Sequential(
            GatedConv2d(c * 4, c * 4, dilation=2),
            GatedConv2d(c * 4, c * 4, dilation=4),
            GatedConv2d(c * 4, c * 4, dilation=8),
        )
        self.decoder = nn.Sequential(
            GatedDeconv2d(c * 4, c * 2),
            GatedConv2d(c * 2, c * 2),
            GatedDeconv2d(c * 2, c),
            GatedConv2d(c, c // 2),
        )
        self.to_rgb = nn.Conv2d(c // 2, 3, kernel_size=3, padding=1)

    def forward(self, masked_image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = torch.cat([masked_image, mask], dim=1)
        x = self.encoder(x)
        x = self.dilated(x)
        x = self.decoder(x)
        return torch.sigmoid(self.to_rgb(x))


def composite(image: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Собирает финальный результат: предсказание внутри маски, оригинал — снаружи."""
    return image * (1 - mask) + pred * mask
