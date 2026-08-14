import math

import torch

from ml.training.metrics import psnr, ssim


def test_psnr_is_infinite_for_identical_images():
    image = torch.rand(1, 3, 16, 16)

    assert psnr(image, image) == float("inf")


def test_psnr_is_lower_for_more_different_images():
    image = torch.rand(1, 3, 16, 16)
    slightly_off = image + 0.05
    very_off = image + 0.5

    assert psnr(slightly_off, image) > psnr(very_off, image)


def test_ssim_is_close_to_one_for_identical_images():
    image = torch.rand(1, 3, 32, 32)

    assert math.isclose(ssim(image, image), 1.0, abs_tol=1e-4)


def test_ssim_is_lower_for_more_different_images():
    image = torch.rand(1, 3, 32, 32)
    slightly_off = image + 0.05
    very_off = torch.rand(1, 3, 32, 32)

    assert ssim(slightly_off, image) > ssim(very_off, image)
