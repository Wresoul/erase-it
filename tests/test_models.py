import torch

from ml.models.discriminator import PatchDiscriminator
from ml.models.generator import InpaintingGenerator, composite


def test_generator_output_shape_matches_input():
    model = InpaintingGenerator(base_channels=8)
    masked_image = torch.rand(2, 3, 32, 32)
    mask = torch.randint(0, 2, (2, 1, 32, 32)).float()

    pred = model(masked_image, mask)

    assert pred.shape == (2, 3, 32, 32)
    assert pred.min() >= 0.0
    assert pred.max() <= 1.0


def test_composite_keeps_original_outside_mask():
    image = torch.rand(1, 3, 8, 8)
    pred = torch.zeros(1, 3, 8, 8)
    mask = torch.zeros(1, 1, 8, 8)  # маска пустая -> вся область "снаружи"

    result = composite(image, pred, mask)

    assert torch.allclose(result, image)


def test_composite_uses_prediction_inside_mask():
    image = torch.zeros(1, 3, 8, 8)
    pred = torch.ones(1, 3, 8, 8)
    mask = torch.ones(1, 1, 8, 8)  # маска полная -> вся область "внутри"

    result = composite(image, pred, mask)

    assert torch.allclose(result, pred)


def test_discriminator_outputs_patch_score_map():
    model = PatchDiscriminator(base_channels=8)
    image = torch.rand(2, 3, 32, 32)
    mask = torch.randint(0, 2, (2, 1, 32, 32)).float()

    scores = model(image, mask)

    assert scores.shape[0] == 2
    assert scores.shape[2] > 1 and scores.shape[3] > 1  # сетка патчей, не одно число
