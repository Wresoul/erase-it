import numpy as np
import pytest
from PIL import Image

from ml.datasets.inpainting_dataset import InpaintingDataset
from ml.datasets.mask_generator import generate_random_mask


def test_generate_random_mask_shape_and_range():
    mask = generate_random_mask(64, 64)

    assert mask.shape == (64, 64)
    assert mask.min() >= 0.0
    assert mask.max() <= 1.0


def test_generate_random_mask_covers_nonzero_area():
    mask = generate_random_mask(128, 128, min_strokes=5, max_strokes=8)

    assert mask.sum() > 0


def _make_dummy_images(tmp_path, count=3):
    for i in range(count):
        image = Image.fromarray((np.random.rand(64, 64, 3) * 255).astype(np.uint8))
        image.save(tmp_path / f"img_{i}.png")
    return tmp_path


def test_inpainting_dataset_getitem_shapes(tmp_path):
    image_dir = _make_dummy_images(tmp_path)
    dataset = InpaintingDataset(str(image_dir), image_size=64)

    assert len(dataset) == 3

    sample = dataset[0]
    assert sample["image"].shape == (3, 64, 64)
    assert sample["mask"].shape == (1, 64, 64)
    assert sample["masked_image"].shape == (3, 64, 64)


def test_inpainting_dataset_raises_on_empty_dir(tmp_path):
    with pytest.raises(ValueError):
        InpaintingDataset(str(tmp_path))
