import io

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from app.backend.main import app

client = TestClient(app)


def _make_test_image_bytes(size: int = 200) -> bytes:
    image = Image.new("RGB", (size, size), (70, 130, 180))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_index_serves_frontend_html():
    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "erase-it" in response.text


def test_static_app_js_is_served():
    response = client.get("/static/app.js")

    assert response.status_code == 200
    assert "fetch" in response.text


def test_segment_rejects_invalid_image():
    response = client.post(
        "/segment",
        files={"file": ("not-an-image.txt", b"hello world", "text/plain")},
        data={"x": 10, "y": 10},
    )

    assert response.status_code == 400


def test_segment_rejects_out_of_bounds_point():
    response = client.post(
        "/segment",
        files={"file": ("photo.png", _make_test_image_bytes(100), "image/png")},
        data={"x": 999, "y": 999},
    )

    assert response.status_code == 400


def test_segment_returns_mask_png():
    response = client.post(
        "/segment",
        files={"file": ("photo.png", _make_test_image_bytes(200), "image/png")},
        data={"x": 100, "y": 100},
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"

    mask = np.array(Image.open(io.BytesIO(response.content)))
    assert mask.shape == (200, 200)
    assert set(np.unique(mask)).issubset({0, 255})


def test_inpaint_rejects_mismatched_mask_size():
    response = client.post(
        "/inpaint",
        files={
            "file": ("photo.png", _make_test_image_bytes(200), "image/png"),
            "mask": ("mask.png", _make_test_image_bytes(64), "image/png"),
        },
    )

    assert response.status_code == 400


def test_inpaint_returns_result_png_same_size():
    size = 64
    image_bytes = _make_test_image_bytes(size)

    mask = Image.new("L", (size, size), 0)
    mask.paste(255, (16, 16, 48, 48))
    mask_buffer = io.BytesIO()
    mask.save(mask_buffer, format="PNG")

    response = client.post(
        "/inpaint",
        files={
            "file": ("photo.png", image_bytes, "image/png"),
            "mask": ("mask.png", mask_buffer.getvalue(), "image/png"),
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"

    result = Image.open(io.BytesIO(response.content))
    assert result.size == (size, size)
