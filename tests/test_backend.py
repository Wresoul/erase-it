import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from app.backend.main import app
from app.backend.validation import MAX_IMAGE_DIMENSION, MAX_UPLOAD_SIZE_BYTES

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


def test_segment_rejects_truncated_image():
    # PIL распознаёт PNG-заголовок и не кидает UnidentifiedImageError, но падает
    # позже с OSError при попытке дочитать данные — раньше это давало голый 500.
    truncated_png = _make_test_image_bytes(100)[:20]

    response = client.post(
        "/segment",
        files={"file": ("broken.png", truncated_png, "image/png")},
        data={"x": 10, "y": 10},
    )

    assert response.status_code == 400


def test_segment_rejects_foreign_origin():
    response = client.post(
        "/segment",
        files={"file": ("photo.png", _make_test_image_bytes(100), "image/png")},
        data={"x": 10, "y": 10},
        headers={"Origin": "https://evil.example.com"},
    )

    assert response.status_code == 403


def test_segment_allows_matching_origin():
    response = client.post(
        "/segment",
        files={"file": ("photo.png", _make_test_image_bytes(100), "image/png")},
        data={"x": 10, "y": 10},
        headers={"Origin": str(client.base_url)},
    )

    assert response.status_code == 200


def test_inpaint_rejects_foreign_origin():
    response = client.post(
        "/inpaint",
        files={
            "file": ("photo.png", _make_test_image_bytes(100), "image/png"),
            "mask": ("mask.png", _make_test_image_bytes(100), "image/png"),
        },
        headers={"Origin": "https://evil.example.com"},
    )

    assert response.status_code == 403


def test_segment_rejects_oversized_upload():
    oversized = b"x" * (MAX_UPLOAD_SIZE_BYTES + 1)

    response = client.post(
        "/segment",
        files={"file": ("huge.png", oversized, "image/png")},
        data={"x": 10, "y": 10},
    )

    assert response.status_code == 413


def test_segment_rejects_oversized_dimensions():
    # Узкая, но очень широкая картинка — дёшево создать (мало пикселей),
    # но превышает лимит по одной стороне, как и было бы у decompression bomb.
    huge_image = Image.new("RGB", (MAX_IMAGE_DIMENSION + 1, 10), (0, 0, 0))
    buffer = io.BytesIO()
    huge_image.save(buffer, format="PNG")

    response = client.post(
        "/segment",
        files={"file": ("huge.png", buffer.getvalue(), "image/png")},
        data={"x": 10, "y": 5},
    )

    assert response.status_code == 400


def test_unknown_path_returns_html_error_page_for_browser_navigation():
    response = client.get("/no-such-page", headers={"accept": "text/html"})

    assert response.status_code == 404
    assert "text/html" in response.headers["content-type"]
    assert "404" in response.text
    assert "Страница не найдена" in response.text


def test_unknown_path_returns_json_for_non_browser_requests():
    response = client.get("/no-such-page")

    assert response.status_code == 404
    assert "application/json" in response.headers["content-type"]
    assert "detail" in response.json()


def test_segment_error_stays_json_even_with_html_accept_header():
    # /segment вызывается из app.js через fetch() и должен всегда возвращать JSON,
    # даже если по случайности будет отправлен Accept: text/html.
    response = client.post(
        "/segment",
        files={"file": ("not-an-image.txt", b"hello world", "text/plain")},
        data={"x": 10, "y": 10},
        headers={"accept": "text/html"},
    )

    assert response.status_code == 400
    assert "application/json" in response.headers["content-type"]


def test_no_cors_headers_are_ever_sent():
    # Same-origin приложение: подтверждаем, что запрос с чужого Origin не получает
    # Access-Control-Allow-* — если кто-то в будущем случайно добавит permissive
    # CORSMiddleware, этот тест должен упасть.
    response = client.get("/health", headers={"Origin": "https://evil.example.com"})

    assert "access-control-allow-origin" not in response.headers


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


def test_inpaint_reports_untrained_model():
    # В репозитории нет ml/checkpoints/generator.pth, поэтому сервис работает на
    # случайных весах — заголовок должен честно сообщать об этом клиенту.
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
    assert response.headers["x-model-trained"] == "false"


def test_inpaint_rejects_truncated_image():
    truncated_png = _make_test_image_bytes(100)[:20]

    response = client.post(
        "/inpaint",
        files={
            "file": ("broken.png", truncated_png, "image/png"),
            "mask": ("mask.png", _make_test_image_bytes(100), "image/png"),
        },
    )

    assert response.status_code == 400


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


def _make_scenario_image_bytes(width: int, height: int, shape: str | None, box: tuple | None) -> bytes:
    image = Image.new("RGB", (width, height), (70, 130, 180))
    if shape is not None:
        draw = ImageDraw.Draw(image)
        getattr(draw, shape)(box, fill=(230, 126, 34))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


# Без настоящих фото пользователя проверяем устойчивость на разнообразных синтетических
# сценариях: маленький квадрат, широкий/узкий формат, нечётные размеры, объект у самого
# края и клик по однородному фону без выраженного объекта.
SHAPE_SCENARIOS = [
    pytest.param(64, 64, "ellipse", (16, 16, 48, 48), 32, 32, id="small-square"),
    pytest.param(400, 200, "rectangle", (250, 50, 350, 150), 300, 100, id="wide-landscape"),
    pytest.param(150, 300, "ellipse", (30, 100, 120, 220), 75, 160, id="tall-portrait"),
    pytest.param(257, 199, "ellipse", (10, 10, 60, 60), 35, 35, id="odd-dims-object-near-corner"),
    pytest.param(200, 200, None, None, 100, 100, id="click-on-plain-background"),
]


@pytest.mark.parametrize("width,height,shape,box,click_x,click_y", SHAPE_SCENARIOS)
def test_full_pipeline_handles_varied_images(width, height, shape, box, click_x, click_y):
    """End-to-end: /segment -> /inpaint на разных по форме/размеру изображениях."""
    image_bytes = _make_scenario_image_bytes(width, height, shape, box)

    segment_response = client.post(
        "/segment",
        files={"file": ("photo.png", image_bytes, "image/png")},
        data={"x": click_x, "y": click_y},
    )
    assert segment_response.status_code == 200
    mask_bytes = segment_response.content
    assert Image.open(io.BytesIO(mask_bytes)).size == (width, height)

    inpaint_response = client.post(
        "/inpaint",
        files={
            "file": ("photo.png", image_bytes, "image/png"),
            "mask": ("mask.png", mask_bytes, "image/png"),
        },
    )
    assert inpaint_response.status_code == 200
    assert Image.open(io.BytesIO(inpaint_response.content)).size == (width, height)
