"""Проверки безопасности для загружаемых изображений.

Два риска, которые тут закрываются:
1. DoS через огромный файл — читаем запрос чанками и обрываем раньше, чем
   весь файл окажется в памяти, если он больше разумного лимита.
2. "Decompression bomb" — маленький по размеру файл, который распаковывается
   в изображение с гигантским разрешением и съедает всю память при декодировании.
   Pillow сам проверяет это (Image.DecompressionBombError), но порог у него
   рассчитан на общий случай; здесь дополнительно ограничиваем разрешение
   явным лимитом, подходящим для веб-приложения с загрузкой фото руками.
"""

import io

from fastapi import HTTPException, UploadFile
from PIL import Image

MAX_UPLOAD_SIZE_BYTES = 20 * 1024 * 1024  # 20 MB
MAX_IMAGE_DIMENSION = 8000  # пикселей по любой стороне
_READ_CHUNK_SIZE = 1024 * 1024


async def read_upload_bounded(file: UploadFile, limit: int = MAX_UPLOAD_SIZE_BYTES) -> bytes:
    """Читает загруженный файл чанками, не давая ему превысить лимит размера."""
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(_READ_CHUNK_SIZE)
        if not chunk:
            break
        total += len(chunk)
        if total > limit:
            raise HTTPException(status_code=413, detail="Файл слишком большой")
        chunks.append(chunk)
    return b"".join(chunks)


def open_validated_image(data: bytes, mode: str = "RGB") -> Image.Image:
    """Открывает изображение с защитой от битых файлов и decompression bomb."""
    try:
        image = Image.open(io.BytesIO(data))
        width, height = image.size
    except (OSError, Image.DecompressionBombError):
        raise HTTPException(status_code=400, detail="Не удалось прочитать изображение")

    if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
        raise HTTPException(status_code=400, detail="Изображение слишком большое")

    try:
        return image.convert(mode)
    except (OSError, Image.DecompressionBombError):
        raise HTTPException(status_code=400, detail="Не удалось прочитать изображение")
