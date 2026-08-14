"""POST /inpaint: изображение + маска -> результат без объекта."""

import io
from functools import lru_cache

import numpy as np
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image, UnidentifiedImageError

from app.backend.services.inpainting_service import InpaintingService

router = APIRouter()


@lru_cache
def get_inpainting_service() -> InpaintingService:
    return InpaintingService()


@router.post("/inpaint")
async def inpaint(
    file: UploadFile = File(...),
    mask: UploadFile = File(...),
    service: InpaintingService = Depends(get_inpainting_service),
) -> Response:
    try:
        image = np.array(Image.open(io.BytesIO(await file.read())).convert("RGB"))
        mask_image = np.array(Image.open(io.BytesIO(await mask.read())).convert("L"))
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Не удалось прочитать изображение или маску")

    if mask_image.shape != image.shape[:2]:
        raise HTTPException(status_code=400, detail="Размер маски не совпадает с размером изображения")

    result = service.predict(image, mask_image)
    result_image = Image.fromarray(result)

    buffer = io.BytesIO()
    result_image.save(buffer, format="PNG")
    return Response(content=buffer.getvalue(), media_type="image/png")
