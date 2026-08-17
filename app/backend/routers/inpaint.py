"""POST /inpaint: изображение + маска -> результат без объекта."""

import io
from functools import lru_cache

import numpy as np
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image

from app.backend.services.inpainting_service import InpaintingService
from app.backend.validation import open_validated_image, read_upload_bounded

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
    image_data = await read_upload_bounded(file)
    mask_data = await read_upload_bounded(mask)
    image = np.array(open_validated_image(image_data, mode="RGB"))
    mask_image = np.array(open_validated_image(mask_data, mode="L"))

    if mask_image.shape != image.shape[:2]:
        raise HTTPException(status_code=400, detail="Размер маски не совпадает с размером изображения")

    result = service.predict(image, mask_image)
    result_image = Image.fromarray(result)

    buffer = io.BytesIO()
    result_image.save(buffer, format="PNG")
    return Response(content=buffer.getvalue(), media_type="image/png")
