"""POST /segment: изображение + точка клика -> маска объекта (SAM)."""

import io
from functools import lru_cache

import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image

from app.backend.services.segmentation_service import SegmentationService
from app.backend.validation import open_validated_image, read_upload_bounded, verify_same_origin

router = APIRouter()


@lru_cache
def get_segmentation_service() -> SegmentationService:
    return SegmentationService()


@router.post("/segment", dependencies=[Depends(verify_same_origin)])
async def segment(
    file: UploadFile = File(...),
    x: int = Form(...),
    y: int = Form(...),
    service: SegmentationService = Depends(get_segmentation_service),
) -> Response:
    data = await read_upload_bounded(file)
    image = np.array(open_validated_image(data, mode="RGB"))

    height, width = image.shape[:2]
    if not (0 <= x < width and 0 <= y < height):
        raise HTTPException(status_code=400, detail="Точка клика вне границ изображения")

    mask = service.predict_mask(image, (x, y))
    mask_image = Image.fromarray((mask.astype(np.uint8)) * 255)

    buffer = io.BytesIO()
    mask_image.save(buffer, format="PNG")
    return Response(content=buffer.getvalue(), media_type="image/png")
