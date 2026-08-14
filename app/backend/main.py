"""FastAPI-приложение erase-it: эндпоинты /segment и /inpaint."""

from fastapi import FastAPI

from app.backend.routers import inpaint, segment
from app.backend.schemas import HealthResponse

app = FastAPI(title="erase-it")
app.include_router(segment.router)
app.include_router(inpaint.router)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")
