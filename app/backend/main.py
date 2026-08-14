"""FastAPI-приложение erase-it: эндпоинты /segment, /inpaint и веб-интерфейс."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.backend.routers import inpaint, segment
from app.backend.schemas import HealthResponse

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

app = FastAPI(title="erase-it")
app.include_router(segment.router)
app.include_router(inpaint.router)
app.mount("/static", StaticFiles(directory=FRONTEND_DIR / "static"), name="static")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")
