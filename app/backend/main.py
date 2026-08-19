"""FastAPI-приложение erase-it: эндпоинты /segment, /inpaint и веб-интерфейс."""

import logging
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.backend.error_pages import render_error_page
from app.backend.routers import inpaint, segment
from app.backend.schemas import HealthResponse

logger = logging.getLogger(__name__)

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

# Сознательно НЕ добавляем CORSMiddleware: фронтенд отдаётся тем же самым FastAPI-
# приложением (см. index() ниже), поэтому все запросы всегда same-origin и никакого
# кросс-доменного доступа не требуется. Самая строгая защита — не открывать его вообще,
# а не открывать и сразу же ограничивать. Если в будущем фронтенд переедет на отдельный
# домен — добавляй CORSMiddleware с явным списком origins (allow_origins=[...]),
# НИКОГДА не используй allow_origins=["*"] вместе с allow_credentials=True.
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


def _wants_html(request: Request) -> bool:
    # Прямая навигация браузера (адресная строка, /unknown-path) — это всегда GET
    # с "text/html" в Accept. /segment и /inpaint вызываются из app.js через
    # fetch(POST) и должны получать JSON независимо от заголовков — поэтому метод
    # проверяем в первую очередь, а не полагаемся на один Accept.
    return request.method == "GET" and "text/html" in request.headers.get("accept", "")


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    if _wants_html(request):
        detail = exc.detail if isinstance(exc.detail, str) else None
        html = render_error_page(exc.status_code, detail)
        return HTMLResponse(content=html, status_code=exc.status_code, headers=exc.headers)
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail}, headers=exc.headers)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception while processing %s %s", request.method, request.url.path)
    if _wants_html(request):
        html = render_error_page(500)
        return HTMLResponse(content=html, status_code=500)
    return JSONResponse(status_code=500, content={"detail": "Внутренняя ошибка сервера"})
