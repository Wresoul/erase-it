"""FastAPI-приложение erase-it: эндпоинты /segment и /inpaint появятся на шаге 5."""

from fastapi import FastAPI

app = FastAPI(title="erase-it")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
