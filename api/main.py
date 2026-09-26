"""FastAPI app entrypoint."""
from __future__ import annotations

from fastapi import FastAPI

from api.db import init_db
from api.routers.predict import router as predict_router
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="Customer Behavior Prediction Platform API",
    description="Churn risk, CLV, and segmentation scoring.",
    version="1.0.0",
)


@app.on_event("startup")
def on_startup() -> None:
    logger.info("Starting up: initializing database schema")
    init_db()


app.include_router(predict_router, tags=["predictions"])


@app.get("/")
def root() -> dict:
    return {
        "service": "Customer Behavior Prediction Platform",
        "docs": "/docs",
        "health": "/health",
    }