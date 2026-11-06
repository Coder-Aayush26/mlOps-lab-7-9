"""FastAPI inference service.

Run:  uvicorn src.api:app --host 127.0.0.1 --port 8000
Env:  MODEL_PATH (default models/model.joblib), MLFLOW_TRACKING_URI (optional), LOG_LEVEL
"""
from __future__ import annotations

import logging
import os
import urllib.request
from contextlib import asynccontextmanager
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from src import __version__
from src.predict import get_model_path, load_metadata, load_model, predict as score

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("api")

state: dict[str, Any] = {"model": None, "meta": {}}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup (path is read here so tests/containers can set MODEL_PATH)."""
    path = get_model_path()
    try:
        state["model"] = load_model(path)
        state["meta"] = load_metadata(path)
        logger.info("Model loaded from %s", path)
    except Exception:  # noqa: BLE001
        state["model"], state["meta"] = None, {}
        logger.exception("Could not load model from %s", path)
    yield
    state["model"] = None


app = FastAPI(
    title="ML Prediction API",
    version=__version__,
    description="Breast-cancer (benign/malignant) classifier served from a saved scikit-learn Pipeline.",
    lifespan=lifespan,
)


class PredictionInput(BaseModel):
    """Request schema. Business constraints: sizes must be > 0; shape metrics are in [0, 1]."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "mean_radius": 14.1, "mean_texture": 19.3, "mean_perimeter": 92.0, "mean_area": 654.9,
                "mean_smoothness": 0.096, "mean_compactness": 0.104, "mean_concavity": 0.089,
                "mean_concave_points": 0.049, "mean_symmetry": 0.181, "mean_fractal_dimension": 0.063,
            }
        },
    )
    mean_radius: float = Field(gt=0, le=50)
    mean_texture: float = Field(gt=0, le=60)
    mean_perimeter: float = Field(gt=0, le=300)
    mean_area: float = Field(gt=0, le=5000)
    mean_smoothness: float = Field(ge=0, le=1)
    mean_compactness: float = Field(ge=0, le=1)
    mean_concavity: float = Field(ge=0, le=1)
    mean_concave_points: float = Field(ge=0, le=1)
    mean_symmetry: float = Field(ge=0, le=1)
    mean_fractal_dimension: float = Field(ge=0, le=1)


class PredictionOutput(BaseModel):
    prediction: int
    label: str
    probability: float
    probabilities: dict[str, float]
    model_version: str


class BatchInput(BaseModel):
    records: list[PredictionInput] = Field(min_length=1, max_length=1000)


class BatchOutput(BaseModel):
    count: int
    predictions: list[PredictionOutput]


def _require_model():
    if state["model"] is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    return state["model"]


def _version() -> str:
    return state["meta"].get("trained_at_utc", "unknown")


@app.get("/health")
def health() -> dict[str, Any]:
    """Liveness/readiness probe."""
    return {"status": "ok", "model_loaded": state["model"] is not None}


@app.get("/info")
def info() -> dict[str, Any]:
    """Model metadata + optional MLflow reachability (used to verify API->MLflow connectivity)."""
    meta = state["meta"]
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    reachable: bool | None = None
    if uri and uri.startswith("http"):
        try:
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(uri.rstrip("/") + "/health", timeout=2) as r:
                reachable = r.status == 200
        except Exception:  # noqa: BLE001
            reachable = False
    return {
        "model_loaded": state["model"] is not None,
        "model_type": meta.get("model_type"),
        "trained_at_utc": meta.get("trained_at_utc"),
        "features": meta.get("features"),
        "metrics": meta.get("metrics"),
        "mlflow_tracking_uri": uri,
        "mlflow_reachable": reachable,
    }


@app.post("/predict", response_model=PredictionOutput)
def predict(payload: PredictionInput) -> dict[str, Any]:
    """Online inference for ONE record."""
    model = _require_model()
    row = pd.DataFrame([payload.model_dump()])
    result = score(row, model=model, class_names=state["meta"].get("class_names"))[0]
    return {**result, "model_version": _version()}


@app.post("/predict/batch", response_model=BatchOutput)
def predict_batch(payload: BatchInput) -> dict[str, Any]:
    """Small synchronous batch (<= 1000 records). Larger jobs: see docs/BATCH_INFERENCE_DESIGN.md."""
    model = _require_model()
    df = pd.DataFrame([r.model_dump() for r in payload.records])
    results = score(df, model=model, class_names=state["meta"].get("class_names"))
    return {"count": len(results), "predictions": [{**r, "model_version": _version()} for r in results]}


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Log details server-side; never leak internals to the client."""
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})
