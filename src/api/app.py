"""
AaroSense - Production Inference API (Milestone 4).

FastAPI application exposing endpoints for PM2.5 forecasting.
Features:
  - Pydantic v2 schemas for request validation.
  - Asynchronous background tasks for drift monitoring (so inference is fast).
  - Health check endpoint integrating ML model state and drift snapshots.
  - Multi-season model routing based on request payload.

Usage:
    uvicorn src.api.app:app --host 0.0.0.0 --port 8000
"""

import logging
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import BackgroundTasks, FastAPI, HTTPException, status

from src.api.inference import InferenceService
from src.api.schemas import (
    HealthResponse,
    ModelInfo,
    PredictBatchRequest,
    PredictionBatchResponse,
    PredictionResponse,
    PredictRequest,
)
from src.monitoring.batch_monitor import BatchMonitor

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AaroSense.API")

# Global singletons
inference_service = InferenceService()
batch_monitor = BatchMonitor()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup actions
    logger.info("Starting AaroSense API...")
    if not inference_service.models:
        logger.warning("No models loaded! API will return 503 for predictions.")
    yield
    # Shutdown actions
    logger.info("Shutting down AaroSense API...")


app = FastAPI(
    title="AaroSense PM2.5 Forecasting API",
    description="Production-grade API for seasonal PM2.5 forecasting with drift detection.",
    version="1.0.0",
    lifespan=lifespan,
)


def _run_drift_monitor_background(season: str, batch_df: pd.DataFrame, batch_id: str):
    """Background task wrapper for drift monitoring."""
    try:
        logger.info(f"Running background drift monitoring for batch {batch_id}...")
        batch_monitor.monitor_batch(season=season, batch_df=batch_df, batch_id=batch_id)
    except Exception as exc:
        logger.error(
            f"Background drift monitoring failed for batch {batch_id}: {exc}",
            exc_info=True,
        )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """System health snapshot: models loaded and drift status."""
    active_models = inference_service.get_all_active_models()
    drift_status = batch_monitor.health_snapshot()

    return HealthResponse(
        status="ok" if active_models else "degraded",
        active_models=[ModelInfo(**m) for m in active_models],
        drift_status=drift_status,
    )


@app.get("/model-info", response_model=list[ModelInfo])
async def get_model_info():
    """Retrieve metadata about currently loaded models for each season."""
    active_models = inference_service.get_all_active_models()
    return [ModelInfo(**m) for m in active_models]


@app.post("/predict", response_model=PredictionResponse)
async def predict_single(request: PredictRequest):
    """
    Predict PM2.5 for a single observation.
    Note: Single predictions are not sent to the drift monitor to avoid noisy PSI/KS tests on n=1.
    """
    season = request.season.capitalize()
    model_info = inference_service.get_model_info(season)

    if not model_info or not model_info["loaded"]:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"No active model for season '{season}'",
        )

    # Convert Pydantic model to DataFrame
    df = pd.DataFrame([request.features.model_dump()])

    try:
        preds = inference_service.predict(season, df)
        return PredictionResponse(
            prediction=preds[0],
            model_used=model_info["model_name"],
            season=season,
            batch_id=request.batch_id,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal inference error",
        )


@app.post("/predict-batch", response_model=PredictionBatchResponse)
async def predict_batch(
    request: PredictBatchRequest, background_tasks: BackgroundTasks
):
    """
    Predict PM2.5 for a batch of observations and trigger asynchronous drift monitoring.
    """
    season = request.season.capitalize()
    model_info = inference_service.get_model_info(season)

    if not model_info or not model_info["loaded"]:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"No active model for season '{season}'",
        )

    # Convert list of Pydantic models to DataFrame
    df = pd.DataFrame([f.model_dump() for f in request.features_list])

    try:
        preds = inference_service.predict(season, df)

        batch_id = request.batch_id or "batch_auto"

        # Dispatch background monitoring if n >= 5 (minimum for KS/PSI)
        if len(df) >= 5:
            background_tasks.add_task(
                _run_drift_monitor_background, season, df, batch_id
            )
        else:
            logger.info("Batch size < 5, skipping drift detection.")

        return PredictionBatchResponse(
            predictions=preds,
            model_used=model_info["model_name"],
            season=season,
            batch_id=batch_id,
        )

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal inference error",
        )



from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

# Mount frontend assets
if os.path.exists("frontend/dist"):
    app.mount("/assets", StaticFiles(directory="frontend/dist/assets"), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_frontend(full_path: str):
        if os.path.exists(f"frontend/dist/{full_path}") and os.path.isfile(f"frontend/dist/{full_path}"):
            return FileResponse(f"frontend/dist/{full_path}")
        return FileResponse("frontend/dist/index.html")
