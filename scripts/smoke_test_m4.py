"""
AaroSense - Milestone 4 Smoke Test.

Tests the FastAPI endpoints using TestClient:
  - GET /health
  - GET /model-info
  - POST /predict (single prediction)
  - POST /predict-batch (batch prediction, triggers async drift monitor)

Run from project root:
    python3 scripts/smoke_test_m4.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from fastapi.testclient import TestClient

from src.api.app import app

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger("SmokeTest.M4")

client = TestClient(app)

SEASON = "Winter"
DATA_PATH = Path("data/raw/pm25_dataset.csv")


def get_test_record():
    """Extract a single preprocessed record from the Winter test split."""
    from src.preprocessing import PM25DataPreprocessor, PreprocessingConfig

    raw_df = pd.read_csv(DATA_PATH)
    pp = PM25DataPreprocessor(config=PreprocessingConfig(train_ratio=0.80))
    cleaned = pp.clean_and_prepare(raw_df)
    splits = pp.split_by_season(cleaned)
    test_df = splits[SEASON].X_test
    # Use the 26 numeric columns needed by schemas
    feature_cols = [
        "temperature",
        "humidity",
        "wind_speed",
        "wind_direction",
        "pressure",
        "precipitation",
        "pm25",
        "pm25_lag_1",
        "pm25_lag_3",
        "pm25_lag_6",
        "pm25_lag_12",
        "pm25_lag_24",
        "pm25_mean_3h",
        "pm25_mean_6h",
        "pm25_mean_12h",
        "pm25_mean_24h",
        "wind_x",
        "wind_y",
        "hour_sin",
        "hour_cos",
        "day_of_week_sin",
        "day_of_week_cos",
        "month_sin",
        "month_cos",
        "day_of_year_sin",
        "day_of_year_cos",
    ]
    single_record = test_df[feature_cols].iloc[0].to_dict()
    # To test batch, grab a chunk
    batch_records = test_df[feature_cols].iloc[0:10].to_dict(orient="records")
    return single_record, batch_records


def main() -> None:
    all_passed = True
    logger.info("Starting M4 Smoke Test (FastAPI)...")

    # Ensure models are loaded
    response = client.get("/health")
    data = response.json()
    if data["status"] != "ok" or len(data["active_models"]) == 0:
        logger.error(f"Health check failed or no active models: {data}")
        sys.exit(1)

    logger.info(
        f"✅ /health check passed. Active models: {[m['model_name'] for m in data['active_models']]}"
    )

    # Check model info
    response = client.get("/model-info")
    info = response.json()
    if not isinstance(info, list) or len(info) == 0:
        logger.error(f"/model-info returned invalid data: {info}")
        all_passed = False
    else:
        logger.info(f"✅ /model-info passed. Found {len(info)} seasons loaded.")

    single_record, batch_records = get_test_record()

    # Test single prediction
    single_payload = {
        "season": SEASON,
        "features": single_record,
        "batch_id": "test_single_001",
    }

    response = client.post("/predict", json=single_payload)
    if response.status_code == 200:
        pred = response.json()
        logger.info(f"✅ /predict passed. Predicted PM2.5: {pred['prediction']:.2f}")
    else:
        logger.error(f"❌ /predict failed: {response.status_code} - {response.text}")
        all_passed = False

    # Test batch prediction
    batch_payload = {
        "season": SEASON,
        "features_list": batch_records,
        "batch_id": "test_batch_001",
    }

    response = client.post("/predict-batch", json=batch_payload)
    if response.status_code == 200:
        pred_batch = response.json()
        logger.info(
            f"✅ /predict-batch passed. Returned {len(pred_batch['predictions'])} predictions."
        )
    else:
        logger.error(
            f"❌ /predict-batch failed: {response.status_code} - {response.text}"
        )
        all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("  ✅ MILESTONE 4 SMOKE TEST: ALL ENDPOINTS PASSED")
    else:
        print("  ❌ MILESTONE 4 SMOKE TEST: SOME ENDPOINTS FAILED")
    print("=" * 60)
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
