"""
AaroSense - Seasonal Model Training & Evaluation Pipeline.
Trains seasonal forecasting models for Delhi PM2.5 concentrations.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.preprocessing import PM25DataPreprocessor, PreprocessingConfig

logger = logging.getLogger("AaroSense.Training")


def compute_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    """Computes standard regression evaluation metrics."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": round(rmse, 4), "mae": round(mae, 4), "r2": round(r2, 4)}


def train_seasonal_models(
    raw_data_path: Path,
    output_model_dir: Path = Path("models"),
    baseline_dir: Path = Path("baselines"),
    train_ratio: float = 0.80,
) -> dict[str, dict[str, Any]]:
    """
    Complete pipeline:
    1. Loads raw CSV data.
    2. Runs data validation, cleaning, cyclical encoding, and wind decomposition.
    3. Splits data into chronological seasonal train/test sets.
    4. Exports KS drift test baseline Parquet files.
    5. Trains dedicated seasonal gradient boosting models.
    6. Evaluates test set performance and saves model artifacts (.joblib).
    """
    output_model_dir.mkdir(parents=True, exist_ok=True)
    baseline_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading raw dataset from %s ...", raw_data_path)
    if not raw_data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{raw_data_path}'. "
            f"Please place your CSV file in 'data/raw/' and verify the path."
        )

    raw_df = pd.read_csv(raw_data_path)
    logger.info("Raw dataset loaded successfully with shape: %s", raw_df.shape)

    # 1. Preprocessing & Feature Engineering
    config = PreprocessingConfig(train_ratio=train_ratio, baseline_dir=baseline_dir)
    preprocessor = PM25DataPreprocessor(config=config)
    cleaned_df = preprocessor.clean_and_prepare(raw_df)

    # 2. Split by Season chronologically
    seasonal_data = preprocessor.split_by_season(cleaned_df)

    # 3. Export MLOps drift baselines
    preprocessor.export_drift_baselines(seasonal_data, output_dir=baseline_dir)

    # 4. Train seasonal models
    results: dict[str, dict[str, Any]] = {}

    for season, splits in seasonal_data.items():
        logger.info("========== Training Model for Season: %s ==========", season)
        X_train, y_train = splits.X_train, splits.y_train
        X_test, y_test = splits.X_test, splits.y_test

        logger.info(
            "Season '%s' - Training set shape: %s | Test set shape: %s",
            season,
            X_train.shape,
            X_test.shape,
        )

        # Using HistGradientBoostingRegressor (fast, native categorical/missing support, tree-based)
        model = HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.05,
            max_depth=6,
            random_state=42,
        )
        model.fit(X_train, y_train)

        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        train_metrics = compute_metrics(y_train, y_train_pred)
        test_metrics = compute_metrics(y_test, y_test_pred)

        logger.info("Season '%s' Train Metrics: %s", season, train_metrics)
        logger.info("Season '%s' Test Metrics:  %s", season, test_metrics)

        # Save model artifact
        sanitized_season = season.strip().lower().replace(" ", "_")
        model_filename = output_model_dir / f"{sanitized_season}_pm25_model.joblib"
        joblib.dump(model, model_filename)
        logger.info("Saved model artifact to %s", model_filename.as_posix())

        results[season] = {
            "model_path": str(model_filename),
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "features": list(X_train.columns),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
        }

    # Save summary report
    summary_path = output_model_dir / "training_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    logger.info("Saved training summary report to %s", summary_path.as_posix())

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AaroSense Seasonal PM2.5 Training Pipeline"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/raw/delhi_pm25_raw.csv",
        help="Path to the raw input CSV file (default: data/raw/delhi_pm25_raw.csv)",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models",
        help="Directory to save trained model artifacts",
    )
    parser.add_argument(
        "--baseline-dir",
        type=str,
        default="baselines",
        help="Directory to export Parquet drift reference baselines",
    )
    args = parser.parse_args()

    train_seasonal_models(
        raw_data_path=Path(args.data_path),
        output_model_dir=Path(args.model_dir),
        baseline_dir=Path(args.baseline_dir),
    )
