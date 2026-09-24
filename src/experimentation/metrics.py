"""
AaroSense - Metrics Utilities.

Computes standard and custom regression metrics for PM2.5 forecasting tasks.
All functions operate on numpy arrays / pandas Series for framework agnosticism.
"""

from __future__ import annotations

import logging
from typing import Dict

import numpy as np
import pandas as pd

logger = logging.getLogger("AaroSense.Metrics")


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Mean Absolute Percentage Error (MAPE).

    Zero-target observations are excluded to prevent division-by-zero.

    Args:
        y_true: Ground-truth target values.
        y_pred: Model prediction values.

    Returns:
        MAPE as a percentage (e.g., 12.5 means 12.5%).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    nonzero_mask = y_true != 0.0
    if not nonzero_mask.any():
        logger.warning("All target values are zero; MAPE is undefined. Returning NaN.")
        return float("nan")

    mape = np.mean(np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask]))
    return float(round(mape * 100.0, 4))


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute the full suite of regression evaluation metrics.

    Metrics:
        - RMSE  (Root Mean Squared Error)
        - MAE   (Mean Absolute Error)
        - R²    (Coefficient of Determination)
        - MAPE  (Mean Absolute Percentage Error, %)

    Args:
        y_true: Ground-truth target values (PM2.5 µg/m³).
        y_pred: Predicted values (PM2.5 µg/m³).

    Returns:
        Dictionary mapping metric names to float values.
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)

    rmse = float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr)))
    mae = float(mean_absolute_error(y_true_arr, y_pred_arr))
    r2 = float(r2_score(y_true_arr, y_pred_arr))
    mape = mean_absolute_percentage_error(y_true_arr, y_pred_arr)

    metrics = {
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "r2": round(r2, 4),
        "mape": round(mape, 4),
    }
    return metrics


def compute_persistence_baseline(y_true: np.ndarray) -> Dict[str, float]:
    """
    Compute metrics for a naïve persistence forecast (yhat_t = y_{t-1}).

    The persistence baseline is the canonical lower bound for time-series
    regression models; all deployed models must beat it.

    Args:
        y_true: Full target series (sorted chronologically).

    Returns:
        Dictionary of persistence-model metrics (RMSE, MAE, R², MAPE).
    """
    y_arr = np.asarray(y_true, dtype=float)
    if len(y_arr) < 2:
        raise ValueError("Need at least 2 samples to compute persistence baseline.")

    y_true_trimmed = y_arr[1:]    # t=1..N
    y_pred_persistence = y_arr[:-1]  # t=0..N-1  (previous step as prediction)

    metrics = compute_all_metrics(y_true_trimmed, y_pred_persistence)
    logger.info("Persistence baseline metrics: %s", metrics)
    return metrics
