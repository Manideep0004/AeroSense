"""
AaroSense - Milestone 1 Smoke Test.

Fast integration test that runs a trimmed (2-trial, 2-fold, 1-season)
version of the experimentation pipeline to validate end-to-end wiring
without a full training run.

Run from the project root:
    python scripts/smoke_test_m1.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Ensure src is importable from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.experimentation.config import ExperimentConfig
from src.experimentation.runner import ExperimentationRunner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    stream=sys.stdout,
)

if __name__ == "__main__":
    cfg = ExperimentConfig(
        n_optuna_trials=2,  # Very fast — just validate wiring
        n_cv_splits=2,
        optuna_timeout=120,
        model_candidates=["lightgbm"],  # Single model for speed
        model_artifact_dir=Path("models"),
        report_dir=Path("reports/plots"),
        mlflow_tracking_uri="sqlite:///mlruns/mlflow.db",
    )

    runner = ExperimentationRunner(
        config=cfg,
        data_path=Path("data/raw/pm25_dataset.csv"),
    )

    results = runner.run(
        seasons=["Winter"],  # Single season for speed
        models=["lightgbm"],
    )

    print("\n=== SMOKE TEST PASSED ===")
    for season, model_results in results.items():
        for model, result in model_results.items():
            if "error" in result:
                print(f"  ERROR [{season}|{model}]: {result['error']}")
                sys.exit(1)
            tm = result["test_metrics"]
            print(
                f"  [{season}|{model}] "
                f"Test RMSE={tm['rmse']:.4f} | MAE={tm['mae']:.4f} | R²={tm['r2']:.4f}"
            )
    print("=========================\n")
