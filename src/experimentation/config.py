"""
AaroSense - Experimentation Configuration.

Centralised hyperparameter search-space definitions and global experiment settings
for all four seasonal PM2.5 forecasting model candidates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Experiment-level constants
# ---------------------------------------------------------------------------

EXPERIMENT_NAME: str = "AaroSense-PM25-Seasonal-Forecasting"
MLFLOW_TRACKING_URI: str = "sqlite:///mlruns/mlflow.db"
RANDOM_STATE: int = 42
N_OPTUNA_TRIALS: int = 40  # Trials per model per season
N_CV_SPLITS: int = 5  # TimeSeriesSplit folds for temporal CV
OPTUNA_TIMEOUT_SECONDS: int = 600  # Hard cap per study (10 min)
OPTUNA_N_JOBS: int = 1  # Sequential to avoid MLflow log collision
METRICS_OPTIMIZE: str = "rmse"  # Primary metric Optuna minimises

# Evaluation thresholds for the model gate (Milestone 2)
RMSE_THRESHOLD_GATE: float = 50.0  # µg/m³ — reject models above this RMSE

# ---------------------------------------------------------------------------
# Dataclass containers for typed search-space definitions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LightGBMSearchSpace:
    """Optuna hyperparameter search space for LightGBM."""

    num_leaves: tuple[int, int] = (20, 200)
    max_depth: tuple[int, int] = (3, 12)
    n_estimators: tuple[int, int] = (100, 800)
    learning_rate: tuple[float, float] = (1e-3, 0.3)
    min_child_samples: tuple[int, int] = (5, 100)
    subsample: tuple[float, float] = (0.5, 1.0)
    colsample_bytree: tuple[float, float] = (0.5, 1.0)
    reg_alpha: tuple[float, float] = (1e-8, 10.0)
    reg_lambda: tuple[float, float] = (1e-8, 10.0)


@dataclass(frozen=True)
class XGBoostSearchSpace:
    """Optuna hyperparameter search space for XGBoost."""

    n_estimators: tuple[int, int] = (100, 800)
    max_depth: tuple[int, int] = (3, 10)
    learning_rate: tuple[float, float] = (1e-3, 0.3)
    subsample: tuple[float, float] = (0.5, 1.0)
    colsample_bytree: tuple[float, float] = (0.5, 1.0)
    min_child_weight: tuple[int, int] = (1, 10)
    gamma: tuple[float, float] = (0.0, 5.0)
    reg_alpha: tuple[float, float] = (1e-8, 10.0)
    reg_lambda: tuple[float, float] = (1e-8, 10.0)


@dataclass(frozen=True)
class CatBoostSearchSpace:
    """Optuna hyperparameter search space for CatBoost."""

    iterations: tuple[int, int] = (100, 800)
    depth: tuple[int, int] = (4, 10)
    learning_rate: tuple[float, float] = (1e-3, 0.3)
    l2_leaf_reg: tuple[float, float] = (1.0, 10.0)
    bagging_temperature: tuple[float, float] = (0.0, 1.0)
    border_count: tuple[int, int] = (32, 255)


@dataclass(frozen=True)
class RandomForestSearchSpace:
    """Optuna hyperparameter search space for Random Forest."""

    n_estimators: tuple[int, int] = (50, 500)
    max_depth: tuple[int, int] = (5, 30)
    min_samples_split: tuple[int, int] = (2, 20)
    min_samples_leaf: tuple[int, int] = (1, 10)
    max_features: tuple[float, float] = (0.3, 1.0)


@dataclass
class ExperimentConfig:
    """Master configuration dataclass for the full experimentation suite."""

    experiment_name: str = EXPERIMENT_NAME
    mlflow_tracking_uri: str = MLFLOW_TRACKING_URI
    random_state: int = RANDOM_STATE
    n_optuna_trials: int = N_OPTUNA_TRIALS
    n_cv_splits: int = N_CV_SPLITS
    optuna_timeout: int = OPTUNA_TIMEOUT_SECONDS
    optuna_n_jobs: int = OPTUNA_N_JOBS
    metrics_optimize: str = METRICS_OPTIMIZE
    rmse_gate_threshold: float = RMSE_THRESHOLD_GATE
    model_artifact_dir: Path = Path("models")
    report_dir: Path = Path("reports/plots")

    lgbm_space: LightGBMSearchSpace = field(default_factory=LightGBMSearchSpace)
    xgb_space: XGBoostSearchSpace = field(default_factory=XGBoostSearchSpace)
    cb_space: CatBoostSearchSpace = field(default_factory=CatBoostSearchSpace)
    rf_space: RandomForestSearchSpace = field(default_factory=RandomForestSearchSpace)

    # Candidate model names – controls which models are included in the sweep
    model_candidates: list[str] = field(
        default_factory=lambda: ["lightgbm", "xgboost", "catboost", "random_forest"]
    )
