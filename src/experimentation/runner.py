"""
AaroSense - Multi-Model Experimentation Runner (Milestone 1).

Orchestrates the full experimental sweep:
  1. Load & preprocess raw CSV data.
  2. For each season, run Optuna-tuned experiments across all four model candidates.
  3. Retrain the best hyperparameter configuration on the full seasonal training set.
  4. Evaluate on the held-out test set.
  5. Log all metrics, parameters, and model artifacts to MLflow.
  6. Persist the best model artifact per season as a .joblib file.

MLflow Run Hierarchy:
    [Parent Run]  season + model
        └── [Child Runs]  individual Optuna trials (nested)

Usage:
    python -m src.experimentation.runner \
        --data-path data/raw/pm25_dataset.csv \
        --n-trials 40 \
        --models lightgbm xgboost catboost random_forest
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from src.experimentation.config import ExperimentConfig
from src.experimentation.metrics import compute_all_metrics, compute_persistence_baseline
from src.experimentation.model_factory import get_model_builder, get_search_space
from src.experimentation.tuner import OptunaSeasonalTuner
from src.preprocessing import PM25DataPreprocessor, PreprocessingConfig

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("AaroSense.Runner")


# ---------------------------------------------------------------------------
# Helper: rebuild best model from tuned parameters
# ---------------------------------------------------------------------------

def _build_best_model(
    model_name: str,
    best_params: Dict[str, Any],
    config: ExperimentConfig,
) -> Any:
    """
    Instantiate a model directly from the best Optuna parameters (no trial needed).

    We re-use the model factory logic by creating a FrozenTrial-compatible object.

    Args:
        model_name:  Algorithm identifier.
        best_params: Dictionary of best hyperparameters from Optuna.
        config:      Experiment configuration.

    Returns:
        Fitted-ready model instance.
    """
    import optuna

    # Create a frozen trial to reuse builder signatures
    frozen = optuna.trial.FixedTrial(best_params)
    space = get_search_space(model_name, config)
    builder = get_model_builder(model_name)
    return builder(frozen, space, config.random_state)


# ---------------------------------------------------------------------------
# Core experimentation class
# ---------------------------------------------------------------------------

class ExperimentationRunner:
    """
    Drives the full multi-model, multi-season MLOps experimentation pipeline.

    Args:
        config:    Experiment configuration.
        data_path: Path to the raw CSV file.
    """

    def __init__(self, config: ExperimentConfig, data_path: Path) -> None:
        self.config = config
        self.data_path = data_path
        self._data_path = data_path   # alias used by the M2 evaluation hook
        self._setup_mlflow()


    def _setup_mlflow(self) -> None:
        """Configure MLflow tracking URI and create/retrieve experiment."""
        mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
        experiment = mlflow.get_experiment_by_name(self.config.experiment_name)
        if experiment is None:
            mlflow.create_experiment(
                self.config.experiment_name,
                tags={"project": "AaroSense", "task": "PM25-Forecasting"},
            )
        mlflow.set_experiment(self.config.experiment_name)
        logger.info(
            "MLflow configured. Tracking URI: %s | Experiment: %s",
            self.config.mlflow_tracking_uri,
            self.config.experiment_name,
        )

    def _load_and_preprocess(self) -> Dict[str, Any]:
        """
        Load raw CSV, run preprocessing pipeline, and return seasonal splits.

        Returns:
            Dictionary mapping season names to ``SeasonalSplits`` objects.
        """
        logger.info("Loading dataset from '%s'...", self.data_path)
        if not self.data_path.exists():
            raise FileNotFoundError(f"Raw data not found at '{self.data_path}'.")

        raw_df = pd.read_csv(self.data_path)
        logger.info("Raw dataset loaded: shape=%s", raw_df.shape)

        pp_config = PreprocessingConfig(
            train_ratio=0.80,
            baseline_dir=Path("baselines"),
        )
        preprocessor = PM25DataPreprocessor(config=pp_config)
        cleaned_df = preprocessor.clean_and_prepare(raw_df)
        seasonal_data = preprocessor.split_by_season(cleaned_df)
        preprocessor.export_drift_baselines(seasonal_data)
        return seasonal_data

    def _run_single_experiment(
        self,
        season: str,
        model_name: str,
        splits: Any,
        tuner: OptunaSeasonalTuner,
    ) -> Dict[str, Any]:
        """
        Run the full experiment for one (season, model) pair:
        1. Optuna hyperparameter search (nested child runs).
        2. Retrain on full season training set with best params.
        3. Evaluate on held-out test set.
        4. Log everything to parent MLflow run.
        5. Save artifact to disk.

        Args:
            season:     Season identifier string.
            model_name: Algorithm identifier string.
            splits:     ``SeasonalSplits`` object for this season.
            tuner:      Initialised ``OptunaSeasonalTuner`` instance.

        Returns:
            Result dictionary with metrics, artifact path, and params.
        """
        X_train, y_train = splits.X_train, splits.y_train
        X_test, y_test = splits.X_test, splits.y_test
        parent_run_name = f"{season}__{model_name}__final"

        logger.info("=" * 65)
        logger.info("EXPERIMENT: Season='%s' | Model='%s'", season, model_name)
        logger.info("  Train size: %d | Test size: %d", len(X_train), len(X_test))

        with mlflow.start_run(
            run_name=parent_run_name,
            tags={
                "season": season,
                "model": model_name,
                "pipeline_stage": "final_evaluation",
                "project": "AaroSense",
            },
        ) as parent_run:
            parent_run_id = parent_run.info.run_id

            # ---- 1. Hyperparameter Tuning (nested child runs) -------------
            logger.info("  [1/4] Starting Optuna tuning (%d trials)...", self.config.n_optuna_trials)
            tuner.parent_run_id = parent_run_id
            best_params, best_cv_rmse = tuner.tune_model(
                model_name=model_name,
                season=season,
                X_train=X_train,
                y_train=y_train,
            )

            # ---- 2. Retrain best model on full training set ---------------
            logger.info("  [2/4] Retraining best model on full training set...")
            best_model = _build_best_model(model_name, best_params, self.config)
            best_model.fit(X_train, y_train)

            # ---- 3. Compute metrics (train + test + persistence baseline) -
            logger.info("  [3/4] Evaluating on held-out test set...")
            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            train_metrics = compute_all_metrics(y_train.to_numpy(), y_train_pred)
            test_metrics = compute_all_metrics(y_test.to_numpy(), y_test_pred)
            persistence_metrics = compute_persistence_baseline(y_test.to_numpy())

            # ---- 4. Log to MLflow parent run ------------------------------
            logger.info("  [4/4] Logging to MLflow...")

            # Tags / params
            mlflow.log_param("season", season)
            mlflow.log_param("model", model_name)
            mlflow.log_param("n_train_samples", len(X_train))
            mlflow.log_param("n_test_samples", len(X_test))
            mlflow.log_param("n_features", X_train.shape[1])
            mlflow.log_param("n_optuna_trials", self.config.n_optuna_trials)
            mlflow.log_param("n_cv_splits", self.config.n_cv_splits)
            mlflow.log_params({f"best__{k}": v for k, v in best_params.items()})

            # Tuning metric
            mlflow.log_metric("cv_best_rmse", best_cv_rmse)

            # Train metrics (prefixed)
            mlflow.log_metrics({f"train_{k}": v for k, v in train_metrics.items()})

            # Test metrics (prefixed)
            mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

            # Persistence baseline (for gate comparison)
            mlflow.log_metrics(
                {f"persistence_{k}": v for k, v in persistence_metrics.items()}
            )

            # Feature list as artifact
            feature_list_path = (
                self.config.model_artifact_dir / f"{season}_{model_name}_features.json"
            )
            feature_list_path.parent.mkdir(parents=True, exist_ok=True)
            with open(feature_list_path, "w") as fh:
                json.dump(list(X_train.columns), fh, indent=2)
            mlflow.log_artifact(str(feature_list_path), artifact_path="metadata")

            # Serialise model artifact to disk + log to MLflow
            sanitised_season = season.strip().lower().replace(" ", "_")
            model_path = (
                self.config.model_artifact_dir
                / sanitised_season
                / f"{model_name}_model.joblib"
            )
            model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(best_model, model_path)
            mlflow.log_artifact(str(model_path), artifact_path="model")

            # Log mlflow model (sklearn-compatible flavour).
            # MLflow 3.x uses skops for serialisation and requires an explicit
            # trusted-type allowlist for third-party estimator internals.
            _SKOPS_TRUSTED: Dict[str, list[str]] = {
                "lightgbm": [
                    "lightgbm.sklearn.LGBMRegressor",
                    "lightgbm.basic.Booster",
                    "collections.OrderedDict",
                ],
                "xgboost": [
                    "xgboost.sklearn.XGBRegressor",
                    "xgboost.core.Booster",
                    "collections.OrderedDict",
                    "numpy.dtype",
                ],
                "catboost": [
                    "catboost.core.CatBoostRegressor",
                    "catboost.core.CatBoost",
                    "collections.OrderedDict",
                ],
                "random_forest": [
                    "sklearn.ensemble._forest.RandomForestRegressor",
                    "sklearn.tree._classes.DecisionTreeRegressor",
                    "collections.OrderedDict",
                    "numpy.dtype",
                    "numpy.random.mtrand.RandomState",
                ],
            }
            trusted_types = _SKOPS_TRUSTED.get(model_name, [])
            mlflow.sklearn.log_model(
                best_model,
                artifact_path="sklearn_model",
                registered_model_name=None,  # Registration handled in Milestone 2
                input_example=X_train.head(5),
                skops_trusted_types=trusted_types if trusted_types else None,
            )

        result = {
            "season": season,
            "model": model_name,
            "best_params": best_params,
            "best_cv_rmse": best_cv_rmse,
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "persistence_metrics": persistence_metrics,
            "model_path": str(model_path),
            "mlflow_run_id": parent_run_id,
            "features": list(X_train.columns),
        }

        logger.info(
            "  RESULT  Test RMSE=%.4f | MAE=%.4f | R²=%.4f | MAPE=%.2f%% "
            "(Persistence RMSE=%.4f)",
            test_metrics["rmse"],
            test_metrics["mae"],
            test_metrics["r2"],
            test_metrics["mape"],
            persistence_metrics["rmse"],
        )
        return result

    def run(
        self,
        seasons: Optional[List[str]] = None,
        models: Optional[List[str]] = None,
        run_evaluation: bool = True,
        enable_shap: bool = True,
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Execute the full multi-model, multi-season experimentation sweep.

        Args:
            seasons:         Subset of seasons to run. Defaults to all available.
            models:          Subset of model candidates. Defaults to config candidates.
            run_evaluation:  If True, auto-run Milestone 2 evaluation after the sweep.
            enable_shap:     Whether to compute SHAP plots (only used if run_evaluation=True).

        Returns:
            Nested dict: ``results[season][model_name] = result_dict``.
        """
        seasonal_data = self._load_and_preprocess()
        tuner = OptunaSeasonalTuner(config=self.config)

        active_seasons = seasons if seasons else list(seasonal_data.keys())
        active_models = models if models else self.config.model_candidates

        logger.info(
            "Launching sweep: %d seasons × %d models = %d total experiments.",
            len(active_seasons),
            len(active_models),
            len(active_seasons) * len(active_models),
        )

        all_results: Dict[str, Dict[str, Dict[str, Any]]] = {}

        for season in active_seasons:
            if season not in seasonal_data:
                logger.warning("Season '%s' not found in data. Skipping.", season)
                continue

            splits = seasonal_data[season]
            all_results[season] = {}

            for model_name in active_models:
                try:
                    result = self._run_single_experiment(
                        season=season,
                        model_name=model_name,
                        splits=splits,
                        tuner=tuner,
                    )
                    all_results[season][model_name] = result
                except Exception as exc:
                    logger.error(
                        "Experiment failed for [%s | %s]: %s",
                        season,
                        model_name,
                        exc,
                        exc_info=True,
                    )
                    all_results[season][model_name] = {"error": str(exc)}

        # Save consolidated results JSON
        summary_path = self.config.model_artifact_dir / "experiment_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        _serialisable = _make_json_serialisable(all_results)
        with open(summary_path, "w") as fh:
            json.dump(_serialisable, fh, indent=2)
        logger.info("Consolidated experiment summary saved to '%s'.", summary_path)

        # ── Milestone 2: Evaluation, Explainability & Registry ─────────────
        if run_evaluation:
            logger.info("Starting Milestone 2 — Model Evaluation & Registry promotion...")
            try:
                from src.evaluation.evaluator import ModelEvaluator
                from src.evaluation.model_gate import GateConfig

                evaluator = ModelEvaluator(
                    report_dir=self.config.report_dir,
                    gate_config=GateConfig(
                        rmse_threshold=self.config.rmse_gate_threshold,
                    ),
                    tracking_uri=self.config.mlflow_tracking_uri,
                    enable_shap=enable_shap,
                )
                evaluator.evaluate_from_summary(
                    summary_path=summary_path,
                    data_path=self._data_path,
                )
            except Exception as exc:
                logger.warning(
                    "Milestone 2 evaluation raised an error (non-fatal): %s", exc,
                    exc_info=True,
                )

        _log_leaderboard(all_results)
        return all_results


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _make_json_serialisable(obj: Any) -> Any:
    """Recursively convert numpy/Path types to JSON-serialisable forms."""
    if isinstance(obj, dict):
        return {k: _make_json_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_json_serialisable(v) for v in obj]
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _log_leaderboard(
    all_results: Dict[str, Dict[str, Dict[str, Any]]]
) -> None:
    """Print a formatted leaderboard table to stdout after sweep completion."""
    rows = []
    for season, model_results in all_results.items():
        for model_name, result in model_results.items():
            if "error" in result:
                continue
            tm = result.get("test_metrics", {})
            rows.append(
                {
                    "Season": season,
                    "Model": model_name,
                    "Test RMSE": tm.get("rmse", float("nan")),
                    "Test MAE": tm.get("mae", float("nan")),
                    "Test R²": tm.get("r2", float("nan")),
                    "Test MAPE%": tm.get("mape", float("nan")),
                    "CV RMSE": result.get("best_cv_rmse", float("nan")),
                }
            )

    if not rows:
        logger.warning("No successful results to display in leaderboard.")
        return

    df = pd.DataFrame(rows).sort_values(["Season", "Test RMSE"])
    separator = "-" * 85
    logger.info("\n%s\n  LEADERBOARD — AaroSense Multi-Model Sweep\n%s", separator, separator)
    logger.info("\n%s", df.to_string(index=False))
    logger.info(separator)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AaroSense Multi-Model Experimentation Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/raw/pm25_dataset.csv"),
        help="Path to raw input CSV.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models"),
        help="Directory to save model artifacts.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("reports/plots"),
        help="Directory to save evaluation plots.",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=40,
        help="Number of Optuna trials per (model, season) study.",
    )
    parser.add_argument(
        "--n-cv-splits",
        type=int,
        default=5,
        help="Number of TimeSeriesSplit folds for CV.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["lightgbm", "xgboost", "catboost", "random_forest"],
        choices=["lightgbm", "xgboost", "catboost", "random_forest"],
        help="Model candidates to include in the sweep.",
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        default=None,
        help="Seasons to include (default: all detected seasons).",
    )
    parser.add_argument(
        "--mlflow-uri",
        type=str,
        default="sqlite:///mlruns/mlflow.db",
        help="MLflow tracking server URI.",
    )
    parser.add_argument(
        "--rmse-gate",
        type=float,
        default=50.0,
        help="Maximum acceptable test RMSE (µg/m³) for model gate.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    cfg = ExperimentConfig(
        model_artifact_dir=args.model_dir,
        report_dir=args.report_dir,
        n_optuna_trials=args.n_trials,
        n_cv_splits=args.n_cv_splits,
        model_candidates=args.models,
        mlflow_tracking_uri=args.mlflow_uri,
        rmse_gate_threshold=args.rmse_gate,
    )

    runner = ExperimentationRunner(config=cfg, data_path=args.data_path)
    results = runner.run(seasons=args.seasons, models=args.models)
