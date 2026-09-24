"""
AaroSense - Optuna Hyperparameter Tuner.

Runs per-model, per-season Optuna studies with TimeSeriesSplit cross-validation.
Each trial is logged as a nested MLflow child run for full lineage tracking.

Architecture:
    OptunaSeasonalTuner
        └── tune_model()          – one Optuna study per (model, season) pair
              └── _objective()   – single CV fold evaluation, nested MLflow run
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import mlflow
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from src.experimentation.config import ExperimentConfig
from src.experimentation.metrics import compute_all_metrics
from src.experimentation.model_factory import get_model_builder, get_search_space

# Suppress verbose Optuna logging in production; INFO is enough
optuna.logging.set_verbosity(optuna.logging.WARNING)

logger = logging.getLogger("AaroSense.Tuner")


class OptunaSeasonalTuner:
    """
    Orchestrates Optuna hyperparameter optimisation for each (model, season) pair
    and logs all trial metrics and parameters to MLflow nested runs.

    Args:
        config:          Experiment configuration dataclass.
        parent_run_id:   MLflow parent run ID for nested run hierarchy.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        parent_run_id: Optional[str] = None,
    ) -> None:
        self.config = config
        self.parent_run_id = parent_run_id
        self._tss = TimeSeriesSplit(n_splits=config.n_cv_splits)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _objective(
        self,
        trial: optuna.Trial,
        model_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        season: str,
    ) -> float:
        """
        Optuna objective function.

        Evaluates one set of hyperparameters via ``n_cv_splits``-fold
        TimeSeriesSplit CV and returns the mean validation RMSE.
        Logs individual trial parameters + CV metric to a nested MLflow run.

        Args:
            trial:      Active Optuna trial.
            model_name: Model algorithm name.
            X_train:    Full seasonal training features.
            y_train:    Full seasonal training targets.
            season:     Season label for logging.

        Returns:
            Mean validation RMSE across all folds (to minimise).
        """
        space = get_search_space(model_name, self.config)
        builder = get_model_builder(model_name)

        model = builder(trial, space, self.config.random_state)

        fold_rmse_list: list[float] = []

        for fold_idx, (train_idx, val_idx) in enumerate(
            self._tss.split(X_train)
        ):
            X_fold_train = X_train.iloc[train_idx]
            y_fold_train = y_train.iloc[train_idx]
            X_fold_val = X_train.iloc[val_idx]
            y_fold_val = y_train.iloc[val_idx]

            model.fit(X_fold_train, y_fold_train)
            y_fold_pred = model.predict(X_fold_val)

            fold_metrics = compute_all_metrics(y_fold_val.to_numpy(), y_fold_pred)
            fold_rmse_list.append(fold_metrics["rmse"])

            # Report intermediate value for Optuna pruning
            trial.report(fold_metrics["rmse"], step=fold_idx)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        mean_rmse = float(np.mean(fold_rmse_list))

        # ------------------------------------------------------------------
        # Log this trial as a nested MLflow child run
        # ------------------------------------------------------------------
        run_name = f"{season}__{model_name}__trial_{trial.number}"
        with mlflow.start_run(
            run_name=run_name,
            nested=True,
            tags={
                "season": season,
                "model": model_name,
                "trial_number": str(trial.number),
                "pipeline_stage": "hyperparameter_tuning",
            },
        ):
            # Log all hyperparameters sampled by Optuna
            mlflow.log_params(trial.params)
            # Log primary CV metric
            mlflow.log_metric("cv_mean_rmse", mean_rmse)
            mlflow.log_metric("cv_std_rmse", float(np.std(fold_rmse_list)))

        logger.debug(
            "[%s | %s] Trial %d — CV RMSE: %.4f",
            season,
            model_name,
            trial.number,
            mean_rmse,
        )
        return mean_rmse

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def tune_model(
        self,
        model_name: str,
        season: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> Tuple[Dict[str, Any], float]:
        """
        Run a full Optuna study for a single (model, season) combination.

        Uses ``TPESampler`` (Tree-structured Parzen Estimator) with
        ``MedianPruner`` for early stopping of underperforming trials.

        Args:
            model_name: Algorithm identifier (``lightgbm``, ``xgboost``, etc.).
            season:     Season label (``Winter``, ``Summer``, etc.).
            X_train:    Training features for this season.
            y_train:    Training target for this season.

        Returns:
            Tuple of:
                - best_params (Dict[str, Any]): Best hyperparameters found.
                - best_cv_rmse (float): Corresponding mean CV RMSE.
        """
        study_name = f"{self.config.experiment_name}__{season}__{model_name}"

        sampler = optuna.samplers.TPESampler(seed=self.config.random_state)
        pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)

        study = optuna.create_study(
            study_name=study_name,
            direction="minimize",
            sampler=sampler,
            pruner=pruner,
        )

        logger.info(
            "Starting Optuna study '%s' (%d trials, timeout=%ds).",
            study_name,
            self.config.n_optuna_trials,
            self.config.optuna_timeout,
        )

        study.optimize(
            lambda trial: self._objective(trial, model_name, X_train, y_train, season),
            n_trials=self.config.n_optuna_trials,
            timeout=self.config.optuna_timeout,
            n_jobs=self.config.optuna_n_jobs,
            catch=(Exception,),   # Gracefully skip malformed trials
            show_progress_bar=False,
        )

        if len(study.trials) == 0 or study.best_trial is None:
            raise RuntimeError(
                f"Optuna study '{study_name}' completed with no successful trials."
            )

        best_params = study.best_trial.params
        best_cv_rmse = study.best_value

        logger.info(
            "[%s | %s] Optimisation complete. Best CV RMSE=%.4f | Best params: %s",
            season,
            model_name,
            best_cv_rmse,
            best_params,
        )
        return best_params, best_cv_rmse
