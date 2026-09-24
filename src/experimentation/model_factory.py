"""
AaroSense - Model Factory.

Provides factory functions that build sklearn-compatible model instances from
Optuna trial parameter suggestions for each supported algorithm.
All models are wrapped with consistent interfaces (fit / predict).
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import optuna

logger = logging.getLogger("AaroSense.ModelFactory")

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------
ModelInstance = Any  # sklearn-compatible estimator


def build_lightgbm(trial: optuna.Trial, space: Any, random_state: int) -> ModelInstance:
    """
    Construct a LightGBM regressor with Optuna-sampled hyperparameters.

    Args:
        trial:        Active Optuna trial object.
        space:        LightGBMSearchSpace dataclass instance.
        random_state: Reproducibility seed.

    Returns:
        Configured ``lgb.LGBMRegressor`` instance.
    """
    import lightgbm as lgb  # lazy import — avoids cost if model not selected

    params: Dict[str, Any] = {
        "num_leaves": trial.suggest_int("num_leaves", *space.num_leaves),
        "max_depth": trial.suggest_int("max_depth", *space.max_depth),
        "n_estimators": trial.suggest_int("n_estimators", *space.n_estimators),
        "learning_rate": trial.suggest_float("learning_rate", *space.learning_rate, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", *space.min_child_samples),
        "subsample": trial.suggest_float("subsample", *space.subsample),
        "colsample_bytree": trial.suggest_float("colsample_bytree", *space.colsample_bytree),
        "reg_alpha": trial.suggest_float("reg_alpha", *space.reg_alpha, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", *space.reg_lambda, log=True),
        "random_state": random_state,
        "n_jobs": -1,
        "verbose": -1,
    }
    logger.debug("LightGBM trial params: %s", params)
    return lgb.LGBMRegressor(**params)


def build_xgboost(trial: optuna.Trial, space: Any, random_state: int) -> ModelInstance:
    """
    Construct an XGBoost regressor with Optuna-sampled hyperparameters.

    Args:
        trial:        Active Optuna trial object.
        space:        XGBoostSearchSpace dataclass instance.
        random_state: Reproducibility seed.

    Returns:
        Configured ``xgb.XGBRegressor`` instance.
    """
    import xgboost as xgb  # lazy import

    params: Dict[str, Any] = {
        "n_estimators": trial.suggest_int("n_estimators", *space.n_estimators),
        "max_depth": trial.suggest_int("max_depth", *space.max_depth),
        "learning_rate": trial.suggest_float("learning_rate", *space.learning_rate, log=True),
        "subsample": trial.suggest_float("subsample", *space.subsample),
        "colsample_bytree": trial.suggest_float("colsample_bytree", *space.colsample_bytree),
        "min_child_weight": trial.suggest_int("min_child_weight", *space.min_child_weight),
        "gamma": trial.suggest_float("gamma", *space.gamma),
        "reg_alpha": trial.suggest_float("reg_alpha", *space.reg_alpha, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", *space.reg_lambda, log=True),
        "random_state": random_state,
        "n_jobs": -1,
        "tree_method": "hist",
        "verbosity": 0,
    }
    logger.debug("XGBoost trial params: %s", params)
    return xgb.XGBRegressor(**params)


def build_catboost(trial: optuna.Trial, space: Any, random_state: int) -> ModelInstance:
    """
    Construct a CatBoost regressor with Optuna-sampled hyperparameters.

    Args:
        trial:        Active Optuna trial object.
        space:        CatBoostSearchSpace dataclass instance.
        random_state: Reproducibility seed.

    Returns:
        Configured ``CatBoostRegressor`` instance.
    """
    from catboost import CatBoostRegressor  # lazy import

    params: Dict[str, Any] = {
        "iterations": trial.suggest_int("iterations", *space.iterations),
        "depth": trial.suggest_int("depth", *space.depth),
        "learning_rate": trial.suggest_float("learning_rate", *space.learning_rate, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", *space.l2_leaf_reg),
        "bagging_temperature": trial.suggest_float(
            "bagging_temperature", *space.bagging_temperature
        ),
        "border_count": trial.suggest_int("border_count", *space.border_count),
        "random_seed": random_state,
        "verbose": 0,
        "allow_writing_files": False,
    }
    logger.debug("CatBoost trial params: %s", params)
    return CatBoostRegressor(**params)


def build_random_forest(trial: optuna.Trial, space: Any, random_state: int) -> ModelInstance:
    """
    Construct a Random Forest regressor with Optuna-sampled hyperparameters.

    Args:
        trial:        Active Optuna trial object.
        space:        RandomForestSearchSpace dataclass instance.
        random_state: Reproducibility seed.

    Returns:
        Configured ``RandomForestRegressor`` instance.
    """
    from sklearn.ensemble import RandomForestRegressor

    params: Dict[str, Any] = {
        "n_estimators": trial.suggest_int("n_estimators", *space.n_estimators),
        "max_depth": trial.suggest_int("max_depth", *space.max_depth),
        "min_samples_split": trial.suggest_int("min_samples_split", *space.min_samples_split),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", *space.min_samples_leaf),
        "max_features": trial.suggest_float("max_features", *space.max_features),
        "random_state": random_state,
        "n_jobs": -1,
    }
    logger.debug("RandomForest trial params: %s", params)
    return RandomForestRegressor(**params)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

MODEL_BUILDERS: Dict[str, Any] = {
    "lightgbm": build_lightgbm,
    "xgboost": build_xgboost,
    "catboost": build_catboost,
    "random_forest": build_random_forest,
}


def get_model_builder(model_name: str):
    """
    Retrieve the model factory function for the given algorithm name.

    Args:
        model_name: One of ``lightgbm``, ``xgboost``, ``catboost``, ``random_forest``.

    Returns:
        Callable factory function ``(trial, space, random_state) -> estimator``.

    Raises:
        KeyError: If ``model_name`` is not a registered model.
    """
    if model_name not in MODEL_BUILDERS:
        raise KeyError(
            f"Unknown model '{model_name}'. "
            f"Registered models: {list(MODEL_BUILDERS.keys())}"
        )
    return MODEL_BUILDERS[model_name]


def get_search_space(model_name: str, config: Any) -> Any:
    """
    Return the hyperparameter search space dataclass for a given model name.

    Args:
        model_name: Algorithm identifier string.
        config:     ``ExperimentConfig`` instance holding all search spaces.

    Returns:
        Search space dataclass instance.
    """
    space_map = {
        "lightgbm": config.lgbm_space,
        "xgboost": config.xgb_space,
        "catboost": config.cb_space,
        "random_forest": config.rf_space,
    }
    if model_name not in space_map:
        raise KeyError(f"No search space registered for model '{model_name}'.")
    return space_map[model_name]
