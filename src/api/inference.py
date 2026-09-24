"""
AaroSense - ML Inference Service (Milestone 4).

Loads the best seasonal models (from experiment summary) and handles
predictions and drift monitoring delegation.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import joblib

logger = logging.getLogger("AaroSense.InferenceService")

class InferenceService:
    def __init__(self, summary_path: Path = Path("models/experiment_summary.json")):
        self.summary_path = summary_path
        self.models: Dict[str, Any] = {}
        self.feature_cols: Dict[str, List[str]] = {}
        self.model_names: Dict[str, str] = {}
        self._load_best_models()

    def _load_best_models(self) -> None:
        """Parse the experiment summary and load the best model (lowest RMSE) per season."""
        if not self.summary_path.exists():
            logger.warning(f"Experiment summary not found at {self.summary_path}. Run M1 first.")
            return

        try:
            with open(self.summary_path, "r") as f:
                summary = json.load(f)

            for season, models_dict in summary.items():
                best_model = None
                best_rmse = float("inf")
                best_path = None
                best_features = None

                for model_name, metrics in models_dict.items():
                    if "error" in metrics:
                        continue
                    test_rmse = metrics.get("test_metrics", {}).get("rmse", float("inf"))
                    if test_rmse < best_rmse:
                        best_rmse = test_rmse
                        best_model = model_name
                        best_path = metrics.get("model_path")
                        best_features = metrics.get("features")

                if best_model and best_path and best_features:
                    model_filepath = Path(best_path)
                    if model_filepath.exists():
                        self.models[season] = joblib.load(model_filepath)
                        self.feature_cols[season] = best_features
                        self.model_names[season] = best_model
                        logger.info(f"Loaded {season} best model: {best_model} (RMSE: {best_rmse:.2f})")
                    else:
                        logger.error(f"Model file {model_filepath} not found for {season}.")
        except Exception as e:
            logger.error(f"Failed to load models: {e}")

    def predict(self, season: str, df: pd.DataFrame) -> List[float]:
        """Run inference for a given season and batch DataFrame."""
        season = season.capitalize()
        if season not in self.models:
            raise ValueError(f"No active model found for season: '{season}'")

        expected_features = self.feature_cols[season]
        
        # Validate columns
        missing = [f for f in expected_features if f not in df.columns]
        if missing:
            raise ValueError(f"Missing required features for {season}: {missing}")

        # Ensure correct order
        X = df[expected_features]
        model = self.models[season]
        
        preds = model.predict(X)
        # Ensure output is standard python float
        return [float(p) for p in preds]

    def get_model_info(self, season: str) -> Optional[Dict[str, Any]]:
        """Return metadata about the loaded model for a season."""
        season = season.capitalize()
        if season in self.model_names:
            return {
                "season": season,
                "model_name": self.model_names[season],
                "status": "active",
                "loaded": True
            }
        return {"season": season, "model_name": "unknown", "status": "missing", "loaded": False}

    def get_all_active_models(self) -> List[Dict[str, Any]]:
        return [self.get_model_info(s) for s in self.models.keys() if self.get_model_info(s) is not None]
