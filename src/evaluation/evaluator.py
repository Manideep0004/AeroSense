"""
AaroSense - Model Evaluator Orchestrator (Milestone 2).

Wires together:
  1. Diagnostic plots  (residuals, feature importance, SHAP)
  2. Automated gate    (RMSE / R² / MAE / persistence checks)
  3. Registry promotion (MLflow Staging / Production tagging)

Designed to be called **after** a model is trained and its MLflow run ID
is known — either from ``runner.py`` or standalone via CLI.

Usage (standalone):
    python -m src.evaluation.evaluator \
        --summary-path models/experiment_summary.json \
        --data-path data/raw/pm25_dataset.csv \
        --report-dir reports/plots
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
import numpy as np
import pandas as pd

from src.evaluation.model_gate import GateConfig, GateReport, GateVerdict, ModelEvaluationGate
from src.evaluation.plots import (
    plot_feature_importance,
    plot_residual_analysis,
    plot_shap_summary,
    plot_shap_waterfall,
)
from src.evaluation.registry import ModelRegistryManager
from src.preprocessing import PM25DataPreprocessor, PreprocessingConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("AaroSense.Evaluator")


class ModelEvaluator:
    """
    Post-training evaluation pipeline for AaroSense seasonal models.

    For each (season, model) result in the experiment summary:
      1. Load the saved ``.joblib`` artifact.
      2. Reload seasonal test data from the raw CSV.
      3. Generate all diagnostic plots.
      4. Run the automated evaluation gate.
      5. Register gate-passing models to the MLflow Registry as ``Staging``.
      6. Log all plots and gate reports back to the originating MLflow run.

    Args:
        report_dir:      Directory to save generated plots.
        gate_config:     Evaluation gate thresholds.
        tracking_uri:    MLflow tracking URI.
        enable_shap:     Whether to compute SHAP explanations (slow for large datasets).
        shap_sample_size: Number of samples to use for SHAP computation.
    """

    def __init__(
        self,
        report_dir: Path = Path("reports/plots"),
        gate_config: Optional[GateConfig] = None,
        tracking_uri: str = "sqlite:///mlruns/mlflow.db",
        enable_shap: bool = True,
        shap_sample_size: int = 500,
    ) -> None:
        self.report_dir = report_dir
        self.gate = ModelEvaluationGate(config=gate_config or GateConfig())
        self.registry = ModelRegistryManager(tracking_uri=tracking_uri)
        self.enable_shap = enable_shap
        self.shap_sample_size = shap_sample_size
        mlflow.set_tracking_uri(tracking_uri)
        self.report_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Data helpers
    # ------------------------------------------------------------------

    def _load_seasonal_test_data(
        self,
        data_path: Path,
        season: str,
        feature_cols: List[str],
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Re-run the preprocessing pipeline and return the test split
        for the requested season.

        Args:
            data_path:    Path to raw CSV.
            season:       Season identifier.
            feature_cols: Expected feature column order.

        Returns:
            Tuple of (X_test DataFrame, y_test Series).
        """
        raw_df = pd.read_csv(data_path)
        pp = PM25DataPreprocessor(config=PreprocessingConfig(train_ratio=0.80))
        cleaned = pp.clean_and_prepare(raw_df)
        splits = pp.split_by_season(cleaned)

        if season not in splits:
            raise KeyError(f"Season '{season}' not found in data. Available: {list(splits.keys())}")

        X_test = splits[season].X_test[feature_cols]
        y_test = splits[season].y_test
        return X_test, y_test

    # ------------------------------------------------------------------
    # Core per-model evaluation
    # ------------------------------------------------------------------

    def evaluate_single(
        self,
        result: Dict[str, Any],
        data_path: Path,
    ) -> GateReport:
        """
        Run the full evaluation pipeline for a single (season, model) result.

        Args:
            result:    Result dict from ``ExperimentationRunner`` (or JSON summary).
            data_path: Path to the raw CSV for reloading test data.

        Returns:
            ``GateReport`` containing the gate verdict and per-criterion results.
        """
        season = result["season"]
        model_name = result["model"]
        run_id = result.get("mlflow_run_id")
        model_path = Path(result["model_path"])
        feature_cols = result["features"]
        test_metrics = result["test_metrics"]
        persistence_metrics = result["persistence_metrics"]

        logger.info("=" * 65)
        logger.info("EVALUATING: Season='%s' | Model='%s'", season, model_name)

        # ── Load model artifact ───────────────────────────────────────────
        if not model_path.exists():
            logger.error("Model artifact not found at '%s'. Skipping.", model_path)
            return GateReport(
                season=season, model_name=model_name,
                verdict=GateVerdict.SKIPPED,
                summary=f"Model artifact missing at '{model_path}'.",
            )

        model = joblib.load(model_path)
        logger.info("Loaded model from '%s'.", model_path)

        # ── Load test data ────────────────────────────────────────────────
        X_test, y_test = self._load_seasonal_test_data(data_path, season, feature_cols)
        y_pred = model.predict(X_test)

        # ── 1. Generate diagnostic plots ──────────────────────────────────
        plot_paths: Dict[str, Path] = {}

        logger.info("  [1/3] Generating diagnostic plots...")

        plot_paths["residual_analysis"] = plot_residual_analysis(
            y_true=y_test.to_numpy(),
            y_pred=y_pred,
            season=season,
            model_name=model_name,
            report_dir=self.report_dir,
            metrics=test_metrics,
        )

        plot_paths["feature_importance"] = plot_feature_importance(
            model=model,
            feature_names=feature_cols,
            season=season,
            model_name=model_name,
            report_dir=self.report_dir,
            top_n=20,
        )

        if self.enable_shap:
            bee_path, bar_path = plot_shap_summary(
                model=model,
                X_sample=X_test,
                season=season,
                model_name=model_name,
                report_dir=self.report_dir,
                sample_size=self.shap_sample_size,
            )
            plot_paths["shap_beeswarm"] = bee_path
            plot_paths["shap_bar"] = bar_path

            plot_paths["shap_waterfall"] = plot_shap_waterfall(
                model=model,
                X_sample=X_test,
                season=season,
                model_name=model_name,
                report_dir=self.report_dir,
                instance_idx=0,
            )

        # ── 2. Run evaluation gate ────────────────────────────────────────
        logger.info("  [2/3] Running evaluation gate...")
        gate_report = self.gate.evaluate(
            season=season,
            model_name=model_name,
            test_metrics=test_metrics,
            persistence_metrics=persistence_metrics,
        )

        # ── 3. Log plots + gate report back to MLflow run ─────────────────
        if run_id:
            logger.info("  [3/3] Logging artifacts to MLflow run '%s'...", run_id)
            try:
                with mlflow.start_run(run_id=run_id):
                    # Log all generated plots
                    for plot_name, plot_path in plot_paths.items():
                        if plot_path.exists() and plot_path.suffix == ".png":
                            mlflow.log_artifact(
                                str(plot_path),
                                artifact_path=f"evaluation_plots/{plot_name}",
                            )

                    # Log gate report as JSON artifact
                    gate_report_path = (
                        self.report_dir
                        / season.lower()
                        / model_name
                        / "gate_report.json"
                    )
                    gate_report_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(gate_report_path, "w") as fh:
                        json.dump(gate_report.to_dict(), fh, indent=2)
                    mlflow.log_artifact(str(gate_report_path), artifact_path="evaluation_plots")

                    # Log gate verdict as metric (1=pass, 0=fail)
                    mlflow.log_metric(
                        "gate_passed",
                        1.0 if gate_report.verdict == GateVerdict.PASSED else 0.0,
                    )
            except Exception as exc:
                logger.warning("Could not log to MLflow run '%s': %s", run_id, exc)

        # ── 4. Registry promotion (gate-passing models only) ───────────────
        if gate_report.verdict == GateVerdict.PASSED and run_id:
            logger.info(
                "  Gate PASSED — registering '%s/%s' to MLflow Registry (Staging).",
                season, model_name,
            )
            try:
                version = self.registry.register_model(
                    run_id=run_id,
                    season=season,
                    model_name=model_name,
                    artifact_path="sklearn_model",
                )
                self.registry.transition_to_staging(
                    season=season,
                    model_name=model_name,
                    version=version.version,
                    description=(
                        f"Auto-promoted: RMSE={test_metrics['rmse']:.4f}, "
                        f"R²={test_metrics['r2']:.4f}, "
                        f"Beats persistence by "
                        f"{((persistence_metrics['rmse'] - test_metrics['rmse']) / persistence_metrics['rmse'] * 100):.1f}%"
                    ),
                )
            except Exception as exc:
                logger.warning("Registry promotion failed: %s", exc)
        elif gate_report.verdict == GateVerdict.FAILED and run_id:
            logger.warning(
                "  Gate FAILED — '%s/%s' will NOT be registered.",
                season, model_name,
            )

        return gate_report

    # ------------------------------------------------------------------
    # Batch evaluation from summary JSON
    # ------------------------------------------------------------------

    def evaluate_from_summary(
        self,
        summary_path: Path,
        data_path: Path,
    ) -> Dict[str, Dict[str, GateReport]]:
        """
        Load experiment results from a JSON summary file and evaluate all models.

        Args:
            summary_path: Path to ``models/experiment_summary.json``.
            data_path:    Path to raw CSV for test data reconstruction.

        Returns:
            Nested dict: ``reports[season][model_name] = GateReport``.
        """
        if not summary_path.exists():
            raise FileNotFoundError(f"Experiment summary not found at '{summary_path}'.")

        with open(summary_path) as fh:
            summary = json.load(fh)

        all_reports: Dict[str, Dict[str, GateReport]] = {}

        for season, model_results in summary.items():
            all_reports[season] = {}
            for model_name, result in model_results.items():
                if "error" in result:
                    logger.warning(
                        "Skipping [%s | %s] — recorded error: %s",
                        season, model_name, result["error"],
                    )
                    continue
                result["season"] = season
                result["model"] = model_name
                report = self.evaluate_single(result=result, data_path=data_path)
                all_reports[season][model_name] = report

        # Print final gate summary table
        self._print_gate_summary(all_reports)
        return all_reports

    # ------------------------------------------------------------------
    # Summary printer
    # ------------------------------------------------------------------

    @staticmethod
    def _print_gate_summary(
        reports: Dict[str, Dict[str, GateReport]]
    ) -> None:
        """Print a formatted gate result table to stdout."""
        rows = []
        for season, model_reports in reports.items():
            for model_name, report in model_reports.items():
                rows.append({
                    "Season": season,
                    "Model": model_name,
                    "Verdict": report.verdict.name,
                    "Failed Criteria": ", ".join(
                        c.name for c in report.criteria if not c.passed
                    ) or "—",
                })

        if not rows:
            return

        df = pd.DataFrame(rows)
        sep = "=" * 80
        logger.info("\n%s\n  EVALUATION GATE SUMMARY\n%s", sep, sep)
        logger.info("\n%s", df.to_string(index=False))
        logger.info(sep)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AaroSense Model Evaluator — Milestone 2",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--summary-path", type=Path,
        default=Path("models/experiment_summary.json"),
        help="Path to experiment_summary.json produced by the runner.",
    )
    parser.add_argument(
        "--data-path", type=Path,
        default=Path("data/raw/pm25_dataset.csv"),
        help="Path to raw CSV (for reconstructing test sets).",
    )
    parser.add_argument(
        "--report-dir", type=Path,
        default=Path("reports/plots"),
        help="Directory to save diagnostic plots.",
    )
    parser.add_argument(
        "--rmse-gate", type=float, default=50.0,
        help="Max acceptable test RMSE (µg/m³).",
    )
    parser.add_argument(
        "--r2-min", type=float, default=0.70,
        help="Minimum acceptable test R².",
    )
    parser.add_argument(
        "--no-shap", action="store_true",
        help="Disable SHAP computation (faster evaluation).",
    )
    parser.add_argument(
        "--shap-samples", type=int, default=500,
        help="Number of samples to use for SHAP computation.",
    )
    parser.add_argument(
        "--mlflow-uri", type=str,
        default="sqlite:///mlruns/mlflow.db",
        help="MLflow tracking server URI.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    evaluator = ModelEvaluator(
        report_dir=args.report_dir,
        gate_config=GateConfig(
            rmse_threshold=args.rmse_gate,
            r2_min=args.r2_min,
        ),
        tracking_uri=args.mlflow_uri,
        enable_shap=not args.no_shap,
        shap_sample_size=args.shap_samples,
    )

    reports = evaluator.evaluate_from_summary(
        summary_path=args.summary_path,
        data_path=args.data_path,
    )
