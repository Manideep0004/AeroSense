"""
AaroSense - Batch Monitor (Milestone 3).

High-level orchestrator that wires DriftDetector + DriftAlertManager
into a production-ready monitoring loop.

Responsibilities:
  - Accept incoming inference batches (from the FastAPI endpoint or a
    scheduled batch job).
  - Route each batch to the correct seasonal detector.
  - Persist every DriftReport to a JSONL audit log.
  - Emit PSI heatmap snapshots for the Streamlit dashboard.
  - Maintain an in-memory rolling summary for the /health endpoint.

Usage (programmatic):
    from src.monitoring.batch_monitor import BatchMonitor
    monitor = BatchMonitor()
    report  = monitor.monitor_batch(season="Winter", batch_df=df)

Usage (CLI — simulates a production batch from the test set):
    python3 -m src.monitoring.batch_monitor \
        --data-path data/raw/pm25_dataset.csv \
        --season Winter \
        --batch-size 300
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.monitoring.alert_manager import (
    DriftAlertManager,
    log_retraining_trigger_callback,
    mlflow_alert_callback,
)
from src.monitoring.drift_detector import DriftDetector
from src.monitoring.types import DriftReport, DriftSeverity

logger = logging.getLogger("AaroSense.BatchMonitor")


class BatchMonitor:
    """
    Production monitoring orchestrator for AaroSense inference batches.

    Integrates DriftDetector and DriftAlertManager into a single entry
    point that the FastAPI prediction endpoint calls after each batch.

    Args:
        baseline_dir:          Directory containing baseline Parquet files.
        report_log_dir:        Directory to write JSONL drift report logs.
        alert_log_dir:         Directory to write JSONL alert logs.
        ks_alpha:              KS test significance level (default 0.05).
        psi_threshold:         PSI drift trigger level (default 0.25).
        psi_warning:           PSI warning level (default 0.10).
        drift_fraction_alert:  Fraction of drifted features to trigger alert.
        enable_mlflow_alerts:  Whether to log alerts as MLflow runs.
        features_to_monitor:   Optional whitelist of features to test.
    """

    def __init__(
        self,
        baseline_dir: Path = Path("baselines"),
        report_log_dir: Path = Path("logs/drift_reports"),
        alert_log_dir: Path = Path("logs/drift_alerts"),
        ks_alpha: float = 0.05,
        psi_threshold: float = 0.25,
        psi_warning: float = 0.10,
        drift_fraction_alert: float = 0.20,
        enable_mlflow_alerts: bool = False,
        features_to_monitor: list[str] | None = None,
    ) -> None:
        self.report_log_dir = Path(report_log_dir)
        self.report_log_dir.mkdir(parents=True, exist_ok=True)

        # Detector
        self.detector = DriftDetector(
            baseline_dir=baseline_dir,
            ks_alpha=ks_alpha,
            psi_threshold=psi_threshold,
            psi_warning=psi_warning,
            n_psi_bins=10,
            drift_fraction_alert=drift_fraction_alert,
            features_to_monitor=features_to_monitor,
        )

        # Alert manager with built-in callbacks
        callbacks = [log_retraining_trigger_callback]
        if enable_mlflow_alerts:
            callbacks.append(mlflow_alert_callback)

        self.alert_manager = DriftAlertManager(
            alert_log_dir=alert_log_dir,
            retraining_callbacks=callbacks,
            min_severity_to_alert=DriftSeverity.WARNING,
        )

        # In-memory rolling summary (last N reports per season)
        self._recent_reports: dict[str, list[DriftReport]] = {}
        self._max_history: int = 50

        logger.info("BatchMonitor ready. Baseline dir: '%s'.", baseline_dir)

    # ------------------------------------------------------------------
    # Primary public interface
    # ------------------------------------------------------------------

    def monitor_batch(
        self,
        season: str,
        batch_df: pd.DataFrame,
        batch_id: str | None = None,
    ) -> DriftReport:
        """
        Run drift detection on an incoming inference batch and dispatch
        alerts if drift thresholds are exceeded.

        Args:
            season:    Season identifier (e.g., ``Winter``).
            batch_df:  Feature DataFrame from the inference pipeline.
                       Must contain the same columns as the baseline.
            batch_id:  Optional unique identifier for this batch
                       (e.g., request ID from the API).

        Returns:
            Completed ``DriftReport`` with per-feature results and verdict.
        """
        metadata = {
            "batch_id": batch_id or _generate_batch_id(),
            "monitored_at": datetime.now(tz=timezone.utc).isoformat(),
            "season": season,
        }

        # Run detection
        report = self.detector.run(
            season=season,
            production_batch=batch_df,
            batch_metadata=metadata,
        )

        # Persist report to JSONL audit log
        self._persist_report(report)

        # Dispatch alerts if needed
        self.alert_manager.dispatch(report)

        # Update rolling in-memory summary
        self._update_history(season, report)

        return report

    def monitor_batch_from_dict(
        self,
        season: str,
        records: list[dict[str, Any]],
        batch_id: str | None = None,
    ) -> DriftReport:
        """
        Convenience wrapper: accepts a list-of-dicts (JSON payload) and
        converts to a DataFrame before running drift detection.

        Args:
            season:   Season identifier.
            records:  List of feature dicts (one per inference row).
            batch_id: Optional batch identifier.

        Returns:
            ``DriftReport``.
        """
        batch_df = pd.DataFrame(records)
        return self.monitor_batch(season=season, batch_df=batch_df, batch_id=batch_id)

    # ------------------------------------------------------------------
    # Dashboard / health query helpers
    # ------------------------------------------------------------------

    def get_recent_summary(self, season: str, n: int = 10) -> list[dict]:
        """
        Return the last ``n`` drift report summaries for a season.
        Used by the Streamlit dashboard's rolling drift chart.

        Args:
            season: Season identifier.
            n:      Number of recent reports to return.

        Returns:
            List of report summary dicts (most recent last).
        """
        history = self._recent_reports.get(season, [])
        recent = history[-n:]
        return [
            {
                "monitored_at": r.metadata.get("monitored_at"),
                "batch_id": r.metadata.get("batch_id"),
                "drift_fraction": r.drift_fraction,
                "n_drifted": len(r.drifted_features),
                "severity": r.overall_severity.name,
                "alert": r.alert_triggered,
            }
            for r in recent
        ]

    def get_psi_heatmap(self, season: str, batch_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute a PSI heatmap DataFrame for dashboard rendering.

        Args:
            season:   Season identifier.
            batch_df: Incoming batch for comparison.

        Returns:
            DataFrame with columns: feature, psi, severity_label, psi_bin.
        """
        return self.detector.get_psi_heatmap_data(season, batch_df)

    def health_snapshot(self) -> dict[str, Any]:
        """
        Return a compact health/status dict for the /health API endpoint.

        Returns:
            Dict with per-season alert counts and latest severity.
        """
        snapshot: dict[str, Any] = {}
        for season, history in self._recent_reports.items():
            if not history:
                continue
            latest = history[-1]
            snapshot[season] = {
                "latest_severity": latest.overall_severity.name,
                "latest_drift_fraction": latest.drift_fraction,
                "latest_alert": latest.alert_triggered,
                "total_reports": len(history),
            }
        return snapshot

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _persist_report(self, report: DriftReport) -> None:
        """
        Append the DriftReport to a daily rotating JSONL log file:
        ``logs/drift_reports/reports_YYYY-MM-DD.jsonl``
        """
        date_str = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
        log_file = self.report_log_dir / f"reports_{date_str}.jsonl"
        try:
            with open(log_file, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(report.to_dict(), default=str) + "\n")
        except OSError as exc:
            logger.error("Failed to persist drift report: %s", exc)

    def _update_history(self, season: str, report: DriftReport) -> None:
        """Maintain rolling in-memory history of reports per season."""
        if season not in self._recent_reports:
            self._recent_reports[season] = []
        self._recent_reports[season].append(report)
        # Trim to max history window
        if len(self._recent_reports[season]) > self._max_history:
            self._recent_reports[season] = self._recent_reports[season][
                -self._max_history :
            ]


def _generate_batch_id() -> str:
    """Generate a short unique batch identifier."""
    import uuid

    return uuid.uuid4().hex[:12]


# ---------------------------------------------------------------------------
# CLI entry point — simulates a production inference batch
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AaroSense Batch Drift Monitor — simulate a production batch",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/raw/pm25_dataset.csv"),
        help="Path to raw CSV (test split used as simulated production batch).",
    )
    parser.add_argument(
        "--season",
        type=str,
        default="Winter",
        help="Season to simulate (must match a baseline Parquet file).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=300,
        help="Number of rows to sample from the test split as the batch.",
    )
    parser.add_argument(
        "--inject-drift",
        action="store_true",
        help="Artificially inject Gaussian drift into the batch features.",
    )
    parser.add_argument(
        "--drift-scale",
        type=float,
        default=3.0,
        help="Standard deviation scale for injected drift noise.",
    )
    parser.add_argument(
        "--ks-alpha",
        type=float,
        default=0.05,
        help="KS test significance level.",
    )
    parser.add_argument(
        "--psi-threshold",
        type=float,
        default=0.25,
        help="PSI drift trigger threshold.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        stream=sys.stdout,
    )

    args = _parse_args()

    # ── Load + preprocess data to get the seasonal test split ─────────────
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    import numpy as np

    from src.preprocessing import PM25DataPreprocessor, PreprocessingConfig

    raw_df = pd.read_csv(args.data_path)
    pp = PM25DataPreprocessor(config=PreprocessingConfig(train_ratio=0.80))
    cleaned = pp.clean_and_prepare(raw_df)
    splits = pp.split_by_season(cleaned)

    if args.season not in splits:
        print(
            f"ERROR: Season '{args.season}' not in data. Available: {list(splits.keys())}"
        )
        sys.exit(1)

    test_df = splits[args.season].X_test
    batch = test_df.sample(
        n=min(args.batch_size, len(test_df)), random_state=42
    ).reset_index(drop=True)

    # ── Optionally inject synthetic drift ─────────────────────────────────
    if args.inject_drift:
        logger.warning(
            "Injecting synthetic drift (scale=%.1f) into batch...", args.drift_scale
        )
        numeric_cols = batch.select_dtypes(include=[float, int]).columns.tolist()
        for col in numeric_cols[:8]:  # Drift first 8 features
            noise = np.random.normal(
                loc=batch[col].mean() * args.drift_scale,
                scale=batch[col].std() * args.drift_scale,
                size=len(batch),
            )
            batch[col] = batch[col] + noise

    # ── Run BatchMonitor ──────────────────────────────────────────────────
    monitor = BatchMonitor(
        ks_alpha=args.ks_alpha,
        psi_threshold=args.psi_threshold,
        enable_mlflow_alerts=False,
    )

    report = monitor.monitor_batch(
        season=args.season,
        batch_df=batch,
        batch_id="cli_simulation_001",
    )

    # ── Print full report ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"  DRIFT REPORT — {args.season} | Batch size: {len(batch)}")
    print("=" * 70)
    print(f"  Overall severity : {report.overall_severity.name}")
    print(f"  Alert triggered  : {report.alert_triggered}")
    print(f"  Drift fraction   : {report.drift_fraction:.1%}")
    print(f"  Drifted features : {report.drifted_features or 'None'}")
    print("\n  Per-feature breakdown:")
    print(f"  {'Feature':<26} {'KS-stat':>8} {'p-value':>10} {'PSI':>8}  Status")
    print("  " + "-" * 62)
    for r in sorted(report.feature_results, key=lambda x: x.psi.psi, reverse=True):
        status = f"{'⚠ DRIFT' if r.is_drifted else '   OK  '}"
        print(
            f"  {r.feature:<26} {r.ks.statistic:8.4f} {r.ks.p_value:10.4f}"
            f" {r.psi.psi:8.4f}  {status}"
        )
    print("=" * 70)
