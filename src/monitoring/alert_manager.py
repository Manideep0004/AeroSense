"""
AaroSense - Drift Alert Manager (Milestone 3).

Receives a ``DriftReport`` from ``DriftDetector`` and dispatches alerts
through configurable channels when the alert threshold is crossed.

Supported alert channels:
  1. Structured logging (always active)
  2. JSON alert log file (rotating, audit-trail compliant)
  3. Retraining workflow trigger callback (pluggable — webhook, MLflow
     run, Airflow DAG trigger, etc.)

The alert manager is intentionally decoupled from the detector so that
new channels can be added without modifying detection logic.

Alert payload schema (JSON):
{
    "alert_id":          "<uuid>",
    "triggered_at":      "<ISO-8601 timestamp>",
    "season":            "Winter",
    "batch_size":        500,
    "drift_fraction":    0.35,
    "overall_severity":  "CRITICAL",
    "drifted_features":  ["pm25", "temperature", ...],
    "recommended_action":"retrain",
    "report":            { ... full DriftReport dict ... }
}
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from src.monitoring.types import DriftReport, DriftSeverity

logger = logging.getLogger("AaroSense.AlertManager")


# ---------------------------------------------------------------------------
# Alert payload builder
# ---------------------------------------------------------------------------


def _build_alert_payload(report: DriftReport) -> dict:
    """Construct a standardised alert payload dict from a DriftReport."""
    severity = report.overall_severity
    recommended_action = (
        "retrain"
        if severity == DriftSeverity.CRITICAL
        else "monitor_closely" if severity == DriftSeverity.WARNING else "none"
    )
    return {
        "alert_id": str(uuid.uuid4()),
        "triggered_at": datetime.now(tz=timezone.utc).isoformat(),
        "season": report.season,
        "batch_size": report.batch_size,
        "n_features_tested": report.n_features_tested,
        "drift_fraction": round(report.drift_fraction, 4),
        "overall_severity": severity.name,
        "drifted_features": report.drifted_features,
        "recommended_action": recommended_action,
        "metadata": report.metadata,
        "report": report.to_dict(),
    }


# ---------------------------------------------------------------------------
# Alert Manager
# ---------------------------------------------------------------------------


class DriftAlertManager:
    """
    Dispatches drift alerts through multiple channels when a ``DriftReport``
    has ``alert_triggered=True``.

    Args:
        alert_log_dir:       Directory to write JSON alert log files.
                             Set to None to disable file logging.
        retraining_callbacks: List of callables invoked with the alert payload
                              dict when an alert fires. Each callable must accept
                              a single ``dict`` argument.
        min_severity_to_alert: Minimum severity level that triggers dispatch.
                               Defaults to WARNING (both WARNING and CRITICAL alert).
    """

    def __init__(
        self,
        alert_log_dir: Path | None = Path("logs/drift_alerts"),
        retraining_callbacks: list[Callable[[dict], None]] | None = None,
        min_severity_to_alert: DriftSeverity = DriftSeverity.WARNING,
    ) -> None:
        self.alert_log_dir = Path(alert_log_dir) if alert_log_dir else None
        self.retraining_callbacks = retraining_callbacks or []
        self.min_severity_to_alert = min_severity_to_alert

        if self.alert_log_dir:
            self.alert_log_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "DriftAlertManager initialised. Log dir: '%s' | "
            "Callbacks: %d | Min severity: %s",
            self.alert_log_dir,
            len(self.retraining_callbacks),
            self.min_severity_to_alert.name,
        )

    def dispatch(self, report: DriftReport) -> dict | None:
        """
        Evaluate the DriftReport and dispatch alerts if thresholds are crossed.

        Args:
            report: Completed ``DriftReport`` from ``DriftDetector.run()``.

        Returns:
            Alert payload dict if an alert was dispatched, else ``None``.
        """
        severity_order = [
            DriftSeverity.NONE,
            DriftSeverity.WARNING,
            DriftSeverity.CRITICAL,
        ]
        should_alert = report.alert_triggered and severity_order.index(
            report.overall_severity
        ) >= severity_order.index(self.min_severity_to_alert)

        if not should_alert:
            logger.info(
                "No alert dispatched for [%s] — severity=%s, alert_triggered=%s.",
                report.season,
                report.overall_severity.name,
                report.alert_triggered,
            )
            return None

        payload = _build_alert_payload(report)
        self._log_alert(payload)
        self._write_alert_file(payload)
        self._invoke_callbacks(payload)

        return payload

    # ------------------------------------------------------------------
    # Private dispatch channels
    # ------------------------------------------------------------------

    def _log_alert(self, payload: dict) -> None:
        """Emit a structured WARNING log for the alert."""
        logger.warning(
            "🚨 DRIFT ALERT DISPATCHED | ID=%s | Season=%s | Severity=%s | "
            "Drift=%.1f%% | Action=%s | Drifted features: %s",
            payload["alert_id"],
            payload["season"],
            payload["overall_severity"],
            payload["drift_fraction"] * 100,
            payload["recommended_action"],
            payload["drifted_features"],
        )

    def _write_alert_file(self, payload: dict) -> None:
        """
        Append the alert as a newline-delimited JSON record to a rotating
        daily alert log file: ``logs/drift_alerts/alerts_YYYY-MM-DD.jsonl``.
        """
        if not self.alert_log_dir:
            return
        date_str = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
        log_file = self.alert_log_dir / f"alerts_{date_str}.jsonl"
        try:
            with open(log_file, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload, default=str) + "\n")
            logger.info("Alert written to '%s'.", log_file)
        except OSError as exc:
            logger.error("Failed to write alert log file: %s", exc)

    def _invoke_callbacks(self, payload: dict) -> None:
        """
        Call all registered retraining/webhook callbacks with the alert payload.

        Each callback receives the full alert payload dict. If a callback raises,
        the error is logged but other callbacks still execute (fault-isolated).
        """
        for i, callback in enumerate(self.retraining_callbacks):
            try:
                callback(payload)
                logger.info("Callback %d executed successfully.", i)
            except Exception as exc:
                logger.error("Callback %d failed: %s", i, exc, exc_info=True)

    def add_callback(self, callback: Callable[[dict], None]) -> None:
        """
        Register a new retraining/webhook callback at runtime.

        Args:
            callback: Callable that accepts an alert payload dict.
        """
        self.retraining_callbacks.append(callback)
        logger.info(
            "Registered new drift alert callback. Total: %d",
            len(self.retraining_callbacks),
        )


# ---------------------------------------------------------------------------
# Built-in callback helpers
# ---------------------------------------------------------------------------


def log_retraining_trigger_callback(payload: dict) -> None:
    """
    Built-in callback: logs a structured retraining trigger event.
    Plug this into ``DriftAlertManager`` as the default callback.
    In production, replace / augment with an Airflow/Prefect/MLflow trigger.
    """
    logger.warning(
        "🔄 RETRAINING WORKFLOW TRIGGERED | Season=%s | Alert ID=%s | "
        "Drifted features (%d): %s",
        payload.get("season"),
        payload.get("alert_id"),
        len(payload.get("drifted_features", [])),
        payload.get("drifted_features"),
    )


def mlflow_alert_callback(payload: dict) -> None:
    """
    Built-in callback: logs a drift alert as an MLflow run for traceability.
    Creates a standalone run tagged ``drift_alert`` in the AaroSense experiment.
    """
    try:
        import mlflow

        with mlflow.start_run(
            run_name=f"drift_alert_{payload['season']}_{payload['alert_id'][:8]}",
            tags={
                "alert_type": "drift",
                "season": payload["season"],
                "severity": payload["overall_severity"],
                "pipeline_stage": "monitoring",
            },
        ):
            mlflow.log_metric("drift_fraction", payload["drift_fraction"])
            mlflow.log_metric("n_drifted_features", len(payload["drifted_features"]))
            mlflow.log_param("recommended_action", payload["recommended_action"])
            mlflow.log_dict(payload, artifact_file="drift_alert_payload.json")
        logger.info("Drift alert logged to MLflow for season '%s'.", payload["season"])
    except Exception as exc:
        logger.warning("MLflow alert callback failed: %s", exc)
