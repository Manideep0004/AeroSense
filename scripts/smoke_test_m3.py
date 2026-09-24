"""
AaroSense - Milestone 3 Smoke Test.

Tests three scenarios end-to-end:
  1. CLEAN batch  — from same distribution as baseline (expect no alert).
  2. DRIFTED batch — Gaussian-shifted features (expect CRITICAL + alert).
  3. PSI heatmap  — validates heatmap DataFrame structure for dashboard.

Run from project root:
    python3 scripts/smoke_test_m3.py
"""

from __future__ import annotations

import logging
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.monitoring.batch_monitor import BatchMonitor
from src.monitoring.types import DriftSeverity
from src.preprocessing import PM25DataPreprocessor, PreprocessingConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("SmokeTest.M3")

SEASON = "Winter"
DATA_PATH = Path("data/raw/pm25_dataset.csv")
BATCH_SIZE = 300


def load_clean_batch() -> pd.DataFrame:
    """Load a batch from the Winter test split (same distribution as baseline)."""
    raw_df = pd.read_csv(DATA_PATH)
    pp = PM25DataPreprocessor(config=PreprocessingConfig(train_ratio=0.80))
    cleaned = pp.clean_and_prepare(raw_df)
    splits = pp.split_by_season(cleaned)
    return splits[SEASON].X_test.sample(n=BATCH_SIZE, random_state=42).reset_index(drop=True)


def inject_drift(df: pd.DataFrame, scale: float = 4.0) -> pd.DataFrame:
    """Inject strong Gaussian shift into the first 12 numeric features."""
    drifted = df.copy()
    numeric_cols = drifted.select_dtypes(include=[float, int]).columns.tolist()[:12]
    for col in numeric_cols:
        noise = np.random.normal(
            loc=drifted[col].mean() * scale,
            scale=drifted[col].std() * scale,
            size=len(drifted),
        )
        drifted[col] = drifted[col] + noise
    return drifted


def run_test(name: str, batch: pd.DataFrame, monitor: BatchMonitor,
             expect_alert: bool) -> bool:
    """Run a single scenario and return True if it meets expectations."""
    logger.info("\n%s\n  Scenario: %s\n%s", "=" * 60, name, "=" * 60)
    report = monitor.monitor_batch(season=SEASON, batch_df=batch, batch_id=name)

    alert_ok = report.alert_triggered == expect_alert
    status = "✅ PASS" if alert_ok else "❌ FAIL"
    logger.info(
        "  %s | Alert expected=%s | Got=%s | Severity=%s | "
        "Drift=%.1f%% | Drifted features: %s",
        status, expect_alert, report.alert_triggered,
        report.overall_severity.name,
        report.drift_fraction * 100,
        report.drifted_features[:5] if report.drifted_features else "None",
    )
    return alert_ok


def main() -> None:
    np.random.seed(42)
    monitor = BatchMonitor(
        ks_alpha=0.05,
        psi_threshold=0.25,
        psi_warning=0.10,
        drift_fraction_alert=0.20,
        enable_mlflow_alerts=False,
    )

    all_passed = True

    # ── Scenario 1: Clean batch ───────────────────────────────────────────
    clean_batch = load_clean_batch()
    passed = run_test(
        name="clean_batch",
        batch=clean_batch,
        monitor=monitor,
        expect_alert=False,
    )
    all_passed = all_passed and passed

    # ── Scenario 2: Drifted batch ─────────────────────────────────────────
    drifted_batch = inject_drift(clean_batch, scale=4.0)
    passed = run_test(
        name="drifted_batch",
        batch=drifted_batch,
        monitor=monitor,
        expect_alert=True,
    )
    all_passed = all_passed and passed

    # ── Scenario 3: PSI heatmap structure ────────────────────────────────
    logger.info("\n%s\n  Scenario: PSI heatmap\n%s", "=" * 60, "=" * 60)
    heatmap_df = monitor.get_psi_heatmap(SEASON, drifted_batch)
    required_cols = {"feature", "psi", "ks_statistic", "severity_label", "psi_bin"}
    cols_ok = required_cols.issubset(set(heatmap_df.columns))
    rows_ok = len(heatmap_df) > 0
    heatmap_ok = cols_ok and rows_ok
    all_passed = all_passed and heatmap_ok
    logger.info(
        "  %s | Heatmap shape=%s | Columns=%s",
        "✅ PASS" if heatmap_ok else "❌ FAIL",
        heatmap_df.shape,
        list(heatmap_df.columns),
    )
    print("\nTop 5 features by PSI:")
    print(heatmap_df[["feature", "psi", "severity_label"]].head())

    # ── Scenario 4: Health snapshot ───────────────────────────────────────
    logger.info("\n%s\n  Scenario: Health snapshot\n%s", "=" * 60, "=" * 60)
    snapshot = monitor.health_snapshot()
    snap_ok = SEASON in snapshot
    all_passed = all_passed and snap_ok
    logger.info(
        "  %s | Snapshot: %s",
        "✅ PASS" if snap_ok else "❌ FAIL", snapshot,
    )

    # ── Verify JSONL audit log was written ────────────────────────────────
    report_logs = list(Path("logs/drift_reports").glob("*.jsonl"))
    alert_logs  = list(Path("logs/drift_alerts").glob("*.jsonl"))
    logs_ok = len(report_logs) > 0 and len(alert_logs) > 0
    all_passed = all_passed and logs_ok
    logger.info(
        "  %s | Report logs: %d | Alert logs: %d",
        "✅ PASS" if logs_ok else "❌ FAIL",
        len(report_logs), len(alert_logs),
    )

    # ── Final verdict ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    if all_passed:
        print("  ✅ MILESTONE 3 SMOKE TEST: ALL SCENARIOS PASSED")
    else:
        print("  ❌ MILESTONE 3 SMOKE TEST: SOME SCENARIOS FAILED")
    print("=" * 60)
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
