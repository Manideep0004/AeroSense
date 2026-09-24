"""
AaroSense - Feature Drift Detector (Milestone 3).

Implements two complementary statistical drift detection methods:

1. Kolmogorov-Smirnov (KS) Two-Sample Test
   - Non-parametric test comparing CDFs of reference vs production distributions.
   - Triggers if p-value < alpha (default 0.05).
   - Best for: detecting any distributional shift in continuous features.

2. Population Stability Index (PSI)
   - Measures divergence between two binned probability distributions.
   - Triggers if PSI > threshold (default 0.25).
   - Best for: quantifying the magnitude of drift for business reporting.

For each incoming production inference batch the detector:
  - Loads the seasonal reference baseline from Parquet.
  - Runs KS + PSI on every numerical feature.
  - Computes an aggregate DriftReport with per-feature results.
  - Fires an alert if any feature crosses either threshold.

PSI Formula:
    PSI = Σ (P_actual - P_expected) * ln(P_actual / P_expected)
    where P_actual and P_expected are bin proportions of the two distributions.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from src.monitoring.types import (
    DriftReport,
    DriftSeverity,
    FeatureDriftResult,
    KSTestResult,
    PSIResult,
)

logger = logging.getLogger("AaroSense.DriftDetector")


# ---------------------------------------------------------------------------
# PSI computation
# ---------------------------------------------------------------------------

def _compute_psi(
    reference: np.ndarray,
    production: np.ndarray,
    n_bins: int = 10,
    epsilon: float = 1e-8,
) -> float:
    """
    Compute Population Stability Index between reference and production distributions.

    Bin edges are derived from the reference distribution only — ensuring
    a stable reference frame regardless of production range.

    Args:
        reference:  Reference (training) feature values.
        production: Production (incoming batch) feature values.
        n_bins:     Number of equally-spaced percentile bins.
        epsilon:    Small constant to avoid log(0) or division by zero.

    Returns:
        PSI score (float). Higher = more drift.
    """
    # Derive bin edges from reference using percentile spacing
    breakpoints = np.nanpercentile(reference, np.linspace(0, 100, n_bins + 1))
    # Deduplicate edges (can occur in sparse/discrete distributions)
    breakpoints = np.unique(breakpoints)

    if len(breakpoints) < 3:
        # Degenerate case: constant feature — no meaningful PSI
        logger.debug("Degenerate bin breakpoints for PSI (constant feature?). Returning 0.")
        return 0.0

    # Bin proportions — clip production values to reference range
    ref_counts, _ = np.histogram(reference, bins=breakpoints)
    prod_counts, _ = np.histogram(production, bins=breakpoints)

    ref_props  = ref_counts  / (ref_counts.sum()  + epsilon)
    prod_props = prod_counts / (prod_counts.sum() + epsilon)

    # Add epsilon to avoid log(0)
    ref_props  = np.clip(ref_props,  epsilon, None)
    prod_props = np.clip(prod_props, epsilon, None)

    psi = float(np.sum((prod_props - ref_props) * np.log(prod_props / ref_props)))
    return psi


# ---------------------------------------------------------------------------
# Per-feature drift tests
# ---------------------------------------------------------------------------

def _run_ks_test(
    reference: np.ndarray,
    production: np.ndarray,
    feature: str,
    alpha: float = 0.05,
) -> KSTestResult:
    """
    Run the Kolmogorov-Smirnov two-sample test for a single feature.

    Args:
        reference:  Reference distribution values.
        production: Production distribution values.
        feature:    Feature name (for result labelling).
        alpha:      Significance level for rejecting the null hypothesis.

    Returns:
        ``KSTestResult`` with statistic, p-value, and drift flag.
    """
    # Drop NaNs before testing
    ref_clean  = reference[~np.isnan(reference)]
    prod_clean = production[~np.isnan(production)]

    if len(ref_clean) < 5 or len(prod_clean) < 5:
        logger.warning(
            "Feature '%s': insufficient non-null samples for KS test "
            "(ref=%d, prod=%d). Skipping.", feature, len(ref_clean), len(prod_clean),
        )
        return KSTestResult(feature=feature, statistic=0.0, p_value=1.0,
                            drifted=False, alpha=alpha)

    ks_stat, p_value = stats.ks_2samp(ref_clean, prod_clean)
    drifted = bool(p_value < alpha)
    return KSTestResult(
        feature=feature,
        statistic=float(ks_stat),
        p_value=float(p_value),
        drifted=drifted,
        alpha=alpha,
    )


def _run_psi_test(
    reference: np.ndarray,
    production: np.ndarray,
    feature: str,
    psi_threshold: float = 0.25,
    n_bins: int = 10,
) -> PSIResult:
    """
    Compute PSI for a single feature and evaluate against threshold.

    Args:
        reference:     Reference distribution values.
        production:    Production distribution values.
        feature:       Feature name (for result labelling).
        psi_threshold: Drift trigger threshold (default 0.25).
        n_bins:        Number of bins for PSI discretization.

    Returns:
        ``PSIResult`` with PSI score and drift flag.
    """
    ref_clean  = reference[~np.isnan(reference)]
    prod_clean = production[~np.isnan(production)]

    if len(ref_clean) < 5 or len(prod_clean) < 5:
        logger.warning(
            "Feature '%s': insufficient samples for PSI (ref=%d, prod=%d). Skipping.",
            feature, len(ref_clean), len(prod_clean),
        )
        return PSIResult(feature=feature, psi=0.0, drifted=False,
                         psi_threshold=psi_threshold, n_bins=n_bins)

    psi = _compute_psi(ref_clean, prod_clean, n_bins=n_bins)
    drifted = bool(psi > psi_threshold)
    return PSIResult(
        feature=feature,
        psi=psi,
        drifted=drifted,
        psi_threshold=psi_threshold,
        n_bins=n_bins,
    )


# ---------------------------------------------------------------------------
# Main DriftDetector class
# ---------------------------------------------------------------------------

class DriftDetector:
    """
    Production drift detector for AaroSense seasonal PM2.5 models.

    Loads seasonal reference baseline distributions from Parquet files
    and monitors incoming inference batches for statistical drift using
    KS tests and PSI scores.

    Args:
        baseline_dir:     Directory containing ``<season>_baseline.parquet`` files.
        ks_alpha:         KS test significance level (default 0.05).
        psi_threshold:    PSI drift trigger threshold (default 0.25).
        psi_warning:      PSI warning level (default 0.10).
        n_psi_bins:       Number of PSI bins (default 10).
        drift_fraction_alert: Fraction of drifted features that triggers
                              an overall alert (default 0.20 = 20%).
        features_to_monitor: Optional whitelist of feature names to test.
                             Defaults to all numeric columns in the baseline.
    """

    def __init__(
        self,
        baseline_dir: Path = Path("baselines"),
        ks_alpha: float = 0.05,
        psi_threshold: float = 0.25,
        psi_warning: float = 0.10,
        n_psi_bins: int = 10,
        drift_fraction_alert: float = 0.20,
        features_to_monitor: Optional[List[str]] = None,
    ) -> None:
        self.baseline_dir = Path(baseline_dir)
        self.ks_alpha = ks_alpha
        self.psi_threshold = psi_threshold
        self.psi_warning = psi_warning
        self.n_psi_bins = n_psi_bins
        self.drift_fraction_alert = drift_fraction_alert
        self.features_to_monitor = features_to_monitor

        # Cache: season → baseline DataFrame
        self._baseline_cache: Dict[str, pd.DataFrame] = {}
        logger.info(
            "DriftDetector initialised. Baseline dir: '%s' | "
            "KS α=%.3f | PSI threshold=%.3f",
            self.baseline_dir, self.ks_alpha, self.psi_threshold,
        )

    # ------------------------------------------------------------------
    # Baseline loading
    # ------------------------------------------------------------------

    def _load_baseline(self, season: str) -> pd.DataFrame:
        """
        Load and cache the reference baseline for a given season.

        Args:
            season: Season identifier (e.g., ``Winter``).

        Returns:
            Reference baseline DataFrame.

        Raises:
            FileNotFoundError: If baseline Parquet file is missing.
        """
        if season in self._baseline_cache:
            return self._baseline_cache[season]

        sanitised = season.strip().lower().replace(" ", "_")
        parquet_path = self.baseline_dir / f"{sanitised}_baseline.parquet"

        if not parquet_path.exists():
            raise FileNotFoundError(
                f"Baseline Parquet not found: '{parquet_path}'. "
                f"Run the preprocessing pipeline first to generate baselines."
            )

        baseline = pd.read_parquet(parquet_path)
        self._baseline_cache[season] = baseline
        logger.info(
            "Loaded '%s' baseline: shape=%s from '%s'.",
            season, baseline.shape, parquet_path,
        )
        return baseline

    def _get_numeric_features(self, df: pd.DataFrame) -> List[str]:
        """Return list of numeric columns to test (respecting whitelist)."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if self.features_to_monitor:
            return [f for f in self.features_to_monitor if f in numeric_cols]
        return numeric_cols

    # ------------------------------------------------------------------
    # Core detection methods
    # ------------------------------------------------------------------

    def detect_feature_drift(
        self,
        season: str,
        production_batch: pd.DataFrame,
    ) -> List[FeatureDriftResult]:
        """
        Run KS + PSI tests for every monitored feature in the batch.

        Args:
            season:            Season identifier for baseline lookup.
            production_batch:  Incoming inference feature DataFrame.

        Returns:
            List of ``FeatureDriftResult`` objects (one per tested feature).
        """
        baseline = self._load_baseline(season)
        features = self._get_numeric_features(baseline)

        # Only test features present in both baseline and batch
        common_features = [f for f in features if f in production_batch.columns]
        missing = set(features) - set(common_features)
        if missing:
            logger.warning(
                "Features in baseline but missing from batch: %s", missing
            )

        results: List[FeatureDriftResult] = []

        for feature in common_features:
            ref_vals  = baseline[feature].to_numpy(dtype=float)
            prod_vals = production_batch[feature].to_numpy(dtype=float)

            ks_result  = _run_ks_test(ref_vals, prod_vals, feature, self.ks_alpha)
            psi_result = _run_psi_test(
                ref_vals, prod_vals, feature, self.psi_threshold, self.n_psi_bins
            )

            results.append(FeatureDriftResult(
                feature=feature,
                ks=ks_result,
                psi=psi_result,
            ))

        # Log summary row per feature
        for r in results:
            log_fn = logger.warning if r.is_drifted else logger.debug
            log_fn(
                "  Feature %-22s | KS=%.4f (p=%.4f, %s) | PSI=%.4f (%s)",
                r.feature,
                r.ks.statistic, r.ks.p_value,
                "DRIFT" if r.ks.drifted else "OK",
                r.psi.psi,
                "DRIFT" if r.psi.drifted else "OK",
            )

        return results

    def run(
        self,
        season: str,
        production_batch: pd.DataFrame,
        batch_metadata: Optional[Dict] = None,
    ) -> DriftReport:
        """
        Execute the complete drift detection pipeline for a batch.

        Steps:
          1. Load seasonal reference baseline.
          2. Run KS + PSI tests per feature.
          3. Aggregate results into a ``DriftReport``.
          4. Determine overall severity and alert flag.

        Args:
            season:            Season identifier for model + baseline routing.
            production_batch:  Feature DataFrame from the inference API.
            batch_metadata:    Optional dict (batch_id, timestamp, source, etc.).

        Returns:
            ``DriftReport`` with per-feature results and overall assessment.
        """
        logger.info(
            "Running drift detection: season='%s', batch_size=%d",
            season, len(production_batch),
        )

        feature_results = self.detect_feature_drift(season, production_batch)

        drifted = [r.feature for r in feature_results if r.is_drifted]
        n_tested = len(feature_results)
        drift_fraction = len(drifted) / n_tested if n_tested > 0 else 0.0

        # Determine overall severity (worst case across all features)
        severities = [r.severity for r in feature_results]
        if DriftSeverity.CRITICAL in severities:
            overall_severity = DriftSeverity.CRITICAL
        elif DriftSeverity.WARNING in severities:
            overall_severity = DriftSeverity.WARNING
        else:
            overall_severity = DriftSeverity.NONE

        # Alert fires if drift fraction exceeds threshold OR any CRITICAL feature
        alert_triggered = (
            drift_fraction >= self.drift_fraction_alert
            or overall_severity == DriftSeverity.CRITICAL
        )

        report = DriftReport(
            season=season,
            batch_size=len(production_batch),
            n_features_tested=n_tested,
            feature_results=feature_results,
            drifted_features=drifted,
            overall_severity=overall_severity,
            alert_triggered=alert_triggered,
            drift_fraction=drift_fraction,
            metadata=batch_metadata or {},
        )

        self._log_report_summary(report)
        return report

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _log_report_summary(report: DriftReport) -> None:
        """Log a concise summary of the drift report."""
        severity_emoji = {
            DriftSeverity.NONE: "🟢",
            DriftSeverity.WARNING: "🟡",
            DriftSeverity.CRITICAL: "🔴",
        }
        emoji = severity_emoji.get(report.overall_severity, "⚪")
        log_fn = logger.warning if report.alert_triggered else logger.info

        log_fn(
            "%s DRIFT REPORT [%s] | Batch=%d | Features tested=%d | "
            "Drifted=%d (%.1f%%) | Severity=%s | Alert=%s",
            emoji,
            report.season,
            report.batch_size,
            report.n_features_tested,
            len(report.drifted_features),
            report.drift_fraction * 100,
            report.overall_severity.name,
            "TRIGGERED 🚨" if report.alert_triggered else "None",
        )

        if report.drifted_features:
            logger.warning("  Drifted features: %s", report.drifted_features)

    def get_psi_heatmap_data(
        self,
        season: str,
        production_batch: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute PSI scores for all features and return as a tidy DataFrame
        suitable for dashboard heatmap rendering.

        Args:
            season:            Season identifier.
            production_batch:  Incoming production feature batch.

        Returns:
            DataFrame with columns: feature, psi, severity_label, psi_bin.
        """
        feature_results = self.detect_feature_drift(season, production_batch)
        rows = []
        for r in feature_results:
            if r.psi.psi < self.psi_warning:
                severity_label = "No Drift"
                psi_bin = 0
            elif r.psi.psi < self.psi_threshold:
                severity_label = "Warning"
                psi_bin = 1
            else:
                severity_label = "Drift"
                psi_bin = 2
            rows.append({
                "feature": r.feature,
                "psi": round(r.psi.psi, 4),
                "ks_statistic": round(r.ks.statistic, 4),
                "ks_p_value": round(r.ks.p_value, 4),
                "severity_label": severity_label,
                "psi_bin": psi_bin,
                "season": season,
            })
        return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
