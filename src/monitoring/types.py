"""
AaroSense - Drift Detection Result Types.

Typed dataclasses representing per-feature and aggregate drift outcomes.
Kept in a separate module so they can be imported by both the detector
and downstream consumers (API, dashboard) without circular imports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Severity levels
# ---------------------------------------------------------------------------

class DriftSeverity(Enum):
    """Categorical severity of detected drift."""
    NONE     = auto()   # No drift detected
    WARNING  = auto()   # Marginal drift — monitor closely
    CRITICAL = auto()   # Significant drift — trigger retraining


# ---------------------------------------------------------------------------
# Per-feature result containers
# ---------------------------------------------------------------------------

@dataclass
class KSTestResult:
    """
    Result of a Kolmogorov-Smirnov two-sample test for a single feature.

    Attributes:
        feature:     Feature column name.
        statistic:   KS test statistic D ∈ [0, 1].
        p_value:     Two-tailed p-value.
        drifted:     True if p_value < alpha threshold.
        alpha:       Significance level used (default 0.05).
    """
    feature: str
    statistic: float
    p_value: float
    drifted: bool
    alpha: float = 0.05


@dataclass
class PSIResult:
    """
    Result of a Population Stability Index calculation for a single feature.

    PSI interpretation:
        PSI < 0.10  → No meaningful change
        PSI 0.10–0.25 → Moderate change — monitor
        PSI > 0.25  → Significant change — action required

    Attributes:
        feature:       Feature column name.
        psi:           Computed PSI score.
        drifted:       True if psi > psi_threshold.
        psi_threshold: Threshold used (default 0.25).
        n_bins:        Number of bins used for discretization.
    """
    feature: str
    psi: float
    drifted: bool
    psi_threshold: float = 0.25
    n_bins: int = 10


# ---------------------------------------------------------------------------
# Aggregate feature drift result
# ---------------------------------------------------------------------------

@dataclass
class FeatureDriftResult:
    """Combined drift assessment for a single feature (KS + PSI)."""
    feature: str
    ks: KSTestResult
    psi: PSIResult

    @property
    def is_drifted(self) -> bool:
        """True if either KS or PSI test flags drift."""
        return self.ks.drifted or self.psi.drifted

    @property
    def severity(self) -> DriftSeverity:
        """
        Severity based on combined KS + PSI signals:
          CRITICAL → both KS and PSI flag drift
          WARNING  → only one test flags drift
          NONE     → neither flags drift
        """
        both = self.ks.drifted and self.psi.drifted
        either = self.ks.drifted or self.psi.drifted
        if both:
            return DriftSeverity.CRITICAL
        if either:
            return DriftSeverity.WARNING
        return DriftSeverity.NONE

    def to_dict(self) -> Dict:
        return {
            "feature": self.feature,
            "ks_statistic": round(self.ks.statistic, 6),
            "ks_p_value": round(self.ks.p_value, 6),
            "ks_drifted": self.ks.drifted,
            "psi": round(self.psi.psi, 6),
            "psi_drifted": self.psi.drifted,
            "is_drifted": self.is_drifted,
            "severity": self.severity.name,
        }


# ---------------------------------------------------------------------------
# Full drift report for an inference batch
# ---------------------------------------------------------------------------

@dataclass
class DriftReport:
    """
    Aggregate drift monitoring report for a full inference batch.

    Attributes:
        season:             Season the report applies to.
        batch_size:         Number of rows in the incoming batch.
        n_features_tested:  Number of features included in the analysis.
        feature_results:    Per-feature drift results.
        drifted_features:   Names of features flagged as drifted.
        overall_severity:   Worst-case severity across all features.
        alert_triggered:    True if retraining workflow should fire.
        drift_fraction:     Fraction of features showing drift.
        metadata:           Optional dict for batch ID, timestamp, etc.
    """
    season: str
    batch_size: int
    n_features_tested: int
    feature_results: List[FeatureDriftResult] = field(default_factory=list)
    drifted_features: List[str] = field(default_factory=list)
    overall_severity: DriftSeverity = DriftSeverity.NONE
    alert_triggered: bool = False
    drift_fraction: float = 0.0
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "season": self.season,
            "batch_size": self.batch_size,
            "n_features_tested": self.n_features_tested,
            "drifted_features": self.drifted_features,
            "drift_fraction": round(self.drift_fraction, 4),
            "overall_severity": self.overall_severity.name,
            "alert_triggered": self.alert_triggered,
            "metadata": self.metadata,
            "feature_details": [r.to_dict() for r in self.feature_results],
        }
