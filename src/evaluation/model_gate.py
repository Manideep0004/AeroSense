"""
AaroSense - Automated Model Evaluation Gate (Milestone 2).

Enforces production-readiness standards before a model can be promoted
to the MLflow Model Registry.

Gate Criteria (ALL must pass):
  1. Test RMSE < ``rmse_threshold`` (µg/m³) — absolute performance bound.
  2. Test RMSE < Persistence Baseline RMSE — model must beat naïve forecast.
  3. Test R² > ``r2_min`` — explains at least ``r2_min`` of variance.
  4. Test MAE < ``mae_threshold`` (µg/m³) — practical utility bound.

If any criterion fails, the model is marked as REJECTED and is NOT
registered to the MLflow Model Registry.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto

logger = logging.getLogger("AaroSense.ModelGate")


# ---------------------------------------------------------------------------
# Gate verdict enum
# ---------------------------------------------------------------------------


class GateVerdict(Enum):
    """Possible outcomes of the model evaluation gate."""

    PASSED = auto()  # All criteria satisfied — promote to Registry
    FAILED = auto()  # One or more criteria failed — reject
    SKIPPED = auto()  # Gate was not run (e.g., missing metrics)


# ---------------------------------------------------------------------------
# Per-criterion result
# ---------------------------------------------------------------------------


@dataclass
class CriterionResult:
    """Result of evaluating a single gate criterion."""

    name: str
    passed: bool
    actual_value: float
    threshold_value: float
    message: str


# ---------------------------------------------------------------------------
# Full gate report
# ---------------------------------------------------------------------------


@dataclass
class GateReport:
    """
    Complete evaluation gate report for one (season, model) pair.

    Attributes:
        season:     Season identifier.
        model_name: Algorithm identifier.
        verdict:    PASSED / FAILED / SKIPPED.
        criteria:   List of individual criterion results.
        summary:    Human-readable summary string.
    """

    season: str
    model_name: str
    verdict: GateVerdict
    criteria: list[CriterionResult] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> dict:
        """Serialise report to a JSON-compatible dictionary."""
        return {
            "season": self.season,
            "model": self.model_name,
            "verdict": self.verdict.name,
            "criteria": [
                {
                    "name": c.name,
                    "passed": c.passed,
                    "actual": round(c.actual_value, 4),
                    "threshold": round(c.threshold_value, 4),
                    "message": c.message,
                }
                for c in self.criteria
            ],
            "summary": self.summary,
        }


# ---------------------------------------------------------------------------
# Gate configuration
# ---------------------------------------------------------------------------


@dataclass
class GateConfig:
    """
    Configurable thresholds for the automated model evaluation gate.

    Attributes:
        rmse_threshold:  Maximum acceptable test RMSE (µg/m³). Default 50.
        mae_threshold:   Maximum acceptable test MAE (µg/m³). Default 35.
        r2_min:          Minimum acceptable test R². Default 0.70.
        beat_persistence: Whether the model must beat persistence baseline. Default True.
    """

    rmse_threshold: float = 50.0
    mae_threshold: float = 35.0
    r2_min: float = 0.70
    beat_persistence: bool = True


# ---------------------------------------------------------------------------
# Gate evaluator
# ---------------------------------------------------------------------------


class ModelEvaluationGate:
    """
    Evaluates a trained model's performance metrics against production
    acceptance criteria and returns a structured ``GateReport``.

    Args:
        config: ``GateConfig`` instance defining acceptance thresholds.
    """

    def __init__(self, config: GateConfig | None = None) -> None:
        self.config = config or GateConfig()

    def evaluate(
        self,
        season: str,
        model_name: str,
        test_metrics: dict[str, float],
        persistence_metrics: dict[str, float],
    ) -> GateReport:
        """
        Run all gate criteria and return a structured report.

        Args:
            season:              Season identifier (e.g., ``Winter``).
            model_name:          Algorithm identifier (e.g., ``lightgbm``).
            test_metrics:        Dict with keys ``rmse``, ``mae``, ``r2``, ``mape``.
            persistence_metrics: Persistence baseline metrics dict (same keys).

        Returns:
            ``GateReport`` with per-criterion results and overall verdict.
        """
        criteria: list[CriterionResult] = []

        # ── Criterion 1: Absolute RMSE bound ──────────────────────────────
        rmse = test_metrics.get("rmse", float("inf"))
        c1 = CriterionResult(
            name="RMSE < Threshold",
            passed=rmse < self.config.rmse_threshold,
            actual_value=rmse,
            threshold_value=self.config.rmse_threshold,
            message=(
                f"RMSE={rmse:.4f} < {self.config.rmse_threshold} ✅"
                if rmse < self.config.rmse_threshold
                else f"RMSE={rmse:.4f} ≥ {self.config.rmse_threshold} ❌"
            ),
        )
        criteria.append(c1)

        # ── Criterion 2: Beat persistence baseline ────────────────────────
        persistence_rmse = persistence_metrics.get("rmse", float("inf"))
        if self.config.beat_persistence:
            c2 = CriterionResult(
                name="Beats Persistence Baseline",
                passed=rmse < persistence_rmse,
                actual_value=rmse,
                threshold_value=persistence_rmse,
                message=(
                    f"RMSE={rmse:.4f} < Persistence={persistence_rmse:.4f} ✅ "
                    f"(improvement: {((persistence_rmse - rmse) / persistence_rmse * 100):.1f}%)"
                    if rmse < persistence_rmse
                    else f"RMSE={rmse:.4f} ≥ Persistence={persistence_rmse:.4f} ❌"
                ),
            )
            criteria.append(c2)

        # ── Criterion 3: Minimum R² ───────────────────────────────────────
        r2 = test_metrics.get("r2", -float("inf"))
        c3 = CriterionResult(
            name=f"R² > {self.config.r2_min}",
            passed=r2 > self.config.r2_min,
            actual_value=r2,
            threshold_value=self.config.r2_min,
            message=(
                f"R²={r2:.4f} > {self.config.r2_min} ✅"
                if r2 > self.config.r2_min
                else f"R²={r2:.4f} ≤ {self.config.r2_min} ❌"
            ),
        )
        criteria.append(c3)

        # ── Criterion 4: Absolute MAE bound ───────────────────────────────
        mae = test_metrics.get("mae", float("inf"))
        c4 = CriterionResult(
            name="MAE < Threshold",
            passed=mae < self.config.mae_threshold,
            actual_value=mae,
            threshold_value=self.config.mae_threshold,
            message=(
                f"MAE={mae:.4f} < {self.config.mae_threshold} ✅"
                if mae < self.config.mae_threshold
                else f"MAE={mae:.4f} ≥ {self.config.mae_threshold} ❌"
            ),
        )
        criteria.append(c4)

        # ── Final verdict ─────────────────────────────────────────────────
        all_passed = all(c.passed for c in criteria)
        verdict = GateVerdict.PASSED if all_passed else GateVerdict.FAILED

        failed_names = [c.name for c in criteria if not c.passed]
        summary = (
            f"[{season} | {model_name}] Gate {'PASSED ✅' if all_passed else 'FAILED ❌'}"
            + (f" — Failed criteria: {failed_names}" if not all_passed else "")
        )

        report = GateReport(
            season=season,
            model_name=model_name,
            verdict=verdict,
            criteria=criteria,
            summary=summary,
        )

        # Structured log output
        log_fn = logger.info if all_passed else logger.warning
        log_fn(summary)
        for c in criteria:
            log_level = logging.INFO if c.passed else logging.WARNING
            logger.log(log_level, "    [%s] %s", "✅" if c.passed else "❌", c.message)

        return report
