"""
AaroSense - Milestone 2 Smoke Test.

Validates the full Milestone 2 pipeline:
  - Residual analysis plots
  - Feature importance plots
  - SHAP summary + waterfall plots
  - Evaluation gate
  - MLflow artifact logging
  - Registry promotion (Staging)

Uses the Winter/LightGBM result from the M1 smoke test JSON.
Run from project root:
    python3 scripts/smoke_test_m2.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.evaluator import ModelEvaluator
from src.evaluation.model_gate import GateConfig, GateVerdict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    stream=sys.stdout,
)

SUMMARY_PATH = Path("models/experiment_summary.json")
DATA_PATH = Path("data/raw/pm25_dataset.csv")
REPORT_DIR = Path("reports/plots")


def main() -> None:
    if not SUMMARY_PATH.exists():
        print(
            f"ERROR: '{SUMMARY_PATH}' not found. "
            "Run the M1 smoke test first: python3 scripts/smoke_test_m1.py",
            file=sys.stderr,
        )
        sys.exit(1)

    evaluator = ModelEvaluator(
        report_dir=REPORT_DIR,
        gate_config=GateConfig(
            rmse_threshold=50.0,
            mae_threshold=35.0,
            r2_min=0.70,
        ),
        tracking_uri="sqlite:///mlruns/mlflow.db",
        enable_shap=True,
        shap_sample_size=200,  # Fast for smoke test
    )

    reports = evaluator.evaluate_from_summary(
        summary_path=SUMMARY_PATH,
        data_path=DATA_PATH,
    )

    print("\n=== MILESTONE 2 SMOKE TEST RESULTS ===")
    all_ok = True
    for season, season_reports in reports.items():
        for model_name, report in season_reports.items():
            status = (
                "✅ PASSED" if report.verdict == GateVerdict.PASSED else "❌ FAILED"
            )
            print(f"  [{season} | {model_name}] Gate: {status}")
            for c in report.criteria:
                mark = "✅" if c.passed else "❌"
                print(f"    {mark} {c.message}")
            if report.verdict == GateVerdict.FAILED:
                all_ok = False

    # Verify plots were generated
    plot_files = list(REPORT_DIR.rglob("*.png"))
    print(f"\n  Generated {len(plot_files)} plot(s):")
    for p in plot_files:
        print(f"    → {p.relative_to(REPORT_DIR)}")

    print(
        "\n" + ("=== ALL CHECKS PASSED ===" if all_ok else "=== SOME CHECKS FAILED ===")
    )
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
