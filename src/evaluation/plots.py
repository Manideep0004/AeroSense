"""
AaroSense - Model Evaluation Plots (Milestone 2).

Generates publication-quality diagnostic figures for each seasonal model:

1. Residual Analysis Panel (3 sub-plots):
   - Predicted vs Actual scatter with 1:1 identity line
   - Residual distribution histogram with KDE
   - Residual vs Predicted (homoscedasticity check)

2. Model-native Feature Importance (mean gain / split count).

3. SHAP Summary Beeswarm Plot (global feature attribution).

4. SHAP Waterfall Plot (single-instance local explanation).

All figures are saved as PNG to ``report_dir/<season>/<model>/``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend — safe for headless CI/CD

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger("AaroSense.Plots")

# Consistent visual style across all figures
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
FIGURE_DPI: int = 150
ACCENT_COLOR: str = "#2563eb"  # Blue-600
WARN_COLOR: str = "#dc2626"  # Red-600
GRID_COLOR: str = "#e5e7eb"  # Gray-200


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_output_dir(report_dir: Path, season: str, model_name: str) -> Path:
    """Create and return ``report_dir/<season>/<model>/``."""
    out = report_dir / season.lower() / model_name
    out.mkdir(parents=True, exist_ok=True)
    return out


def _save_fig(fig: plt.Figure, path: Path) -> Path:
    """Save figure, close it, and return path."""
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved plot → %s", path)
    return path


# ---------------------------------------------------------------------------
# 1. Residual Analysis Panel
# ---------------------------------------------------------------------------


def plot_residual_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    season: str,
    model_name: str,
    report_dir: Path,
    metrics: dict[str, float] | None = None,
) -> Path:
    """
    Generate a 3-panel residual analysis figure.

    Panels:
      - Left:   Predicted vs Actual scatter (identity line + R² annotation)
      - Centre: Residual distribution histogram + KDE (zero-mean check)
      - Right:  Residuals vs Predicted (heteroscedasticity check)

    Args:
        y_true:      Ground-truth PM2.5 values.
        y_pred:      Model predictions.
        season:      Season label for plot title.
        model_name:  Model algorithm name.
        report_dir:  Base directory for saving plots.
        metrics:     Optional dict of pre-computed metrics for annotation.

    Returns:
        Path to the saved PNG file.
    """
    residuals = y_true - y_pred
    out_dir = _make_output_dir(report_dir, season, model_name)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        f"Residual Analysis — {season} | {model_name.upper()}",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )

    # ── Panel 1: Predicted vs Actual ──────────────────────────────────────
    ax1 = axes[0]
    ax1.scatter(y_true, y_pred, alpha=0.35, s=12, color=ACCENT_COLOR, rasterized=True)

    lims = [
        min(y_true.min(), y_pred.min()) * 0.95,
        max(y_true.max(), y_pred.max()) * 1.05,
    ]
    ax1.plot(
        lims, lims, "--", color=WARN_COLOR, linewidth=1.5, label="Perfect fit (1:1)"
    )
    ax1.set_xlim(lims)
    ax1.set_ylim(lims)
    ax1.set_xlabel("Actual PM2.5 (µg/m³)")
    ax1.set_ylabel("Predicted PM2.5 (µg/m³)")
    ax1.set_title("Predicted vs Actual")
    ax1.legend(fontsize=9)

    if metrics:
        annotation = (
            f"RMSE = {metrics.get('rmse', 0):.2f}\n"
            f"MAE  = {metrics.get('mae', 0):.2f}\n"
            f"R²   = {metrics.get('r2', 0):.4f}"
        )
        ax1.text(
            0.04,
            0.95,
            annotation,
            transform=ax1.transAxes,
            va="top",
            fontsize=9,
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8),
        )

    # ── Panel 2: Residual Distribution ────────────────────────────────────
    ax2 = axes[1]
    sns.histplot(residuals, bins=50, kde=True, ax=ax2, color=ACCENT_COLOR, alpha=0.65)
    ax2.axvline(
        0, color=WARN_COLOR, linestyle="--", linewidth=1.5, label="Zero residual"
    )
    ax2.axvline(
        residuals.mean(),
        color="#16a34a",
        linestyle=":",
        linewidth=1.5,
        label=f"Mean = {residuals.mean():.2f}",
    )
    ax2.set_xlabel("Residual (Actual − Predicted, µg/m³)")
    ax2.set_ylabel("Count")
    ax2.set_title("Residual Distribution")
    ax2.legend(fontsize=9)

    # ── Panel 3: Residuals vs Predicted ───────────────────────────────────
    ax3 = axes[2]
    ax3.scatter(
        y_pred, residuals, alpha=0.35, s=12, color=ACCENT_COLOR, rasterized=True
    )
    ax3.axhline(0, color=WARN_COLOR, linestyle="--", linewidth=1.5)
    ax3.axhline(
        residuals.mean() + 2 * residuals.std(),
        color="#f59e0b",
        linestyle=":",
        linewidth=1.0,
        alpha=0.7,
        label="±2σ band",
    )
    ax3.axhline(
        residuals.mean() - 2 * residuals.std(),
        color="#f59e0b",
        linestyle=":",
        linewidth=1.0,
        alpha=0.7,
    )
    ax3.set_xlabel("Predicted PM2.5 (µg/m³)")
    ax3.set_ylabel("Residual (µg/m³)")
    ax3.set_title("Residuals vs Predicted")
    ax3.legend(fontsize=9)

    fig.tight_layout()
    out_path = out_dir / "residual_analysis.png"
    return _save_fig(fig, out_path)


# ---------------------------------------------------------------------------
# 2. Feature Importance Plot (model-native)
# ---------------------------------------------------------------------------


def plot_feature_importance(
    model: Any,
    feature_names: list[str],
    season: str,
    model_name: str,
    report_dir: Path,
    top_n: int = 20,
) -> Path:
    """
    Plot model-native feature importances (gain-based where available).

    Supports LightGBM, XGBoost, CatBoost, and sklearn RandomForest.
    Falls back to ``feature_importances_`` attribute for other estimators.

    Args:
        model:         Trained model instance.
        feature_names: Ordered list of feature column names.
        season:        Season label.
        model_name:    Algorithm identifier.
        report_dir:    Base output directory.
        top_n:         Number of top features to display.

    Returns:
        Path to the saved PNG file.
    """
    out_dir = _make_output_dir(report_dir, season, model_name)
    importances: np.ndarray | None = None

    try:
        algo = model_name.lower()
        if algo == "lightgbm":
            importances = model.booster_.feature_importance(importance_type="gain")
        elif algo == "xgboost":
            raw = model.get_booster().get_score(importance_type="gain")
            # XGBoost returns a dict keyed by 'f<index>'
            importances = np.array(
                [raw.get(f"f{i}", 0.0) for i in range(len(feature_names))]
            )
        elif algo == "catboost":
            importances = np.array(model.get_feature_importance())
        else:
            # sklearn-style: feature_importances_ (mean decrease impurity)
            importances = model.feature_importances_
    except Exception as exc:
        logger.warning(
            "Could not extract native importance for '%s': %s", model_name, exc
        )
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_

    if importances is None or len(importances) == 0:
        logger.warning(
            "No feature importances available for '%s'. Skipping plot.", model_name
        )
        return out_dir / "feature_importance_unavailable.txt"

    # Sort and take top_n
    importance_series = pd.Series(importances, index=feature_names).sort_values(
        ascending=False
    )
    top_features = importance_series.head(top_n)

    fig, ax = plt.subplots(figsize=(10, max(5, top_n * 0.38)))
    colors = [ACCENT_COLOR] * len(top_features)
    bars = ax.barh(
        top_features.index[::-1], top_features.values[::-1], color=colors, alpha=0.85
    )

    # Add value labels on bars
    for bar, val in zip(bars, top_features.values[::-1]):
        ax.text(
            bar.get_width() * 1.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:,.1f}",
            va="center",
            ha="left",
            fontsize=8,
        )

    ax.set_xlabel("Feature Importance (Gain)")
    ax.set_title(
        f"Top {top_n} Feature Importances — {season} | {model_name.upper()}",
        fontweight="bold",
    )
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.grid(axis="x", color=GRID_COLOR)
    ax.set_axisbelow(True)

    fig.tight_layout()
    out_path = out_dir / "feature_importance.png"
    return _save_fig(fig, out_path)


# ---------------------------------------------------------------------------
# 3. SHAP Summary Plot
# ---------------------------------------------------------------------------


def plot_shap_summary(
    model: Any,
    X_sample: pd.DataFrame,
    season: str,
    model_name: str,
    report_dir: Path,
    max_display: int = 20,
    sample_size: int = 500,
) -> tuple[Path, Path]:
    """
    Generate SHAP beeswarm summary and bar importance plots.

    Uses ``shap.TreeExplainer`` for tree-based models (fast, exact).
    Samples up to ``sample_size`` rows for computational efficiency.

    Args:
        model:        Trained model instance.
        X_sample:     Feature DataFrame (typically the test set).
        season:       Season label.
        model_name:   Algorithm identifier.
        report_dir:   Base output directory.
        max_display:  Number of features to display in plots.
        sample_size:  Max rows to use for SHAP computation.

    Returns:
        Tuple of (beeswarm_path, bar_path) PNG file paths.
    """
    import shap

    out_dir = _make_output_dir(report_dir, season, model_name)

    # Sub-sample for speed
    if len(X_sample) > sample_size:
        X_shap = X_sample.sample(n=sample_size, random_state=42).reset_index(drop=True)
    else:
        X_shap = X_sample.copy()

    logger.info(
        "[%s | %s] Computing SHAP values for %d samples...",
        season,
        model_name,
        len(X_shap),
    )

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_shap)
    except Exception as exc:
        logger.warning(
            "TreeExplainer failed for '%s' (%s). Falling back to KernelExplainer.",
            model_name,
            exc,
        )
        # Slow fallback for non-tree models
        background = shap.sample(X_shap, 50)
        explainer = shap.KernelExplainer(model.predict, background)
        shap_values = explainer.shap_values(X_shap)

    # ── 3a. Beeswarm Summary Plot ─────────────────────────────────────────
    fig_bee, ax_bee = plt.subplots(figsize=(10, max(6, max_display * 0.42)))
    shap.summary_plot(
        shap_values,
        X_shap,
        max_display=max_display,
        show=False,
        plot_size=None,
    )
    ax_bee = plt.gca()
    ax_bee.set_title(
        f"SHAP Beeswarm — {season} | {model_name.upper()}",
        fontweight="bold",
        pad=12,
    )
    bee_path = out_dir / "shap_beeswarm.png"
    fig_bee = plt.gcf()
    fig_bee.tight_layout()
    _save_fig(fig_bee, bee_path)

    # ── 3b. SHAP Bar Importance Plot ──────────────────────────────────────
    fig_bar, ax_bar = plt.subplots(figsize=(10, max(5, max_display * 0.38)))
    shap.summary_plot(
        shap_values,
        X_shap,
        plot_type="bar",
        max_display=max_display,
        show=False,
        plot_size=None,
    )
    ax_bar = plt.gca()
    ax_bar.set_title(
        f"SHAP Feature Importance (|Mean SHAP|) — {season} | {model_name.upper()}",
        fontweight="bold",
        pad=12,
    )
    bar_path = out_dir / "shap_bar.png"
    fig_bar = plt.gcf()
    fig_bar.tight_layout()
    _save_fig(fig_bar, bar_path)

    logger.info("[%s | %s] SHAP plots saved.", season, model_name)
    return bee_path, bar_path


# ---------------------------------------------------------------------------
# 4. SHAP Waterfall (single-instance local explanation)
# ---------------------------------------------------------------------------


def plot_shap_waterfall(
    model: Any,
    X_sample: pd.DataFrame,
    season: str,
    model_name: str,
    report_dir: Path,
    instance_idx: int = 0,
) -> Path:
    """
    Generate a SHAP waterfall plot for a single prediction instance.

    Waterfall plots explain exactly how each feature contributed to pushing
    the model output from the expected base value to the final prediction.

    Args:
        model:        Trained model instance.
        X_sample:     Feature DataFrame.
        season:       Season label.
        model_name:   Algorithm identifier.
        report_dir:   Base output directory.
        instance_idx: Row index in X_sample to explain.

    Returns:
        Path to the saved PNG file.
    """
    import shap

    out_dir = _make_output_dir(report_dir, season, model_name)

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_sample.iloc[[instance_idx]])
    except Exception as exc:
        logger.warning("Waterfall plot failed for '%s': %s. Skipping.", model_name, exc)
        return out_dir / "shap_waterfall_unavailable.txt"

    fig, ax = plt.subplots(figsize=(12, 7))
    shap.waterfall_plot(shap_values[0], max_display=15, show=False)
    ax = plt.gca()
    ax.set_title(
        f"SHAP Waterfall (Instance #{instance_idx}) — {season} | {model_name.upper()}",
        fontweight="bold",
        pad=12,
    )
    out_path = out_dir / "shap_waterfall.png"
    fig = plt.gcf()
    fig.tight_layout()
    return _save_fig(fig, out_path)
