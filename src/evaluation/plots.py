"""
plots.py
--------
All publication-quality figures for the project.

Figure inventory:
  F1  Event study plot (L1/L2) with pre-trend shading
  F2  Callaway-Sant'Anna ATT(g,t) dynamic plot
  F3  Dose-response curve (L3)
  F4  Dose-response non-parametric bins (L3)
  F5  DML multi-outcome coefficient plot (L5a)
  F6  CATE choropleth map by zone (L5b)
  F7  CATE distribution + feature importance (L5b)
  F8  Policy tree text output (L5b)
  F9  RATE / TOC curve (L5b)
  F10 Robustness forest plot
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from loguru import logger

try:
    import geopandas as gpd
    GEO_AVAILABLE = True
except ImportError:
    GEO_AVAILABLE = False

plt.rcParams.update({
    "figure.dpi": 150,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

BLUE   = "#2563EB"
RED    = "#DC2626"
GRAY   = "#6B7280"
GREEN  = "#16A34A"
ORANGE = "#EA580C"
PURPLE = "#7C3AED"


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  ✓ Saved: {path.name}")


# ── F1: Event Study ───────────────────────────────────────────────────────────

def plot_event_study(
    es_df: pd.DataFrame,
    outcome_label: str = "log(Trip Count)",
    out_path: Optional[Path] = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(11, 5))

    pre  = es_df[es_df["rel_week"] < 0]
    post = es_df[es_df["rel_week"] >= 0]

    ax.fill_between(pre["rel_week"],  pre["ci_low"],  pre["ci_high"],
                    color=GRAY,  alpha=0.15)
    ax.fill_between(post["rel_week"], post["ci_low"], post["ci_high"],
                    color=BLUE,  alpha=0.15)

    ax.plot(pre["rel_week"],  pre["coef"],  "o--", color=GRAY,
            linewidth=1.5, markersize=4, label="Pre-period (parallel trend check)")
    ax.plot(post["rel_week"], post["coef"], "o-",  color=BLUE,
            linewidth=2, markersize=5,   label="Post-period treatment effect")

    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color=RED, linewidth=1.8, linestyle="--", alpha=0.8,
               label="Congestion pricing start (2025-01-05)")

    pre_trend_p = es_df.attrs.get("pre_trend_pvalue", None)
    if pre_trend_p is not None:
        ok = pre_trend_p > 0.05
        ax.annotate(
            f"{'✓ Pre-trends hold' if ok else '⚠ Pre-trend violation'}\n"
            f"(joint F-test p={pre_trend_p:.3f})",
            xy=(pre["rel_week"].min() + 1, es_df["ci_high"].max() * 0.88),
            fontsize=8, color=GREEN if ok else RED,
        )

    ax.set_xlabel("Weeks relative to congestion pricing (t = 0)", fontsize=11)
    ax.set_ylabel(f"Estimated effect on {outcome_label}", fontsize=11)
    ax.set_title("Dynamic Treatment Effects: NYC Congestion Pricing",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(4))

    if out_path:
        _save(fig, out_path)
    return fig


# ── F2: CS-DiD ────────────────────────────────────────────────────────────────

def plot_cs_did(
    att_df: pd.DataFrame,
    out_path: Optional[Path] = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(11, 5))
    pre  = att_df[att_df["is_pre"]]
    post = att_df[~att_df["is_pre"]]

    ax.fill_between(pre["rel_week"],  pre["ci_low"],  pre["ci_high"],
                    color=GRAY, alpha=0.2)
    ax.fill_between(post["rel_week"], post["ci_low"], post["ci_high"],
                    color=BLUE, alpha=0.2)
    ax.plot(pre["rel_week"],  pre["att"],  "s--", color=GRAY, markersize=4)
    ax.plot(post["rel_week"], post["att"], "s-",  color=BLUE, markersize=5)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color=RED, linewidth=1.8, linestyle="--", alpha=0.8)

    ax.set_xlabel("Weeks relative to shock", fontsize=11)
    ax.set_ylabel("ATT(g,t)", fontsize=11)
    ax.set_title("Callaway-Sant'Anna ATT(g,t): Dynamic Effects",
                 fontsize=13, fontweight="bold")

    if out_path:
        _save(fig, out_path)
    return fig


# ── F3: Dose-Response Curve ────────────────────────────────────────────────────

def plot_dose_response(
    dose_response: pd.DataFrame,
    beta_lin: float,
    outcome_label: str = "log(Trip Count)",
    out_path: Optional[Path] = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.fill_between(dose_response["dose"], dose_response["ci_low"],
                    dose_response["ci_high"], color=ORANGE, alpha=0.2, label="95% CI")
    ax.plot(dose_response["dose"], dose_response["effect"],
            color=ORANGE, linewidth=2.5, label="Dose-response curve")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("CBD Congestion Fee ($)", fontsize=11)
    ax.set_ylabel(f"Effect on {outcome_label}", fontsize=11)
    ax.set_title("Dose-Response: Congestion Fee vs Trip Volume",
                 fontsize=13, fontweight="bold")
    ax.annotate(
        f"β = {beta_lin:.4f} per $1",
        xy=(0.05, 0.12), xycoords="axes fraction",
        fontsize=10, color=ORANGE,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=ORANGE),
    )
    ax.legend(fontsize=9)
    if out_path:
        _save(fig, out_path)
    return fig


# ── F4: Dose-Response Bins ─────────────────────────────────────────────────────

def plot_dose_bins(
    bins_df: pd.DataFrame,
    out_path: Optional[Path] = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(bins_df["mean_dose"], bins_df["att"], width=0.3,
           color=ORANGE, alpha=0.7, label="ATT per dose bin")
    ax.errorbar(bins_df["mean_dose"], bins_df["att"],
                yerr=1.96 * bins_df["se"],
                fmt="none", color="black", capsize=5)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Mean CBD Fee ($) in dose bin", fontsize=11)
    ax.set_ylabel("ATT (DiD estimate)", fontsize=11)
    ax.set_title("Non-Parametric Dose-Response: ATT by Fee Quantile",
                 fontsize=13, fontweight="bold")
    if out_path:
        _save(fig, out_path)
    return fig


# ── F5: DML Multi-Outcome Coefficient Plot ────────────────────────────────────

def plot_dml_results(
    dml_df: pd.DataFrame,
    out_path: Optional[Path] = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, max(4, len(dml_df) * 0.9)))
    colors = [BLUE if p < 0.05 else GRAY for p in dml_df["p_value"]]
    y_pos  = np.arange(len(dml_df))

    ax.barh(y_pos, dml_df["theta"], color=colors, alpha=0.75, height=0.5)
    ax.errorbar(
        dml_df["theta"], y_pos,
        xerr=1.96 * dml_df["se"],
        fmt="none", color="black", capsize=4,
    )
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(dml_df["outcome"], fontsize=10)
    ax.set_xlabel("DML-ATE  (θ̂)", fontsize=11)
    ax.set_title("Double ML: Treatment Effects Across Outcomes",
                 fontsize=13, fontweight="bold")

    sig_patch = plt.Rectangle((0, 0), 1, 1, fc=BLUE, alpha=0.75)
    ns_patch  = plt.Rectangle((0, 0), 1, 1, fc=GRAY, alpha=0.75)
    ax.legend([sig_patch, ns_patch], ["p < 0.05", "p ≥ 0.05"], fontsize=9)

    if out_path:
        _save(fig, out_path)
    return fig


# ── F6: CATE Choropleth Map ────────────────────────────────────────────────────

def plot_cate_map(
    zone_cate: pd.DataFrame,
    shapefile_dir: str,
    out_path: Optional[Path] = None,
) -> Optional[plt.Figure]:
    if not GEO_AVAILABLE:
        logger.warning("geopandas not available, skipping CATE map")
        return None

    shp_path = next(Path(shapefile_dir).glob("*.shp"), None)
    if shp_path is None:
        logger.warning(f"No shapefile found in {shapefile_dir}")
        return None

    gdf = gpd.read_file(shp_path)
    gdf = gdf.rename(columns={"LocationID": "zone_id"})
    gdf["zone_id"] = gdf["zone_id"].astype(int)
    gdf = gdf.merge(zone_cate, on="zone_id", how="left")

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Left: CATE point estimate
    gdf.plot(
        column="cate", ax=axes[0],
        cmap="RdBu_r", legend=True,
        legend_kwds={"label": "CATE (τ̂)", "shrink": 0.6},
        missing_kwds={"color": "#E5E7EB"},
        vmin=-gdf["cate"].abs().quantile(0.95),
        vmax= gdf["cate"].abs().quantile(0.95),
    )
    axes[0].set_title("Conditional Treatment Effects by Zone\n(Causal Forest CATE)",
                      fontsize=12, fontweight="bold")
    axes[0].axis("off")

    # Right: Statistical significance (|t| > 1.96)
    gdf["t_stat"] = (gdf["cate"] / (gdf["cate_se"] + 1e-8)).abs()
    gdf["sig"] = gdf["t_stat"] > 1.96
    gdf.plot(
        column="sig", ax=axes[1],
        cmap="coolwarm", legend=True,
        legend_kwds={"label": "Significant (|t|>1.96)"},
        missing_kwds={"color": "#E5E7EB"},
    )
    axes[1].set_title("Statistical Significance of Zone-Level CATEs",
                      fontsize=12, fontweight="bold")
    axes[1].axis("off")

    fig.suptitle("NYC Zone-Level Heterogeneous Treatment Effects\nCongestion Pricing Impact",
                 fontsize=14, fontweight="bold", y=1.02)

    if out_path:
        _save(fig, out_path)
    return fig


# ── F7: CATE Distribution + Feature Importance ────────────────────────────────

def plot_cate_diagnostics(
    cate_series: pd.Series,
    feature_importance: pd.Series,
    out_path: Optional[Path] = None,
) -> plt.Figure:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # CATE distribution
    ax1.hist(cate_series, bins=50, color=BLUE, alpha=0.7, edgecolor="white")
    ax1.axvline(cate_series.mean(), color=RED, linewidth=2,
                linestyle="--", label=f"Mean CATE = {cate_series.mean():.4f}")
    ax1.axvline(0, color="black", linewidth=0.8)
    ax1.set_xlabel("CATE τ̂(x)", fontsize=11)
    ax1.set_ylabel("Count", fontsize=11)
    ax1.set_title("Distribution of CATEs", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=9)

    # Feature importance (top 15)
    top_fi = feature_importance.head(15)
    colors = [BLUE if v > 0 else RED for v in top_fi.values]
    ax2.barh(range(len(top_fi)), top_fi.values, color=colors, alpha=0.8)
    ax2.set_yticks(range(len(top_fi)))
    ax2.set_yticklabels(top_fi.index, fontsize=9)
    ax2.set_xlabel("Feature Importance / BLP Coefficient", fontsize=10)
    ax2.set_title("CATE Heterogeneity Drivers\n(Best Linear Projection)",
                  fontsize=12, fontweight="bold")

    if out_path:
        _save(fig, out_path)
    return fig


# ── F9: TOC / RATE Curve ──────────────────────────────────────────────────────

def plot_rate_curve(
    toc_df: pd.DataFrame,
    rate_score: float,
    out_path: Optional[Path] = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(toc_df["fraction"], toc_df["ate_top_q"],
            "o-", color=PURPLE, linewidth=2.5, markersize=6,
            label="TOC: ATE for top-q targeted zones")
    overall = toc_df["ate_top_q"].iloc[-1]
    ax.axhline(overall, color=GRAY, linewidth=1.5, linestyle="--",
               label=f"Uniform ATE = {overall:.4f}")
    ax.set_xlabel("Fraction of zones targeted (by CATE rank)", fontsize=11)
    ax.set_ylabel("Average Treatment Effect on targeted zones", fontsize=11)
    ax.set_title(f"RATE Targeting Curve\n(RATE score = {rate_score:.4f})",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    if out_path:
        _save(fig, out_path)
    return fig


# ── F10: Robustness Forest Plot ───────────────────────────────────────────────

def plot_robustness(
    rob_df: pd.DataFrame,
    main_estimate: float,
    main_se: float,
    out_path: Optional[Path] = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, max(5, len(rob_df) * 0.5 + 2)))

    y_pos = np.arange(len(rob_df))
    colors = [BLUE if p < 0.05 else GRAY for p in rob_df["p"]]

    ax.scatter(rob_df["theta"], y_pos, color=colors, s=60, zorder=3)
    ax.errorbar(
        rob_df["theta"], y_pos,
        xerr=1.96 * rob_df["se"],
        fmt="none", color="black", alpha=0.5, capsize=3,
    )

    # Main estimate band
    ax.axvspan(
        main_estimate - 1.96 * main_se,
        main_estimate + 1.96 * main_se,
        color=BLUE, alpha=0.07, label="Main estimate 95% CI",
    )
    ax.axvline(main_estimate, color=BLUE, linewidth=1.5,
               linestyle="--", label=f"Main estimate = {main_estimate:.4f}")
    ax.axvline(0, color="black", linewidth=0.8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(rob_df["check"], fontsize=8)
    ax.set_xlabel("Estimated ATT / θ̂", fontsize=11)
    ax.set_title("Robustness Checks: Sensitivity of Main Estimate",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)

    if out_path:
        _save(fig, out_path)
    return fig
