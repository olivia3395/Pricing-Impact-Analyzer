"""
tables.py
---------
Generate publication-ready results tables.

Tables:
  T1  Main DiD results (L1: Naive, L2: CS, L4: SDID) — side-by-side
  T2  DML multi-outcome results (L5a)
  T3  CATE summary by borough and time-of-day (L5b)
  T4  Robustness checks summary
  T5  Pre-trend test summary
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger


def _stars(p: float) -> str:
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return ""


def _fmt(x, digits=4):
    if pd.isna(x):
        return "—"
    return f"{x:.{digits}f}"


# ── T1: Main DiD comparison table ─────────────────────────────────────────────

def make_main_did_table(
    l1_results: dict,
    cs_results: dict,
    sdid_results: dict,
    outcome: str = "log_trip_count",
    out_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Three-column table: L1 Naive DiD | L2 CS-DiD | L4 SDID
    Rows: ATT, SE, CI, N, pre-trend p
    """
    def _extract(r, name):
        if name == "cs":
            return {
                "ATT":           r.get("agg_att", np.nan),
                "SE":            np.nan,
                "CI Low":        np.nan,
                "CI High":       np.nan,
                "N Obs":         np.nan,
                "Pre-trend p":   r.get("pre_trend_pvalue", np.nan),
            }
        theta = r.get("tau_hat", r.get("ate", np.nan))
        se    = r.get("se", np.nan)
        return {
            "ATT":          theta,
            "SE":           se,
            "CI Low":       r.get("ci_low", theta - 1.96 * se if not np.isnan(se) else np.nan),
            "CI High":      r.get("ci_high", theta + 1.96 * se if not np.isnan(se) else np.nan),
            "N Obs":        r.get("n_obs", np.nan),
            "Pre-trend p":  np.nan,
        }

    rows = {
        "L1: Naive DiD (TWFE)":        _extract(l1_results, "l1"),
        "L2: Callaway-Sant'Anna":       _extract(cs_results,   "cs"),
        "L4: Synthetic DiD":            _extract(sdid_results, "sdid"),
    }

    df = pd.DataFrame(rows).T
    df.index.name = "Estimator"

    # Format significance stars on ATT
    for est in df.index:
        p_val = l1_results.get("p_value") if "Naive" in est else np.nan
        if not np.isnan(df.loc[est, "ATT"]) and not np.isnan(p_val):
            df.loc[est, "ATT_display"] = f"{df.loc[est,'ATT']:.4f}{_stars(p_val)}"
        else:
            df.loc[est, "ATT_display"] = _fmt(df.loc[est, "ATT"])

    logger.info("T1: Main DiD comparison table")
    logger.info(df[["ATT", "SE", "CI Low", "CI High"]].to_string())

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path.with_suffix(".csv"))
        _to_latex(df, out_path.with_suffix(".tex"),
                  caption="Main DiD Results: Effect of Congestion Pricing",
                  label="tab:main_did")
        logger.info(f"  ✓ Tables saved: {out_path.stem}.*")

    return df


# ── T2: DML multi-outcome table ────────────────────────────────────────────────

def make_dml_table(
    dml_df: pd.DataFrame,
    out_path: Optional[Path] = None,
) -> pd.DataFrame:
    display = dml_df[["outcome", "theta", "se", "p_value", "ci_low", "ci_high", "n_obs"]].copy()
    display["sig"]   = display["p_value"].apply(_stars)
    display["theta_fmt"] = display.apply(
        lambda r: f"{r['theta']:.4f}{r['sig']}", axis=1)
    display["se_fmt"]    = display["se"].apply(lambda x: f"({x:.4f})")
    display["ci_fmt"]    = display.apply(
        lambda r: f"[{r['ci_low']:.4f}, {r['ci_high']:.4f}]", axis=1)

    display = display.rename(columns={
        "outcome": "Outcome",
        "theta_fmt": "θ̂ (DML-ATE)",
        "se_fmt": "SE",
        "ci_fmt": "95% CI",
        "n_obs": "N",
    })

    logger.info("T2: DML results table")
    logger.info(display[["Outcome", "θ̂ (DML-ATE)", "SE", "95% CI"]].to_string(index=False))

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        display.to_csv(out_path.with_suffix(".csv"), index=False)
        _to_latex(
            display[["Outcome", "θ̂ (DML-ATE)", "SE", "95% CI", "N"]],
            out_path.with_suffix(".tex"),
            caption="Double ML Treatment Effect Estimates Across Outcomes",
            label="tab:dml",
        )

    return display


# ── T3: CATE by group ──────────────────────────────────────────────────────────

def make_cate_group_table(
    cate_by_borough: pd.DataFrame,
    cate_by_peak: Optional[pd.DataFrame] = None,
    out_path: Optional[Path] = None,
) -> pd.DataFrame:
    frames = [cate_by_borough.assign(dimension="Borough")]
    if cate_by_peak is not None:
        frames.append(cate_by_peak.assign(dimension="Time of Day"))

    display = pd.concat(frames, ignore_index=True)
    display["sig"] = display.apply(
        lambda r: _stars(
            2 * (1 - __import__("scipy").stats.norm.cdf(
                abs(r["mean_cate"]) / (r["se_cate"] + 1e-8)))
        ),
        axis=1,
    )
    display["cate_fmt"] = display.apply(
        lambda r: f"{r['mean_cate']:.4f}{r['sig']}", axis=1)

    logger.info("T3: CATE by group")
    logger.info(display.to_string(index=False))

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        display.to_csv(out_path.with_suffix(".csv"), index=False)

    return display


# ── T4: Robustness summary ────────────────────────────────────────────────────

def make_robustness_table(
    rob_df: pd.DataFrame,
    main_estimate: float,
    out_path: Optional[Path] = None,
) -> pd.DataFrame:
    rob_df = rob_df.copy()
    rob_df["sig"]   = rob_df["p"].apply(_stars)
    rob_df["theta_fmt"] = rob_df.apply(
        lambda r: f"{r['theta']:.4f}{r['sig']}", axis=1)
    rob_df["se_fmt"]    = rob_df["se"].apply(lambda x: f"({x:.4f})" if not pd.isna(x) else "—")
    rob_df["vs_main"]   = rob_df["theta"] - main_estimate

    logger.info("T4: Robustness table")
    logger.info(rob_df[["group", "check", "theta_fmt", "se_fmt"]].to_string(index=False))

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        rob_df.to_csv(out_path.with_suffix(".csv"), index=False)
        _to_latex(
            rob_df[["group", "check", "theta_fmt", "se_fmt"]],
            out_path.with_suffix(".tex"),
            caption="Robustness Checks",
            label="tab:robustness",
        )

    return rob_df


# ── LaTeX helper ──────────────────────────────────────────────────────────────

def _to_latex(df: pd.DataFrame, path: Path, caption: str = "", label: str = ""):
    latex = df.to_latex(
        index=True,
        escape=False,
        float_format="%.4f",
        caption=caption,
        label=label,
        na_rep="—",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(latex)


# ── Master summary print ──────────────────────────────────────────────────────

def print_executive_summary(
    l1: dict,
    dml_df: pd.DataFrame,
    cf_results: dict,
):
    print("\n" + "=" * 60)
    print("  CONGESTION PRICING IMPACT — EXECUTIVE SUMMARY")
    print("=" * 60)
    print(f"\n  Policy shock: 2025-01-05 (MTA Congestion Relief Zone)")
    print(f"\n  ── L1 Naive DiD (primary outcome: log trip count) ──")
    print(f"     ATT       = {l1['ate']:+.4f}  (≈ {(np.exp(l1['ate'])-1)*100:+.1f}% change)")
    print(f"     SE        = {l1['se']:.4f}")
    print(f"     p-value   = {l1['p_value']:.4f}  {_stars(l1['p_value'])}")

    print(f"\n  ── L5a DML (multiple outcomes) ──")
    for _, row in dml_df.iterrows():
        print(f"     {row['outcome']:20s}  θ = {row['theta']:+.4f}  "
              f"p = {row['p_value']:.4f}  {_stars(row['p_value'])}")

    print(f"\n  ── L5b Causal Forest ──")
    print(f"     Mean CATE = {cf_results['mean_cate']:+.4f}")
    print(f"     Std CATE  = {cf_results['std_cate']:.4f}")
    print(f"     Top driver: {cf_results['feature_importance'].index[0]}")
    print("=" * 60 + "\n")
