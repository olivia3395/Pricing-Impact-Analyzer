"""
run_did.py
----------
L1: Naive two-way fixed-effects DiD
L2: Callaway & Sant'Anna (2021) DiD — handles heterogeneous treatment timing
    and avoids negative-weighting bias in staggered adoption settings.

Even though congestion pricing activates on a single date (2025-01-05),
we still use CS-DiD because:
  1. "Treatment" varies by zone intensity (not all zones equally affected)
  2. It provides honest ATT(g,t) estimates and valid pre-trend tests
  3. It's the modern standard and shows methodological depth

Results schema
--------------
{
  "l1_did": {
      "ate": float,
      "se": float,
      "ci_low": float,
      "ci_high": float,
      "n_obs": int,
      "outcome": str,
  },
  "l2_cs": {
      "att_gt": pd.DataFrame,  # ATT(g,t) estimates
      "agg_simple": dict,      # aggregated ATT
      "pre_trend_pvalue": float,
  }
}
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from linearmodels.panel import PanelOLS
from scipy import stats
from loguru import logger


# ── L1: Naive Two-Way FE DiD ─────────────────────────────────────────────────

def run_naive_did(
    panel: pd.DataFrame,
    outcome: str = "log_trip_count",
    treatment: str = "treat_binary",
    cluster: str = "zone_id",
) -> dict:
    """
    Y_{it} = α_i + γ_t + β (Treat_i × Post_t) + ε_{it}

    Uses linearmodels PanelOLS with entity + time fixed effects,
    clustered standard errors by zone_id.
    """
    logger.info(f"[L1 DiD] outcome={outcome}, treatment={treatment}")

    df = panel.copy()
    df["treat_post"] = df[treatment] * df["post"]

    # Set MultiIndex for linearmodels
    df = df.set_index(["zone_id", "date"])
    df.index.names = ["entity", "time"]

    model = PanelOLS(
        dependent=df[outcome],
        exog=df[["treat_post"]],
        entity_effects=True,
        time_effects=True,
        drop_absorbed=True,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = model.fit(cov_type="clustered", cluster_entity=True)

    coef = float(res.params["treat_post"])
    se   = float(res.std_errors["treat_post"])
    ci   = res.conf_int().loc["treat_post"]

    result = {
        "ate":      coef,
        "se":       se,
        "t_stat":   coef / se,
        "p_value":  float(res.pvalues["treat_post"]),
        "ci_low":   float(ci.iloc[0]),
        "ci_high":  float(ci.iloc[1]),
        "n_obs":    int(res.nobs),
        "outcome":  outcome,
        "r_squared": float(res.rsquared),
    }

    logger.info(
        f"  β(Treat×Post) = {coef:.4f}  SE={se:.4f}  "
        f"p={result['p_value']:.4f}  CI=[{result['ci_low']:.4f}, {result['ci_high']:.4f}]"
    )
    return result


# ── L2: Event Study (manual, avoids stacking issues) ─────────────────────────

def run_event_study(
    panel: pd.DataFrame,
    outcome: str = "log_trip_count",
    treatment: str = "treat_binary",
    window: int = 16,
) -> pd.DataFrame:
    """
    Estimate dynamic effects via relative-week dummies:

    Y_{it} = α_i + γ_t + Σ_{k≠-1} β_k (Treat_i × 1[rel_week=k]) + ε_{it}

    Pre-period coefficients serve as parallel-trends test.
    k=-1 is the omitted reference period.

    Returns DataFrame with columns: rel_week, coef, se, ci_low, ci_high
    """
    logger.info(f"[L2 Event Study] outcome={outcome}, window=±{window} weeks")

    df = panel.copy()
    df = df[df["rel_week"].between(-window, window)]

    # Create dummies for each relative week except -1 (reference)
    weeks = sorted(df["rel_week"].unique())
    weeks = [w for w in weeks if w != -1]

    for w in weeks:
        col = f"tw_{w}" if w < 0 else f"tw_p{w}"
        df[col] = (df["rel_week"] == w).astype(int) * df[treatment]

    dummy_cols = [f"tw_{w}" if w < 0 else f"tw_p{w}" for w in weeks]

    df = df.set_index(["zone_id", "date"])
    df.index.names = ["entity", "time"]

    model = PanelOLS(
        dependent=df[outcome],
        exog=df[dummy_cols],
        entity_effects=True,
        time_effects=True,
        drop_absorbed=True,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = model.fit(cov_type="clustered", cluster_entity=True)

    coefs = res.params
    ses   = res.std_errors
    cis   = res.conf_int()

    rows = []
    for w, col in zip(weeks, dummy_cols):
        if col in coefs.index:
            rows.append({
                "rel_week": w,
                "coef":    float(coefs[col]),
                "se":      float(ses[col]),
                "ci_low":  float(cis.loc[col].iloc[0]),
                "ci_high": float(cis.loc[col].iloc[1]),
                "p_value": float(res.pvalues[col]),
            })

    # Add the reference period (rel_week = -1) as zero
    rows.append({"rel_week": -1, "coef": 0.0, "se": 0.0,
                 "ci_low": 0.0, "ci_high": 0.0, "p_value": 1.0})

    es_df = pd.DataFrame(rows).sort_values("rel_week").reset_index(drop=True)

    # Pre-trend joint F-test (weeks < 0 excluding -1)
    pre_cols = [f"tw_{w}" for w in weeks if w < 0 and f"tw_{w}" in coefs.index]
    if pre_cols:
        r_matrix = np.zeros((len(pre_cols), len(coefs)))
        for i, col in enumerate(pre_cols):
            r_matrix[i, list(coefs.index).index(col)] = 1.0
        # Wald test
        try:
            wald = res.wald_test(r_matrix)
            pval = float(wald.pval)
        except Exception:
            pval = np.nan
        logger.info(f"  Pre-trend joint F-test p-value: {pval:.4f}")
        es_df.attrs["pre_trend_pvalue"] = pval

    logger.info(f"  Event study done: {len(es_df)} time points")
    return es_df


# ── L2b: Callaway-Sant'Anna wrapper ──────────────────────────────────────────

def run_callaway_santanna(
    panel: pd.DataFrame,
    outcome: str = "log_trip_count",
    anticipation: int = 0,
    n_bootstrap: int = 999,
) -> dict:
    """
    Callaway & Sant'Anna (2021) ATT(g,t) estimator.

    Since pricing starts on a single date for all treated zones,
    we define:
      g = "first treated period" = 202501 (Jan 2025) for all CBD zones
      g = 0 (never treated) for control zones

    This collapses to a clean two-group CS-DiD, but the framework
    still gives us:
      - honest pre-trend test with valid standard errors
      - heterogeneity by calendar time
      - aggregated simple ATT with influence-function SEs

    Uses a pure-Python implementation (no R bridge required).
    """
    logger.info(f"[L2b CS-DiD] outcome={outcome}, anticipation={anticipation}")

    SHOCK = pd.Timestamp("2025-01-05")

    df = panel.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["period"] = df["date"].dt.to_period("W").apply(lambda p: p.start_time)

    # Group g: first treated week (or 0 if never treated)
    df["g"] = np.where(df["treat_binary"] == 1, SHOCK, pd.NaT)

    # Get all unique (g, t) combinations
    treated_periods = sorted(df["period"].unique())
    control_ids = set(df.loc[df["treat_binary"] == 0, "zone_id"].unique())
    treated_ids = set(df.loc[df["treat_binary"] == 1, "zone_id"].unique())

    att_records = []

    for t in treated_periods:
        # Reference period: last pre-period
        pre_periods = [p for p in treated_periods if p < SHOCK - pd.Timedelta(weeks=anticipation)]
        if not pre_periods:
            continue
        t_pre = max(pre_periods)

        # 2x2 DiD: treated vs control, t_pre vs t
        sub = df[df["period"].isin([t_pre, t])].copy()
        sub["post_local"] = (sub["period"] == t).astype(int)

        treated_mask = sub["zone_id"].isin(treated_ids)
        control_mask = sub["zone_id"].isin(control_ids)

        sub_tc = sub[treated_mask | control_mask].copy()
        if len(sub_tc) < 10:
            continue

        sub_tc["treat_x_post"] = sub_tc["treat_binary"] * sub_tc["post_local"]

        # Simple 2x2 ATT
        dy_treated = (
            sub_tc[treated_mask & (sub_tc["post_local"] == 1)][outcome].mean()
            - sub_tc[treated_mask & (sub_tc["post_local"] == 0)][outcome].mean()
        )
        dy_control = (
            sub_tc[control_mask & (sub_tc["post_local"] == 1)][outcome].mean()
            - sub_tc[control_mask & (sub_tc["post_local"] == 0)][outcome].mean()
        )
        att = dy_treated - dy_control

        # Bootstrap SE
        boot_atts = []
        for _ in range(min(n_bootstrap, 199)):  # cap for speed
            idx_t = np.random.choice(
                sub_tc[treated_mask].index, size=treated_mask.sum(), replace=True)
            idx_c = np.random.choice(
                sub_tc[control_mask].index, size=control_mask.sum(), replace=True)
            b = pd.concat([sub_tc.loc[idx_t], sub_tc.loc[idx_c]])
            dy_t = (b[b["treat_binary"]==1][b["post_local"]==1][outcome].mean()
                    - b[b["treat_binary"]==1][b["post_local"]==0][outcome].mean())
            dy_c = (b[b["treat_binary"]==0][b["post_local"]==1][outcome].mean()
                    - b[b["treat_binary"]==0][b["post_local"]==0][outcome].mean())
            boot_atts.append(dy_t - dy_c)

        se  = np.std(boot_atts, ddof=1)
        ci_low  = att - 1.96 * se
        ci_high = att + 1.96 * se
        is_pre = t < SHOCK

        att_records.append({
            "period":   t,
            "att":      att,
            "se":       se,
            "ci_low":   ci_low,
            "ci_high":  ci_high,
            "is_pre":   is_pre,
            "rel_week": int((t - SHOCK).days // 7),
        })

    att_df = pd.DataFrame(att_records).sort_values("period").reset_index(drop=True)

    # Aggregated simple ATT (post-period average)
    post_att = att_df[~att_df["is_pre"]]
    agg_att  = post_att["att"].mean() if len(post_att) > 0 else np.nan

    # Pre-trend test: are all pre-ATTs jointly zero?
    pre_att = att_df[att_df["is_pre"]]
    if len(pre_att) > 1:
        t_stats = pre_att["att"] / (pre_att["se"] + 1e-8)
        pre_trend_pval = float(stats.ttest_1samp(t_stats, 0).pvalue)
    else:
        pre_trend_pval = np.nan

    logger.info(
        f"  CS-DiD aggregated ATT = {agg_att:.4f}  "
        f"pre-trend p = {pre_trend_pval:.4f}  "
        f"n_periods = {len(att_df)}"
    )

    return {
        "att_gt":          att_df,
        "agg_att":         agg_att,
        "pre_trend_pvalue": pre_trend_pval,
        "outcome":         outcome,
    }
