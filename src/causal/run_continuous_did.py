"""
run_continuous_did.py
---------------------
L3: Continuous Treatment DiD

Instead of binary Treat ∈ {0,1}, we use the actual cbd_congestion_fee
as a continuous dose variable D_it.

Model:
  Y_{it} = α_i + γ_t + β₁·Post_t·D_i + β₂·D_i + ε_{it}

where D_i = average cbd_congestion_fee for zone i in the post period.

This lets us:
  1. Estimate the dose-response curve: how does each additional $1 in
     congestion fee change trip volume?
  2. Draw a smooth elasticity curve across zones
  3. Test for non-linearity (add D²·Post term)

Reference: Callaway, Goodman-Bacon & Sant'Anna (2024) "DiD with a
Continuous Treatment"
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS
from sklearn.preprocessing import PolynomialFeatures
from loguru import logger


def _build_zone_dose(panel: pd.DataFrame, fee_col: str = "avg_cbd_fee") -> pd.Series:
    """
    Compute each zone's average fee in the post period as its 'dose'.
    Control zones always have dose ≈ 0.
    """
    SHOCK = pd.Timestamp("2025-01-05")
    post = panel[panel["date"] >= SHOCK]
    dose = post.groupby("zone_id")[fee_col].mean().rename("zone_dose")
    return dose


def run_continuous_did(
    panel: pd.DataFrame,
    outcome: str = "log_trip_count",
    fee_col: str = "avg_cbd_fee",
    quadratic: bool = True,
) -> dict:
    """
    Estimate dose-response DiD with zone + time FE.

    Returns
    -------
    dict with:
        beta_linear   : β on Post × Dose (linear term)
        beta_quad     : β on Post × Dose² (if quadratic=True)
        dose_response : DataFrame(dose_value, predicted_effect) for plotting
        result_obj    : linearmodels result object
    """
    logger.info(f"[L3 Continuous DiD] outcome={outcome}, quadratic={quadratic}")

    df = panel.copy()

    # Merge zone-level dose
    zone_dose = _build_zone_dose(df, fee_col)
    df = df.join(zone_dose, on="zone_id")
    df["zone_dose"] = df["zone_dose"].fillna(0)

    # Interaction terms
    df["post_x_dose"]  = df["post"] * df["zone_dose"]
    if quadratic:
        df["zone_dose_sq"]      = df["zone_dose"] ** 2
        df["post_x_dose_sq"]    = df["post"] * df["zone_dose_sq"]

    reg_cols = ["post_x_dose"]
    if quadratic:
        reg_cols += ["post_x_dose_sq"]

    df_idx = df.set_index(["zone_id", "date"])
    df_idx.index.names = ["entity", "time"]

    model = PanelOLS(
        dependent=df_idx[outcome],
        exog=df_idx[reg_cols],
        entity_effects=True,
        time_effects=True,
        drop_absorbed=True,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = model.fit(cov_type="clustered", cluster_entity=True)

    beta_lin  = float(res.params["post_x_dose"])
    beta_quad = float(res.params["post_x_dose_sq"]) if quadratic else 0.0

    # Build dose-response curve
    dose_grid  = np.linspace(0, df["zone_dose"].max(), 100)
    predicted  = beta_lin * dose_grid + beta_quad * dose_grid ** 2

    dose_response = pd.DataFrame({
        "dose": dose_grid,
        "effect": predicted,
        "ci_low":  predicted - 1.96 * np.sqrt(
            (res.std_errors["post_x_dose"] * dose_grid) ** 2),
        "ci_high": predicted + 1.96 * np.sqrt(
            (res.std_errors["post_x_dose"] * dose_grid) ** 2),
    })

    logger.info(
        f"  β_linear={beta_lin:.4f}  β_quad={beta_quad:.4f}  "
        f"R²={res.rsquared:.4f}  N={res.nobs:,}"
    )

    return {
        "beta_linear":    beta_lin,
        "beta_quad":      beta_quad,
        "se_linear":      float(res.std_errors["post_x_dose"]),
        "p_linear":       float(res.pvalues["post_x_dose"]),
        "dose_response":  dose_response,
        "result_obj":     res,
        "outcome":        outcome,
    }


def run_dose_response_by_bin(
    panel: pd.DataFrame,
    outcome: str = "log_trip_count",
    n_bins: int = 5,
) -> pd.DataFrame:
    """
    Non-parametric dose-response: run separate DiD for each dose quantile bin.

    This relaxes the linear dose assumption and reveals non-linearities.
    Returns DataFrame: bin | bin_label | att | se | ci_low | ci_high
    """
    logger.info(f"[L3 Dose-Response Bins] outcome={outcome}, bins={n_bins}")

    df = panel.copy()
    zone_dose = _build_zone_dose(df)
    df = df.join(zone_dose, on="zone_id")
    df["zone_dose"] = df["zone_dose"].fillna(0)

    # Assign bins based on treated zone doses
    treated_doses = df.loc[df["treat_binary"] == 1, "zone_dose"].drop_duplicates()
    try:
        bins = pd.qcut(treated_doses, q=n_bins, labels=False, duplicates="drop")
    except ValueError:
        bins = pd.cut(treated_doses, bins=n_bins, labels=False)

    dose_bin_map = dict(zip(treated_doses.index, bins))
    df["dose_bin"] = df.index.map(dose_bin_map)

    results = []
    for b in sorted(df["dose_bin"].dropna().unique()):
        # Bin b treated zones vs all control zones
        sub = df[(df["dose_bin"] == b) | (df["treat_binary"] == 0)].copy()
        sub["treat_bin"] = ((sub["dose_bin"] == b) & (sub["treat_binary"] == 1)).astype(int)
        sub["treat_post"] = sub["treat_bin"] * sub["post"]

        sub_idx = sub.set_index(["zone_id", "date"])
        sub_idx.index.names = ["entity", "time"]

        try:
            m = PanelOLS(
                dependent=sub_idx[outcome],
                exog=sub_idx[["treat_post"]],
                entity_effects=True, time_effects=True, drop_absorbed=True,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r = m.fit(cov_type="clustered", cluster_entity=True)

            mean_dose = df.loc[df["dose_bin"] == b, "zone_dose"].mean()
            results.append({
                "bin":       b,
                "mean_dose": mean_dose,
                "att":       float(r.params["treat_post"]),
                "se":        float(r.std_errors["treat_post"]),
                "ci_low":    float(r.conf_int().loc["treat_post"].iloc[0]),
                "ci_high":   float(r.conf_int().loc["treat_post"].iloc[1]),
                "p_value":   float(r.pvalues["treat_post"]),
            })
        except Exception as e:
            logger.warning(f"  Bin {b} failed: {e}")

    return pd.DataFrame(results)
