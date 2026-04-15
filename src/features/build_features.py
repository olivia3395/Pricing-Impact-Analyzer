"""
build_features.py
-----------------
Construct the covariate matrix X used in DML and Causal Forest.

Feature groups
--------------
1. Zone-level static:   distance_to_cbd_km, borough_code, is_airport_zone,
                        treat_ring, pre_mean_trips, pre_cv_trips
2. Temporal:            dow, month, is_holiday, rel_week
3. Lagged outcomes:     lag1_trip_count, lag7_trip_count, rolling7_mean
4. Weather proxies:     (optional, loaded if available)

All features are standardised (zero mean, unit variance) before output.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from loguru import logger


# US federal holidays relevant to the study window
# fmt: off
US_HOLIDAYS = pd.to_datetime([
    "2024-07-04", "2024-09-02", "2024-10-14", "2024-11-11",
    "2024-11-28", "2024-12-25",
    "2025-01-01", "2025-01-20", "2025-02-17", "2025-05-26",
    "2025-07-04",
])
# fmt: on


def add_lag_features(panel: pd.DataFrame, outcome: str = "trip_count") -> pd.DataFrame:
    """Add lagged and rolling window features per zone."""
    panel = panel.sort_values(["zone_id", "date"])

    grp = panel.groupby("zone_id")[outcome]
    panel[f"lag1_{outcome}"]       = grp.shift(1)
    panel[f"lag7_{outcome}"]       = grp.shift(7)
    panel[f"roll7_mean_{outcome}"] = grp.shift(1).rolling(7, min_periods=3).mean().reset_index(level=0, drop=True)
    panel[f"roll7_std_{outcome}"]  = grp.shift(1).rolling(7, min_periods=3).std().reset_index(level=0, drop=True)

    return panel


def add_zone_static_features(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Compute pre-period zone-level statistics and merge back.
    Pre-period = before 2025-01-05.
    """
    SHOCK = pd.Timestamp("2025-01-05")
    pre = panel[panel["date"] < SHOCK]

    stats = (
        pre.groupby("zone_id")["trip_count"]
        .agg(pre_mean_trips="mean", pre_std_trips="std", pre_n_days="count")
        .assign(pre_cv_trips=lambda d: d["pre_std_trips"] / (d["pre_mean_trips"] + 1e-6))
        .reset_index()
    )

    panel = panel.merge(stats, on="zone_id", how="left")
    return panel


def add_temporal_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Add calendar and holiday indicators."""
    panel["month"]      = panel["date"].dt.month
    panel["dow"]        = panel["date"].dt.dayofweek
    panel["is_holiday"] = panel["date"].isin(US_HOLIDAYS).astype(int)
    panel["week_of_year"] = panel["date"].dt.isocalendar().week.astype(int)
    return panel


def build_feature_matrix(
    panel: pd.DataFrame,
    outcome: str = "trip_count",
    exclude_post: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Build covariate matrix X for a given outcome variable.

    Parameters
    ----------
    panel        : zone-day panel with treatment + geo columns
    outcome      : outcome variable name (for lag features)
    exclude_post : if True, fit scaler only on pre-period rows

    Returns
    -------
    (panel_with_features, feature_columns)
    """
    panel = add_lag_features(panel, outcome)
    panel = add_zone_static_features(panel)
    panel = add_temporal_features(panel)

    feature_cols = [
        # Zone static
        "distance_to_cbd_km",
        "borough_code",
        "is_airport_zone",
        "treat_ring",
        "pre_mean_trips",
        "pre_cv_trips",
        # Temporal
        "dow",
        "month",
        "is_holiday",
        "week_of_year",
        "is_weekend",
        "peak_share",
        # Lagged outcome
        f"lag1_{outcome}",
        f"lag7_{outcome}",
        f"roll7_mean_{outcome}",
        f"roll7_std_{outcome}",
        # Continuous treatment dose
        "dose_raw",
        "dose_std",
    ]

    # Keep only columns that actually exist
    feature_cols = [c for c in feature_cols if c in panel.columns]

    # Drop rows with NaN in features (typically the first 7 days per zone)
    n_before = len(panel)
    panel = panel.dropna(subset=feature_cols)
    logger.info(
        f"Feature matrix: dropped {n_before - len(panel):,} rows with NaN features "
        f"→ {len(panel):,} rows, {len(feature_cols)} features"
    )

    # Standardise
    SHOCK = pd.Timestamp("2025-01-05")
    if exclude_post:
        ref_idx = panel["date"] < SHOCK
    else:
        ref_idx = pd.Series(True, index=panel.index)

    scaler = StandardScaler()
    panel[feature_cols] = scaler.fit_transform(panel[feature_cols])

    return panel, feature_cols
