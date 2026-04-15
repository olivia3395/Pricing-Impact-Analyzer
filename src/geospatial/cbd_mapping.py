"""
cbd_mapping.py
--------------
Assign treatment definitions to each zone_id:

  treat_binary    : 1 if zone is inside the MTA Congestion Relief Zone, else 0
  treat_intensity : continuous score in [0, 1] based on share of trips
                    that cross the CBD boundary (proxy for exposure)
  treat_ring      : 0 = far control | 1 = adjacent buffer | 2 = CBD core

These are used in:
  - L1/L2: binary DiD
  - L3: continuous treatment DiD
  - L5b: as heterogeneity variables in Causal Forest
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger


# ── Official CBD zone list (from configs/default.yaml) ───────────────────────
CBD_ZONE_IDS: set[int] = {
    4, 12, 13, 24, 41, 42, 43, 45, 48, 50, 68, 79, 87, 88, 89, 90,
    100, 103, 104, 105, 107, 113, 116, 120, 125, 126, 127, 128,
    144, 148, 151, 152, 153, 158, 161, 162, 163, 164, 166, 170,
    186, 194, 202, 209, 211, 224, 229, 230, 231, 232, 233, 234,
    236, 237, 238, 239, 243, 244, 246, 249, 261, 262, 263,
}

# Zones that border the CBD — partial exposure (for ring treatment)
BUFFER_ZONE_IDS: set[int] = {
    17, 25, 36, 37, 74, 75, 76, 77, 112, 140, 168, 255,
}

# Airport zones — receive flat surcharges regardless of CBD
AIRPORT_ZONE_IDS: set[int] = {1, 132, 138}  # EWR, JFK, LGA


def assign_treatment(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Add treatment columns to the zone-day panel.

    Parameters
    ----------
    panel : pd.DataFrame
        Must have column `zone_id`.

    Returns
    -------
    panel with new columns:
        treat_binary, treat_ring, is_airport_zone
    """
    ids = panel["zone_id"]

    panel["treat_binary"]   = ids.isin(CBD_ZONE_IDS).astype(int)
    panel["is_airport_zone"] = ids.isin(AIRPORT_ZONE_IDS).astype(int)

    # Three-way ring
    def _ring(z):
        if z in CBD_ZONE_IDS:
            return 2
        if z in BUFFER_ZONE_IDS:
            return 1
        return 0

    panel["treat_ring"] = ids.map(_ring).astype(int)

    n_treated = panel["treat_binary"].sum()
    logger.info(
        f"Treatment assignment: {panel['treat_binary'].nunique()} binary levels | "
        f"{(panel['treat_binary']==1).mean()*100:.1f}% of zone-day cells treated"
    )
    return panel


def assign_continuous_treatment(
    panel: pd.DataFrame,
    fee_col: str = "avg_cbd_fee",
) -> pd.DataFrame:
    """
    Build a continuous treatment intensity from the cbd_congestion_fee.

    For pre-2025 rows the fee is 0 (no pricing).
    For post-2025 rows the fee reflects actual billing.

    Adds:
        dose_raw      : raw avg_cbd_fee
        dose_std      : standardised (z-score over post-period treated cells)
        dose_quantile : quantile bin (1–5) for dose-response plots
    """
    panel["dose_raw"] = panel[fee_col].fillna(0).clip(lower=0)

    # Standardise using post-period treated cells as reference
    ref = panel.loc[(panel["post"] == 1) & (panel["treat_binary"] == 1), "dose_raw"]
    mu, sigma = ref.mean(), ref.std()
    panel["dose_std"] = (panel["dose_raw"] - mu) / (sigma + 1e-8)

    # Quantile bins (computed only on positive dose values)
    pos = panel["dose_raw"] > 0
    if pos.sum() > 0:
        panel.loc[pos, "dose_quantile"] = pd.qcut(
            panel.loc[pos, "dose_raw"], q=5,
            labels=[1, 2, 3, 4, 5], duplicates="drop",
        ).astype(float)
    panel["dose_quantile"] = panel["dose_quantile"].fillna(0).astype(int)

    return panel


def get_never_treated_zones(panel: pd.DataFrame) -> set:
    """
    Return zone_ids that never appear in CBD zone list.
    Used as 'never treated' control group in Callaway-Sant'Anna.
    """
    all_zones   = set(panel["zone_id"].unique())
    ever_treated = set(panel.loc[panel["treat_binary"] == 1, "zone_id"].unique())
    return all_zones - ever_treated
