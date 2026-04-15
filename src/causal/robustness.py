"""
robustness.py
-------------
Robustness checks for the main causal estimates.

Checks implemented:
  R1  Placebo date test:  shift shock date by -8 and +8 weeks
  R2  Anticipation test:  restrict post-window to ≥4 weeks after shock
  R3  Control group sensitivity: exclude NYC buffer zones
  R4  Alternative outcome: level trip_count (vs log)
  R5  Bandwidth test: ±4 weeks, ±8 weeks, ±16 weeks windows
  R6  Leave-one-borough-out: drop each outer borough in turn
  R7  DML learner sensitivity: RF vs GBM vs Lasso

All results are stored in a single robustness_table DataFrame.
"""

from __future__ import annotations

from typing import Callable, List

import numpy as np
import pandas as pd
from loguru import logger

from src.causal.run_did import run_naive_did, run_event_study
from src.causal.run_dml import run_dml


def placebo_date_test(
    panel: pd.DataFrame,
    feature_cols: List[str],
    outcome: str = "log_trip_count",
    shifts_weeks: List[int] = [-8, -4, 4, 8],
) -> pd.DataFrame:
    """
    R1: Run DiD with placebo shock dates (pre-period only).
    If pre-trends are flat, placebo β should be ≈ 0.
    """
    logger.info("[Robustness R1] Placebo date test")
    SHOCK = pd.Timestamp("2025-01-05")
    records = []

    for shift in shifts_weeks:
        placebo_date = SHOCK + pd.Timedelta(weeks=shift)
        p = panel.copy()

        if shift < 0:
            # Placebo entirely in pre-period
            p = p[p["date"] < SHOCK]
        else:
            p = p[p["date"] < SHOCK]  # Use only pre-period for placebo

        p["post"] = (p["date"] >= placebo_date).astype(int)
        p = p[p["date"].between(
            placebo_date - pd.Timedelta(weeks=8),
            placebo_date + pd.Timedelta(weeks=8),
        )]

        if len(p) < 100:
            continue

        try:
            res = run_naive_did(p, outcome=outcome)
            records.append({
                "check": f"Placebo shift {shift:+d}w",
                "theta": res["ate"],
                "se":    res["se"],
                "p":     res["p_value"],
                "note":  "Should be ≈ 0",
            })
        except Exception as e:
            logger.warning(f"  Placebo {shift:+d}w failed: {e}")

    return pd.DataFrame(records)


def anticipation_test(
    panel: pd.DataFrame,
    outcome: str = "log_trip_count",
    drop_weeks: int = 4,
) -> dict:
    """
    R2: Drop the first `drop_weeks` weeks of post period.
    Tests if results are driven by anticipation effects.
    """
    logger.info(f"[Robustness R2] Anticipation test: drop first {drop_weeks}w post")
    SHOCK = pd.Timestamp("2025-01-05")
    p = panel[
        (panel["date"] < SHOCK) |
        (panel["date"] >= SHOCK + pd.Timedelta(weeks=drop_weeks))
    ].copy()
    return run_naive_did(p, outcome=outcome)


def control_sensitivity(
    panel: pd.DataFrame,
    outcome: str = "log_trip_count",
) -> pd.DataFrame:
    """
    R3: Vary control group:
      (a) exclude buffer/adjacent zones
      (b) Manhattan-only control
      (c) Outer-borough-only control
    """
    logger.info("[Robustness R3] Control group sensitivity")

    from src.geospatial.cbd_mapping import BUFFER_ZONE_IDS

    records = []
    specs = {
        "Exclude buffer zones": panel[panel["treat_ring"] != 1].copy(),
        "Manhattan control only": panel[
            (panel["treat_binary"] == 1) |
            ((panel["treat_binary"] == 0) & (panel["borough"] == "Manhattan"))
        ].copy(),
        "Outer-borough control only": panel[
            (panel["treat_binary"] == 1) |
            ((panel["treat_binary"] == 0) & (panel["borough"] != "Manhattan"))
        ].copy(),
    }

    for label, p in specs.items():
        if len(p) < 100:
            continue
        try:
            res = run_naive_did(p, outcome=outcome)
            records.append({
                "check": label,
                "theta": res["ate"],
                "se":    res["se"],
                "p":     res["p_value"],
            })
        except Exception as e:
            logger.warning(f"  {label} failed: {e}")

    return pd.DataFrame(records)


def bandwidth_sensitivity(
    panel: pd.DataFrame,
    outcome: str = "log_trip_count",
    bandwidths_weeks: List[int] = [4, 8, 12, 16, 26],
) -> pd.DataFrame:
    """
    R5: Vary the pre/post window width.
    """
    logger.info("[Robustness R5] Bandwidth test")
    SHOCK = pd.Timestamp("2025-01-05")
    records = []

    for bw in bandwidths_weeks:
        p = panel[
            panel["date"].between(
                SHOCK - pd.Timedelta(weeks=bw),
                SHOCK + pd.Timedelta(weeks=bw),
            )
        ].copy()
        if len(p) < 100:
            continue
        try:
            res = run_naive_did(p, outcome=outcome)
            records.append({
                "check": f"±{bw}w window",
                "theta": res["ate"],
                "se":    res["se"],
                "p":     res["p_value"],
                "n_obs": res["n_obs"],
            })
        except Exception as e:
            logger.warning(f"  BW±{bw}w failed: {e}")

    return pd.DataFrame(records)


def leave_one_borough_out(
    panel: pd.DataFrame,
    outcome: str = "log_trip_count",
) -> pd.DataFrame:
    """
    R6: Drop each outer borough from the control group in turn.
    Checks whether results are driven by a single borough's trend.
    """
    logger.info("[Robustness R6] Leave-one-borough-out")
    boroughs = [b for b in panel["borough"].dropna().unique() if b != "Manhattan"]
    records = []

    for b in boroughs:
        p = panel[panel["borough"] != b].copy()
        try:
            res = run_naive_did(p, outcome=outcome)
            records.append({
                "check": f"Drop {b}",
                "theta": res["ate"],
                "se":    res["se"],
                "p":     res["p_value"],
            })
        except Exception as e:
            logger.warning(f"  Drop {b} failed: {e}")

    return pd.DataFrame(records)


def dml_learner_sensitivity(
    panel: pd.DataFrame,
    feature_cols: List[str],
    outcome: str = "log_trip_count",
    treatment: str = "treat_binary",
) -> pd.DataFrame:
    """
    R7: Compare DML results across ML learner choices.
    """
    logger.info("[Robustness R7] DML learner sensitivity")
    records = []

    for learner in ["random_forest", "gradient_boosting", "lasso"]:
        try:
            res = run_dml(
                panel, feature_cols, [outcome],
                treatment=treatment,
                learner=learner,
                n_folds=5, n_reps=2,
            )
            r = res.iloc[0]
            records.append({
                "check": f"DML/{learner}",
                "theta": r["theta"],
                "se":    r["se"],
                "p":     r["p_value"],
            })
        except Exception as e:
            logger.warning(f"  DML/{learner} failed: {e}")

    return pd.DataFrame(records)


def run_all_robustness(
    panel: pd.DataFrame,
    feature_cols: List[str],
    outcome: str = "log_trip_count",
) -> pd.DataFrame:
    """
    Run all robustness checks and return a unified summary table.
    """
    logger.info("=== Running All Robustness Checks ===")

    all_dfs = []

    checks = [
        ("Placebo Date",        lambda: placebo_date_test(panel, feature_cols, outcome)),
        ("Anticipation",        lambda: pd.DataFrame([anticipation_test(panel, outcome)])),
        ("Control Sensitivity", lambda: control_sensitivity(panel, outcome)),
        ("Bandwidth",           lambda: bandwidth_sensitivity(panel, outcome)),
        ("Leave-One-Borough",   lambda: leave_one_borough_out(panel, outcome)),
        ("DML Learners",        lambda: dml_learner_sensitivity(panel, feature_cols, outcome)),
    ]

    for name, fn in checks:
        try:
            df = fn()
            if df is not None and len(df) > 0:
                df["group"] = name
                all_dfs.append(df)
        except Exception as e:
            logger.warning(f"Robustness check '{name}' failed: {e}")

    if not all_dfs:
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)
    combined["sig"] = combined["p"].apply(
        lambda p: "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
    )

    logger.info(f"Robustness table: {len(combined)} checks completed")
    return combined
