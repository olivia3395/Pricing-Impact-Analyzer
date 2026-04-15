"""
run_sdid.py
-----------
L4: Synthetic Difference-in-Differences (Arkhangelsky et al. 2021)

SDID combines:
  - Synthetic Control: reweights control units to match pre-trends
  - DiD: adds time weights to balance pre- vs post-period
  - Placebo SE via unit/time bootstrap

Key advantage over standard DiD:
  Removes parallel-trends assumption — instead finds ω weights such that
  the weighted control trajectory is parallel to the treated trajectory
  in the pre-period.

This is the cleanest robustness check for our main DiD result.
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from loguru import logger


def _balance_weights(
    Y_control_pre: np.ndarray,
    Y_treated_pre: np.ndarray,
    regularize: float = 1e-6,
) -> np.ndarray:
    """
    Solve for unit weights ω ∈ Δ^{N_c} that minimise:
      ||Y_treated_pre_mean - Y_control_pre.T @ ω||² + λ·||ω||²

    Returns ω of shape (N_control,)
    """
    N_c, T_pre = Y_control_pre.shape
    y_target = Y_treated_pre.mean(axis=0)  # (T_pre,)

    def objective(w):
        diff = Y_control_pre.T @ w - y_target
        return np.dot(diff, diff) + regularize * np.dot(w, w)

    def gradient(w):
        diff = Y_control_pre.T @ w - y_target
        return 2 * (Y_control_pre @ diff) + 2 * regularize * w

    w0 = np.ones(N_c) / N_c
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    bounds = [(0, 1)] * N_c

    res = minimize(
        objective, w0, jac=gradient,
        method="SLSQP", bounds=bounds, constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-9},
    )
    return res.x


def _time_weights(
    Y_control_pre: np.ndarray,
    Y_control_post: np.ndarray,
) -> np.ndarray:
    """
    Solve for time weights λ ∈ Δ^{T_pre} such that:
      Y_control_post_mean ≈ Y_control_pre.T @ λ

    Balances post-period mean with pre-period weighted average.
    Returns λ of shape (T_pre,)
    """
    T_pre  = Y_control_pre.shape[1]
    y_post = Y_control_post.mean(axis=1)  # (N_c,)

    def objective(l):
        diff = Y_control_pre @ l - y_post
        return np.dot(diff, diff)

    def gradient(l):
        diff = Y_control_pre @ l - y_post
        return 2 * Y_control_pre.T @ diff

    l0 = np.ones(T_pre) / T_pre
    constraints = [{"type": "eq", "fun": lambda l: l.sum() - 1}]
    bounds = [(0, 1)] * T_pre

    res = minimize(
        objective, l0, jac=gradient,
        method="SLSQP", bounds=bounds, constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-9},
    )
    return res.x


def sdid_estimator(
    Y: np.ndarray,
    treat_mask: np.ndarray,
    T_pre: int,
) -> float:
    """
    Core SDID estimator on outcome matrix Y (N × T).

    Parameters
    ----------
    Y          : (N_total × T) outcome matrix, rows = units, cols = time
    treat_mask : boolean array of length N_total (True = treated unit)
    T_pre      : number of pre-treatment time periods

    Returns
    -------
    tau_hat : scalar SDID ATT estimate
    """
    Y_treat   = Y[treat_mask]            # (N_treat × T)
    Y_control = Y[~treat_mask]           # (N_control × T)

    Y_treat_pre   = Y_treat[:, :T_pre]
    Y_treat_post  = Y_treat[:, T_pre:]
    Y_control_pre  = Y_control[:, :T_pre]
    Y_control_post = Y_control[:, T_pre:]

    omega = _balance_weights(Y_control_pre, Y_treat_pre)
    lam   = _time_weights(Y_control_pre, Y_control_post)

    # SDID DiD formula
    tau = (
        (Y_treat_post.mean() - Y_treat_pre @ lam)
        - (omega @ Y_control_post.mean(axis=1) - omega @ (Y_control_pre @ lam))
    )
    return float(tau)


def run_sdid(
    panel: pd.DataFrame,
    outcome: str = "log_trip_count",
    n_bootstrap: int = 200,
    seed: int = 42,
) -> dict:
    """
    Run Synthetic DiD on the zone-day panel.

    Pivots the panel to (zone × week) matrix, runs SDID,
    bootstraps SE by randomly permuting control unit assignment.

    Returns
    -------
    dict: tau_hat, se, ci_low, ci_high, omega_weights, lambda_weights
    """
    logger.info(f"[L4 SDID] outcome={outcome}, bootstrap={n_bootstrap}")
    np.random.seed(seed)

    SHOCK = pd.Timestamp("2025-01-05")

    # Aggregate to zone × week (reduces noise, speeds up optimisation)
    df = panel.copy()
    df["week"] = df["date"].dt.to_period("W").apply(lambda p: p.start_time)
    weekly = (
        df.groupby(["zone_id", "week", "treat_binary"])[outcome]
        .mean()
        .reset_index()
    )

    # Pivot to matrix
    mat = weekly.pivot(index="zone_id", columns="week", values=outcome).fillna(0)
    treat_info = (
        weekly[["zone_id", "treat_binary"]]
        .drop_duplicates()
        .set_index("zone_id")["treat_binary"]
    )
    treat_mask = treat_info.reindex(mat.index).fillna(0).astype(bool).values

    # T_pre: number of weeks before shock
    weeks = list(mat.columns)
    T_pre = sum(w < SHOCK for w in weeks)
    Y     = mat.values.astype(float)

    tau_hat = sdid_estimator(Y, treat_mask, T_pre)
    logger.info(f"  τ̂ (SDID) = {tau_hat:.4f}")

    # Bootstrap SE via placebo unit permutation
    n_treated  = treat_mask.sum()
    n_control  = (~treat_mask).sum()
    boot_taus  = []

    for _ in range(n_bootstrap):
        # Randomly assign treatment to n_treated control units (placebo)
        control_idx = np.where(~treat_mask)[0]
        perm_treat  = np.zeros(len(treat_mask), dtype=bool)
        chosen      = np.random.choice(control_idx, size=n_treated, replace=False)
        perm_treat[chosen] = True

        # Use only control units in this placebo run
        Y_ctrl = Y[~treat_mask]
        perm_mask = perm_treat[~treat_mask]
        if perm_mask.sum() == 0 or (~perm_mask).sum() == 0:
            continue
        try:
            tau_b = sdid_estimator(Y_ctrl, perm_mask, T_pre)
            boot_taus.append(tau_b)
        except Exception:
            continue

    se      = float(np.std(boot_taus, ddof=1)) if boot_taus else np.nan
    ci_low  = tau_hat - 1.96 * se
    ci_high = tau_hat + 1.96 * se

    logger.info(f"  SE={se:.4f}  CI=[{ci_low:.4f}, {ci_high:.4f}]")

    return {
        "tau_hat":  tau_hat,
        "se":       se,
        "ci_low":   ci_low,
        "ci_high":  ci_high,
        "outcome":  outcome,
        "n_treated": int(n_treated),
        "n_control": int(n_control),
        "T_pre":    T_pre,
        "T_post":   len(weeks) - T_pre,
    }
