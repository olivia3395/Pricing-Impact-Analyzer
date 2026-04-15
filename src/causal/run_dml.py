"""
run_dml.py
----------
L5a: Double / Debiased Machine Learning (Chernozhukov et al. 2018)

Why DML here?
  Standard DiD controls for zone + time FEs, but ignores high-dimensional
  confounders X (lag features, zone characteristics, weather proxies).
  DML uses ML to partial out X from both Y and D, then runs OLS on the
  residuals — giving an asymptotically unbiased ATE even when X is
  high-dimensional.

Partially linear model:
  Y = θ·D + g(X) + ε        E[ε|D,X] = 0
  D = m(X) + v              E[v|X] = 0

  1. Fit ĝ(X): predict Y from X (no D)  →  Ỹ = Y - ĝ(X)
  2. Fit m̂(X): predict D from X         →  D̃ = D - m̂(X)
  3. OLS of Ỹ ~ D̃                        →  θ̂ = DML-ATE

Cross-fitting (K-fold) eliminates regularisation bias from step 1/2.

We run DML for multiple outcomes and report a joint results table.
"""

from __future__ import annotations

import warnings
from typing import List

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
)
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from scipy import stats
from loguru import logger


# ── ML Learner factory ────────────────────────────────────────────────────────

def _get_learner(name: str, seed: int = 42):
    if name == "random_forest":
        return RandomForestRegressor(
            n_estimators=300, max_depth=8, min_samples_leaf=5,
            n_jobs=-1, random_state=seed,
        )
    if name == "gradient_boosting":
        return GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=5,
            random_state=seed,
        )
    if name == "lasso":
        return LassoCV(cv=5, random_state=seed, max_iter=5000)
    if name == "ridge":
        return RidgeCV(cv=5)
    raise ValueError(f"Unknown learner: {name}")


# ── Core cross-fitted residuals ───────────────────────────────────────────────

def _cross_fit_residuals(
    X: np.ndarray,
    Z: np.ndarray,
    n_folds: int,
    learner_name: str,
    seed: int,
) -> np.ndarray:
    """
    For each fold k:
      1. Fit learner on training folds
      2. Predict on held-out fold
      3. Compute residual Z - Ẑ

    Returns residuals array of same length as Z.
    """
    residuals = np.zeros_like(Z, dtype=float)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        learner = _get_learner(learner_name, seed=seed + fold)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            learner.fit(X[train_idx], Z[train_idx])
        Z_hat = learner.predict(X[test_idx])
        residuals[test_idx] = Z[test_idx] - Z_hat

    return residuals


# ── DML estimator ─────────────────────────────────────────────────────────────

def dml_ate(
    Y: np.ndarray,
    D: np.ndarray,
    X: np.ndarray,
    n_folds: int = 5,
    n_reps: int = 3,
    learner: str = "random_forest",
    seed: int = 42,
) -> dict:
    """
    DML-ATE via cross-fitting repeated n_reps times (median aggregation).

    Parameters
    ----------
    Y : outcome (n,)
    D : treatment indicator or continuous dose (n,)
    X : covariate matrix (n, p)

    Returns
    -------
    dict: theta, se, t_stat, p_value, ci_low, ci_high
    """
    thetas, ses = [], []

    for rep in range(n_reps):
        s = seed + rep * 1000

        # Partial out Y ~ X
        Y_res = _cross_fit_residuals(X, Y, n_folds, learner, s)

        # Partial out D ~ X
        D_res = _cross_fit_residuals(X, D, n_folds, learner, s + 1)

        # OLS: Y_res ~ D_res (no intercept, moments estimator)
        denom = np.dot(D_res, D_res)
        if denom < 1e-12:
            continue
        theta_rep = np.dot(D_res, Y_res) / denom

        # Heteroskedasticity-robust SE (sandwich)
        n = len(Y_res)
        eps = Y_res - theta_rep * D_res
        meat = np.dot(D_res ** 2, eps ** 2)
        se_rep = np.sqrt(meat / (denom ** 2))

        thetas.append(theta_rep)
        ses.append(se_rep)

    theta = float(np.median(thetas))
    se    = float(np.median(ses))
    t     = theta / (se + 1e-12)
    p     = float(2 * stats.norm.sf(abs(t)))

    return {
        "theta":   theta,
        "se":      se,
        "t_stat":  t,
        "p_value": p,
        "ci_low":  theta - 1.96 * se,
        "ci_high": theta + 1.96 * se,
        "n_reps":  len(thetas),
    }


# ── Multi-outcome DML wrapper ─────────────────────────────────────────────────

def run_dml(
    panel: pd.DataFrame,
    feature_cols: List[str],
    outcomes: List[str],
    treatment: str = "treat_binary",
    learner: str = "random_forest",
    n_folds: int = 5,
    n_reps: int = 3,
) -> pd.DataFrame:
    """
    Run DML-ATE for each outcome in `outcomes`.

    Parameters
    ----------
    panel        : zone-day panel with features already built
    feature_cols : list of covariate column names
    outcomes     : list of outcome column names
    treatment    : treatment variable (binary or continuous dose)

    Returns
    -------
    DataFrame: outcome | theta | se | t_stat | p_value | ci_low | ci_high | sig
    """
    logger.info(f"[L5a DML] learner={learner}, folds={n_folds}, reps={n_reps}")
    logger.info(f"  Outcomes: {outcomes}")
    logger.info(f"  Features: {len(feature_cols)} columns")

    df = panel.dropna(subset=feature_cols + outcomes + [treatment]).copy()
    X  = df[feature_cols].values.astype(float)
    D  = df[treatment].values.astype(float)

    records = []
    for outcome in outcomes:
        Y = df[outcome].values.astype(float)
        logger.info(f"  → {outcome}")
        result = dml_ate(Y, D, X, n_folds=n_folds, n_reps=n_reps, learner=learner)
        result["outcome"] = outcome
        result["n_obs"]   = len(Y)
        records.append(result)

    results_df = pd.DataFrame(records)
    results_df["sig"] = results_df["p_value"].apply(
        lambda p: "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
    )

    logger.info("DML Results:")
    for _, row in results_df.iterrows():
        logger.info(
            f"  {row['outcome']:20s}  θ={row['theta']:+.4f}  "
            f"SE={row['se']:.4f}  p={row['p_value']:.4f}  {row['sig']}"
        )

    return results_df
