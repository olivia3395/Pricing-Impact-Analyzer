"""
run_causal_forest.py
--------------------
L5b: Causal Forest (Wager & Athey 2018; Athey, Tibshirani & Wager 2019)

Estimates Conditional Average Treatment Effects (CATE):
  τ(x) = E[Y(1) - Y(0) | X = x]

Key outputs:
  1. Per-zone CATE estimates τ̂(x_i)  → choropleth map
  2. Variable importance: which features drive heterogeneity most?
  3. Best linear projection: τ̂(x) ~ Λ(x)  (Chernozhukov et al. 2022)
  4. Policy tree: interpretable depth-2 tree on CATE
  5. RATE curve: targeting gain from heterogeneous treatment assignment

Uses EconML's CausalForestDML (honest splitting + DML nuisance estimates)
which is the Python equivalent of the R GRF package.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.linear_model import LinearRegression
from scipy import stats
from loguru import logger

try:
    from econml.dml import CausalForestDML
    ECONML_AVAILABLE = True
except ImportError:
    ECONML_AVAILABLE = False
    logger.warning("econml not installed. Using fallback Causal Forest implementation.")


# ── EconML Causal Forest (preferred) ────────────────────────────────────────

def run_causal_forest_econml(
    panel: pd.DataFrame,
    feature_cols: List[str],
    outcome: str = "log_trip_count",
    treatment: str = "treat_binary",
    n_estimators: int = 2000,
    seed: int = 42,
) -> dict:
    """
    Fit EconML CausalForestDML and return CATE estimates + diagnostics.

    CausalForestDML:
      - Uses DML to partial out nuisance (Y~X and D~X)
      - Fits causal forest on residuals
      - Honest splitting for valid confidence intervals
    """
    logger.info(f"[L5b CausalForest/EconML] outcome={outcome}")

    df = panel.dropna(subset=feature_cols + [outcome, treatment]).copy()
    X  = df[feature_cols].values.astype(float)
    Y  = df[outcome].values.astype(float)
    T  = df[treatment].values.astype(float)

    # Nuisance learners
    model_y = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=seed)
    model_t = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=seed)

    cf = CausalForestDML(
        model_y=model_y,
        model_t=model_t,
        n_estimators=n_estimators,
        min_samples_leaf=5,
        max_features="auto",
        honest=True,
        n_jobs=-1,
        random_state=seed,
    )

    cf.fit(Y, T, X=X, W=None)

    # Point estimates and CIs
    tau_hat  = cf.effect(X)
    tau_ci   = cf.effect_interval(X, alpha=0.05)
    tau_se   = (tau_ci[1] - tau_ci[0]) / (2 * 1.96)

    df["cate"]     = tau_hat
    df["cate_low"] = tau_ci[0]
    df["cate_high"]= tau_ci[1]
    df["cate_se"]  = tau_se

    # Variable importance
    feat_imp = pd.Series(
        cf.feature_importances_, index=feature_cols
    ).sort_values(ascending=False)

    # Best Linear Projection of CATE onto features
    # τ̂(x) = β₀ + β₁·f₁(x) + ...
    blp = _best_linear_projection(tau_hat, X, feature_cols)

    # Policy tree (depth-2 interpretable rule)
    policy_tree = _fit_policy_tree(tau_hat, X, feature_cols, max_depth=2)

    # RATE (Rank-Weighted Average Treatment Effect)
    rate = _compute_rate(tau_hat, Y, T, X)

    # Aggregate CATE by zone
    zone_cate = (
        df.groupby("zone_id")[["cate", "cate_low", "cate_high", "cate_se"]]
        .mean()
        .reset_index()
    )

    logger.info(
        f"  Mean CATE = {tau_hat.mean():.4f}  "
        f"Std CATE = {tau_hat.std():.4f}  "
        f"% significant = {(np.abs(tau_hat / (tau_se + 1e-8)) > 1.96).mean()*100:.1f}%"
    )

    return {
        "cate_estimates": df[["zone_id", "date", "cate", "cate_low",
                               "cate_high", "cate_se"]],
        "zone_cate":       zone_cate,
        "feature_importance": feat_imp,
        "blp":             blp,
        "policy_tree":     policy_tree,
        "rate":            rate,
        "model":           cf,
        "outcome":         outcome,
        "mean_cate":       float(tau_hat.mean()),
        "std_cate":        float(tau_hat.std()),
    }


# ── Fallback Causal Forest (no EconML) ───────────────────────────────────────

def run_causal_forest_fallback(
    panel: pd.DataFrame,
    feature_cols: List[str],
    outcome: str = "log_trip_count",
    treatment: str = "treat_binary",
    n_estimators: int = 500,
    seed: int = 42,
) -> dict:
    """
    Honest causal forest approximation using T-learner + cross-fitting.

    T-learner:
      μ̂₁(x) = E[Y|T=1, X=x]   fit on treated
      μ̂₀(x) = E[Y|T=0, X=x]   fit on controls
      τ̂(x)  = μ̂₁(x) - μ̂₀(x)

    Honesty: fit on one half, estimate on other half.
    """
    logger.info(f"[L5b CausalForest/Fallback T-Learner] outcome={outcome}")

    df = panel.dropna(subset=feature_cols + [outcome, treatment]).copy()
    X  = df[feature_cols].values.astype(float)
    Y  = df[outcome].values.astype(float)
    T  = df[treatment].values.astype(float)

    np.random.seed(seed)
    train_idx, est_idx = train_test_split(np.arange(len(Y)), test_size=0.5, random_state=seed)

    # Fit on training half
    X_train, Y_train, T_train = X[train_idx], Y[train_idx], T[train_idx]

    m1 = RandomForestRegressor(n_estimators=n_estimators, n_jobs=-1, random_state=seed)
    m0 = RandomForestRegressor(n_estimators=n_estimators, n_jobs=-1, random_state=seed)
    m1.fit(X_train[T_train == 1], Y_train[T_train == 1])
    m0.fit(X_train[T_train == 0], Y_train[T_train == 0])

    # Estimate on held-out half
    X_est = X[est_idx]
    tau_est = m1.predict(X_est) - m0.predict(X_est)

    # Bootstrap SE
    boot_taus = []
    for b in range(200):
        bi = np.random.choice(len(train_idx), size=len(train_idx), replace=True)
        Xb, Yb, Tb = X_train[bi], Y_train[bi], T_train[bi]
        if Tb.sum() < 5 or (~Tb.astype(bool)).sum() < 5:
            continue
        m1b = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=b)
        m0b = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=b)
        m1b.fit(Xb[Tb == 1], Yb[Tb == 1])
        m0b.fit(Xb[Tb == 0], Yb[Tb == 0])
        boot_taus.append(m1b.predict(X_est) - m0b.predict(X_est))

    tau_se = np.std(boot_taus, axis=0, ddof=1) if boot_taus else np.ones_like(tau_est) * np.nan

    df_est = df.iloc[est_idx].copy()
    df_est["cate"]     = tau_est
    df_est["cate_se"]  = tau_se
    df_est["cate_low"] = tau_est - 1.96 * tau_se
    df_est["cate_high"]= tau_est + 1.96 * tau_se

    feat_imp = pd.Series(
        (m1.feature_importances_ + m0.feature_importances_) / 2,
        index=feature_cols,
    ).sort_values(ascending=False)

    zone_cate = (
        df_est.groupby("zone_id")[["cate", "cate_low", "cate_high", "cate_se"]]
        .mean()
        .reset_index()
    )

    blp = _best_linear_projection(tau_est, X_est, feature_cols)
    policy_tree = _fit_policy_tree(tau_est, X_est, feature_cols, max_depth=2)

    logger.info(
        f"  Mean CATE = {tau_est.mean():.4f}  Std CATE = {tau_est.std():.4f}"
    )

    return {
        "cate_estimates": df_est[["zone_id", "date", "cate", "cate_low",
                                   "cate_high", "cate_se"]],
        "zone_cate":       zone_cate,
        "feature_importance": feat_imp,
        "blp":             blp,
        "policy_tree":     policy_tree,
        "rate":            None,
        "model":           (m1, m0),
        "outcome":         outcome,
        "mean_cate":       float(tau_est.mean()),
        "std_cate":        float(tau_est.std()),
    }


def run_causal_forest(panel, feature_cols, outcome="log_trip_count",
                      treatment="treat_binary", **kwargs) -> dict:
    """Dispatch to EconML or fallback based on availability."""
    if ECONML_AVAILABLE:
        return run_causal_forest_econml(panel, feature_cols, outcome, treatment, **kwargs)
    else:
        return run_causal_forest_fallback(panel, feature_cols, outcome, treatment, **kwargs)


# ── Supporting analyses ───────────────────────────────────────────────────────

def _best_linear_projection(
    tau_hat: np.ndarray,
    X: np.ndarray,
    feature_names: List[str],
) -> pd.DataFrame:
    """
    Project CATE estimates onto features to find which X dimensions
    drive the most heterogeneity.
    Best Linear Projection: E[τ(x)] = β₀ + X β
    """
    lr = LinearRegression()
    lr.fit(X, tau_hat)
    return pd.DataFrame({
        "feature":    feature_names,
        "coefficient": lr.coef_,
    }).sort_values("coefficient", key=abs, ascending=False)


def _fit_policy_tree(
    tau_hat: np.ndarray,
    X: np.ndarray,
    feature_names: List[str],
    max_depth: int = 2,
) -> str:
    """
    Fit a shallow decision tree to CATE estimates.
    Returns human-readable rule text.
    """
    tree = DecisionTreeRegressor(max_depth=max_depth, random_state=0)
    tree.fit(X, tau_hat)
    return export_text(tree, feature_names=feature_names)


def _compute_rate(
    tau_hat: np.ndarray,
    Y: np.ndarray,
    T: np.ndarray,
    X: np.ndarray,
    n_bootstrap: int = 100,
) -> dict:
    """
    RATE (Rank-Weighted Average Treatment Effect) measures:
    how much do we gain by targeting treatment to high-CATE units?

    Computes Targeting Operator Characteristic (TOC) curve.
    """
    n = len(tau_hat)
    rank_order = np.argsort(-tau_hat)  # descending CATE

    # TOC curve: fraction targeted q → average Y(1)-Y(0) for top-q fraction
    fracs = np.linspace(0.1, 1.0, 10)
    toc = []
    for q in fracs:
        k = int(np.ceil(q * n))
        top_idx = rank_order[:k]
        if T[top_idx].sum() > 0 and (~T[top_idx].astype(bool)).sum() > 0:
            ate_q = (Y[top_idx][T[top_idx] == 1].mean()
                     - Y[top_idx][T[top_idx] == 0].mean())
        else:
            ate_q = np.nan
        toc.append({"fraction": q, "ate_top_q": ate_q})

    return {
        "toc_curve": pd.DataFrame(toc),
        "rate_score": float(np.nanmean([r["ate_top_q"] for r in toc])),
    }


# ── Heterogeneity decomposition ───────────────────────────────────────────────

def cate_by_group(
    cate_df: pd.DataFrame,
    group_var: str,
    panel: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate CATE by a grouping variable (borough, peak_flag, etc.)
    with bootstrapped confidence intervals.
    """
    merged = cate_df.merge(panel[["zone_id", "date", group_var]], on=["zone_id", "date"])
    return (
        merged.groupby(group_var)["cate"]
        .agg(
            mean_cate="mean",
            std_cate="std",
            n="count",
        )
        .assign(se_cate=lambda d: d["std_cate"] / np.sqrt(d["n"]))
        .assign(ci_low=lambda d: d["mean_cate"] - 1.96 * d["se_cate"])
        .assign(ci_high=lambda d: d["mean_cate"] + 1.96 * d["se_cate"])
        .reset_index()
    )
