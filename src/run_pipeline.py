"""
run_pipeline.py
---------------
Master pipeline: runs all steps from raw data to final results.

Steps:
  0. Download TLC data + auxiliary files
  1. Preprocess raw parquets
  2. Build zone-day panel
  3. Enrich panel: zone lookup + CBD treatment assignment
  4. Build feature matrix
  5. L1: Naive DiD
  6. L1b: Event Study
  7. L2: Callaway-Sant'Anna DiD
  8. L3: Continuous Treatment DiD + dose-response
  9. L4: Synthetic DiD
  10. L5a: Double ML
  11. L5b: Causal Forest
  12. Robustness checks
  13. Generate all tables + figures
  14. Print executive summary

Usage:
  python src/run_pipeline.py                    # full run
  python src/run_pipeline.py --skip-download    # skip step 0 (data already present)
  python src/run_pipeline.py --steps 5,6,7      # run specific steps only
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml
from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _save_result(obj, name: str, results_dir: Path):
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / f"{name}.pkl", "wb") as f:
        pickle.dump(obj, f)
    if isinstance(obj, pd.DataFrame):
        obj.to_parquet(results_dir / f"{name}.parquet", index=False)
    logger.info(f"  ✓ Saved result: {name}")


def run_pipeline(
    config_path: str = "configs/default.yaml",
    skip_download: bool = False,
    steps: Optional[str] = None,
):
    # ── Config ────────────────────────────────────────────────────────────────
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    raw_dir     = Path(cfg["paths"]["raw_data"])
    proc_dir    = Path(cfg["paths"]["processed"])
    results_dir = Path(cfg["paths"]["results"])
    figures_dir = Path(cfg["paths"]["figures"])
    tables_dir  = Path(cfg["paths"]["tables"])

    run_steps = set(map(int, steps.split(","))) if steps else set(range(20))

    # ── Step 0: Download ──────────────────────────────────────────────────────
    if 0 in run_steps and not skip_download:
        logger.info("=== Step 0: Download TLC data ===")
        from src.data.download_tlc import download_trip_records, download_auxiliary
        download_trip_records(cfg["data"]["trip_types"], cfg["data"]["months"], raw_dir)
        download_auxiliary(raw_dir)

    # ── Step 1: Preprocess ────────────────────────────────────────────────────
    if 1 in run_steps:
        logger.info("=== Step 1: Preprocess raw parquets ===")
        from src.data.preprocess import process_all
        process_all(config_path)

    # ── Step 2: Build zone-day panel ──────────────────────────────────────────
    if 2 in run_steps:
        logger.info("=== Step 2: Build zone-day panel ===")
        from src.data.build_zone_panel import build_panel
        panel_path = proc_dir / "zone_day_panel.parquet"
        build_panel(proc_dir, cfg["data"]["trip_types"], panel_path,
                    min_trips=cfg["data"]["min_trips_per_cell"])

    # ── Step 3: Enrich panel ──────────────────────────────────────────────────
    if 3 in run_steps:
        logger.info("=== Step 3: Enrich panel (geo + treatment) ===")
        panel_path = proc_dir / "zone_day_panel.parquet"
        panel = pd.read_parquet(panel_path)
        panel["date"] = pd.to_datetime(panel["date"])

        aux_dir = raw_dir / "auxiliary"
        csv_path = aux_dir / "taxi_zone_lookup.csv"
        shp_dir  = aux_dir / "taxi_zones_shp"

        if csv_path.exists():
            from src.geospatial.zone_lookup import enrich_panel
            panel = enrich_panel(
                panel, str(csv_path),
                shapefile_dir=str(shp_dir) if shp_dir.exists() else None,
            )

        from src.geospatial.cbd_mapping import assign_treatment, assign_continuous_treatment
        panel = assign_treatment(panel)
        panel = assign_continuous_treatment(panel, cfg["treatment"]["fee_col"])

        out = proc_dir / "zone_day_panel_enriched.parquet"
        panel.to_parquet(out, index=False)
        logger.info(f"  ✓ Enriched panel: {out}")

    # ── Step 4: Feature matrix ────────────────────────────────────────────────
    if 4 in run_steps:
        logger.info("=== Step 4: Build feature matrix ===")
        panel = pd.read_parquet(proc_dir / "zone_day_panel_enriched.parquet")
        panel["date"] = pd.to_datetime(panel["date"])
        from src.features.build_features import build_feature_matrix
        panel, feature_cols = build_feature_matrix(panel)
        panel.to_parquet(proc_dir / "zone_day_panel_features.parquet", index=False)
        _save_result(feature_cols, "feature_cols", results_dir)

    # ── Load panel + features ─────────────────────────────────────────────────
    feat_path = proc_dir / "zone_day_panel_features.parquet"
    if feat_path.exists():
        panel = pd.read_parquet(feat_path)
        panel["date"] = pd.to_datetime(panel["date"])
        with open(results_dir / "feature_cols.pkl", "rb") as f:
            feature_cols = pickle.load(f)
    elif (proc_dir / "zone_day_panel_enriched.parquet").exists():
        panel = pd.read_parquet(proc_dir / "zone_day_panel_enriched.parquet")
        panel["date"] = pd.to_datetime(panel["date"])
        feature_cols = []
    else:
        logger.error("No panel found. Run steps 0–4 first.")
        return

    # ── Step 5: L1 Naive DiD ──────────────────────────────────────────────────
    if 5 in run_steps:
        logger.info("=== Step 5: L1 Naive DiD ===")
        from src.causal.run_did import run_naive_did
        outcomes = cfg["outcomes"]["secondary"] + [cfg["outcomes"]["primary"]]
        l1_all = {}
        for outcome in outcomes:
            if outcome in panel.columns:
                l1_all[outcome] = run_naive_did(panel, outcome=outcome)
        l1 = l1_all.get(cfg["outcomes"]["primary"], {})
        _save_result(l1,     "l1_did",     results_dir)
        _save_result(l1_all, "l1_did_all", results_dir)

    # ── Step 6: Event Study ────────────────────────────────────────────────────
    if 6 in run_steps:
        logger.info("=== Step 6: Event Study ===")
        from src.causal.run_did import run_event_study
        es_df = run_event_study(
            panel,
            outcome=cfg["outcomes"]["primary"],
            window=cfg["policy"]["event_window_weeks"],
        )
        _save_result(es_df, "event_study", results_dir)

    # ── Step 7: CS-DiD ────────────────────────────────────────────────────────
    if 7 in run_steps:
        logger.info("=== Step 7: Callaway-Sant'Anna DiD ===")
        from src.causal.run_did import run_callaway_santanna
        cs = run_callaway_santanna(
            panel,
            outcome=cfg["outcomes"]["primary"],
            anticipation=cfg["causal"]["callaway_santanna"]["anticipation"],
        )
        _save_result(cs, "cs_did", results_dir)

    # ── Step 8: Continuous DiD ────────────────────────────────────────────────
    if 8 in run_steps:
        logger.info("=== Step 8: Continuous Treatment DiD ===")
        from src.causal.run_continuous_did import run_continuous_did, run_dose_response_by_bin
        cont = run_continuous_did(
            panel,
            outcome=cfg["outcomes"]["primary"],
            fee_col=cfg["causal"]["continuous_did"]["dose_var"],
        )
        bins = run_dose_response_by_bin(
            panel,
            outcome=cfg["outcomes"]["primary"],
            n_bins=cfg["causal"]["continuous_did"]["bins"],
        )
        _save_result(cont, "continuous_did", results_dir)
        _save_result(bins, "dose_bins",      results_dir)

    # ── Step 9: SDID ──────────────────────────────────────────────────────────
    if 9 in run_steps:
        logger.info("=== Step 9: Synthetic DiD ===")
        from src.causal.run_sdid import run_sdid
        sdid = run_sdid(panel, outcome=cfg["outcomes"]["primary"])
        _save_result(sdid, "sdid", results_dir)

    # ── Step 10: DML ──────────────────────────────────────────────────────────
    if 10 in run_steps:
        logger.info("=== Step 10: Double ML ===")
        if not feature_cols:
            logger.warning("  feature_cols empty, skipping DML")
        else:
            from src.causal.run_dml import run_dml
            outcomes = [o for o in cfg["causal"]["dml"]["outcomes"]
                        if o in panel.columns]
            dml_df = run_dml(
                panel, feature_cols, outcomes,
                treatment=cfg["treatment"].get("treatment_col", "treat_binary"),
                learner=cfg["causal"]["dml"]["ml_learner"],
                n_folds=cfg["causal"]["dml"]["n_folds"],
                n_reps=cfg["causal"]["dml"]["n_reps"],
            )
            _save_result(dml_df, "dml_results", results_dir)

    # ── Step 11: Causal Forest ─────────────────────────────────────────────────
    if 11 in run_steps:
        logger.info("=== Step 11: Causal Forest ===")
        if not feature_cols:
            logger.warning("  feature_cols empty, skipping Causal Forest")
        else:
            from src.causal.run_causal_forest import run_causal_forest, cate_by_group
            cf_cfg = cfg["causal"]["causal_forest"]

            for outcome in cf_cfg["outcomes"]:
                if outcome not in panel.columns:
                    continue
                cf = run_causal_forest(
                    panel, feature_cols, outcome=outcome,
                    n_estimators=cf_cfg["n_estimators"],
                )
                _save_result(cf["zone_cate"],          f"zone_cate_{outcome}",    results_dir)
                _save_result(cf["feature_importance"], f"feature_importance_{outcome}", results_dir)
                _save_result(cf["cate_estimates"],     f"cate_estimates_{outcome}", results_dir)
                if cf.get("rate"):
                    _save_result(cf["rate"]["toc_curve"], f"toc_curve_{outcome}", results_dir)

                # Heterogeneity tables
                for grp in ["borough", "is_weekend", "is_peak"]:
                    if grp in panel.columns:
                        grp_df = cate_by_group(cf["cate_estimates"], grp, panel)
                        _save_result(grp_df, f"cate_by_{grp}_{outcome}", results_dir)

                # Save primary outcome results under canonical names
                if outcome == cfg["outcomes"]["primary"].replace("log_", ""):
                    _save_result(cf["zone_cate"],          "zone_cate",          results_dir)
                    _save_result(cf["feature_importance"], "feature_importance",  results_dir)

    # ── Step 12: Robustness ────────────────────────────────────────────────────
    if 12 in run_steps:
        logger.info("=== Step 12: Robustness checks ===")
        from src.causal.robustness import run_all_robustness
        rob = run_all_robustness(panel, feature_cols, cfg["outcomes"]["primary"])
        _save_result(rob, "robustness", results_dir)

    # ── Step 13: Figures + Tables ─────────────────────────────────────────────
    if 13 in run_steps:
        logger.info("=== Step 13: Generate figures + tables ===")
        from src.evaluation.plots import (
            plot_event_study, plot_cs_did, plot_dose_response,
            plot_dose_bins, plot_dml_results, plot_cate_diagnostics,
            plot_rate_curve, plot_robustness,
        )
        from src.evaluation.tables import (
            make_main_did_table, make_dml_table, make_robustness_table,
            print_executive_summary,
        )

        def _load(name):
            p = results_dir / f"{name}.pkl"
            return pickle.load(open(p, "rb")) if p.exists() else None

        es_df  = _load("event_study")
        cs     = _load("cs_did")
        cont   = _load("continuous_did")
        bins   = _load("dose_bins")
        dml_df = _load("dml_results")
        sdid   = _load("sdid")
        l1     = _load("l1_did")
        rob    = _load("robustness")
        fi     = _load("feature_importance")
        zcate  = _load("zone_cate")
        rate   = _load(f"toc_curve_{cfg['outcomes']['primary']}")

        if es_df is not None:
            plot_event_study(es_df, out_path=figures_dir / "F1_event_study.png")
        if cs and isinstance(cs, dict) and "att_gt" in cs:
            plot_cs_did(cs["att_gt"], out_path=figures_dir / "F2_cs_did.png")
        if cont and isinstance(cont, dict):
            plot_dose_response(cont["dose_response"], cont["beta_linear"],
                               out_path=figures_dir / "F3_dose_response.png")
        if bins is not None and len(bins) > 0:
            plot_dose_bins(bins, out_path=figures_dir / "F4_dose_bins.png")
        if dml_df is not None:
            plot_dml_results(dml_df, out_path=figures_dir / "F5_dml_results.png")
        if zcate is not None and fi is not None:
            plot_cate_diagnostics(
                pd.read_parquet(results_dir / "zone_cate.parquet")["cate"],
                fi,
                out_path=figures_dir / "F7_cate_diagnostics.png",
            )
        if rate is not None:
            plot_rate_curve(
                rate, rate_score=rate["ate_top_q"].mean(),
                out_path=figures_dir / "F9_rate_curve.png",
            )
        if rob is not None and l1 is not None:
            plot_robustness(
                rob, main_estimate=l1.get("ate", 0), main_se=l1.get("se", 0),
                out_path=figures_dir / "F10_robustness.png",
            )

        # Tables
        if l1 and cs and sdid:
            make_main_did_table(l1, cs, sdid,
                                out_path=tables_dir / "T1_main_did")
        if dml_df is not None:
            make_dml_table(dml_df, out_path=tables_dir / "T2_dml")
        if rob is not None and l1 is not None:
            make_robustness_table(rob, l1.get("ate", 0),
                                  out_path=tables_dir / "T4_robustness")

    # ── Step 14: Executive summary ─────────────────────────────────────────────
    if 14 in run_steps:
        logger.info("=== Step 14: Executive Summary ===")

        def _load(name):
            p = results_dir / f"{name}.pkl"
            return pickle.load(open(p, "rb")) if p.exists() else {}

        l1     = _load("l1_did")
        dml_df = _load("dml_results")
        cf     = _load("zone_cate")

        if isinstance(dml_df, dict):
            dml_df = pd.DataFrame(dml_df)

        from src.evaluation.tables import print_executive_summary
        if l1 and dml_df is not None:
            cf_stub = {
                "mean_cate": float(cf["cate"].mean()) if isinstance(cf, pd.DataFrame) else 0,
                "std_cate":  float(cf["cate"].std())  if isinstance(cf, pd.DataFrame) else 0,
                "feature_importance": pd.Series({"distance_to_cbd_km": 1}),
            }
            print_executive_summary(l1, dml_df, cf_stub)

    logger.info("✅ Pipeline complete.")


if __name__ == "__main__":
    import typer

    def main(
        config: str = "configs/default.yaml",
        skip_download: bool = False,
        steps: Optional[str] = None,
    ):
        run_pipeline(config, skip_download, steps)

    typer.run(main)
