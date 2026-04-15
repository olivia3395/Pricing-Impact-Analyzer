"""
build_zone_panel.py
-------------------
Aggregate cleaned trip-level parquets into a balanced zone × day panel.

Output schema (one row = one zone × day):
  zone_id, date, trip_count, avg_distance, avg_duration, avg_fare,
  surcharge_total, avg_cbd_fee, cbd_fee_total, airport_share,
  peak_share, weekend_flag, borough, trip_type_agg
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import duckdb
import pandas as pd
import numpy as np
from loguru import logger
import yaml


def _get_cleaned_files(proc_dir: Path, trip_types: List[str]) -> List[Path]:
    files = []
    for tt in trip_types:
        files.extend(sorted((proc_dir / tt).glob("clean_*.parquet")))
    return files


def build_panel(
    proc_dir: Path,
    trip_types: List[str],
    out_path: Path,
    min_trips: int = 5,
) -> pd.DataFrame:
    """
    Use DuckDB for fast aggregation over all cleaned parquets.
    Returns zone × day panel DataFrame and writes to out_path.
    """
    files = _get_cleaned_files(proc_dir, trip_types)
    if not files:
        raise FileNotFoundError(f"No cleaned parquets found in {proc_dir}")

    logger.info(f"Building panel from {len(files)} files...")

    # Build DuckDB query over all files at once
    file_list = ", ".join(f"'{str(f)}'" for f in files)

    con = duckdb.connect()
    panel = con.execute(f"""
        SELECT
            pickup_zone                             AS zone_id,
            pickup_date                             AS date,
            COUNT(*)                                AS trip_count,
            AVG(trip_distance)                      AS avg_distance,
            AVG(trip_duration_min)                  AS avg_duration,
            AVG(fare_amount)                        AS avg_fare,
            SUM(surcharge_total)                    AS surcharge_sum,
            AVG(cbd_congestion_fee)                 AS avg_cbd_fee,
            SUM(cbd_congestion_fee)                 AS cbd_fee_total,
            AVG(is_airport_trip)                    AS airport_share,
            AVG(is_peak)                            AS peak_share,
            MAX(is_weekend)                         AS weekend_flag,
            FIRST(trip_type)                        AS trip_type_agg
        FROM read_parquet([{file_list}])
        GROUP BY pickup_zone, pickup_date
        HAVING COUNT(*) >= {min_trips}
        ORDER BY zone_id, date
    """).df()

    con.close()

    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.sort_values(["zone_id", "date"]).reset_index(drop=True)

    # Log-transform trip_count (common for count outcomes in panel regressions)
    panel["log_trip_count"] = np.log1p(panel["trip_count"])

    # Week and relative-week columns (for event study)
    SHOCK = pd.Timestamp("2025-01-05")
    panel["week"] = panel["date"].dt.to_period("W").apply(lambda p: p.start_time)
    panel["rel_week"] = ((panel["week"] - SHOCK).dt.days // 7).astype(int)
    panel["post"] = (panel["date"] >= SHOCK).astype(int)

    # Year-month label
    panel["year_month"] = panel["date"].dt.to_period("M").astype(str)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(out_path, index=False, compression="snappy")
    logger.info(f"✓ Panel written: {out_path}  shape={panel.shape}")
    return panel


def load_panel(out_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(out_path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def main(config_path: str = "configs/default.yaml"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    proc_dir = Path(cfg["paths"]["processed"])
    out_path = Path(cfg["paths"]["processed"]) / "zone_day_panel.parquet"

    build_panel(
        proc_dir=proc_dir,
        trip_types=cfg["data"]["trip_types"],
        out_path=out_path,
        min_trips=cfg["data"]["min_trips_per_cell"],
    )


if __name__ == "__main__":
    import typer
    typer.run(main)
