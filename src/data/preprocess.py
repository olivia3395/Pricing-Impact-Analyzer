"""
preprocess.py
-------------
Clean and unify raw TLC parquet files across Yellow / Green / HVFHV schemas.
Outputs a standardised Parquet per (trip_type, month) to artifacts/processed/.

Unified schema
--------------
pickup_datetime, dropoff_datetime, pickup_zone, dropoff_zone,
trip_distance, fare_amount, tolls_amount, surcharge_total,
cbd_congestion_fee, passenger_count, trip_type, year_month
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger
import yaml


# ---------------------------------------------------------------------------
# Column maps per trip type
# ---------------------------------------------------------------------------

YELLOW_COLS = {
    "tpep_pickup_datetime":  "pickup_datetime",
    "tpep_dropoff_datetime": "dropoff_datetime",
    "PULocationID":          "pickup_zone",
    "DOLocationID":          "dropoff_zone",
    "trip_distance":         "trip_distance",
    "fare_amount":           "fare_amount",
    "tolls_amount":          "tolls_amount",
    "passenger_count":       "passenger_count",
    # 2025+ field
    "cbd_congestion_surcharge": "cbd_congestion_fee",   # may not exist pre-2025
}

GREEN_COLS = {
    "lpep_pickup_datetime":  "pickup_datetime",
    "lpep_dropoff_datetime": "dropoff_datetime",
    "PULocationID":          "pickup_zone",
    "DOLocationID":          "dropoff_zone",
    "trip_distance":         "trip_distance",
    "fare_amount":           "fare_amount",
    "tolls_amount":          "tolls_amount",
    "passenger_count":       "passenger_count",
    "cbd_congestion_surcharge": "cbd_congestion_fee",
}

FHVHV_COLS = {
    "pickup_datetime":       "pickup_datetime",
    "dropoff_datetime":      "dropoff_datetime",
    "PULocationID":          "pickup_zone",
    "DOLocationID":          "dropoff_zone",
    "trip_miles":            "trip_distance",
    "base_passenger_fare":   "fare_amount",
    "tolls":                 "tolls_amount",
    "congestion_surcharge":  "cbd_congestion_fee",   # HVFHV naming
    "bcf":                   "black_car_fund",        # keep for surcharge calc
    "sales_tax":             "sales_tax",
    "tips":                  "tips",
}

# Surcharge-related fields used to compute surcharge_total
SURCHARGE_FIELDS_YELLOW = ["improvement_surcharge", "extra", "mta_tax",
                            "congestion_surcharge", "airport_fee"]
SURCHARGE_FIELDS_FHVHV  = ["bcf", "sales_tax", "congestion_surcharge",
                             "airport_fee"]


def _rename_and_select(df: pd.DataFrame, col_map: dict) -> pd.DataFrame:
    """Rename columns using col_map; add missing ones as NaN."""
    present = {k: v for k, v in col_map.items() if k in df.columns}
    df = df.rename(columns=present)
    for tgt in set(col_map.values()):
        if tgt not in df.columns:
            df[tgt] = np.nan
    keep = list(set(col_map.values()))
    return df[[c for c in keep if c in df.columns]]


def _compute_surcharge_total(df: pd.DataFrame, trip_type: str) -> pd.Series:
    fields = (SURCHARGE_FIELDS_YELLOW
              if trip_type in ("yellow", "green") else SURCHARGE_FIELDS_FHVHV)
    cols = [c for c in fields if c in df.columns]
    return df[cols].clip(lower=0).sum(axis=1)


def _clean_common(df: pd.DataFrame, trip_type: str, year_month: str) -> pd.DataFrame:
    """Apply quality filters shared across all trip types."""
    n0 = len(df)

    # Parse datetimes
    df["pickup_datetime"]  = pd.to_datetime(df["pickup_datetime"],  errors="coerce")
    df["dropoff_datetime"] = pd.to_datetime(df["dropoff_datetime"], errors="coerce")

    # Drop null datetimes
    df = df.dropna(subset=["pickup_datetime", "dropoff_datetime"])

    # Keep only rows within the declared month (avoid cross-month leakage)
    ym = pd.Period(year_month, freq="M")
    mask = (df["pickup_datetime"].dt.to_period("M") == ym)
    df = df[mask]

    # Trip duration (minutes)
    df["trip_duration_min"] = (
        (df["dropoff_datetime"] - df["pickup_datetime"]).dt.total_seconds() / 60
    )

    # Sanity filters
    df = df[
        (df["trip_distance"]    > 0)    & (df["trip_distance"]    < 300)   &
        (df["trip_duration_min"] > 0.5) & (df["trip_duration_min"] < 360)  &
        (df["fare_amount"]      > 0)    & (df["fare_amount"]       < 1000) &
        (df["pickup_zone"].between(1, 265)) &
        (df["dropoff_zone"].between(1, 265))
    ]

    # Fill cbd_congestion_fee: NaN → 0 for pre-2025 records
    df["cbd_congestion_fee"] = df["cbd_congestion_fee"].fillna(0).clip(lower=0)

    # Surcharge total
    df["surcharge_total"] = _compute_surcharge_total(df, trip_type)

    # Meta
    df["trip_type"]   = trip_type
    df["year_month"]  = year_month
    df["pickup_date"] = df["pickup_datetime"].dt.date
    df["hour"]        = df["pickup_datetime"].dt.hour
    df["dow"]         = df["pickup_datetime"].dt.dayofweek   # 0=Mon
    df["is_weekend"]  = df["dow"].isin([5, 6]).astype(int)
    df["is_peak"]     = df["hour"].isin(list(range(7, 10)) + list(range(16, 20))).astype(int)

    # Airport flag: JFK=132, LGA=138, EWR=1
    airport_zones = {1, 132, 138}
    df["is_airport_trip"] = (
        df["pickup_zone"].isin(airport_zones) | df["dropoff_zone"].isin(airport_zones)
    ).astype(int)

    logger.info(f"  {trip_type} {year_month}: {n0:,} raw → {len(df):,} clean rows")
    return df


def process_file(
    parquet_path: Path,
    trip_type: str,
    year_month: str,
    out_dir: Path,
) -> Optional[Path]:
    """Process a single raw parquet file and write cleaned output."""
    out_path = out_dir / trip_type / f"clean_{year_month}.parquet"
    if out_path.exists():
        logger.info(f"  ✓ Already processed: {out_path.name}")
        return out_path

    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        df = pd.read_parquet(parquet_path)
    except Exception as exc:
        logger.error(f"  ✗ Cannot read {parquet_path}: {exc}")
        return None

    col_map = {
        "yellow": YELLOW_COLS,
        "green":  GREEN_COLS,
        "fhvhv":  FHVHV_COLS,
    }[trip_type]

    df = _rename_and_select(df, col_map)
    df = _clean_common(df, trip_type, year_month)

    final_cols = [
        "pickup_datetime", "dropoff_datetime", "pickup_date",
        "pickup_zone", "dropoff_zone",
        "trip_distance", "trip_duration_min",
        "fare_amount", "tolls_amount", "surcharge_total", "cbd_congestion_fee",
        "passenger_count",
        "hour", "dow", "is_weekend", "is_peak", "is_airport_trip",
        "trip_type", "year_month",
    ]
    df = df[[c for c in final_cols if c in df.columns]]
    df.to_parquet(out_path, index=False, compression="snappy")
    logger.info(f"  ✓ Written: {out_path}")
    return out_path


def process_all(config_path: str = "configs/default.yaml"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    raw_dir  = Path(cfg["paths"]["raw_data"])
    proc_dir = Path(cfg["paths"]["processed"])

    for trip_type in cfg["data"]["trip_types"]:
        for month in cfg["data"]["months"]:
            prefix = {
                "yellow": "yellow_tripdata",
                "green":  "green_tripdata",
                "fhvhv":  "fhv_tripdata",
            }[trip_type]
            src = raw_dir / trip_type / f"{prefix}_{month}.parquet"
            if not src.exists():
                logger.warning(f"  Missing raw file: {src}")
                continue
            process_file(src, trip_type, month, proc_dir)


if __name__ == "__main__":
    import typer
    typer.run(process_all)
