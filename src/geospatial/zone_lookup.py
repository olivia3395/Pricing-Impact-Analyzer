"""
zone_lookup.py
--------------
Load TLC zone lookup CSV and shapefile.
Provides zone_id → borough / zone_name / geometry / distance_to_cbd.
"""

from __future__ import annotations

from pathlib import Path
from functools import lru_cache

import pandas as pd
import numpy as np
import geopandas as gpd
from loguru import logger


BOROUGH_CODE = {
    "Manhattan":    1,
    "Bronx":        2,
    "Brooklyn":     3,
    "Queens":       4,
    "Staten Island":5,
    "EWR":          6,
}

# Approximate centroid of NYC Congestion Relief Zone (mid-Manhattan)
CBD_CENTROID_LAT = 40.754
CBD_CENTROID_LON = -73.984


@lru_cache(maxsize=1)
def load_zone_lookup(csv_path: str) -> pd.DataFrame:
    """
    Load taxi_zone_lookup.csv.
    Returns DataFrame indexed by LocationID with columns:
      borough, zone, service_zone, borough_code
    """
    df = pd.read_csv(csv_path)
    df = df.rename(columns={
        "LocationID": "zone_id",
        "Borough":    "borough",
        "Zone":       "zone_name",
        "service_zone": "service_zone",
    })
    df["borough_code"] = df["borough"].map(BOROUGH_CODE).fillna(0).astype(int)
    df = df.set_index("zone_id")
    logger.info(f"Zone lookup loaded: {len(df)} zones")
    return df


@lru_cache(maxsize=1)
def load_zone_geodata(shapefile_dir: str) -> gpd.GeoDataFrame:
    """
    Load TLC taxi zones shapefile.
    Returns GeoDataFrame with zone_id, geometry, centroid_lat/lon,
    distance_to_cbd_km.
    """
    shp_path = next(Path(shapefile_dir).glob("*.shp"), None)
    if shp_path is None:
        raise FileNotFoundError(f"No .shp in {shapefile_dir}")

    gdf = gpd.read_file(shp_path)
    gdf = gdf.rename(columns={"LocationID": "zone_id", "zone": "zone_name"})
    gdf["zone_id"] = gdf["zone_id"].astype(int)

    # Project to WGS84 for lat/lon centroids
    gdf_wgs = gdf.to_crs(epsg=4326)
    gdf["centroid_lat"] = gdf_wgs.geometry.centroid.y
    gdf["centroid_lon"] = gdf_wgs.geometry.centroid.x

    # Haversine distance to CBD centroid
    gdf["distance_to_cbd_km"] = gdf.apply(
        lambda r: _haversine(r["centroid_lat"], r["centroid_lon"],
                             CBD_CENTROID_LAT, CBD_CENTROID_LON),
        axis=1,
    )

    logger.info(f"Zone geodata loaded: {len(gdf)} zones")
    return gdf.set_index("zone_id")


def _haversine(lat1, lon1, lat2, lon2) -> float:
    """Return great-circle distance in km."""
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def enrich_panel(
    panel: pd.DataFrame,
    csv_path: str,
    shapefile_dir: str | None = None,
) -> pd.DataFrame:
    """
    Merge zone metadata into zone-day panel.
    Adds: borough, borough_code, zone_name, [distance_to_cbd_km].
    """
    lookup = load_zone_lookup(csv_path)
    panel = panel.join(
        lookup[["borough", "borough_code", "zone_name"]],
        on="zone_id", how="left",
    )

    if shapefile_dir:
        geo = load_zone_geodata(shapefile_dir)
        panel = panel.join(
            geo[["centroid_lat", "centroid_lon", "distance_to_cbd_km"]],
            on="zone_id", how="left",
        )

    return panel
