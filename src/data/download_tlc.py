"""
download_tlc.py
---------------
Download NYC TLC trip record parquet files from the official AWS S3 bucket.
Covers Yellow Taxi, Green Taxi, and HVFHV (Uber/Lyft) for the study window.

TLC data URL pattern:
  https://d37ci6vzurychx.cloudfront.net/trip-data/{type}_tripdata_{YYYY-MM}.parquet
"""

import os
import time
import requests
from pathlib import Path
from typing import List
from loguru import logger
from tqdm import tqdm
import yaml


BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data"
ZONE_LOOKUP_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"
ZONE_SHAPE_URL  = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip"

TYPE_PREFIX = {
    "yellow": "yellow_tripdata",
    "green":  "green_tripdata",
    "fhvhv":  "fhv_tripdata",   # High-Volume FHV (Uber/Lyft)
}


def load_config(path: str = "configs/default.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _download_file(url: str, dest: Path, retries: int = 3, backoff: float = 5.0) -> bool:
    """Stream-download a single file with retry logic."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        logger.info(f"  ✓ Already exists: {dest.name}")
        return True

    for attempt in range(1, retries + 1):
        try:
            logger.info(f"  ↓ Downloading {dest.name}  (attempt {attempt})")
            resp = requests.get(url, stream=True, timeout=120)
            if resp.status_code == 404:
                logger.warning(f"  ✗ 404 Not Found: {url}")
                return False
            resp.raise_for_status()

            total = int(resp.headers.get("content-length", 0))
            with open(dest, "wb") as fh, tqdm(
                total=total, unit="B", unit_scale=True,
                desc=dest.name, leave=False,
            ) as bar:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)
                    bar.update(len(chunk))
            return True

        except Exception as exc:
            logger.warning(f"  ✗ Attempt {attempt} failed: {exc}")
            if attempt < retries:
                time.sleep(backoff * attempt)

    logger.error(f"  ✗ Failed after {retries} attempts: {url}")
    return False


def download_trip_records(
    trip_types: List[str],
    months: List[str],
    raw_dir: Path,
) -> dict:
    """
    Download parquet files for all combinations of trip_type × month.

    Returns a manifest dict: {(type, month): local_path}
    """
    manifest = {}

    for trip_type in trip_types:
        prefix = TYPE_PREFIX[trip_type]
        type_dir = raw_dir / trip_type
        type_dir.mkdir(parents=True, exist_ok=True)

        for month in months:
            filename = f"{prefix}_{month}.parquet"
            url = f"{BASE_URL}/{filename}"
            dest = type_dir / filename

            ok = _download_file(url, dest)
            if ok:
                manifest[(trip_type, month)] = dest

    return manifest


def download_auxiliary(raw_dir: Path) -> dict:
    """Download zone lookup CSV and shapefile ZIP."""
    aux_dir = raw_dir / "auxiliary"
    aux_dir.mkdir(parents=True, exist_ok=True)

    paths = {}

    lookup_dest = aux_dir / "taxi_zone_lookup.csv"
    if _download_file(ZONE_LOOKUP_URL, lookup_dest):
        paths["zone_lookup"] = lookup_dest

    shape_dest = aux_dir / "taxi_zones.zip"
    if _download_file(ZONE_SHAPE_URL, shape_dest):
        # unzip
        import zipfile
        shape_dir = aux_dir / "taxi_zones_shp"
        shape_dir.mkdir(exist_ok=True)
        with zipfile.ZipFile(shape_dest, "r") as zf:
            zf.extractall(shape_dir)
        paths["zone_shapefile"] = shape_dir
        logger.info(f"  ✓ Shapefile extracted to {shape_dir}")

    return paths


def main():
    import typer

    def run(config: str = "configs/default.yaml"):
        cfg = load_config(config)
        raw_dir = Path(cfg["paths"]["raw_data"])

        logger.info("=== Downloading TLC Trip Records ===")
        manifest = download_trip_records(
            trip_types=cfg["data"]["trip_types"],
            months=cfg["data"]["months"],
            raw_dir=raw_dir,
        )
        logger.info(f"Downloaded {len(manifest)} trip record files.")

        logger.info("=== Downloading Auxiliary Files ===")
        aux = download_auxiliary(raw_dir)
        logger.info(f"Auxiliary files: {list(aux.keys())}")

    typer.run(run)


if __name__ == "__main__":
    main()
