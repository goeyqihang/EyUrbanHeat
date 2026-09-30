"""
End-to-end pipeline: GeoTIFFs + challenge files -> features -> XGBoost -> submission CSV.

Usage:
    python -m uhi.pipeline                      # defaults from uhi.config
    python -m uhi.pipeline --output outputs/submission.csv

The GeoTIFFs must exist in data/processed (notebooks/01_satellite_data_extraction.ipynb) and the
challenge files in data/raw (see data/README.md). Feature selection is not re-run: the model uses
config.SELECTED_FEATURES.
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, modeling, raster
from .features import buildings, spatial, spectral, weather

log = logging.getLogger("uhi")


def build_pixel_features(processed_dir=config.PROCESSED_DIR, raw_dir=config.RAW_DIR, method="nearest"):
    """One row per 10 m pixel: satellite bands, station weather, spectral indices and interactions."""
    pixels = raster.pixel_table(processed_dir, method)
    stations = pd.ExcelFile(raw_dir / config.WEATHER_FILE)
    bronx = weather.survey_hour_means(weather.read_station(stations, "Bronx"))
    manhattan = weather.survey_hour_means(weather.read_station(stations, "Manhattan"))
    weather.add_station_weather(pixels, bronx, manhattan)

    pixels.dropna(inplace=True)
    spectral.add_spectral_indices(pixels)
    spectral.add_temporal_change(pixels)
    weather.add_interactions(pixels)
    # Drop rows with NaN / inf (e.g. from divisions by zero)
    return pixels[np.isfinite(pixels).all(axis=1)]


def add_neighbourhood_features(pixels, footprints):
    """Building features and 100 m focal means for the (filtered) pixels. footprints in EPSG:3857."""
    pixels = buildings.nearest_building_distance(pixels, footprints)
    pixels = buildings.buffer_stats(pixels, buildings.shape_metrics(footprints), radius=config.BUILDING_RADIUS)
    return spatial.focal_means(pixels, config.FOCAL_FEATURES, radius=config.FOCAL_RADIUS,
                               chunk_size=config.FOCAL_CHUNK_SIZE)


def run(raw_dir=config.RAW_DIR, processed_dir=config.PROCESSED_DIR, method="nearest",
        output=config.OUTPUT_DIR / "submission.csv"):
    t0 = time.time()

    def step(message):
        log.info("[%6.0fs] %s", time.time() - t0, message)

    step("building the pixel feature table")
    pixels = build_pixel_features(processed_dir, raw_dir, method)
    ground = spatial.points_to_gdf(pd.read_csv(raw_dir / config.TRAINING_FILE))
    template = pd.read_csv(raw_dir / config.SUBMISSION_TEMPLATE_FILE)
    test = spatial.points_to_gdf(template)

    step(f"keeping pixels within {config.PIXEL_FILTER_RADIUS} of the {len(ground) + len(test):,} points")
    near, info = spatial.pixels_near_points(spatial.points_to_gdf(pixels), [ground, test],
                                            radius=config.PIXEL_FILTER_RADIUS)
    del pixels
    step(f"kept {info['kept']:,} of {info['pixels']:,} pixels; adding building and focal features")
    footprints = buildings.read_footprints(raw_dir / config.BUILDINGS_FILE).to_crs(epsg=3857)
    near = add_neighbourhood_features(near, footprints)

    step("mapping the pixel features onto the points")
    ground_mapped = spatial.map_nearest_pixel(ground[["Longitude", "Latitude", config.TARGET, "geometry"]],
                                              near).fillna(0)
    test_mapped = spatial.map_nearest_pixel(test, near).fillna(0)
    del near

    step("training XGBoost")
    trained = modeling.train_and_evaluate(modeling.prepare_training_table(ground_mapped))
    predictions = modeling.predict(trained, pd.DataFrame(test_mapped.drop(columns="geometry")))

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"Longitude": template["Longitude"].values, "Latitude": template["Latitude"].values,
                  config.TARGET: predictions}).to_csv(output, index=False)
    metrics = {"cv_r2": trained.cv_scores.tolist(), "cv_r2_mean": float(trained.cv_scores.mean()),
               "train_r2": float(trained.train_r2), "test_r2": float(trained.test_r2),
               "n_features": len(trained.features)}
    output.with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=2))
    step(f"hold-out R² {trained.test_r2:.4f}, CV R² {trained.cv_scores.mean():.4f} -> {output}")
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train the UHI model and predict the submission points.")
    parser.add_argument("--raw-dir", type=Path, default=config.RAW_DIR, help="challenge files")
    parser.add_argument("--processed-dir", type=Path, default=config.PROCESSED_DIR, help="GeoTIFFs from notebook 01")
    parser.add_argument("--method", default="nearest", help="resampling method in the GeoTIFF file names")
    parser.add_argument("--output", type=Path, default=config.OUTPUT_DIR / "submission.csv",
                        help="submission CSV (metrics are written next to it)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    run(args.raw_dir, args.processed_dir, args.method, args.output)


if __name__ == "__main__":
    main()
