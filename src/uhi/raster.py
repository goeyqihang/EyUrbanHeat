"""GeoTIFF helpers: write band stacks, read them back as one row per pixel, and plot them."""

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_bounds

from . import config


def write_geotiff(path, bands, bounds, crs="EPSG:4326"):
    """
    Write equally-shaped 2-D arrays as one float64 GeoTIFF (LZW-compressed).

    Parameters:
      - path: output file
      - bands: dict {band name: 2-D array}, written in this order; the names become band descriptions
      - bounds: (min_lon, min_lat, max_lon, max_lat) covered by the arrays
    """
    arrays = list(bands.values())
    height, width = arrays[0].shape
    transform = from_bounds(*bounds, width, height)
    with rasterio.open(path, "w", driver="GTiff", width=width, height=height, crs=crs,
                       transform=transform, count=len(arrays), compress="lzw", dtype="float64") as dst:
        for i, (name, array) in enumerate(bands.items(), start=1):
            dst.write(np.asarray(array, dtype="float64"), i)
            dst.set_band_description(i, name)
    return path


def extract_bands(tiff_path, band_names):
    """
    Read every band of a GeoTIFF into a DataFrame with one row per pixel.

    Parameters:
      - tiff_path: path to the GeoTIFF
      - band_names: column names in band order (any extra band is named "Band_<n>")

    Returns:
      - DataFrame of shape (height * width, n_bands), pixels in row-major order
    """
    with rasterio.open(tiff_path) as src:
        bands_data = src.read()  # shape (bands, height, width)

    # Flatten every band to a 1-D array, shape (bands, height * width)
    bands_flat = bands_data.reshape(bands_data.shape[0], -1)

    df = pd.DataFrame({})
    for i in range(bands_data.shape[0]):
        col_name = band_names[i] if i < len(band_names) else f"Band_{i + 1}"
        df[col_name] = bands_flat[i]
    return df


def extract_coordinates(tiff_path):
    """Longitude / latitude of every pixel centre, in the same row-major order as extract_bands()."""
    with rasterio.open(tiff_path) as src:
        transform = src.transform
        height, width = src.height, src.width

    rows, cols = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    xs, ys = rasterio.transform.xy(transform, rows, cols)
    return pd.DataFrame({"Longitude": np.array(xs).flatten(), "Latitude": np.array(ys).flatten()})


def pixel_table(processed_dir=config.PROCESSED_DIR, method="nearest"):
    """
    Combine the six GeoTIFFs of notebook 01 into one table with one row per 10 m pixel.

    Columns: Longitude, Latitude, the Sentinel-2 bands (<band>_single, <band>_median), the Landsat
    bands (<band>_single, <band>_median) and the MODIS composites (<band>_mean).
    """
    names = config.geotiff_names(method)
    path = {key: processed_dir / name for key, name in names.items()}
    parts = [
        extract_coordinates(path["landsat_single"]),
        extract_bands(path["s2_single"], config.S2_BANDS).add_suffix("_single"),
        extract_bands(path["s2_median"], config.S2_BANDS).add_suffix("_median"),
        extract_bands(path["landsat_single"], config.LANDSAT_BANDS).add_suffix("_single"),
        extract_bands(path["landsat_median"], config.LANDSAT_BANDS).add_suffix("_median"),
        extract_bands(path["modis_lst"], config.MODIS_LST_BANDS).add_suffix("_mean"),
        extract_bands(path["modis_vi"], config.MODIS_VI_BANDS).add_suffix("_mean"),
    ]
    return pd.concat(parts, axis=1)


def plot_bands(tiff_path, band_labels, cmap="viridis", n_cols=3):
    """Plot every band of a GeoTIFF on a grid of subplots (n_cols per row)."""
    with rasterio.open(tiff_path) as src:
        band_data = [src.read(i + 1) for i in range(len(band_labels))]

    n_plots = len(band_data)
    n_rows = math.ceil(n_plots / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
    axes = np.atleast_1d(axes).flatten()

    for ax, data, label in zip(axes, band_data, band_labels, strict=False):  # axes may outnumber bands
        im = ax.imshow(data, cmap=cmap)
        ax.set_title(label, fontsize=10)
        fig.colorbar(im, ax=ax)

    # Hide the unused subplots
    for ax in axes[n_plots:]:
        ax.axis("off")

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    plt.show()
