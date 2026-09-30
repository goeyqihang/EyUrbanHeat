"""
Satellite layers from the Microsoft Planetary Computer STAC API, resampled onto one 10 m grid.

- Sentinel-2 L2A: single-day scene (survey date) and median composite, 10 / 20 / 60 m bands
- Landsat 8 Collection 2 Level-2: single scene and median composite, 30 m bands + 100 m thermal band
- MODIS: 8-day land surface temperature / emissivity (1 km) and 16-day vegetation indices (250 m)

Every layer is reprojected onto the grid of the single-day Sentinel-2 10 m bands.
"""

import numpy as np
import planetary_computer
import pystac_client
import rioxarray  # noqa: F401  (registers the .rio accessor used for reprojection)
import xarray as xr
from dask.diagnostics import ProgressBar
from odc.stac import stac_load

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
SCALE_10M = 10 / 111320.0  # 10 m expressed in degrees (EPSG:4326)

S2_10M_BANDS = ["B02", "B03", "B04", "B08", "AOT", "WVP", "visual"]
S2_20M_BANDS = ["B05", "B06", "B07", "B8A", "B11", "B12", "SCL"]
S2_60M_BANDS = ["B01", "B09"]
# Resolution that provides each band of the Sentinel-2 GeoTIFFs, in GeoTIFF band order
S2_BAND_SOURCES = {
    "B01": "60m", "B02": "10m", "B03": "10m", "B04": "10m", "B05": "20m", "B06": "20m",
    "B07": "20m", "B08": "10m", "B8A": "20m", "B09": "60m", "B11": "20m", "B12": "20m",
    "AOT": "10m", "WVP": "10m", "visual": "10m", "SCL": "20m",
}

LANDSAT_UINT16_BANDS = ["red", "green", "blue", "nir08", "swir16", "swir22", "coastal",
                        "qa_pixel", "qa_radsat", "qa_aerosol"]
LANDSAT_INT16_BANDS = ["emis", "emsd", "trad", "urad", "atran", "cdist"]
# Scale factors / offsets of the Landsat Collection 2 Level-2 bands
LANDSAT_SCALE_30M = {
    "red": 2.75e-05, "green": 2.75e-05, "blue": 2.75e-05, "nir08": 2.75e-05, "swir16": 2.75e-05,
    "swir22": 2.75e-05, "coastal": 2.75e-05, "emis": 0.0001, "emsd": 0.0001, "trad": 0.001,
    "urad": 0.001, "atran": 0.0001, "cdist": 0.01,
}
LANDSAT_OFFSET_30M = {
    "red": -0.2, "green": -0.2, "blue": -0.2, "nir08": -0.2, "swir16": -0.2, "swir22": -0.2,
    "coastal": -0.2, "emis": 0, "emsd": 0, "trad": 0, "urad": 0, "atran": 0, "cdist": 0,
}
LANDSAT_SCALE_100M = {"lwir11": 0.00341802}
LANDSAT_OFFSET_100M = {"lwir11": 149.0}


def open_catalog():
    return pystac_client.Client.open(STAC_URL)


def search(catalog, collection, bounds, time_window, query=None):
    """STAC items of a collection that intersect bounds within time_window."""
    return list(catalog.search(bbox=bounds, datetime=time_window, collections=[collection],
                               query=query).items())


def load(items, bands, resolution, bounds, dtype):
    """Lazy (dask-backed) xarray.Dataset of the bands in EPSG:4326 at `resolution` degrees."""
    return stac_load(items, bands=bands, crs="EPSG:4326", resolution=resolution,
                     chunks={"x": 2048, "y": 2048}, dtype=dtype,
                     patch_url=planetary_computer.sign, bbox=bounds)


def _stack(sources, band_sources):
    """dict {band: float64 2-D array} taking each band from its source dataset."""
    return {band: sources[src][band].squeeze().astype("float64").values
            for band, src in band_sources.items()}


# ----------------------------------------------------------------------------- Sentinel-2
def load_sentinel2(items, bounds):
    """Lazy Sentinel-2 datasets at their native resolutions: {'10m', '20m', '60m'}."""
    return {
        "10m": load(items, S2_10M_BANDS, SCALE_10M, bounds, "uint16"),
        "20m": load(items, S2_20M_BANDS, SCALE_10M * 2, bounds, "uint16"),
        "60m": load(items, S2_60M_BANDS, SCALE_10M * 6, bounds, "uint16"),
    }


def sentinel2_composites(datasets, date, resampling):
    """
    Single-day and median composites of all 16 Sentinel-2 layers on the 10 m grid.

    Returns (single, median, grid): two dicts {band: float64 2-D array} and the single-day
    10 m dataset, which defines the target grid for every other layer.
    """
    with ProgressBar():
        loaded = {res: ds.compute() for res, ds in datasets.items()}

    # Single day: select the date and resample the 20 m / 60 m bands onto the 10 m grid
    day = {res: ds.sel(time=date, drop=True) for res, ds in loaded.items()}
    with ProgressBar():
        day["20m"] = day["20m"].rio.reproject_match(day["10m"], resampling=resampling)
        day["60m"] = day["60m"].rio.reproject_match(day["10m"], resampling=resampling)
    single = _stack(day, S2_BAND_SOURCES)

    # Median over all scenes, resampled onto the 10 m median grid
    with ProgressBar():
        med = {res: ds.median(dim="time") for res, ds in loaded.items()}
        med["20m"] = med["20m"].rio.reproject_match(med["10m"], resampling=resampling)
        med["60m"] = med["60m"].rio.reproject_match(med["10m"], resampling=resampling)
    median = _stack(med, S2_BAND_SOURCES)

    return single, median, day["10m"]


# ----------------------------------------------------------------------------- Landsat 8
def load_landsat(items, bounds):
    """Lazy Landsat datasets: {'30m': reflectance + auxiliary bands, '100m': thermal band lwir11}."""
    data_uint16 = load(items, LANDSAT_UINT16_BANDS, SCALE_10M * 3, bounds, "uint16")
    data_int16 = load(items, LANDSAT_INT16_BANDS, SCALE_10M * 3, bounds, "int16")
    return {
        "30m": xr.merge([data_uint16, data_int16]),
        "100m": load(items, ["lwir11"], SCALE_10M * 10, bounds, "uint16"),
    }


def scale_landsat(datasets):
    """
    Apply the Collection 2 Level-2 scale factors and offsets.

    Returns two dicts {band: DataArray}: the 30 m bands, and the 100 m surface temperature
    lwir11 converted to degrees Celsius.
    """
    scaled_30m = {band: datasets["30m"][band].astype(float) * LANDSAT_SCALE_30M[band]
                  + LANDSAT_OFFSET_30M.get(band, 0)
                  for band in LANDSAT_SCALE_30M if band in datasets["30m"]}
    scaled_100m = {band: datasets["100m"][band].astype(float) * LANDSAT_SCALE_100M[band]
                   + LANDSAT_OFFSET_100M.get(band, 0) - 273.15
                   for band in LANDSAT_SCALE_100M if band in datasets["100m"]}
    return scaled_30m, scaled_100m


def _resample_bands(bands, grid, resampling):
    with ProgressBar():
        return {band: da.rio.reproject_match(grid, resampling=resampling) for band, da in bands.items()}


def landsat_single_scene(scaled_30m, scaled_100m, time_index, grid, resampling):
    """One Landsat scene (position `time_index` on the time axis) on the 10 m grid."""
    bands = {band: da.isel(time=time_index) for band, da in {**scaled_30m, **scaled_100m}.items()}
    resampled = _resample_bands(bands, grid, resampling)
    return {band: da.squeeze().astype("float64").values for band, da in resampled.items()}


def landsat_median(scaled_30m, scaled_100m, grid, resampling):
    """
    Per-pixel median over all Landsat scenes on the 10 m grid.

    A raw value of 0 scales to exactly -0.2 in the red band, i.e. a no-data pixel; those pixels
    are excluded from the median of the 30 m bands.
    """
    red = scaled_30m["red"].astype(float)
    no_data = red == -0.2
    masked_30m = {band: da.where(~no_data, np.nan) for band, da in scaled_30m.items()}
    with ProgressBar():
        medians = {band: da.median(dim="time", skipna=True)
                   for band, da in {**masked_30m, **scaled_100m}.items()}
    resampled = _resample_bands(medians, grid, resampling)
    return {band: da.squeeze().astype("float64").values for band, da in resampled.items()}


# ----------------------------------------------------------------------------- MODIS
def load_modis_lst(items, bounds):
    """8-day daytime land surface temperature (degrees C) and band 31/32 emissivity at 1 km."""
    resolution = 1000 / 111320.0
    lst = load(items, "LST_Day_1km", resolution, bounds, "uint16")
    emis = load(items, ["Emis_31", "Emis_32"], resolution, bounds, "uint8")
    return xr.merge([
        xr.Dataset({"LST_Celsius": lst["LST_Day_1km"] * 0.02 - 273.15}),
        xr.Dataset({"Emis_31": emis["Emis_31"] * 0.002, "Emis_32": emis["Emis_32"] * 0.002}),
    ])


def load_modis_vi(items, bounds):
    """16-day NDVI and EVI at 250 m."""
    resolution = 250 / 111320.0
    ndvi = load(items, "250m_16_days_NDVI", resolution, bounds, "int16")
    evi = load(items, "250m_16_days_EVI", resolution, bounds, "int16")
    return xr.merge([
        xr.Dataset({"NDVI": ndvi["250m_16_days_NDVI"].astype(float) * 0.0001}),
        xr.Dataset({"EVI": evi["250m_16_days_EVI"].astype(float) * 0.0001}),
    ])


def composite_on_grid(dataset, time_index, grid, resampling):
    """One composite (position `time_index` on the time axis) of every variable, on the 10 m grid."""
    with ProgressBar():
        resampled = dataset.isel(time=time_index).rio.reproject_match(grid, resampling=resampling)
    return {band: resampled[band].values.astype("float64") for band in resampled.data_vars}
