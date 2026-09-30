"""Project-wide settings: study area, file locations, feature lists and model parameters."""

from pathlib import Path

# ----------------------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"              # files provided by the challenge
PROCESSED_DIR = DATA_DIR / "processed"  # GeoTIFFs written by notebooks/01_satellite_data_extraction.ipynb
OUTPUT_DIR = ROOT / "outputs"           # submissions and metrics

TRAINING_FILE = "Training_data_uhi_index_UHI2025-v2.csv"
SUBMISSION_TEMPLATE_FILE = "Submission_template_UHI2025-v2.csv"
WEATHER_FILE = "NY_Mesonet_Weather.xlsx"
BUILDINGS_FILE = "Building_Footprint.kml"

# ----------------------------------------------------------------------------- study area
LOWER_LEFT = (40.75, -74.01)   # (latitude, longitude)
UPPER_RIGHT = (40.88, -73.86)
BOUNDS = (LOWER_LEFT[1], LOWER_LEFT[0], UPPER_RIGHT[1], UPPER_RIGHT[0])  # (min_lon, min_lat, max_lon, max_lat)
TIME_WINDOW = "2021-06-01/2021-09-01"
SURVEY_DATE = "2021-07-24"      # day of the ground-truth temperature campaign (15:00-16:00)

# Positions on the time axis of the scenes used as single-date layers
LANDSAT_TIME_INDEX = 4          # Landsat 8 scene of 2021-06-23
MODIS_LST_TIME_INDEX = 8        # 8-day composite starting 2021-07-28
MODIS_VI_TIME_INDEX = 9         # 16-day composite starting 2021-07-28

# ----------------------------------------------------------------------------- GeoTIFF bands
S2_BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08",
            "B8A", "B09", "B11", "B12", "AOT", "WVP", "visual", "SCL"]
LANDSAT_BANDS = ["red", "green", "blue", "nir08", "swir16", "swir22", "coastal",
                 "emis", "emsd", "trad", "urad", "atran", "cdist", "lwir11"]
MODIS_LST_BANDS = ["LST", "Emis_31", "Emis_32"]
MODIS_VI_BANDS = ["NDVI", "EVI"]


def geotiff_names(method="nearest"):
    """File names of the six GeoTIFFs written by notebook 01 for a resampling method."""
    return {
        "s2_single": f"Sentinel2_SingleDay_{method}.tiff",
        "s2_median": f"Sentinel2_Median_{method}.tiff",
        "landsat_single": f"LST_Singleday_{method}_{LANDSAT_TIME_INDEX}.tiff",
        "landsat_median": f"LST_Median_{method}_{LANDSAT_TIME_INDEX}.tiff",
        "modis_lst": f"LST_Emis_mean_{method}_{MODIS_LST_TIME_INDEX}.tiff",
        "modis_vi": f"VI_mean_{method}_{MODIS_VI_TIME_INDEX}.tiff",
    }


# ----------------------------------------------------------------------------- features
# Radii are in EPSG:3857 units (at 40.8° N one unit is about 0.76 m on the ground)
PIXEL_FILTER_RADIUS = 500       # keep pixels within this distance of a training / submission point
BUILDING_RADIUS = 500           # building statistics around each pixel
FOCAL_RADIUS = 100              # neighbourhood means (buffered_<feature>)
FOCAL_CHUNK_SIZE = 100_000      # pixels per chunk; neighbours are searched within a chunk

# Bands that have a single-day and a median version (used for the diff / ratio features)
CHANGE_BANDS = S2_BANDS + LANDSAT_BANDS

# Features whose 100 m neighbourhood mean is added as buffered_<feature>
FOCAL_FEATURES = [
    # Sentinel-2 bands (single day)
    "B01_single", "B02_single", "B03_single", "B04_single", "B05_single", "B06_single", "B07_single",
    "B08_single", "B8A_single", "B09_single", "B11_single", "B12_single", "AOT_single", "WVP_single",
    "visual_single", "SCL_single",
    # Landsat bands (single scene)
    "red_single", "green_single", "blue_single", "nir08_single", "swir16_single", "swir22_single",
    "coastal_single", "emis_single", "emsd_single", "trad_single", "urad_single", "atran_single",
    "cdist_single", "lwir11_single",
    # Vegetation, water and built-up indices (single day)
    "NDVI_single", "NDVI_single_lst", "NDBI_single", "NDBIa_single", "NDWI_single", "NDTI_single",
    "Albedo_single", "Albedo_NYC_single", "EVI_single", "BSR_single",
    # Weather interaction features
    "Wind_Solar_Impact", "Temp_Humidity_Ratio", "Wind_Dir_Radians", "Wind_U_Component",
    "Wind_V_Component", "Green_Heat_Reflection", "Urban_Water_Buffer", "Temp_Humidity_Diff",
    "Temp_Solar_Impact", "Wind_Temperature_Ratio", "NDVI_NDBI_Ratio", "Wind_Direction_Speed_Impact",
    # Building features
    "nearest_building_distance", "building_density", "building_count", "total_building_area",
    "avg_building_area", "total_building_perimeter", "avg_building_perimeter", "std_building_area",
    "avg_building_compactness", "total_building_compactness", "building_coverage_ratio",
    "area_to_perimeter_ratio",
]

# ----------------------------------------------------------------------------- model
TARGET = "UHI Index"
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Tuned with a 5,832-configuration grid search (see notebooks/02_uhi_model_nearest.ipynb)
XGB_PARAMS = {
    "colsample_bytree": 0.7,
    "gamma": 0,
    "learning_rate": 0.04,
    "max_depth": 15,
    "min_child_weight": 7,
    "n_estimators": 800,
    "subsample": 0.6,
    "reg_alpha": 0.001,
    "reg_lambda": 3.5,
    "objective": "reg:squarederror",
    "random_state": RANDOM_STATE,
}

# The 52 features selected by RFECV with the tuned model during the competition
SELECTED_FEATURES = [
    # Satellite bands (single day / median)
    "B01_single", "B09_single", "urad_single", "atran_single", "cdist_single",
    "urad_median", "atran_median", "cdist_median", "lwir11_median",
    # MODIS composites
    "LST_mean", "Emis_31_mean", "Emis_32_mean", "NDVI_mean", "EVI_mean",
    # Single-day vs. median change
    "B01_diff", "B01_ratio", "B09_diff", "B09_ratio", "urad_diff", "urad_ratio",
    "atran_diff", "atran_ratio", "cdist_diff", "cdist_ratio",
    # Buildings within 500 m
    "building_count", "total_building_area", "avg_building_area", "total_building_perimeter",
    "avg_building_perimeter", "std_building_area", "avg_building_compactness",
    "total_building_compactness", "area_to_perimeter_ratio",
    # 100 m focal means
    "buffered_urad_single", "buffered_atran_single", "buffered_cdist_single", "buffered_lwir11_single",
    "buffered_NDVI_single_lst", "buffered_NDBIa_single", "buffered_NDWI_single", "buffered_NDTI_single",
    "buffered_building_density", "buffered_building_count", "buffered_total_building_area",
    "buffered_avg_building_area", "buffered_total_building_perimeter", "buffered_avg_building_perimeter",
    "buffered_std_building_area", "buffered_avg_building_compactness",
    "buffered_total_building_compactness", "buffered_building_coverage_ratio",
    "buffered_area_to_perimeter_ratio",
]
