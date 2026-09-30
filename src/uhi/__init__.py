"""Urban Heat Island (UHI) index prediction for the EY Open Science AI & Data Challenge 2025.

Modules
-------
config      study area, file locations, feature lists and model settings
satellite   Planetary Computer search / loading and 10 m composites (Sentinel-2, Landsat 8, MODIS)
raster      GeoTIFF writing and pixel-table extraction
features    spectral, weather, building and spatial (neighbourhood / mapping) features
modeling    training-table preparation, feature selection, XGBoost training and evaluation
pipeline    end-to-end pipeline from the GeoTIFFs and challenge files to a submission (CLI)
"""

__version__ = "1.0.0"
