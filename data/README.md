# Data

Apart from this file, nothing in `data/` is tracked by git. Put the files here before running the notebooks:

```
data/
├── raw/                                    # provided by the EY challenge
│   ├── Training_data_uhi_index_UHI2025-v2.csv
│   ├── Submission_template_UHI2025-v2.csv
│   ├── NY_Mesonet_Weather.xlsx
│   └── Building_Footprint.kml
└── processed/                              # written by notebooks/01_satellite_data_extraction.ipynb
    ├── Sentinel2_SingleDay_nearest.tiff    # 16 Sentinel-2 bands, 2021-07-24
    ├── Sentinel2_Median_nearest.tiff       # 16 Sentinel-2 bands, median Jun–Aug 2021
    ├── LST_Singleday_nearest_4.tiff        # 14 Landsat 8 bands, 2021-06-23
    ├── LST_Median_nearest_4.tiff           # 14 Landsat 8 bands, median Jun–Aug 2021
    ├── LST_Emis_mean_nearest_8.tiff        # MODIS LST + band 31/32 emissivity, 8-day composite
    └── VI_mean_nearest_9.tiff              # MODIS NDVI + EVI, 16-day composite
```

All GeoTIFFs share one grid: 1448 × 1671 pixels of about 10 m, in EPSG:4326, covering 40.75–40.88° N, 74.01–73.86° W.

| Challenge file | Contents |
|---|---|
| `Training_data_uhi_index_UHI2025-v2.csv` | 11,229 points with `Longitude`, `Latitude`, `datetime` and the target `UHI Index` |
| `Submission_template_UHI2025-v2.csv` | The 1,040 points to predict (`Longitude`, `Latitude`, empty `UHI Index`) |
| `NY_Mesonet_Weather.xlsx` | 5-minute readings from the Bronx and Manhattan NY Mesonet stations on 24 July 2021 (one sheet per station) |
| `Building_Footprint.kml` | Building footprint polygons of the study area |

The challenge bundle also contains two sample GeoTIFFs, `S2_sample.tiff` and `Landsat_LST.tiff`, made for the challenge's starter notebook. This project does not use them.

The challenge files come from the [EY Open Science AI & Data Challenge](https://challenge.ey.com) platform and are not redistributed here.
