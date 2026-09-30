"""Spectral indices and single-day vs. median change features.

The functions add columns to the pixel table in place (it holds ~2.4 M rows, so copies are
avoided) and return it for convenience.
"""

from .. import config


def add_spectral_indices(df):
    """
    Add vegetation, built-up and water indices for the single-day and median images.

    Sentinel-2 bands: B02 blue, B03 green, B04 red, B08 NIR, B11 SWIR1, B12 SWIR2.
    NDVI_*_lst uses the Landsat red / NIR bands.
    """
    # NDVI (vegetation) from Sentinel-2 and from Landsat
    df["NDVI_single"] = (df["B08_single"] - df["B04_single"]) / (df["B08_single"] + df["B04_single"])
    df["NDVI_median"] = (df["B08_median"] - df["B04_median"]) / (df["B08_median"] + df["B04_median"])
    df["NDVI_single_lst"] = (df["nir08_single"] - df["red_single"]) / (df["nir08_single"] + df["red_single"])
    df["NDVI_median_lst"] = (df["nir08_median"] - df["red_median"]) / (df["nir08_median"] + df["red_median"])

    # NDBI (built-up areas)
    df["NDBI_single"] = (df["B11_single"] - df["B08_single"]) / (df["B11_single"] + df["B08_single"])
    df["NDBI_median"] = (df["B11_median"] - df["B08_median"]) / (df["B11_median"] + df["B08_median"])

    # NDBIa: NDBI minus NDVI
    df["NDBIa_single"] = df["NDBI_single"] - df["NDVI_single"]
    df["NDBIa_median"] = df["NDBI_median"] - df["NDVI_median"]

    # NDWI (water)
    df["NDWI_single"] = (df["B03_single"] - df["B08_single"]) / (df["B03_single"] + df["B08_single"])
    df["NDWI_median"] = (df["B03_median"] - df["B08_median"]) / (df["B03_median"] + df["B08_median"])

    # NDTI (normalised difference of the two SWIR bands)
    df["NDTI_single"] = (df["B11_single"] - df["B12_single"]) / (df["B11_single"] + df["B12_single"])
    df["NDTI_median"] = (df["B11_median"] - df["B12_median"]) / (df["B11_median"] + df["B12_median"])

    # Broadband albedo (Liang's coefficients applied to the matching Sentinel-2 bands)
    for kind in ("single", "median"):
        df[f"Albedo_{kind}"] = 0.356 * df[f"B02_{kind}"] + 0.130 * df[f"B04_{kind}"] + \
                               0.373 * df[f"B08_{kind}"] + 0.085 * df[f"B11_{kind}"] + \
                               0.072 * df[f"B12_{kind}"] - 0.0018

    # Albedo with alternative coefficients for New York City
    for kind in ("single", "median"):
        df[f"Albedo_NYC_{kind}"] = (0.320 * df[f"B02_{kind}"] + 0.140 * df[f"B04_{kind}"]
                                    + 0.380 * df[f"B08_{kind}"] + 0.100 * df[f"B11_{kind}"]
                                    + 0.060 * df[f"B12_{kind}"])

    # BSR: SWIR1 / NIR ratio
    df["BSR_single"] = df["B11_single"] / df["B08_single"]
    df["BSR_median"] = df["B11_median"] / df["B08_median"]

    # EVI (enhanced vegetation index)
    G, C1, C2, L = 2.5, 6, 7.5, 1
    for kind in ("single", "median"):
        df[f"EVI_{kind}"] = G * (df[f"B08_{kind}"] - df[f"B04_{kind}"]) / \
            (df[f"B08_{kind}"] + C1 * df[f"B04_{kind}"] - C2 * df[f"B02_{kind}"] + L)
    return df


def add_temporal_change(df, bands=config.CHANGE_BANDS):
    """Single-day minus median (<band>_diff) and single-day / median (<band>_ratio) for every band."""
    for band in bands:
        df[f"{band}_diff"] = df[f"{band}_single"] - df[f"{band}_median"]
        df[f"{band}_ratio"] = df[f"{band}_single"] / (df[f"{band}_median"] + 1e-6)
    return df
