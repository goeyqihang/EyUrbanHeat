import pandas as pd
import pytest

from uhi import config
from uhi.features import spectral

S2 = {"B02": 0.1, "B03": 0.2, "B04": 0.1, "B08": 0.5, "B11": 0.3, "B12": 0.2}
LANDSAT = {"red": 0.1, "nir08": 0.3}


def band_frame():
    row = {}
    for kind in ("single", "median"):
        row.update({f"{b}_{kind}": v for b, v in S2.items()})
        row.update({f"{b}_{kind}": v for b, v in LANDSAT.items()})
    return pd.DataFrame([row])


def test_spectral_indices_formulas():
    df = spectral.add_spectral_indices(band_frame())
    r = df.iloc[0]
    assert r["NDVI_single"] == pytest.approx((0.5 - 0.1) / (0.5 + 0.1))
    assert r["NDVI_single_lst"] == pytest.approx((0.3 - 0.1) / (0.3 + 0.1))
    assert r["NDBI_single"] == pytest.approx((0.3 - 0.5) / (0.3 + 0.5))
    assert r["NDBIa_single"] == pytest.approx(r["NDBI_single"] - r["NDVI_single"])
    assert r["NDWI_single"] == pytest.approx((0.2 - 0.5) / (0.2 + 0.5))
    assert r["NDTI_single"] == pytest.approx((0.3 - 0.2) / (0.3 + 0.2))
    assert r["BSR_single"] == pytest.approx(0.3 / 0.5)
    assert r["EVI_single"] == pytest.approx(2.5 * 0.4 / (0.5 + 6 * 0.1 - 7.5 * 0.1 + 1))
    assert r["Albedo_single"] == pytest.approx(0.356 * 0.1 + 0.130 * 0.1 + 0.373 * 0.5 + 0.085 * 0.3
                                               + 0.072 * 0.2 - 0.0018)
    # the median image gets the same indices
    assert r["NDVI_median"] == pytest.approx(r["NDVI_single"])


def test_spectral_indices_column_order():
    # the column order feeds the model, so it must stay stable
    df = spectral.add_spectral_indices(band_frame())
    added = [c for c in df.columns if c not in band_frame().columns]
    assert added == [
        "NDVI_single", "NDVI_median", "NDVI_single_lst", "NDVI_median_lst", "NDBI_single", "NDBI_median",
        "NDBIa_single", "NDBIa_median", "NDWI_single", "NDWI_median", "NDTI_single", "NDTI_median",
        "Albedo_single", "Albedo_median", "Albedo_NYC_single", "Albedo_NYC_median",
        "BSR_single", "BSR_median", "EVI_single", "EVI_median",
    ]


def test_temporal_change():
    df = pd.DataFrame({"B01_single": [3.0], "B01_median": [2.0]})
    spectral.add_temporal_change(df, bands=["B01"])
    assert df.loc[0, "B01_diff"] == 1.0
    assert df.loc[0, "B01_ratio"] == pytest.approx(3.0 / (2.0 + 1e-6))


def test_change_bands_cover_every_single_median_pair():
    assert config.CHANGE_BANDS == config.S2_BANDS + config.LANDSAT_BANDS
    assert len(config.CHANGE_BANDS) == 30
