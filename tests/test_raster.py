import numpy as np
import pytest
import rasterio

from uhi import config, raster

BOUNDS = (-74.0, 40.0, -73.6, 40.3)  # 4 x 3 pixels of 0.1 degrees


def test_write_and_extract_roundtrip(tmp_path):
    a = np.arange(12, dtype=float).reshape(3, 4)
    path = raster.write_geotiff(tmp_path / "t.tiff", {"first": a, "second": a * 10}, BOUNDS)

    with rasterio.open(path) as src:
        assert src.descriptions == ("first", "second")
        assert src.crs.to_epsg() == 4326

    df = raster.extract_bands(path, ["first", "second"])
    assert list(df.columns) == ["first", "second"]
    np.testing.assert_array_equal(df["first"], np.arange(12))  # row-major order

    coords = raster.extract_coordinates(path)
    assert coords.iloc[0].tolist() == pytest.approx([-73.95, 40.25])   # top-left pixel centre
    assert coords.iloc[-1].tolist() == pytest.approx([-73.65, 40.05])  # bottom-right pixel centre


def test_extra_bands_get_generic_names(tmp_path):
    a = np.zeros((3, 4))
    path = raster.write_geotiff(tmp_path / "t.tiff", {"a": a, "b": a}, BOUNDS)
    assert list(raster.extract_bands(path, ["a"]).columns) == ["a", "Band_2"]


def test_pixel_table_column_order(tmp_path):
    counts = {"s2_single": config.S2_BANDS, "s2_median": config.S2_BANDS,
              "landsat_single": config.LANDSAT_BANDS, "landsat_median": config.LANDSAT_BANDS,
              "modis_lst": config.MODIS_LST_BANDS, "modis_vi": config.MODIS_VI_BANDS}
    for key, name in config.geotiff_names("nearest").items():
        raster.write_geotiff(tmp_path / name, {b: np.ones((3, 4)) for b in counts[key]}, BOUNDS)

    table = raster.pixel_table(tmp_path, "nearest")
    assert table.shape == (12, 2 + 16 + 16 + 14 + 14 + 3 + 2)
    assert list(table.columns[:3]) == ["Longitude", "Latitude", "B01_single"]
    assert list(table.columns[-5:]) == ["LST_mean", "Emis_31_mean", "Emis_32_mean", "NDVI_mean", "EVI_mean"]
