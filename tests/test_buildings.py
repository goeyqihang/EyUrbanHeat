import geopandas as gpd
import numpy as np
import pytest
from shapely import affinity
from shapely.geometry import Point, box

from uhi.features import buildings

KML = """<?xml version="1.0" encoding="utf-8" ?>
<kml xmlns="http://www.opengis.net/kml/2.2"><Document><Folder>
<Placemark>
  <ExtendedData><SchemaData schemaUrl="#s"><SimpleData name="id">b1</SimpleData></SchemaData></ExtendedData>
  <MultiGeometry><Polygon><outerBoundaryIs><LinearRing><coordinates>
    -73.90,40.80 -73.89,40.80 -73.89,40.81 -73.90,40.81 -73.90,40.80
  </coordinates></LinearRing></outerBoundaryIs></Polygon></MultiGeometry>
</Placemark>
</Folder></Document></kml>
"""


def test_read_footprints(tmp_path):
    path = tmp_path / "footprints.kml"
    path.write_text(KML)
    gdf = buildings.read_footprints(path)
    assert len(gdf) == 1 and gdf.crs.to_epsg() == 4326
    assert gdf.loc[0, "id"] == "b1"
    assert gdf.geometry.iloc[0].bounds == pytest.approx((-73.90, 40.80, -73.89, 40.81))


def test_shape_metrics_of_a_rotated_rectangle():
    rect = affinity.rotate(box(0, 0, 20, 10), 30, origin=(0, 0))
    m = buildings.shape_metrics(gpd.GeoDataFrame(geometry=[rect], crs=3857)).iloc[0]
    assert m["building_area"] == pytest.approx(200)
    assert m["perimeter"] == pytest.approx(60)
    assert m["compactness"] == pytest.approx(4 * np.pi * 200 / 60 ** 2)
    assert m["solidity"] == pytest.approx(1)
    assert m["long_width_ratio"] == pytest.approx(2)
    assert m["rotation_angle"] == pytest.approx(30)


def test_nearest_building_distance():
    pixels = gpd.GeoDataFrame({"f": [1.0, 2.0]}, geometry=[Point(15, 5), Point(-3, 0)], crs=3857)
    footprints = gpd.GeoDataFrame(geometry=[box(0, 0, 10, 10)], crs=3857)
    out = buildings.nearest_building_distance(pixels, footprints, chunk_size=1)
    assert out["nearest_building_distance"].tolist() == pytest.approx([5, 3])
    assert list(out.columns) == ["f", "geometry", "nearest_building_distance"]


def test_buffer_stats():
    footprints = buildings.shape_metrics(
        gpd.GeoDataFrame(geometry=[box(0, 0, 10, 10), box(20, 0, 30, 20)], crs=3857))
    pixels = gpd.GeoDataFrame({"f": [1.0, 2.0]}, geometry=[Point(15, 5), Point(500, 500)], crs=3857)
    out = buildings.buffer_stats(pixels, footprints, radius=8, npartitions=1)

    near, far = out.iloc[0], out.iloc[1]
    assert near["building_count"] == 2
    assert near["total_building_area"] == pytest.approx(100 + 200)
    assert near["avg_building_perimeter"] == pytest.approx((40 + 60) / 2)
    assert near["std_building_area"] == pytest.approx(np.std([100, 200], ddof=1))
    buffer_area = Point(15, 5).buffer(8).area
    assert near["building_coverage_ratio"] == pytest.approx(300 / buffer_area)
    assert near["area_to_perimeter_ratio"] == pytest.approx(150 / 50)
    assert far[buildings.BUFFER_STATS].eq(0).all()
