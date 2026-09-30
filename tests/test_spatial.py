import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from geopandas.testing import assert_geodataframe_equal
from pyproj import Transformer

from conftest import make_pixel_grid
from uhi.features import spatial


# ---------------------------------------------------------------- reference implementations
# The competition notebook's original code; the refactored versions must match it exactly.
def original_filter(final_gdf, point_layers, radius):
    buffers = []
    for layer in point_layers:
        b = layer[["geometry"]].copy()
        b["geometry"] = b.geometry.buffer(radius)
        buffers.append(b)
    buffers_gdf = gpd.GeoDataFrame(pd.concat(buffers, ignore_index=True), crs=point_layers[0].crs)
    minx, miny, maxx, maxy = buffers_gdf.total_bounds
    final_subset = final_gdf.cx[minx:maxx, miny:maxy]
    results = []
    for i in range(0, len(final_subset), 100):
        chunk = final_subset.iloc[i:i + 100]
        chunk_join = gpd.sjoin(chunk, buffers_gdf, how="inner", predicate="intersects")
        results.append(chunk_join[~chunk_join.index.duplicated(keep="first")])
    return pd.concat(results, ignore_index=True).drop(columns=["index_right"])


def original_focal(final_data, buffered_features, buffer_radius, chunk_size):
    def process_chunk(chunk):
        chunk = chunk.copy()
        chunk["buffer"] = chunk.geometry.buffer(buffer_radius)
        buffered_df = chunk[["buffer"]].reset_index().rename(columns={"index": "buffer_index"})
        buffered_df = buffered_df.set_geometry("buffer")
        feat_df = chunk[buffered_features + ["geometry"]].copy()
        joined = gpd.sjoin(buffered_df, feat_df, how="left", predicate="intersects")
        agg = joined.groupby("buffer_index")[buffered_features].mean()
        for feature in buffered_features:
            chunk[f"buffered_{feature}"] = chunk.index.map(agg[feature])
        return chunk.drop(columns=["buffer"])

    chunks = [process_chunk(final_data.iloc[s:s + chunk_size]) for s in range(0, len(final_data), chunk_size)]
    return pd.concat(chunks, ignore_index=True)


# ---------------------------------------------------------------- tests
def test_points_to_gdf_projects_to_web_mercator():
    df = pd.DataFrame({"Longitude": [-73.95, -73.90], "Latitude": [40.80, 40.85], "v": [1, 2]})
    gdf = spatial.points_to_gdf(df)
    x, y = Transformer.from_crs(4326, 3857, always_xy=True).transform(df["Longitude"], df["Latitude"])
    assert gdf.crs.to_epsg() == 3857
    np.testing.assert_allclose(gdf.geometry.x, x)
    np.testing.assert_allclose(gdf.geometry.y, y)
    assert list(gdf.columns) == ["Longitude", "Latitude", "v", "geometry"]


def test_pixels_near_points_matches_original(pixel_grid):
    rng = np.random.default_rng(2)
    points = gpd.GeoDataFrame(geometry=gpd.points_from_xy(rng.uniform(50, 350, 30), rng.uniform(40, 280, 30)),
                              crs=3857)
    more = gpd.GeoDataFrame(geometry=gpd.points_from_xy([380.0], [300.0]), crs=3857)
    filtered, info = spatial.pixels_near_points(pixel_grid, [points, more], radius=40, chunksize=100)
    assert_geodataframe_equal(filtered, original_filter(pixel_grid, [points, more], radius=40))
    assert 0 < info["kept"] < info["pixels"] == len(pixel_grid)


def test_focal_means_matches_original(pixel_grid):
    pixel_grid.loc[pixel_grid.index[5], "f1"] = np.nan  # missing values are skipped
    features = ["f0", "f1", "f2"]
    new = spatial.focal_means(pixel_grid, features, radius=30, chunk_size=300)
    assert_geodataframe_equal(new, original_focal(pixel_grid, features, 30, 300))


def test_focal_means_is_the_neighbourhood_mean():
    grid = make_pixel_grid(12, 9, n_features=1, seed=3)
    out = spatial.focal_means(grid, ["f0"], radius=25, chunk_size=10_000)
    buffers = grid.geometry.buffer(25)
    for i in [0, 17, 50, 107]:
        inside = grid.geometry.intersects(buffers.iloc[i])
        assert out.loc[i, "buffered_f0"] == pytest.approx(grid.loc[inside, "f0"].mean())


def test_map_nearest_pixel_takes_the_closest_pixel(pixel_grid):
    target = pixel_grid.iloc[[10, 200, 555]]
    points = gpd.GeoDataFrame({"id": [1, 2, 3]},
                              geometry=gpd.points_from_xy(target.geometry.x + 2, target.geometry.y - 3), crs=3857)
    mapped = spatial.map_nearest_pixel(points, pixel_grid)
    assert len(mapped) == 3
    np.testing.assert_array_equal(mapped["f0"].to_numpy(), target["f0"].to_numpy())
    np.testing.assert_allclose(mapped["nearest_distance"], np.hypot(2, 3))


def test_map_idw_weights():
    # ImportError, not only ModuleNotFoundError: an OS policy can block compiled sklearn modules
    pytest.importorskip("sklearn.neighbors", exc_type=ImportError)
    pixels = gpd.GeoDataFrame({"v": [10.0, 20.0, 99.0]},
                              geometry=gpd.points_from_xy([0.0, 4.0, 1000.0], [0.0, 0.0, 0.0]), crs=3857)
    points = gpd.GeoDataFrame(geometry=gpd.points_from_xy([1.0, 4.0], [0.0, 0.0]), crs=3857)
    mapped = spatial.map_idw(points, pixels, k=2, attributes=["v"])
    # distances 1 and 3 -> weights 3/4 and 1/4; the second point sits on a pixel
    assert mapped["mapped_v"].tolist() == pytest.approx([0.75 * 10 + 0.25 * 20, 20.0])
