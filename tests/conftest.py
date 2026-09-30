import geopandas as gpd
import numpy as np
import pandas as pd
import pytest


def make_pixel_grid(nx, ny, n_features=3, seed=0, dx=10.0, dy=13.2, index_offset=0):
    """Regular grid of pixel centres in EPSG:3857 (10 m x 13.2 m, like the 10 m grid at 40.8° N)."""
    rng = np.random.default_rng(seed)
    xx, yy = np.meshgrid(np.arange(nx) * dx, np.arange(ny) * dy)
    data = {f"f{i}": rng.normal(size=xx.size) for i in range(n_features)}
    df = pd.DataFrame(data, index=np.arange(xx.size) + index_offset)
    return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(xx.ravel(), yy.ravel()), crs=3857)


@pytest.fixture
def pixel_grid():
    return make_pixel_grid(40, 25, n_features=4, seed=1, index_offset=1000)
