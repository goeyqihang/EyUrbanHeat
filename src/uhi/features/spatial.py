"""
Spatial operations on the pixel table: filtering, neighbourhood (focal) means and point mapping.

The pixel table has millions of rows and nearby points have heavily overlapping buffers, so the
spatial joins only carry indices / geometry; feature columns are gathered afterwards. This keeps
the memory per chunk to a few GB instead of tens of GB, with identical results.
"""

import geopandas as gpd
import numpy as np
import pandas as pd


def points_to_gdf(df, lon="Longitude", lat="Latitude", crs_out=3857):
    """
    GeoDataFrame of points from longitude / latitude columns (EPSG:4326), projected to `crs_out`.

    Only the geometry is reprojected and the other columns are reused without copying.
    """
    geometry = gpd.points_from_xy(df[lon], df[lat], crs="EPSG:4326").to_crs(epsg=crs_out)
    return gpd.GeoDataFrame(df, geometry=geometry, copy=False)


def pixels_near_points(pixels, point_layers, radius=500, chunksize=100_000):
    """
    Keep the pixels that lie within `radius` of any point of the given point layers.

    A bounding-box pre-filter is followed by a chunked spatial join against the point buffers.
    Returns (filtered pixels with a fresh index, info dict with the counts and the buffer bounds).
    """
    buffers = []
    for layer in point_layers:
        b = layer[["geometry"]].copy()
        b["geometry"] = b.geometry.buffer(radius)
        buffers.append(b)
    buffers = gpd.GeoDataFrame(pd.concat(buffers, ignore_index=True), crs=point_layers[0].crs)

    minx, miny, maxx, maxy = buffers.total_bounds
    subset = pixels.cx[minx:maxx, miny:maxy]

    kept = []
    for start in range(0, len(subset), chunksize):
        chunk = subset.iloc[start:start + chunksize]
        # Only the geometry goes through the join; a pixel inside several buffers is kept once
        pairs = gpd.sjoin(chunk[["geometry"]], buffers, how="inner", predicate="intersects")
        pairs = pairs[~pairs.index.duplicated(keep="first")]
        kept.append(chunk.loc[pairs.index])
    filtered = pd.concat(kept, ignore_index=True)

    info = {"pixels": len(pixels), "in_bounding_box": len(subset), "kept": len(filtered),
            "buffer_bounds": (minx, miny, maxx, maxy)}
    return filtered, info


def focal_means(pixels, features, radius=100, chunk_size=100_000):
    """
    Add 'buffered_<feature>': the mean of each feature over all pixels within `radius`.

    Pixels are processed in chunks of consecutive rows and neighbours are searched within the
    same chunk. Missing values are skipped, as in pandas' groupby mean.
    """
    processed = []
    for start in range(0, len(pixels), chunk_size):
        chunk = pixels.iloc[start:start + chunk_size].copy()

        # One row per (pixel, neighbour inside its buffer) pair
        buffers = gpd.GeoDataFrame(
            {"buffer_index": chunk.index}, geometry=chunk.geometry.buffer(radius).values, crs=chunk.crs)
        pairs = gpd.sjoin(buffers, chunk[["geometry"]], how="left", predicate="intersects")
        group = pairs["buffer_index"].to_numpy()
        rows = chunk.index.get_indexer(pairs["index_right"])
        del pairs, buffers

        for feature in features:
            means = pd.Series(chunk[feature].to_numpy()[rows]).groupby(group).mean()
            chunk[f"buffered_{feature}"] = chunk.index.map(means)
        processed.append(chunk)
    return pd.concat(processed, ignore_index=True)


def map_nearest_pixel(points, pixels):
    """
    Give every point the features of its nearest pixel ('nearest_distance' = distance to it).

    Columns present in both tables get the suffixes _left (point) and _right (pixel).
    """
    mapped = gpd.sjoin_nearest(points, pixels, how="left", distance_col="nearest_distance")
    # A point at exactly the same distance from two pixels would appear twice: keep the first match
    return mapped[~mapped.index.duplicated(keep="first")]


def map_idw(points, pixels, k=5, attributes=None):
    """
    Inverse-distance-weighted mean of the k nearest pixels for every point ('mapped_<attribute>').

    A point that coincides with pixels takes their plain mean. Used in the bilinear experiment
    (notebooks/03_uhi_model_bilinear.ipynb).
    """
    from sklearn.neighbors import NearestNeighbors  # only needed here

    point_xy = np.column_stack([points.geometry.x, points.geometry.y])
    pixel_xy = np.column_stack([pixels.geometry.x, pixels.geometry.y])
    distances, indices = NearestNeighbors(n_neighbors=k).fit(pixel_xy).kneighbors(point_xy)

    exact = distances == 0
    weights = np.where(exact.any(axis=1, keepdims=True), exact.astype(float), 1.0 / np.where(exact, 1, distances))
    weights /= weights.sum(axis=1, keepdims=True)

    if attributes is None:
        attributes = [c for c in pixels.columns if c != pixels.geometry.name]
    out = points.copy()
    for attr in attributes:
        out[f"mapped_{attr}"] = (pixels[attr].to_numpy()[indices] * weights).sum(axis=1)
    return out
