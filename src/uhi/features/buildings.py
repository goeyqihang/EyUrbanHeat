"""Building-footprint features: distance to the nearest building and building statistics around pixels."""

import xml.etree.ElementTree as ET

import dask_geopandas as dgp
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Polygon

KML_NS = {"kml": "http://www.opengis.net/kml/2.2"}
BUFFER_STATS = ["building_count", "total_building_area", "avg_building_area", "total_building_perimeter",
                "avg_building_perimeter", "std_building_area", "avg_building_compactness",
                "total_building_compactness", "building_density", "building_coverage_ratio",
                "area_to_perimeter_ratio"]


def _parse_coordinates(coord_str):
    """KML coordinate string "lon,lat[,alt] lon,lat[,alt] ..." -> list of (lon, lat) tuples."""
    return [tuple(map(float, point.split(",")[:2])) for point in coord_str.strip().split()]


def read_footprints(kml_path):
    """Building polygons of a KML file as a GeoDataFrame (EPSG:4326) with their SimpleData attributes."""
    root = ET.parse(kml_path).getroot()
    records = []
    for placemark in root.findall(".//kml:Placemark", KML_NS):
        record = {}
        schema_data = placemark.find(".//kml:ExtendedData/kml:SchemaData", KML_NS)
        if schema_data is not None:
            for simple in schema_data.findall("kml:SimpleData", KML_NS):
                record[simple.get("name")] = simple.text

        # Outer ring: look inside a MultiGeometry first, then in a plain Polygon
        coords = placemark.find(
            ".//kml:MultiGeometry/kml:Polygon/kml:outerBoundaryIs/kml:LinearRing/kml:coordinates", KML_NS)
        if coords is None:
            coords = placemark.find(".//kml:Polygon/kml:outerBoundaryIs/kml:LinearRing/kml:coordinates", KML_NS)
        record["coordinates"] = coords.text.strip().replace("\n", " ") if coords is not None else None
        records.append(record)

    kml_data = pd.DataFrame(records)
    kml_data["geometry"] = [Polygon(_parse_coordinates(c)) if c else None for c in kml_data["coordinates"]]
    return gpd.GeoDataFrame(kml_data, geometry="geometry", crs="EPSG:4326")


def nearest_building_distance(pixels, buildings, chunk_size=150_000):
    """
    Add 'nearest_building_distance': distance from every pixel to the closest building polygon.

    Both inputs must use a projected CRS (EPSG:3857). The nearest-neighbour join runs in chunks to
    limit memory use.
    """
    buildings = buildings[["geometry"]]
    # Leftover join columns would clash with the join below
    leftover = [c for c in ("index_left", "index_right") if c in pixels.columns]
    if leftover:
        pixels = pixels.drop(columns=leftover)
    columns = list(pixels.columns)

    chunks = []
    for start in range(0, len(pixels), chunk_size):
        chunk = pixels.iloc[start:start + chunk_size].copy()
        joined = gpd.sjoin_nearest(chunk, buildings, how="left", distance_col="nearest_building_distance")
        chunks.append(joined[columns + ["nearest_building_distance"]])
    return pd.concat(chunks, ignore_index=True)


def shape_metrics(buildings):
    """
    Per-building shape metrics (projected CRS): footprint area, perimeter, compactness (1 = circle),
    solidity (area / convex hull area), and the length-to-width ratio and orientation of the
    minimum rotated rectangle.
    """
    out = buildings[["geometry"]].copy()
    out["building_area"] = out.geometry.area
    out["perimeter"] = out.geometry.length
    out["compactness"] = (4 * np.pi * out["building_area"]) / (out["perimeter"] ** 2)
    out["solidity"] = out["building_area"] / out.geometry.convex_hull.area

    try:
        from shapely import minimum_rotated_rectangle
        rectangles = minimum_rotated_rectangle(out.geometry.values)
    except ImportError:  # shapely < 2
        rectangles = out.geometry.apply(lambda g: g.minimum_rotated_rectangle)
    ratio, angle = np.vectorize(_rectangle_features, otypes=[np.float64, np.float64])(rectangles)
    out["long_width_ratio"] = ratio
    out["rotation_angle"] = angle
    return out


def _rectangle_features(rectangle):
    """(longest side / shortest side, angle of the longest side to the x-axis in [0, 180) degrees)."""
    try:
        coords = np.array(rectangle.exterior.coords)
    except AttributeError:
        return np.nan, np.nan
    if coords.shape[0] < 5:
        return np.nan, np.nan
    sides = np.sqrt(((coords[1:5] - coords[0:4]) ** 2).sum(axis=1))
    short_side = sides.min() if sides.min() != 0 else np.nan
    long_width_ratio = sides.max() / short_side if short_side != 0 else np.nan
    longest = sides.argmax()
    dx, dy = coords[longest + 1] - coords[longest]
    angle = np.degrees(np.arctan2(dy, dx))
    if angle < 0:
        angle += 180
    return long_width_ratio, angle


def buffer_stats(pixels, buildings, radius=500, npartitions=10):
    """
    Add statistics of the buildings that intersect a buffer of `radius` around every pixel:
    count, total / mean / std footprint area, total / mean perimeter, mean / total compactness,
    density (buildings per unit area), coverage ratio (footprint area / buffer area) and the mean
    area to mean perimeter ratio. `buildings` must come from shape_metrics().
    """
    sindex = buildings.sindex
    empty = pd.Series({name: 0 for name in BUFFER_STATS})

    def aggregate(buffer_geom):
        possible = list(sindex.query(buffer_geom, predicate="intersects"))
        if len(possible) == 0:
            return empty.copy()
        selected = buildings.iloc[possible]
        selected = selected[selected.geometry.intersects(buffer_geom)]
        if selected.empty:
            return empty.copy()
        count = len(selected)
        total_area = selected["building_area"].sum()
        avg_area = selected["building_area"].mean()
        total_perimeter = selected["perimeter"].sum()
        avg_perimeter = selected["perimeter"].mean()
        buffer_area = buffer_geom.area
        return pd.Series({
            "building_count": count,
            "total_building_area": total_area,
            "avg_building_area": avg_area,
            "total_building_perimeter": total_perimeter,
            "avg_building_perimeter": avg_perimeter,
            "std_building_area": selected["building_area"].std() if count > 1 else 0,
            "avg_building_compactness": selected["compactness"].mean(),
            "total_building_compactness": selected["compactness"].sum(),
            "building_density": count / buffer_area if buffer_area > 0 else 0,
            "building_coverage_ratio": total_area / buffer_area if buffer_area > 0 else 0,
            "area_to_perimeter_ratio": avg_area / avg_perimeter if avg_perimeter > 0 else 0,
        })

    def process_partition(df):
        df = df.copy()
        df["buffer"] = df.geometry.buffer(radius)
        metrics = df.apply(lambda row: aggregate(row["buffer"]), axis=1)
        return df.join(metrics).drop(columns=["buffer"])

    return dgp.from_geopandas(pixels, npartitions=npartitions).map_partitions(process_partition).compute()
