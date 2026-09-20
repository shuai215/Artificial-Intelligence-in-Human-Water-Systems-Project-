"""Inspect the raw Berlin green-roof segmentation dataset with GIS libraries."""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import geopandas as gpd
import mercantile
import numpy as np
import rasterio
from pandas.api.types import is_string_dtype
from pyproj import Geod
from rasterio.errors import NotGeoreferencedWarning
from shapely.geometry import box


def png_metadata(path: Path) -> dict[str, Any]:
    """Return dimensions and color metadata through Rasterio."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with rasterio.open(path) as dataset:
            color_channels = tuple(channel.name for channel in dataset.colorinterp)
            color_type = {
                ("gray",): "grayscale",
                ("red", "green", "blue"): "RGB",
                ("red", "green", "blue", "alpha"): "RGBA",
            }.get(color_channels, "+".join(color_channels))
            dtype = np.dtype(dataset.dtypes[0])
            width = dataset.width
            height = dataset.height
    return {
        "width": width,
        "height": height,
        "bit_depth": dtype.itemsize * 8,
        "color_type": color_type,
    }


def read_vector_layer(path: Path) -> gpd.GeoDataFrame:
    """Read one vector layer through GeoPandas/Pyogrio and normalize to WGS 84."""
    frame = gpd.read_file(path, engine="pyogrio")
    if frame.crs is None:
        raise ValueError(f"Vector layer has no CRS: {path}")
    return frame.to_crs("EPSG:4326")


def shapefile_metadata(path: Path, frame: gpd.GeoDataFrame) -> dict[str, Any]:
    """Summarize Shapefile geometry and attributes with GeoPandas."""
    fields = [
        {"name": name, "dtype": str(frame[name].dtype)}
        for name in frame.columns
        if name != frame.geometry.name
    ]
    categorical_counts: dict[str, dict[str, int]] = {}
    numeric_summaries: dict[str, dict[str, float]] = {}
    for name in (field["name"] for field in fields):
        series = frame[name].dropna()
        if not series.empty and is_string_dtype(series.dtype):
            counts = series.astype(str).value_counts()
            if len(counts) <= 20:
                categorical_counts[name] = {
                    str(value): int(count) for value, count in counts.items()
                }
        elif name.lower() == "area" and not series.empty:
            numeric_values = series.astype(float)
            numeric_summaries[name] = {
                "min": float(numeric_values.min()),
                "p25": float(numeric_values.quantile(0.25)),
                "median": float(numeric_values.median()),
                "p75": float(numeric_values.quantile(0.75)),
                "mean": float(numeric_values.mean()),
                "max": float(numeric_values.max()),
                "zero_count": int((numeric_values == 0).sum()),
            }

    xmin, ymin, xmax, ymax = map(float, frame.total_bounds)
    return {
        "file": str(path),
        "shape_type": sorted(set(frame.geometry.geom_type)),
        "record_count": len(frame),
        "bbox_wgs84": [xmin, ymin, xmax, ymax],
        "fields": fields,
        "categorical_counts": categorical_counts,
        "numeric_summaries": numeric_summaries,
        "projection": frame.crs.to_wkt(),
    }


def geometry_coverage(
    frame: gpd.GeoDataFrame,
    coverage_bbox: list[float],
) -> dict[str, int]:
    """Count features within, intersecting, and outside the imagery extent."""
    coverage = box(*coverage_bbox)
    fully_inside = int(frame.geometry.within(coverage).sum())
    intersecting = int(frame.geometry.intersects(coverage).sum())
    return {
        "fully_inside_image_extent": fully_inside,
        "partly_clipped_by_image_extent": intersecting - fully_inside,
        "outside_image_extent": len(frame) - intersecting,
    }


def inspect_dataset(root: Path) -> dict[str, Any]:
    png_root = root / "orthophotos_2016" / "12357" / "19"
    png_paths = sorted(png_root.glob("*/*.png"))
    if not png_paths:
        raise FileNotFoundError(f"No PNG tiles found below {png_root}")

    tile_coordinates = [(int(path.parent.name), int(path.stem)) for path in png_paths]
    tile_metadata_values = [png_metadata(path) for path in png_paths]
    distinct_tile_metadata = {
        tuple(metadata.items()) for metadata in tile_metadata_values
    }
    tile_sizes = [path.stat().st_size for path in png_paths]
    x_values = [x for x, _ in tile_coordinates]
    y_values = [y for _, y in tile_coordinates]
    x_counts = Counter(x_values)
    y_counts = Counter(y_values)
    zoom = 19
    northwest = mercantile.bounds(min(x_values), min(y_values), zoom)
    southeast = mercantile.bounds(max(x_values), max(y_values), zoom)
    tile_bounds = [
        northwest.west,
        southeast.south,
        southeast.east,
        northwest.north,
    ]
    center_tile = mercantile.bounds(
        round((min(x_values) + max(x_values)) / 2),
        round((min(y_values) + max(y_values)) / 2),
        zoom,
    )
    center_lat = (center_tile.south + center_tile.north) / 2
    _, _, tile_width_m = Geod(ellps="WGS84").inv(
        center_tile.west,
        center_lat,
        center_tile.east,
        center_lat,
    )
    meters_per_pixel = tile_width_m / 256

    shapefiles = [
        root / "Berlin_2016_12357_Green_city.shp",
        root / "training_labels_entire_roofs" / "Berlin_2016_12357_Green.shp",
        root
        / "training_labels_partial_roofs"
        / "Berlin_Gründächer_DachteilflächenGebäude_2016_12357.shp",
    ]

    building_footprints = gpd.read_file(
        root / "building_footprints_2016_12357.geojson",
        engine="pyogrio",
    )

    vector_layers = []
    for path in shapefiles:
        frame = read_vector_layer(path)
        metadata = shapefile_metadata(path, frame)
        metadata["image_extent_coverage"] = geometry_coverage(frame, tile_bounds)
        vector_layers.append(metadata)

    return {
        "dataset_root": str(root.resolve()),
        "orthophotos_2016": {
            "format": "XYZ Web Mercator tiles",
            "zoom": zoom,
            "tile_count": len(png_paths),
            "tile_metadata": tile_metadata_values[0],
            "all_tiles_same_metadata": len(distinct_tile_metadata) == 1,
            "file_size_bytes": {
                "min": min(tile_sizes),
                "mean": sum(tile_sizes) // len(tile_sizes),
                "max": max(tile_sizes),
            },
            "tile_x_range": [min(x_values), max(x_values)],
            "tile_y_range": [min(y_values), max(y_values)],
            "tiles_per_x_min_max": [min(x_counts.values()), max(x_counts.values())],
            "tiles_per_y_min_max": [min(y_counts.values()), max(y_counts.values())],
            "complete_rectangular_grid": len(png_paths)
            == len(x_counts) * len(y_counts),
            "bbox_wgs84": tile_bounds,
            "approx_meters_per_pixel": meters_per_pixel,
        },
        "orthophotos_2025": {
            "jp2_count": len(list((root / "orthophotos_2025").rglob("*.jp2"))),
        },
        "vector_layers": vector_layers,
        "building_footprints": {
            "format": "GeoJSON",
            "feature_count": len(building_footprints),
            "declared_crs": str(building_footprints.crs),
            "geometry_types": dict(
                Counter(building_footprints.geometry.geom_type.fillna("missing"))
            ),
            "property_fields": sorted(
                column
                for column in building_footprints.columns
                if column != building_footprints.geometry.name
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path, help="Path to green_roofs/green_roofs")
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = inspect_dataset(args.dataset_root)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
