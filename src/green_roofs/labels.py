"""Vector-label loading, CRS normalization, and topology summaries."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import geopandas as gpd
import pandas as pd
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry


WEB_MERCATOR_CRS = "EPSG:3857"


def read_polygon_shapefile(path: Path) -> gpd.GeoDataFrame:
    """Read a polygon layer and project it directly to EPSG:3857."""
    frame = gpd.read_file(path, engine="pyogrio")
    if frame.crs is None:
        raise ValueError(f"Shapefile has no declared CRS: {path}")
    if frame.empty:
        raise ValueError(f"Shapefile contains no features: {path}")
    unsupported = set(frame.geometry.geom_type) - {"Polygon", "MultiPolygon"}
    if unsupported:
        raise ValueError(f"Unsupported geometry types {sorted(unsupported)}: {path}")
    invalid = ~frame.geometry.is_valid
    if invalid.any():
        raise ValueError(f"Found {int(invalid.sum())} invalid geometries: {path}")
    return frame.to_crs(WEB_MERCATOR_CRS)[["geometry"]].copy()


def load_polygon_labels(paths: Iterable[Path]) -> gpd.GeoDataFrame:
    """Load source label layers already normalized to EPSG:3857 and combine them."""
    frames = [read_polygon_shapefile(path) for path in paths]
    if not frames:
        raise ValueError("At least one polygon label path is required")
    combined = gpd.GeoDataFrame(
        pd.concat(frames, ignore_index=True),
        geometry="geometry",
        crs=WEB_MERCATOR_CRS,
    )
    return combined


def _polygon_parts(geometry: BaseGeometry) -> tuple[Polygon, ...]:
    if isinstance(geometry, Polygon):
        return (geometry,)
    if isinstance(geometry, MultiPolygon):
        return tuple(geometry.geoms)
    raise ValueError(f"Expected Polygon or MultiPolygon, got {geometry.geom_type}")


def polygon_topology_summary(frame: gpd.GeoDataFrame) -> dict[str, int]:
    """Summarize exterior/interior topology without ring-orientation inference."""
    parts = [
        part
        for geometry in frame.geometry
        for part in _polygon_parts(geometry)
    ]
    return {
        "polygon_parts": len(parts),
        "interior_rings": sum(len(part.interiors) for part in parts),
    }
