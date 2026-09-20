"""Study-area loading and tile-centre selection."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import geopandas as gpd
from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry

from .labels import WEB_MERCATOR_CRS
from .tiles import TileKey, tile_bounds_mercator


def load_study_area(path: Path) -> BaseGeometry:
    """Load polygon boundaries and return their EPSG:3857 union."""
    frame = gpd.read_file(path, engine="pyogrio")
    if frame.crs is None or frame.empty:
        raise ValueError(f"Study area must contain geometries with a CRS: {path}")
    if not set(frame.geometry.geom_type) <= {"Polygon", "MultiPolygon"}:
        raise ValueError(f"Study area must contain only polygons: {path}")
    if not frame.geometry.is_valid.all():
        raise ValueError(f"Study area contains invalid geometries: {path}")
    return frame.to_crs(WEB_MERCATOR_CRS).geometry.union_all()


def select_tiles_by_centre(
    tiles: Mapping[TileKey, Path], study_area: BaseGeometry
) -> dict[TileKey, Path]:
    """Keep tiles whose EPSG:3857 centre is covered by the study area."""
    selected = {}
    for key, path in tiles.items():
        left, bottom, right, top = tile_bounds_mercator(key)
        centre = Point((left + right) / 2, (bottom + top) / 2)
        if study_area.covers(centre):
            selected[key] = path
    if not selected:
        raise ValueError("Study area does not contain any tile centres")
    return selected
