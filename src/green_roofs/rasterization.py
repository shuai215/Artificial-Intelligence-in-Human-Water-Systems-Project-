"""Rasterio-based generation of binary masks for XYZ tiles."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

import geopandas as gpd
import numpy as np
from PIL import Image
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry

from .tiles import TILE_SIZE, ZOOM, TileKey, tile_bounds_mercator


def rasterize_geometries(
    geometries: Iterable[BaseGeometry],
    key: TileKey,
) -> Image.Image:
    """Rasterize all EPSG:3857 geometries intersecting one XYZ tile."""
    shapes = [(geometry, 1) for geometry in geometries if not geometry.is_empty]
    if not shapes:
        return Image.fromarray(np.zeros((TILE_SIZE, TILE_SIZE), dtype=np.uint8))
    transform = from_bounds(*tile_bounds_mercator(key), TILE_SIZE, TILE_SIZE)
    mask = rasterize(
        shapes,
        out_shape=(TILE_SIZE, TILE_SIZE),
        fill=0,
        transform=transform,
        all_touched=False,
        dtype=np.uint8,
    )
    return Image.fromarray(mask)


def write_tile_masks(
    features: gpd.GeoDataFrame,
    tiles: Mapping[TileKey, Path],
    output_root: Path,
) -> tuple[dict[TileKey, int], dict[TileKey, str], int]:
    """Rasterize, save, and count masks for the complete tile grid."""
    spatial_index = features.sindex
    covered_feature_indices: set[int] = set()
    tile_stats: dict[TileKey, int] = {}
    mask_relpaths: dict[TileKey, str] = {}

    for key in sorted(tiles):
        tile_geometry = box(*tile_bounds_mercator(key))
        feature_indices = spatial_index.query(tile_geometry, predicate="intersects")
        covered_feature_indices.update(int(index) for index in feature_indices)
        mask = rasterize_geometries(features.geometry.iloc[feature_indices], key)
        positive_pixels = mask.histogram()[1]
        mask_relpath = Path("masks") / str(ZOOM) / str(key.x) / f"{key.y}.png"
        mask_path = output_root / mask_relpath
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        mask.save(mask_path, optimize=True)
        tile_stats[key] = positive_pixels
        mask_relpaths[key] = mask_relpath.as_posix()

    return tile_stats, mask_relpaths, len(covered_feature_indices)
