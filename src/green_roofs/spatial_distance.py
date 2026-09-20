"""Spatial-distance diagnostics for train and prediction tile locations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import mercantile
import numpy as np
from pyproj import Transformer
from scipy.spatial import cKDTree

from .manifests import load_manifest
from .tiles import ZOOM


DISTANCE_CRS = "EPSG:25833"


@dataclass(frozen=True)
class ManifestTile:
    tile_id: str
    split: str
    x: int
    y: int


def read_manifest(path: Path, split: str) -> list[ManifestTile]:
    """Read tile identifiers and XYZ indices from one prepared manifest."""
    frame = load_manifest(path)
    return [
        ManifestTile(
            tile_id=row.tile_id,
            split=split,
            x=int(row.x),
            y=int(row.y),
        )
        for row in frame.itertuples(index=False)
    ]


def tile_centres_to_projected(
    tile_xy: np.ndarray,
    zoom: int = ZOOM,
    target_crs: str = DISTANCE_CRS,
) -> np.ndarray:
    """Convert XYZ tile centres from Web Mercator to a projected CRS."""
    points = np.asarray(tile_xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) == 0:
        raise ValueError("tile_xy must have non-empty shape (n, 2)")
    web_mercator_centres = np.array(
        [
            (
                (bounds.left + bounds.right) / 2,
                (bounds.bottom + bounds.top) / 2,
            )
            for x, y in points
            for bounds in [mercantile.xy_bounds(int(x), int(y), zoom)]
        ],
        dtype=np.float64,
    )
    transformer = Transformer.from_crs("EPSG:3857", target_crs, always_xy=True)
    projected_x, projected_y = transformer.transform(
        web_mercator_centres[:, 0], web_mercator_centres[:, 1]
    )
    return np.column_stack((projected_x, projected_y))


def nearest_distances(
    query_points: np.ndarray,
    reference_points: np.ndarray,
    *,
    exclude_matching_index: bool = False,
    chunk_size: int = 512,
) -> np.ndarray:
    """Return Euclidean distance to the nearest reference using a k-d tree."""
    query = np.asarray(query_points, dtype=np.float64)
    reference = np.asarray(reference_points, dtype=np.float64)
    if query.ndim != 2 or reference.ndim != 2 or query.shape[1:] != reference.shape[1:]:
        raise ValueError("query_points and reference_points must have shape (n, d)")
    if len(query) == 0 or len(reference) == 0:
        raise ValueError("query_points and reference_points must be non-empty")
    if exclude_matching_index and len(query) != len(reference):
        raise ValueError("Self-exclusion requires equally sized query and reference arrays")
    if exclude_matching_index and len(reference) < 2:
        raise ValueError("At least two reference points are required for self-exclusion")

    tree = cKDTree(reference)
    result = np.empty(len(query), dtype=np.float64)
    for start in range(0, len(query), chunk_size):
        stop = min(start + chunk_size, len(query))
        if exclude_matching_index:
            distances, indices = tree.query(query[start:stop], k=2)
            matching_indices = np.arange(start, stop)
            result[start:stop] = np.where(
                indices[:, 0] == matching_indices,
                distances[:, 1],
                distances[:, 0],
            )
        else:
            result[start:stop] = tree.query(query[start:stop], k=1)[0]
    return result


def log_distance_kde(
    distances_m: np.ndarray,
    log10_grid: np.ndarray,
    *,
    minimum_bandwidth: float = 0.035,
) -> np.ndarray:
    """Estimate density in log10-distance space using Silverman's bandwidth."""
    distances = np.asarray(distances_m, dtype=np.float64)
    if np.any(~np.isfinite(distances)) or np.any(distances <= 0):
        raise ValueError("KDE distances must be finite and strictly positive")
    values = np.log10(distances)
    standard_deviation = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    interquartile_range = float(np.subtract(*np.percentile(values, [75, 25])))
    robust_scale = min(standard_deviation, interquartile_range / 1.34)
    if not math.isfinite(robust_scale) or robust_scale <= 0:
        robust_scale = standard_deviation
    bandwidth = max(0.9 * robust_scale * len(values) ** (-0.2), minimum_bandwidth)
    standardized = (log10_grid[:, None] - values[None, :]) / bandwidth
    density = np.exp(-0.5 * standardized**2).sum(axis=1)
    density /= len(values) * bandwidth * math.sqrt(2.0 * math.pi)
    return density
