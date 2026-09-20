"""XYZ tile identifiers, discovery, and Web-Mercator bounds."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mercantile


TILE_SIZE = 256
ZOOM = 19


@dataclass(frozen=True, order=True)
class TileKey:
    x: int
    y: int

    @property
    def tile_id(self) -> str:
        return f"{ZOOM}_{self.x}_{self.y}"


def tile_bounds_mercator(key: TileKey) -> tuple[float, float, float, float]:
    """Return one XYZ tile's EPSG:3857 bounds."""
    bounds = mercantile.xy_bounds(key.x, key.y, ZOOM)
    return bounds.left, bounds.bottom, bounds.right, bounds.top


def discover_tiles(dataset_root: Path) -> dict[TileKey, Path]:
    """Discover the fixed zoom-level XYZ PNG grid."""
    tile_root = dataset_root / "orthophotos_2016" / "12357" / str(ZOOM)
    tiles: dict[TileKey, Path] = {}
    for path in tile_root.glob("*/*.png"):
        key = TileKey(int(path.parent.name), int(path.stem))
        tiles[key] = path
    if not tiles:
        raise FileNotFoundError(f"No source tiles found below {tile_root}")
    return tiles
