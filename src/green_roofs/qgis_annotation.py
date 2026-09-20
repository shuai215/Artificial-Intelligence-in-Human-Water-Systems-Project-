"""Prepare a georeferenced QGIS workspace for green-roof relabeling."""

from __future__ import annotations

import json
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Mapping

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

from .manifests import load_split_manifests
from .tiles import TILE_SIZE, ZOOM, TileKey, discover_tiles, tile_bounds_mercator


BERLIN_POSTCODE_WFS = (
    "WFS:https://gdi.berlin.de/services/wfs/postleitzahlen"
    "?service=WFS&version=2.0.0&request=GetCapabilities"
)
BERLIN_POSTCODE_LAYER = "postleitzahlen:postleitzahlen"


@dataclass(frozen=True)
class MosaicSpec:
    min_x: int
    max_x: int
    min_y: int
    max_y: int
    width: int
    height: int
    pixel_size: float
    origin_x: float
    origin_y: float

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return (
            self.origin_x,
            self.origin_y - self.height * self.pixel_size,
            self.origin_x + self.width * self.pixel_size,
            self.origin_y,
        )


def mosaic_spec(tile_keys: set[TileKey]) -> MosaicSpec:
    """Describe the Web-Mercator raster covering a rectangular XYZ tile set."""
    if not tile_keys:
        raise ValueError("At least one tile is required")
    min_x = min(key.x for key in tile_keys)
    max_x = max(key.x for key in tile_keys)
    min_y = min(key.y for key in tile_keys)
    max_y = max(key.y for key in tile_keys)
    expected = {
        TileKey(x, y)
        for x in range(min_x, max_x + 1)
        for y in range(min_y, max_y + 1)
    }
    missing = expected - tile_keys
    if missing:
        examples = ", ".join(key.tile_id for key in sorted(missing)[:5])
        raise ValueError(f"Tile grid is not rectangular; missing {len(missing)} tiles: {examples}")

    left, _, right, top = tile_bounds_mercator(TileKey(min_x, min_y))
    pixel_size = (right - left) / TILE_SIZE
    return MosaicSpec(
        min_x=min_x,
        max_x=max_x,
        min_y=min_y,
        max_y=max_y,
        width=(max_x - min_x + 1) * TILE_SIZE,
        height=(max_y - min_y + 1) * TILE_SIZE,
        pixel_size=pixel_size,
        origin_x=left,
        origin_y=top,
    )


def write_mosaic_vrt(tiles: Mapping[TileKey, Path], output_path: Path) -> MosaicSpec:
    """Write a four-band VRT that places local XYZ PNGs in EPSG:3857."""
    spec = mosaic_spec(set(tiles))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    color_names = ("Red", "Green", "Blue", "Alpha")
    lines = [
        f'<VRTDataset rasterXSize="{spec.width}" rasterYSize="{spec.height}">',
        "  <SRS>EPSG:3857</SRS>",
        (
            "  <GeoTransform>"
            f"{spec.origin_x:.15f}, {spec.pixel_size:.15f}, 0, "
            f"{spec.origin_y:.15f}, 0, {-spec.pixel_size:.15f}"
            "</GeoTransform>"
        ),
    ]
    for band_index, color_name in enumerate(color_names, start=1):
        lines.extend(
            [
                f'  <VRTRasterBand dataType="Byte" band="{band_index}">',
                f"    <ColorInterp>{color_name}</ColorInterp>",
            ]
        )
        for key, tile_path in sorted(tiles.items()):
            try:
                source_path = tile_path.resolve().relative_to(
                    output_path.parent.resolve(), walk_up=True
                )
                relative_to_vrt = 1
            except ValueError:
                source_path = tile_path.resolve()
                relative_to_vrt = 0
            source_name = escape(source_path.as_posix())
            lines.extend(
                [
                    "    <SimpleSource>",
                    (
                        f'      <SourceFilename relativeToVRT="{relative_to_vrt}">'
                        f"{source_name}</SourceFilename>"
                    ),
                    f"      <SourceBand>{band_index}</SourceBand>",
                    (
                        f'      <SourceProperties RasterXSize="{TILE_SIZE}" '
                        f'RasterYSize="{TILE_SIZE}" DataType="Byte" '
                        f'BlockXSize="{TILE_SIZE}" BlockYSize="1" />'
                    ),
                    (
                        f'      <SrcRect xOff="0" yOff="0" xSize="{TILE_SIZE}" '
                        f'ySize="{TILE_SIZE}" />'
                    ),
                    (
                        f'      <DstRect xOff="{(key.x - spec.min_x) * TILE_SIZE}" '
                        f'yOff="{(key.y - spec.min_y) * TILE_SIZE}" '
                        f'xSize="{TILE_SIZE}" ySize="{TILE_SIZE}" />'
                    ),
                    "    </SimpleSource>",
                ]
            )
        lines.extend(["  </VRTRasterBand>"])
    lines.extend(["</VRTDataset>", ""])
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return spec


def read_manifest_metadata(processed_root: Path | None) -> dict[TileKey, dict[str, object]]:
    """Read frozen split and mask metadata for the optional QGIS tile grid."""
    if processed_root is None:
        return {}
    metadata: dict[TileKey, dict[str, object]] = {}
    frame = load_split_manifests(processed_root)
    for row in frame.itertuples(index=False):
        key = TileKey(int(row.x), int(row.y))
        metadata[key] = {
            "split": row.split,
            "block_id": str(row.block_id),
            "positive_pixels": int(row.positive_pixels),
            "has_green_roof": int(row.has_green_roof),
        }
    return metadata


def create_annotation_geopackage(
    dataset_root: Path,
    tiles: Mapping[TileKey, Path],
    output_path: Path,
    processed_root: Path | None = None,
) -> dict[str, int]:
    """Merge source labels and add a tile-index layer to a new GeoPackage."""
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing annotation database: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    entire_path = (
        dataset_root
        / "training_labels_entire_roofs"
        / "Berlin_2016_12357_Green.shp"
    )
    partial_path = (
        dataset_root
        / "training_labels_partial_roofs"
        / "Berlin_Gründächer_DachteilflächenGebäude_2016_12357.shp"
    )
    entire = gpd.read_file(entire_path, engine="pyogrio").to_crs("EPSG:3857")
    partial = gpd.read_file(partial_path, engine="pyogrio").to_crs("EPSG:3857")
    entire_labels = gpd.GeoDataFrame(
        {
            "plz": entire["plz"],
            "source_id": entire["bds16_id"],
            "green_type": pd.Series([None] * len(entire), dtype="string"),
            "class_name": "green_roof",
            "label_source": "entire_roof",
            "review_status": "unreviewed",
            "annotator": pd.Series([None] * len(entire), dtype="string"),
            "review_note": pd.Series([None] * len(entire), dtype="string"),
        },
        geometry=entire.geometry,
        crs=entire.crs,
    )
    partial_labels = gpd.GeoDataFrame(
        {
            "plz": partial["plz"],
            "source_id": partial["gt2016_id"],
            "green_type": partial["gruen_kat"],
            "class_name": "green_roof",
            "label_source": "partial_roof",
            "review_status": "unreviewed",
            "annotator": pd.Series([None] * len(partial), dtype="string"),
            "review_note": pd.Series([None] * len(partial), dtype="string"),
        },
        geometry=partial.geometry,
        crs=partial.crs,
    )
    labels = gpd.GeoDataFrame(
        pd.concat([entire_labels, partial_labels], ignore_index=True),
        geometry="geometry",
        crs="EPSG:3857",
    )
    labels.to_file(
        output_path,
        layer="green_roofs",
        driver="GPKG",
        engine="pyogrio",
    )

    metadata = read_manifest_metadata(processed_root)
    rows: list[dict[str, object]] = []
    geometries = []
    for key in sorted(tiles):
        properties: dict[str, object] = {"tile_id": key.tile_id, "x": key.x, "y": key.y}
        properties.update(metadata.get(key, {}))
        rows.append(properties)
        geometries.append(box(*tile_bounds_mercator(key)))
    tile_grid = gpd.GeoDataFrame(rows, geometry=geometries, crs="EPSG:3857")
    tile_grid.to_file(
        output_path,
        layer="tile_grid",
        driver="GPKG",
        engine="pyogrio",
        mode="a",
    )
    return {"label_feature_count": len(labels), "tile_feature_count": len(tile_grid)}


def create_postcode_boundary(
    output_path: Path,
    postcode: str = "12357",
) -> int:
    """Download one current official Berlin postcode polygon into EPSG:3857."""
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite postcode boundary: {output_path}")
    boundary = gpd.read_file(
        BERLIN_POSTCODE_WFS,
        layer=BERLIN_POSTCODE_LAYER,
        where=f"plz = '{postcode}'",
        engine="pyogrio",
    ).to_crs("EPSG:3857")
    if len(boundary) != 1:
        raise ValueError(
            f"Expected exactly one polygon for postcode {postcode}, got {len(boundary)}"
        )
    boundary.to_file(
        output_path,
        layer=f"plz_{postcode}",
        driver="GPKG",
        engine="pyogrio",
    )
    return len(boundary)


def prepare_qgis_annotation(
    dataset_root: Path,
    processed_root: Path,
    output_root: Path,
) -> dict[str, object]:
    """Create the VRT and editable GeoPackage; the QGIS project is a separate step."""
    tiles = discover_tiles(dataset_root)
    vrt_path = output_root / "orthophotos_2016_mosaic.vrt"
    labels_path = output_root / "green_roofs_labels_v2.gpkg"
    postcode_path = output_root / "berlin_plz_12357.gpkg"
    spec = write_mosaic_vrt(tiles, vrt_path)
    feature_counts = create_annotation_geopackage(
        dataset_root,
        tiles,
        labels_path,
        processed_root,
    )
    postcode_feature_count = create_postcode_boundary(postcode_path)
    summary = {
        "tile_count": len(tiles),
        "tile_grid": {
            "x": [spec.min_x, spec.max_x],
            "y": [spec.min_y, spec.max_y],
            "columns": spec.max_x - spec.min_x + 1,
            "rows": spec.max_y - spec.min_y + 1,
        },
        "raster_size": [spec.width, spec.height],
        "crs": "EPSG:3857",
        "pixel_size_map_units": spec.pixel_size,
        "bounds_epsg3857": list(spec.bounds),
        "vrt": str(vrt_path.resolve()),
        "geopackage": str(labels_path.resolve()),
        "postcode_boundary": str(postcode_path.resolve()),
        "postcode_boundary_source": BERLIN_POSTCODE_WFS,
        "postcode_feature_count": postcode_feature_count,
        **feature_counts,
    }
    (output_root / "annotation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary
