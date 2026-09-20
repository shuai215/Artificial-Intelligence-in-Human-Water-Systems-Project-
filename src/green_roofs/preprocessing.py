"""High-level orchestration for the Berlin green-roof preprocessing pipeline.

The public imports previously exposed by this module remain available while
their implementations live in focused modules.
"""

from __future__ import annotations

import json
from pathlib import Path

from .labels import (
    WEB_MERCATOR_CRS,
    load_polygon_labels,
    polygon_topology_summary,
    read_polygon_shapefile,
)
from .manifests import SPLITS, TileRecord, write_manifests
from .prepared_dataset import create_qa_contact_sheet, validate_prepared_dataset
from .rasterization import rasterize_geometries, write_tile_masks
from .spatial_split import assign_spatial_splits
from .study_area import load_study_area, select_tiles_by_centre
from .tiles import TILE_SIZE, ZOOM, TileKey, discover_tiles, tile_bounds_mercator


__all__ = [
    "TILE_SIZE",
    "WEB_MERCATOR_CRS",
    "ZOOM",
    "TileKey",
    "TileRecord",
    "assign_spatial_splits",
    "discover_tiles",
    "prepare_dataset",
    "rasterize_geometries",
    "read_polygon_shapefile",
    "select_tiles_by_centre",
    "tile_bounds_mercator",
    "validate_prepared_dataset",
]


def prepare_dataset(
    dataset_root: Path,
    output_root: Path,
    block_size: int = 8,
    seed: int = 42,
    study_area_path: Path | None = None,
) -> dict[str, object]:
    """Create binary masks, spatial manifests, statistics, and QA previews."""
    if output_root.exists():
        raise FileExistsError(
            f"Output already exists: {output_root}. Remove it or choose another path."
        )
    output_root.mkdir(parents=True)

    discovered_tiles = discover_tiles(dataset_root)
    tiles = (
        select_tiles_by_centre(discovered_tiles, load_study_area(study_area_path))
        if study_area_path is not None
        else discovered_tiles
    )
    label_paths = [
        dataset_root
        / "training_labels_entire_roofs"
        / "Berlin_2016_12357_Green.shp",
        dataset_root
        / "training_labels_partial_roofs"
        / "Berlin_Gründächer_DachteilflächenGebäude_2016_12357.shp",
    ]
    features = load_polygon_labels(label_paths)
    tile_stats, mask_relpaths, covered_feature_count = write_tile_masks(
        features,
        tiles,
        output_root,
    )

    assignments, split_block_stats = assign_spatial_splits(
        tile_stats, block_size=block_size, seed=seed
    )
    records = [
        TileRecord(
            key=key,
            image_relpath=image_path.relative_to(dataset_root).as_posix(),
            mask_relpath=mask_relpaths[key],
            positive_pixels=tile_stats[key],
            block_id=assignments[key][1],
            split=assignments[key][0],
        )
        for key, image_path in tiles.items()
    ]
    write_manifests(output_root, records)
    create_qa_contact_sheet(output_root, dataset_root, records)

    split_stats: dict[str, dict[str, int | float]] = {}
    for split in SPLITS:
        split_records = [record for record in records if record.split == split]
        split_stats[split] = {
            **split_block_stats[split],
            "positive_tiles": sum(record.positive_pixels > 0 for record in split_records),
            "positive_fraction": sum(
                record.positive_pixels for record in split_records
            )
            / (len(split_records) * TILE_SIZE * TILE_SIZE),
        }

    summary: dict[str, object] = {
        "dataset_root": str(dataset_root.resolve()),
        "output_root": str(output_root.resolve()),
        "study_area": (
            {
                "path": str(study_area_path.resolve()),
                "selection_rule": "tile centre covered by polygon",
                "excluded_tile_count": len(discovered_tiles) - len(tiles),
            }
            if study_area_path is not None
            else None
        ),
        "discovered_tile_count": len(discovered_tiles),
        "tile_count": len(tiles),
        "label_feature_count": len(features),
        "label_features_intersecting_images": covered_feature_count,
        "polygon_topology": polygon_topology_summary(features),
        "positive_tile_count": sum(pixels > 0 for pixels in tile_stats.values()),
        "positive_pixel_count": sum(tile_stats.values()),
        "positive_pixel_fraction": sum(tile_stats.values())
        / (len(tile_stats) * TILE_SIZE * TILE_SIZE),
        "split_block_size_tiles": block_size,
        "split_seed": seed,
        "splits": split_stats,
    }
    (output_root / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary
