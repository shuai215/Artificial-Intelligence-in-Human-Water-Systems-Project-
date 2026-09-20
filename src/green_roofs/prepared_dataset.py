"""Prepared-dataset QA output and full image/mask validation."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from PIL import Image, ImageDraw

from .label_qa import create_mask_overlay
from .manifests import SPLITS, TileRecord, load_split_manifests
from .tiles import TILE_SIZE


def create_qa_contact_sheet(
    output_root: Path,
    dataset_root: Path,
    records: Sequence[TileRecord],
    sample_count: int = 8,
) -> None:
    """Create a compact source/mask/overlay QA sheet."""
    positive_records = sorted(records, key=lambda row: row.positive_pixels, reverse=True)
    selected = positive_records[:sample_count]
    label_height = 24
    sheet = Image.new(
        "RGB", (TILE_SIZE * 3, (TILE_SIZE + label_height) * len(selected)), "white"
    )
    sheet_draw = ImageDraw.Draw(sheet)
    for row_index, record in enumerate(selected):
        with Image.open(dataset_root / record.image_relpath) as source:
            image = source.convert("RGB")
        with Image.open(output_root / record.mask_relpath) as source:
            mask = source.convert("L")
        mask_preview = mask.point(lambda value: 255 if value else 0).convert("RGB")
        overlay = create_mask_overlay(image, mask, color=(30, 255, 80))
        top = row_index * (TILE_SIZE + label_height)
        sheet.paste(image, (0, top))
        sheet.paste(mask_preview, (TILE_SIZE, top))
        sheet.paste(overlay, (TILE_SIZE * 2, top))
        sheet_draw.text(
            (4, top + TILE_SIZE + 4),
            f"{record.key.tile_id} | {record.split} | {record.positive_fraction:.2%}",
            fill="black",
        )
    qa_root = output_root / "qa"
    qa_root.mkdir(parents=True, exist_ok=True)
    sheet.save(qa_root / "top_positive_tiles.png")


def validate_prepared_dataset(dataset_root: Path, output_root: Path) -> dict[str, object]:
    """Validate masks, manifest references, counts, and spatial-block isolation."""
    rows = load_split_manifests(output_root)
    blocks_by_split: dict[str, set[str]] = {}
    for split in SPLITS:
        blocks_by_split[split] = set(
            rows.loc[rows["split"] == split, "block_id"]
        )

    tile_ids = rows["tile_id"]
    if tile_ids.duplicated().any():
        raise ValueError("A tile appears in more than one manifest")
    for first_split, first_blocks in blocks_by_split.items():
        for second_split, second_blocks in blocks_by_split.items():
            if first_split < second_split and first_blocks & second_blocks:
                raise ValueError(
                    f"Spatial block leakage between {first_split} and {second_split}"
                )

    positive_pixels = 0
    positive_tiles = 0
    for row in rows.itertuples(index=False):
        image_path = dataset_root / row.image_relpath
        mask_path = output_root / row.mask_relpath
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        if not mask_path.is_file():
            raise FileNotFoundError(mask_path)
        with Image.open(mask_path) as mask:
            if mask.size != (TILE_SIZE, TILE_SIZE) or mask.mode != "L":
                raise ValueError(f"Unexpected mask format: {mask_path}")
            colors = mask.getcolors(maxcolors=3)
            if colors is None or any(value not in {0, 1} for _, value in colors):
                raise ValueError(f"Mask is not binary 0/1: {mask_path}")
            actual_positive = sum(count for count, value in colors if value == 1)
        expected_positive = int(row.positive_pixels)
        if actual_positive != expected_positive:
            raise ValueError(f"Manifest pixel count mismatch: {mask_path}")
        positive_pixels += actual_positive
        positive_tiles += actual_positive > 0

    mask_count = len(list((output_root / "masks").rglob("*.png")))
    if mask_count != len(rows):
        raise ValueError(f"Found {mask_count} masks for {len(rows)} manifest rows")
    return {
        "valid": True,
        "tile_count": len(rows),
        "unique_tile_count": int(tile_ids.nunique()),
        "mask_count": mask_count,
        "positive_tile_count": positive_tiles,
        "positive_pixel_count": positive_pixels,
        "spatial_blocks_by_split": {
            split: len(blocks) for split, blocks in blocks_by_split.items()
        },
        "spatial_block_leakage": False,
    }
