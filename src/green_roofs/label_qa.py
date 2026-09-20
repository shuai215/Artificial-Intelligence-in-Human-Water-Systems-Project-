from __future__ import annotations

import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image, ImageDraw

from .manifests import load_split_manifests


@dataclass(frozen=True)
class ReviewRecord:
    tile_id: str
    split: str
    image_relpath: str
    mask_relpath: str
    positive_pixels: int
    positive_fraction: float


def create_mask_overlay(
    image: Image.Image,
    mask: Image.Image,
    color: tuple[int, int, int] = (20, 255, 70),
    opacity: int = 125,
) -> Image.Image:
    """Overlay a binary mask on an RGB image with one shared rendering rule."""
    if image.size != mask.size:
        raise ValueError(f"Image/mask size mismatch: {image.size} vs {mask.size}")
    alpha = mask.point(lambda value: opacity if value else 0)
    layer = Image.new("RGB", image.size, color)
    return Image.composite(layer, image.convert("RGB"), alpha)


def load_positive_records(processed_root: Path) -> list[ReviewRecord]:
    records: list[ReviewRecord] = []
    frame = load_split_manifests(processed_root)
    for row in frame.loc[frame["has_green_roof"] == 1].itertuples(index=False):
        records.append(
            ReviewRecord(
                tile_id=row.tile_id,
                split=row.split,
                image_relpath=row.image_relpath,
                mask_relpath=row.mask_relpath,
                positive_pixels=int(row.positive_pixels),
                positive_fraction=float(row.positive_fraction),
            )
        )
    return records


def coverage_stratified_sample(
    records: list[ReviewRecord], sample_count: int, seed: int
) -> list[ReviewRecord]:
    """Select one random item from each equal-sized coverage interval."""
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    if sample_count > len(records):
        raise ValueError(
            f"Requested {sample_count} samples, but only {len(records)} are available"
        )

    ordered = sorted(records, key=lambda record: record.positive_fraction)
    generator = random.Random(seed)
    selected: list[ReviewRecord] = []
    for index in range(sample_count):
        start = index * len(ordered) // sample_count
        stop = (index + 1) * len(ordered) // sample_count
        selected.append(generator.choice(ordered[start:stop]))
    return sorted(selected, key=lambda record: record.tile_id)


def create_review_pair(
    image_path: Path,
    mask_path: Path,
    record: ReviewRecord,
) -> Image.Image:
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    with Image.open(mask_path) as source:
        mask = source.convert("L")
    if image.size != mask.size:
        raise ValueError(
            f"Image/mask size mismatch for {record.tile_id}: {image.size} vs {mask.size}"
        )
    if sum(mask.histogram()[2:]) != 0:
        raise ValueError(f"Mask contains values other than 0/1: {mask_path}")

    overlay = create_mask_overlay(image, mask)

    header_height = 28
    footer_height = 24
    canvas = Image.new(
        "RGB", (image.width * 2, header_height + image.height + footer_height), "white"
    )
    canvas.paste(image, (0, header_height))
    canvas.paste(overlay, (image.width, header_height))

    draw = ImageDraw.Draw(canvas)
    draw.text((6, 7), "Original image", fill="black")
    draw.text((image.width + 6, 7), "Label overlay (green)", fill="black")
    draw.text(
        (6, header_height + image.height + 5),
        (
            f"{record.tile_id} | {record.split} | "
            f"positive coverage {record.positive_fraction:.2%}"
        ),
        fill="black",
    )
    return canvas


def write_review_package(
    raw_root: Path,
    processed_root: Path,
    output_root: Path,
    sample_count: int = 50,
    seed: int = 42,
) -> dict[str, object]:
    if output_root.exists():
        raise FileExistsError(
            f"Output already exists: {output_root}. Choose a new path or remove it manually."
        )

    positive_records = load_positive_records(processed_root)
    selected = coverage_stratified_sample(positive_records, sample_count, seed)
    pairs_root = output_root / "pairs"
    pairs_root.mkdir(parents=True)

    review_rows: list[dict[str, object]] = []
    split_counts = {"train": 0, "val": 0, "test": 0}
    for sequence, record in enumerate(selected, start=1):
        image_path = raw_root / record.image_relpath
        mask_path = processed_root / record.mask_relpath
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        if not mask_path.is_file():
            raise FileNotFoundError(mask_path)

        pair = create_review_pair(image_path, mask_path, record)
        pair_name = f"{sequence:03d}_{record.tile_id}_{record.split}.png"
        pair.save(pairs_root / pair_name, optimize=True)
        split_counts[record.split] += 1
        review_rows.append(
            {
                "sequence": sequence,
                "pair_file": f"pairs/{pair_name}",
                **asdict(record),
                "review_status": "",
                "error_types": "",
                "notes": "",
            }
        )

    fieldnames = list(review_rows[0])
    with (output_root / "review.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(review_rows)

    summary = {
        "raw_root": str(raw_root.resolve()),
        "processed_root": str(processed_root.resolve()),
        "positive_tiles_available": len(positive_records),
        "sample_count": len(selected),
        "selection_method": "coverage-stratified random sampling",
        "seed": seed,
        "split_counts": split_counts,
        "minimum_selected_positive_fraction": min(
            record.positive_fraction for record in selected
        ),
        "maximum_selected_positive_fraction": max(
            record.positive_fraction for record in selected
        ),
    }
    (output_root / "selection_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (output_root / "README.md").write_text(
        """# Green-roof label quality review

Each image in `pairs/` shows the original 2016 orthophoto on the left and the
same image with the binary green-roof mask overlaid in green on the right.

Complete `review.csv` using:

- `review_status`: `acceptable`, `incorrect`, or `ambiguous`;
- `error_types`: one or more of `misalignment`, `over_label`, `under_label`,
  `non_roof_vegetation`, `ordinary_roof`, `shadow_or_seasonal`, separated by `;`;
- `notes`: a short explanation when useful.

Do not edit the raw labels or masks during this review. Freeze a reviewed label
version before model training, and regenerate masks/manifests if corrections are
made later.
""",
        encoding="utf-8",
    )
    return summary
