"""Analyze positive/negative tile and pixel balance in prepared masks."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from PIL import Image


from green_roofs.manifests import SPLITS, load_manifest


def safe_ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def analyze_split(processed_root: Path, split: str) -> dict[str, int | float | None]:
    manifest_path = processed_root / "manifests" / f"{split}.csv"
    records = load_manifest(manifest_path)

    positive_tiles = 0
    foreground_pixels = 0
    total_pixels = 0
    for record in records.itertuples(index=False):
        mask_path = processed_root / record.mask_relpath
        with Image.open(mask_path) as mask:
            histogram = mask.convert("L").histogram()
        if sum(histogram[2:]) != 0:
            raise ValueError(f"Mask contains values other than 0/1: {mask_path}")
        mask_foreground = histogram[1]
        positive_tiles += mask_foreground > 0
        foreground_pixels += mask_foreground
        total_pixels += histogram[0] + histogram[1]

    tile_count = len(records)
    negative_tiles = tile_count - positive_tiles
    background_pixels = total_pixels - foreground_pixels
    return {
        "tile_count": tile_count,
        "positive_tiles": positive_tiles,
        "negative_tiles": negative_tiles,
        "positive_tile_fraction": safe_ratio(positive_tiles, tile_count),
        "negative_tile_fraction": safe_ratio(negative_tiles, tile_count),
        "negative_to_positive_tile_ratio": safe_ratio(negative_tiles, positive_tiles),
        "total_pixels": total_pixels,
        "foreground_pixels": foreground_pixels,
        "background_pixels": background_pixels,
        "foreground_pixel_fraction": safe_ratio(foreground_pixels, total_pixels),
        "background_pixel_fraction": safe_ratio(background_pixels, total_pixels),
        "background_to_foreground_pixel_ratio": safe_ratio(
            background_pixels, foreground_pixels
        ),
    }


def combine_splits(results: dict[str, dict[str, int | float | None]]) -> dict[str, int | float | None]:
    additive_fields = (
        "tile_count",
        "positive_tiles",
        "negative_tiles",
        "total_pixels",
        "foreground_pixels",
        "background_pixels",
    )
    combined = {
        field: sum(int(results[split][field]) for split in SPLITS)
        for field in additive_fields
    }
    combined.update(
        {
            "positive_tile_fraction": safe_ratio(
                combined["positive_tiles"], combined["tile_count"]
            ),
            "negative_tile_fraction": safe_ratio(
                combined["negative_tiles"], combined["tile_count"]
            ),
            "negative_to_positive_tile_ratio": safe_ratio(
                combined["negative_tiles"], combined["positive_tiles"]
            ),
            "foreground_pixel_fraction": safe_ratio(
                combined["foreground_pixels"], combined["total_pixels"]
            ),
            "background_pixel_fraction": safe_ratio(
                combined["background_pixels"], combined["total_pixels"]
            ),
            "background_to_foreground_pixel_ratio": safe_ratio(
                combined["background_pixels"], combined["foreground_pixels"]
            ),
        }
    )
    return combined


def percentage(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.4%}"


def write_results(
    output_dir: Path,
    processed_root: Path,
    results: dict[str, dict[str, int | float | None]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "processed_root": str(processed_root.resolve()),
        "definitions": {
            "positive_tile": "A tile whose binary mask contains at least one foreground pixel.",
            "foreground_pixel": "A green-roof mask pixel with value 1.",
            "background_pixel": "A non-green-roof mask pixel with value 0.",
        },
        "splits": results,
    }
    (output_dir / "class_balance_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    fields = ["split", *results["all"].keys()]
    with (output_dir / "class_balance_by_split.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for split in (*SPLITS, "all"):
            writer.writerow({"split": split, **results[split]})

    lines = [
        "# Green-roof class-balance analysis",
        "",
        "A positive tile contains at least one mask pixel with value 1. Pixel-level",
        "foreground is the green-roof class (1); background is class 0.",
        "",
        "| Split | Tiles | Positive tiles | Negative tiles | Positive tile % | Foreground pixels | Background pixels | Foreground pixel % |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split in (*SPLITS, "all"):
        row = results[split]
        lines.append(
            f"| {split} | {row['tile_count']:,} | {row['positive_tiles']:,} | "
            f"{row['negative_tiles']:,} | {percentage(row['positive_tile_fraction'])} | "
            f"{row['foreground_pixels']:,} | {row['background_pixels']:,} | "
            f"{percentage(row['foreground_pixel_fraction'])} |"
        )
    all_results = results["all"]
    lines.extend(
        [
            "",
            "## Overall ratios",
            "",
            f"- Negative tiles per positive tile: {all_results['negative_to_positive_tile_ratio']:.2f}",
            f"- Background pixels per foreground pixel: {all_results['background_to_foreground_pixel_ratio']:.2f}",
            "",
        ]
    )
    (output_dir / "class_balance_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("processed_root", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        help="Default: <processed_root>/class_balance_analysis",
    )
    args = parser.parse_args()
    processed_root = args.processed_root.resolve()
    output_dir = (
        args.output.resolve()
        if args.output
        else processed_root / "class_balance_analysis"
    )

    results = {split: analyze_split(processed_root, split) for split in SPLITS}
    results["all"] = combine_splits(results)
    write_results(output_dir, processed_root, results)
    print(json.dumps({"output_dir": str(output_dir), **results["all"]}, indent=2))


if __name__ == "__main__":
    main()
