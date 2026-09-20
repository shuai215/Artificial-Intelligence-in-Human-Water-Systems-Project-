"""Create 50 representative RGB/model four-panel review images."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from green_roofs.label_qa import create_mask_overlay
from green_roofs.tiles import TILE_SIZE


MODELS = {
    "A": ("unet_plz12357_all", "A  Main U-Net"),
    "B": ("unet_baseline_v3", "B  U-Net v3"),
    "C": ("segformer_b0", "C  SegFormer-B0"),
}
QUOTAS = {
    "ABC_overlap": 9,
    "AB_only_overlap": 7,
    "AC_only_overlap": 12,
    "BC_only_overlap": 1,
    "A_only": 7,
    "B_only": 5,
    "C_only": 5,
    "none": 4,
}


@dataclass
class Candidate:
    tile_id: str
    category: str
    score: float
    masks: dict[str, np.ndarray]
    pixels: dict[str, int]


def iou(first: np.ndarray, second: np.ndarray) -> float:
    union = np.logical_or(first, second).sum()
    return float(np.logical_and(first, second).sum() / union) if union else 0.0


def category(masks: dict[str, np.ndarray]) -> str | None:
    active = "".join(name for name, mask in masks.items() if mask.any())
    if active == "ABC" and np.logical_and.reduce(list(masks.values())).any():
        return "ABC_overlap"
    if active in {"AB", "AC", "BC"}:
        first, second = active
        if np.logical_and(masks[first], masks[second]).any():
            return f"{active}_only_overlap"
    if active in MODELS:
        return f"{active}_only"
    return "none" if not active else None


def select_evenly(candidates: list[Candidate], count: int) -> list[Candidate]:
    ordered = sorted(candidates, key=lambda item: (item.score, item.tile_id))
    if len(ordered) <= count:
        return ordered
    indices = np.linspace(
        0.1 * (len(ordered) - 1), 0.9 * (len(ordered) - 1), count
    ).round().astype(int)
    return [ordered[index] for index in indices]


def overlay(image: Image.Image, mask: np.ndarray) -> Image.Image:
    mask_image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    return create_mask_overlay(image, mask_image, color=(255, 30, 120), opacity=150)


def comparison_image(root: Path, candidate: Candidate) -> Image.Image:
    parts = candidate.tile_id.split("_")
    image_path = root / "images" / parts[0] / parts[1] / f"{parts[2]}.png"
    with Image.open(image_path) as source:
        image = source.convert("RGB")

    header, label_height, size = 36, 24, TILE_SIZE
    canvas = Image.new("RGB", (size * 2, header + 2 * (label_height + size)), "#171717")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (8, 11),
        f"{candidate.tile_id} | {candidate.category} | selection score={candidate.score:.3f}",
        fill="white",
    )
    panels = [("2025 RGB", image)] + [
        (
            f"{label} | {candidate.pixels[key] / (size * size):.3%}",
            overlay(image, candidate.masks[key]),
        )
        for key, (_, label) in MODELS.items()
    ]
    for index, (label, panel) in enumerate(panels):
        column, row = index % 2, index // 2
        x = column * size
        label_y = header + row * (label_height + size)
        draw.text((x + 7, label_y + 6), label, fill="white")
        canvas.paste(panel, (x, label_y + label_height))
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prediction_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = args.prediction_root.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")

    manifests = {
        key: pd.read_csv(root / "predictions" / directory / "manifest.csv").set_index(
            "tile_id"
        )
        for key, (directory, _) in MODELS.items()
    }
    tile_ids = list(manifests["A"].index)
    if any(set(frame.index) != set(tile_ids) for frame in manifests.values()):
        raise ValueError("Model manifests do not contain the same tile IDs")

    grouped = {name: [] for name in QUOTAS}
    for tile_id in tile_ids:
        masks = {
            key: np.array(Image.open(root / frame.loc[tile_id, "mask_relpath"]), dtype=bool)
            for key, frame in manifests.items()
        }
        group = category(masks)
        if group not in grouped:
            continue
        active = [key for key, mask in masks.items() if mask.any()]
        overlaps = [iou(masks[a], masks[b]) for a, b in combinations(active, 2)]
        pixels = {key: int(mask.sum()) for key, mask in masks.items()}
        score = (
            sum(overlaps) / len(overlaps)
            if overlaps
            else sum(pixels.values()) / (TILE_SIZE * TILE_SIZE)
        )
        grouped[group].append(Candidate(tile_id, group, score, masks, pixels))

    selected = [
        candidate
        for group, quota in QUOTAS.items()
        for candidate in select_evenly(grouped[group], quota)
    ]
    if len(selected) != sum(QUOTAS.values()):
        raise ValueError("Not enough candidates to satisfy the review quotas")

    output.mkdir(parents=True)
    rows = []
    for rank, candidate in enumerate(selected, start=1):
        filename = f"{rank:02d}_{candidate.category}_{candidate.tile_id}.png"
        comparison_image(root, candidate).save(output / filename)
        rows.append(
            {
                "rank": rank,
                "category": candidate.category,
                "tile_id": candidate.tile_id,
                "selection_score": candidate.score,
                **{f"{key}_pixels": candidate.pixels[key] for key in MODELS},
                "AB_iou": iou(candidate.masks["A"], candidate.masks["B"]),
                "AC_iou": iou(candidate.masks["A"], candidate.masks["C"]),
                "BC_iou": iou(candidate.masks["B"], candidate.masks["C"]),
                "file": filename,
            }
        )
    with (output / "index.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# 2025 four-panel prediction review",
        "",
        "Each image shows 2025 RGB, A (main U-Net), B (U-Net v3), and C (SegFormer-B0).",
        "Predictions are magenta overlays. The sample is deliberately stratified by model",
        "agreement/disagreement and must not be used to estimate population prevalence.",
        "",
        "## Selection counts",
        "",
        *[f"- `{name}`: {count}" for name, count in QUOTAS.items()],
    ]
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"created={len(selected)} output={output}")


if __name__ == "__main__":
    main()
