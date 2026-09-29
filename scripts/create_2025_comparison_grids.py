"""Create a supplementary 50-case diagnostic for the two PLZ models.

The formal 2025 temporal-transfer output is the better-performing U-Net alone.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from green_roofs.label_qa import create_mask_overlay
from green_roofs.tiles import TILE_SIZE


MODELS = {
    "A": ("unet_plz12357_all", "A  ResNet50 U-Net"),
    "B": ("segformer_b0_plz12357", "B  SegFormer-B0"),
}
QUOTAS = {
    "AB_overlap": 15,
    "AB_disjoint": 5,
    "A_only": 10,
    "B_only": 10,
    "none": 10,
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


def category(masks: dict[str, np.ndarray]) -> str:
    a_active, b_active = masks["A"].any(), masks["B"].any()
    if a_active and b_active:
        return "AB_overlap" if np.logical_and(masks["A"], masks["B"]).any() else "AB_disjoint"
    if a_active:
        return "A_only"
    if b_active:
        return "B_only"
    return "none"


def select_evenly(candidates: list[Candidate], count: int) -> list[Candidate]:
    ordered = sorted(candidates, key=lambda item: (item.score, item.tile_id))
    if len(ordered) <= count:
        return ordered
    indices = np.linspace(0, len(ordered) - 1, count).round().astype(int)
    return [ordered[index] for index in indices]


def overlay(image: Image.Image, mask: np.ndarray) -> Image.Image:
    mask_image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    return create_mask_overlay(image, mask_image, color=(255, 30, 120), opacity=150)


def load_mask(path: Path) -> np.ndarray:
    with Image.open(path) as source:
        return np.array(source, dtype=bool)


def agreement_panel(image: Image.Image, masks: dict[str, np.ndarray]) -> Image.Image:
    pixels = (np.asarray(image.convert("RGB"), dtype=np.float32) * 0.35).astype(np.uint8)
    both = masks["A"] & masks["B"]
    pixels[masks["A"] & ~masks["B"]] = (255, 70, 170)
    pixels[masks["B"] & ~masks["A"]] = (40, 190, 255)
    pixels[both] = (40, 220, 90)
    return Image.fromarray(pixels, mode="RGB")


def comparison_image(root: Path, candidate: Candidate) -> Image.Image:
    zoom, x, y = candidate.tile_id.split("_")
    with Image.open(root / "images" / zoom / x / f"{y}.png") as source:
        image = source.convert("RGB")

    header, label_height, size = 36, 24, TILE_SIZE
    canvas = Image.new("RGB", (size * 2, header + 2 * (label_height + size)), "#171717")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (8, 11),
        f"{candidate.tile_id} | {candidate.category} | score={candidate.score:.3f}",
        fill="white",
    )
    panels = [
        ("2025 RGB", image),
        (f"A  U-Net | {candidate.pixels['A'] / (size * size):.3%}", overlay(image, candidate.masks["A"])),
        (f"B  SegFormer-B0 | {candidate.pixels['B'] / (size * size):.3%}", overlay(image, candidate.masks["B"])),
        ("agreement: green=both, pink=A, blue=B", agreement_panel(image, candidate.masks)),
    ]
    for index, (label, panel) in enumerate(panels):
        column, row = index % 2, index // 2
        x_position = column * size
        label_y = header + row * (label_height + size)
        draw.text((x_position + 7, label_y + 6), label, fill="white")
        canvas.paste(panel, (x_position, label_y + label_height))
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
        key: pd.read_csv(root / "predictions" / directory / "manifest.csv").set_index("tile_id")
        for key, (directory, _) in MODELS.items()
    }
    tile_ids = list(manifests["A"].index)
    if any(set(frame.index) != set(tile_ids) for frame in manifests.values()):
        raise ValueError("Model manifests do not contain the same tile IDs")

    grouped = {name: [] for name in QUOTAS}
    for tile_id in tile_ids:
        masks = {
            key: load_mask(root / frame.loc[tile_id, "mask_relpath"])
            for key, frame in manifests.items()
        }
        group = category(masks)
        pixels = {key: int(mask.sum()) for key, mask in masks.items()}
        score = iou(masks["A"], masks["B"]) if group.startswith("AB") else sum(pixels.values()) / (TILE_SIZE * TILE_SIZE)
        grouped[group].append(Candidate(tile_id, group, score, masks, pixels))

    selected = [
        candidate
        for group, quota in QUOTAS.items()
        for candidate in select_evenly(grouped[group], quota)
    ]
    selected_ids = {(candidate.category, candidate.tile_id) for candidate in selected}
    if len(selected) < 50:
        remaining = [
            candidate
            for candidates in grouped.values()
            for candidate in candidates
            if (candidate.category, candidate.tile_id) not in selected_ids
        ]
        selected.extend(select_evenly(remaining, 50 - len(selected)))
    if len(selected) != 50:
        raise ValueError(f"Expected 50 review candidates, found {len(selected)}")

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
                "A_pixels": candidate.pixels["A"],
                "B_pixels": candidate.pixels["B"],
                "AB_iou": iou(candidate.masks["A"], candidate.masks["B"]),
                "file": filename,
            }
        )
    with (output / "index.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    counts = {name: sum(row["category"] == name for row in rows) for name in QUOTAS}
    lines = [
        "# 2025 PLZ-model prediction review",
        "",
        "Each image shows 2025 RGB, PLZ-trained ResNet50 U-Net, PLZ-trained",
        "SegFormer-B0, and a colour-coded agreement/disagreement panel.",
        "The sample is deliberately stratified and must not be used to estimate",
        "population prevalence.",
        "",
        "## Selection counts",
        "",
        *[f"- `{name}`: {count}" for name, count in counts.items()],
    ]
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"created={len(selected)} output={output}")


if __name__ == "__main__":
    main()
