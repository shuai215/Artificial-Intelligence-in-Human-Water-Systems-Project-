from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


HEADER_HEIGHT = 24
PANEL_SIZE = 256


def load_binary_mask(path: Path) -> np.ndarray:
    """Load a single-channel label image and normalize all foreground to True."""
    with Image.open(path) as image:
        mask = np.asarray(image)

    if mask.ndim != 2:
        raise ValueError(f"Expected a single-channel mask, got shape {mask.shape}: {path}")
    return mask > 0


def binary_preview(mask: np.ndarray) -> Image.Image:
    pixels = mask.astype(np.uint8) * 255
    return Image.fromarray(pixels, mode="L").convert("RGB")


def comparison_overlay(mask_a: np.ndarray, mask_b: np.ndarray) -> Image.Image:
    pixels = np.zeros((*mask_a.shape, 3), dtype=np.uint8)
    pixels[mask_a & mask_b] = (0, 210, 80)  # common foreground: green
    pixels[mask_a & ~mask_b] = (255, 45, 45)  # A only: red
    pixels[~mask_a & mask_b] = (45, 120, 255)  # B only: blue
    return Image.fromarray(pixels, mode="RGB")


def satellite_overlay(
    satellite: Image.Image,
    label_1: np.ndarray,
    label_2: np.ndarray,
    alpha: float = 0.62,
) -> Image.Image:
    base = np.asarray(satellite.convert("RGB"), dtype=np.float32).copy()
    colors = np.zeros_like(base)
    colors[label_1 & label_2] = (0, 255, 80)  # common foreground: green
    colors[label_1 & ~label_2] = (255, 35, 35)  # label 1 only: red
    colors[~label_1 & label_2] = (35, 110, 255)  # label 2 only: blue
    foreground = label_1 | label_2
    base[foreground] = (
        base[foreground] * (1.0 - alpha) + colors[foreground] * alpha
    )
    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), mode="RGB")


def difference_preview(mask_a: np.ndarray, mask_b: np.ndarray) -> Image.Image:
    pixels = np.zeros((*mask_a.shape, 3), dtype=np.uint8)
    pixels[mask_a & ~mask_b] = (255, 45, 45)
    pixels[~mask_a & mask_b] = (45, 120, 255)
    return Image.fromarray(pixels, mode="RGB")


def add_panel(canvas: Image.Image, panel: Image.Image, x: int, y: int, title: str) -> None:
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((x, y, x + PANEL_SIZE - 1, y + HEADER_HEIGHT - 1), fill=(35, 35, 35))
    draw.text((x + 6, y + 6), title, fill=(255, 255, 255))
    canvas.paste(panel, (x, y + HEADER_HEIGHT))


def build_comparison(mask_a: np.ndarray, mask_b: np.ndarray) -> Image.Image:
    if mask_a.shape != mask_b.shape:
        raise ValueError(f"Mask shapes differ: {mask_a.shape} != {mask_b.shape}")
    if mask_a.shape != (PANEL_SIZE, PANEL_SIZE):
        raise ValueError(f"Expected 256x256 masks, got {mask_a.shape}")

    canvas = Image.new(
        "RGB",
        (PANEL_SIZE * 2, (PANEL_SIZE + HEADER_HEIGHT) * 2),
        color=(20, 20, 20),
    )
    add_panel(canvas, binary_preview(mask_a), 0, 0, "A: training label")
    add_panel(canvas, binary_preview(mask_b), PANEL_SIZE, 0, "B: PLZ label")
    add_panel(
        canvas,
        comparison_overlay(mask_a, mask_b),
        0,
        PANEL_SIZE + HEADER_HEIGHT,
        "Overlay: green=both, red=A, blue=B",
    )
    add_panel(
        canvas,
        difference_preview(mask_a, mask_b),
        PANEL_SIZE,
        PANEL_SIZE + HEADER_HEIGHT,
        "Differences only",
    )
    return canvas


def build_satellite_comparison(
    satellite: Image.Image,
    label_1: np.ndarray,
    label_2: np.ndarray,
) -> Image.Image:
    if label_1.shape != label_2.shape:
        raise ValueError(f"Mask shapes differ: {label_1.shape} != {label_2.shape}")
    if label_1.shape != (PANEL_SIZE, PANEL_SIZE):
        raise ValueError(f"Expected 256x256 masks, got {label_1.shape}")
    if satellite.size != (PANEL_SIZE, PANEL_SIZE):
        raise ValueError(f"Expected a 256x256 satellite image, got {satellite.size}")

    canvas = Image.new(
        "RGB",
        (PANEL_SIZE * 2, (PANEL_SIZE + HEADER_HEIGHT) * 2),
        color=(20, 20, 20),
    )
    add_panel(canvas, satellite.convert("RGB"), 0, 0, "1. Original satellite")
    add_panel(
        canvas,
        binary_preview(label_1),
        PANEL_SIZE,
        0,
        "2. Label 1: PLZ",
    )
    add_panel(
        canvas,
        binary_preview(label_2),
        0,
        PANEL_SIZE + HEADER_HEIGHT,
        "3. Label 2: training",
    )
    add_panel(
        canvas,
        satellite_overlay(satellite, label_1, label_2),
        PANEL_SIZE,
        PANEL_SIZE + HEADER_HEIGHT,
        "4. Overlay: green=both, red=L1, blue=L2",
    )
    return canvas


def compare_and_write(
    root_a: Path,
    root_b: Path,
    output_root: Path,
    image_root: Path | None = None,
) -> dict[str, int]:
    files_a = {path.relative_to(root_a): path for path in root_a.rglob("*.png")}
    files_b = {path.relative_to(root_b): path for path in root_b.rglob("*.png")}
    common_paths = sorted(files_a.keys() & files_b.keys())

    written = 0
    different_pixels = 0
    for relative_path in common_paths:
        mask_a = load_binary_mask(files_a[relative_path])
        mask_b = load_binary_mask(files_b[relative_path])
        difference_count = int(np.count_nonzero(mask_a != mask_b))
        if difference_count == 0:
            continue

        output_path = output_root / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if image_root is None:
            preview = build_comparison(mask_a, mask_b)
        else:
            image_path = image_root / relative_path
            if not image_path.is_file():
                raise FileNotFoundError(f"Satellite image not found: {image_path}")
            with Image.open(image_path) as satellite:
                preview = build_satellite_comparison(satellite, mask_a, mask_b)
        preview.save(output_path)
        written += 1
        different_pixels += difference_count

    return {
        "files_a": len(files_a),
        "files_b": len(files_b),
        "common_files": len(common_paths),
        "only_a": len(files_a.keys() - files_b.keys()),
        "only_b": len(files_b.keys() - files_a.keys()),
        "different_files_written": written,
        "different_pixels": different_pixels,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write four-panel previews only for binary masks whose pixels differ."
    )
    parser.add_argument("root_a", type=Path, help="First mask root")
    parser.add_argument("root_b", type=Path, help="Second mask root")
    parser.add_argument("output_root", type=Path, help="Preview output root")
    parser.add_argument(
        "--image-root",
        type=Path,
        help=(
            "Optional satellite-image root. When set, create satellite/label 1/"
            "label 2/satellite-overlay panels."
        ),
    )
    args = parser.parse_args()

    summary = compare_and_write(
        args.root_a,
        args.root_b,
        args.output_root,
        image_root=args.image_root,
    )
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
