"""Align 2025 JP2 imagery to the 2016 XYZ grid and predict binary masks."""

from __future__ import annotations

import argparse
import csv
import json
import tomllib
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import rasterio
import torch
from PIL import Image
from rasterio.enums import Resampling
from rasterio.merge import merge
from rasterio.vrt import WarpedVRT


from green_roofs.manifests import load_split_manifests
from green_roofs.models import build_model
from green_roofs.tiles import TILE_SIZE, ZOOM, TileKey, tile_bounds_mercator
from green_roofs.transforms import image_to_tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("threshold", type=float)
    parser.add_argument("name")
    parser.add_argument("source", type=Path, help="Directory containing 2025 JP2 files")
    parser.add_argument("template", type=Path, help="Prepared 2016 dataset defining the XYZ grid")
    parser.add_argument("output", type=Path)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--limit", type=int, help="Process only the first N tiles for a smoke test")
    return parser.parse_args()


def read_rgb_tile(vrts: list[WarpedVRT], key: TileKey) -> Image.Image:
    bounds = tile_bounds_mercator(key)
    resolution = (bounds[2] - bounds[0]) / TILE_SIZE
    array, _ = merge(
        vrts,
        bounds=bounds,
        res=resolution,
        indexes=(1, 2, 3),
        output_count=3,
        resampling=Resampling.bilinear,
    )
    if array.shape != (3, TILE_SIZE, TILE_SIZE):
        raise ValueError(f"Unexpected aligned tile shape for {key.tile_id}: {array.shape}")
    return Image.fromarray(np.moveaxis(array, 0, -1), mode="RGB")


def output_path(root: Path, key: TileKey) -> Path:
    return root / str(ZOOM) / str(key.x) / f"{key.y}.png"


def main() -> None:
    args = parse_args()
    if not 0.0 < args.threshold < 1.0:
        raise ValueError("threshold must be between 0 and 1")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")

    config_path = args.config.resolve()
    checkpoint_path = args.checkpoint.resolve()
    source_root = args.source.resolve()
    template_root = args.template.resolve()
    output_root = args.output.resolve()
    with config_path.open("rb") as file:
        config = tomllib.load(file)
    if int(config["data"]["image_size"]) != TILE_SIZE:
        raise ValueError(f"2025 grid prediction requires image_size={TILE_SIZE}")

    jp2_paths = sorted(source_root.glob("*.jp2"))
    if not jp2_paths:
        raise FileNotFoundError(f"No JP2 files found in {source_root}")
    rows = load_split_manifests(template_root)
    keys = sorted({TileKey(int(row.x), int(row.y)) for row in rows.itertuples()})
    if args.limit is not None:
        keys = keys[: args.limit]

    prediction_root = output_root / "predictions" / args.name
    if prediction_root.exists():
        raise FileExistsError(f"Prediction output already exists: {prediction_root}")
    image_root = output_root / "images"
    mask_root = prediction_root / "masks"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config["model"], pretrained=False).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    batch_size = args.batch_size or int(config["training"]["batch_size"])

    records: list[dict[str, object]] = []
    with ExitStack() as stack:
        sources = [stack.enter_context(rasterio.open(path)) for path in jp2_paths]
        if any(source.count < 3 for source in sources):
            raise ValueError("Every 2025 JP2 must contain at least three RGB bands")
        vrts = [
            stack.enter_context(
                WarpedVRT(source, crs="EPSG:3857", resampling=Resampling.bilinear)
            )
            for source in sources
        ]
        for start in range(0, len(keys), batch_size):
            batch_keys = keys[start : start + batch_size]
            images = []
            for key in batch_keys:
                image_path = output_path(image_root, key)
                if image_path.is_file():
                    with Image.open(image_path) as cached:
                        image = cached.convert("RGB")
                else:
                    image = read_rgb_tile(vrts, key)
                    image_path.parent.mkdir(parents=True, exist_ok=True)
                    image.save(image_path)
                images.append(image)

            inputs = torch.stack([image_to_tensor(image) for image in images]).to(device)
            with torch.inference_mode():
                probabilities = torch.sigmoid(model(inputs)[:, 0]).cpu()
            for key, probability in zip(batch_keys, probabilities):
                mask = probability.ge(args.threshold).to(torch.uint8).mul(255).numpy()
                mask_path = output_path(mask_root, key)
                mask_path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(mask, mode="L").save(mask_path)
                predicted_pixels = int((mask > 0).sum())
                records.append(
                    {
                        "tile_id": key.tile_id,
                        "x": key.x,
                        "y": key.y,
                        "image_relpath": output_path(Path("images"), key).as_posix(),
                        "mask_relpath": output_path(
                            Path("predictions") / args.name / "masks", key
                        ).as_posix(),
                        "predicted_pixels": predicted_pixels,
                        "predicted_fraction": predicted_pixels / (TILE_SIZE * TILE_SIZE),
                    }
                )
            completed = min(start + batch_size, len(keys))
            if completed == len(keys) or completed % (batch_size * 25) == 0:
                print(f"tiles={completed}/{len(keys)}", flush=True)

    prediction_root.mkdir(parents=True, exist_ok=True)
    with (prediction_root / "manifest.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    summary = {
        "year": 2025,
        "status": "unlabeled_predictions",
        "model": args.name,
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "threshold": args.threshold,
        "source": str(source_root),
        "source_band_selection": "JP2 bands 1-3 interpreted as RGB from RGBI product name",
        "template": str(template_root),
        "grid": "2016 XYZ zoom 19, EPSG:3857, 256x256",
        "tile_count": len(records),
        "tiles_with_predictions": sum(int(row["predicted_pixels"]) > 0 for row in records),
        "predicted_pixels": sum(int(row["predicted_pixels"]) for row in records),
        "device": str(device),
    }
    (prediction_root / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
