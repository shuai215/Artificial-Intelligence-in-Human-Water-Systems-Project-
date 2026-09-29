"""Analyze validation data or evaluate a frozen checkpoint on the test split."""

from __future__ import annotations

import argparse
import csv
import json
import tomllib
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader


from green_roofs.dataset import GreenRoofDataset
from green_roofs.label_qa import create_mask_overlay
from green_roofs.metrics import binary_metrics_from_counts
from green_roofs.models import build_model
from green_roofs.transforms import SegmentationTransform


def resolve_path(config_path: Path, value: str) -> Path:
    return (config_path.parent / value).resolve()


def metric_row(threshold: float, counts: dict[str, int]) -> dict[str, float | int]:
    tp, fp, fn, tn = (counts[key] for key in ("tp", "fp", "fn", "tn"))
    return {
        "threshold": threshold,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        **binary_metrics_from_counts(tp, fp, fn),
    }


def summarize_positive_tiles(
    rows: list[dict[str, int | float | str]],
) -> dict[str, int | float]:
    #Aggregate pixel metrics over tiles that contain foreground.
    true_positive = sum(int(row["true_positive"]) for row in rows)
    false_positive = sum(int(row["false_positive"]) for row in rows)
    false_negative = sum(int(row["false_negative"]) for row in rows)
    return {
        "tile_count": len(rows),
        "target_pixels": sum(int(row["target_pixels"]) for row in rows),
        "predicted_pixels": sum(int(row["predicted_pixels"]) for row in rows),
        **binary_metrics_from_counts(true_positive, false_positive, false_negative),
    }


def probability_image(probability: torch.Tensor) -> Image.Image:
    values = probability.mul(255).clamp(0, 255).to(torch.uint8).cpu().numpy()
    grayscale = Image.fromarray(values, mode="L")
    red = grayscale.point(lambda value: min(255, value * 2))
    green = grayscale.point(lambda value: max(0, 255 - abs(value - 128) * 2))
    blue = grayscale.point(lambda value: max(0, 255 - value * 2))
    return Image.merge("RGB", (red, green, blue))


def create_diagnostic(
    image: Image.Image,
    target: torch.Tensor,
    probability: torch.Tensor,
    threshold: float,
    tile_id: str,
    category: str,
) -> Image.Image:
    target_bool = target.bool().cpu()
    prediction_bool = probability.cpu() >= threshold
    target_mask = Image.fromarray(target_bool.to(torch.uint8).numpy(), mode="L")
    prediction_mask = Image.fromarray(
        prediction_bool.to(torch.uint8).numpy(), mode="L"
    )

    tp = prediction_bool & target_bool
    fp = prediction_bool & ~target_bool
    fn = ~prediction_bool & target_bool
    error_pixels = torch.zeros((*target_bool.shape, 3), dtype=torch.uint8)
    error_pixels[tp] = torch.tensor((0, 200, 0), dtype=torch.uint8)
    error_pixels[fp] = torch.tensor((230, 40, 40), dtype=torch.uint8)
    error_pixels[fn] = torch.tensor((40, 100, 230), dtype=torch.uint8)
    error = Image.fromarray(error_pixels.numpy(), mode="RGB")

    panels = [
        image.convert("RGB"),
        create_mask_overlay(image, target_mask, (0, 220, 0)),
        probability_image(probability),
        create_mask_overlay(image, prediction_mask, (255, 200, 0)),
        error,
    ]
    titles = ("Image", "Ground truth", "Probability", "Prediction", "TP/FP/FN")
    header_height, footer_height = 34, 24
    canvas = Image.new(
        "RGB", (image.width * len(panels), image.height + header_height + footer_height), "white"
    )
    draw = ImageDraw.Draw(canvas)
    for index, (panel, title) in enumerate(zip(panels, titles)):
        x = index * image.width
        canvas.paste(panel, (x, header_height))
        draw.text((x + 8, 10), title, fill="black")
    draw.text(
        (8, header_height + image.height + 5),
        f"{tile_id} | {category} | threshold={threshold:.2f} | error: green=TP, red=FP, blue=FN",
        fill="black",
    )
    return canvas


def accumulate_threshold_counts(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    thresholds: list[float],
) -> dict[float, dict[str, int]]:
    #Run through the entire dataset once, then evaluate multiple thresholds simultaneously.
    counts = {
        threshold: {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        for threshold in thresholds
    }
    with torch.inference_mode():
        for batch_index, (images, targets, _) in enumerate(loader, start=1):
            probabilities = torch.sigmoid(model(images.to(device))[:, 0]).cpu()
            target_positive = targets.bool()
            for threshold in thresholds:
                predicted_positive = probabilities >= threshold
                counts[threshold]["tp"] += int(
                    (predicted_positive & target_positive).sum().item()
                )
                counts[threshold]["fp"] += int(
                    (predicted_positive & ~target_positive).sum().item()
                )
                counts[threshold]["fn"] += int(
                    (~predicted_positive & target_positive).sum().item()
                )
                counts[threshold]["tn"] += int(
                    (~predicted_positive & ~target_positive).sum().item()
                )
            if batch_index % 25 == 0:
                print(f"threshold_pass={batch_index}/{len(loader)}")
    return counts


def collect_tile_rows(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
) -> list[dict[str, int | float | str]]:
    rows: list[dict[str, int | float | str]] = []
    offset = 0
    with torch.inference_mode():
        for batch_index, (images, targets, tile_ids) in enumerate(loader, start=1):
            probabilities = torch.sigmoid(model(images.to(device))[:, 0]).cpu()
            predictions = probabilities >= threshold
            targets_positive = targets.bool()
            for batch_offset, tile_id in enumerate(tile_ids):
                prediction = predictions[batch_offset]
                target_positive = targets_positive[batch_offset]
                tp = int((prediction & target_positive).sum().item())
                fp = int((prediction & ~target_positive).sum().item())
                fn = int((~prediction & target_positive).sum().item())
                denominator = 2 * tp + fp + fn
                rows.append(
                    {
                        "index": offset + batch_offset,
                        "tile_id": tile_id,
                        "target_pixels": int(target_positive.sum().item()),
                        "predicted_pixels": int(prediction.sum().item()),
                        "true_positive": tp,
                        "false_positive": fp,
                        "false_negative": fn,
                        "dice": (2 * tp / denominator) if denominator else 1.0,
                    }
                )
            offset += len(tile_ids)
            if batch_index % 25 == 0:
                print(f"tile_pass={batch_index}/{len(loader)}")
    return rows


def write_diagnostics(
    model: torch.nn.Module,
    dataset: GreenRoofDataset,
    selected: dict[str, list[dict[str, int | float | str]]],
    output_dir: Path,
    device: torch.device,
    threshold: float,
) -> None:
    diagnostic_dir = output_dir / "diagnostics"
    diagnostic_dir.mkdir(exist_ok=True)
    with torch.inference_mode():
        for category, category_rows in selected.items():
            for rank, row in enumerate(category_rows, start=1):
                index = int(row["index"])
                image_tensor, target, tile_id = dataset[index]
                #get[H,W]
                probability = torch.sigmoid(
                    model(image_tensor.unsqueeze(0).to(device))[0, 0]
                ).cpu()
                record = dataset.records[index]
                with Image.open(dataset.dataset_root / record["image_relpath"]) as source:
                    image = source.convert("RGB")
                diagnostic = create_diagnostic(
                    image, target, probability, threshold, tile_id, category
                )
                diagnostic.save(
                    diagnostic_dir / f"{category}_{rank:02d}_{tile_id}.png"
                )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--examples-per-category", type=int, default=10)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--threshold", type=float)
    args = parser.parse_args()
    #Test Threshold Protection
    if args.split == "test" and args.threshold is None:
        parser.error("--threshold is required for test evaluation")
    if args.threshold is not None and not 0.0 < args.threshold < 1.0:
        parser.error("--threshold must be between 0 and 1")

    config_path = args.config.resolve()
    checkpoint_path = args.checkpoint.resolve()
    output_dir = args.output.resolve()
    output_stem = "validation" if args.split == "val" else "test"
    output_dir.mkdir(parents=True, exist_ok=True)
    with config_path.open("rb") as file:
        config = tomllib.load(file)

    data_config = config["data"]
    dataset_root = resolve_path(config_path, data_config["dataset_root"])
    processed_root = resolve_path(config_path, data_config["processed_root"])
    transform = SegmentationTransform(
        training=False,
        augment=False,
        image_size=int(data_config["image_size"]),
    )
    dataset = GreenRoofDataset(
        dataset_root,
        processed_root,
        processed_root / "manifests" / f"{args.split}.csv",
        transform,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["training"]["num_workers"]),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config["model"], pretrained=False).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    thresholds = (
        [args.threshold]
        if args.threshold is not None
        else [step / 100 for step in range(5, 100, 5)]
    )
    counts = accumulate_threshold_counts(model, loader, device, thresholds)
    rows = [metric_row(threshold, counts[threshold]) for threshold in thresholds]
    best = max(rows, key=lambda row: row["foreground_dice"])
    with (output_dir / "threshold_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    threshold = float(best["threshold"])
    tile_rows = collect_tile_rows(model, loader, device, threshold)
    with (output_dir / f"{output_stem}_tiles.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(tile_rows[0]))
        writer.writeheader()
        writer.writerows(tile_rows)

    count = args.examples_per_category
    positive_rows = [row for row in tile_rows if row["target_pixels"] > 0]
    negative_rows = [row for row in tile_rows if row["target_pixels"] == 0]
    positive_summary = summarize_positive_tiles(positive_rows)
    negative_summary = {
        "tile_count": len(negative_rows),
        "tiles_with_false_positives": sum(
            int(row["false_positive"]) > 0 for row in negative_rows
        ),
        "false_positive_pixels": sum(
            int(row["false_positive"]) for row in negative_rows
        ),
    }
    selected = {
        "best_positive": sorted(positive_rows, key=lambda row: row["dice"], reverse=True)[:count],
        "fn_heavy_positive": sorted(
            positive_rows, key=lambda row: row["false_negative"], reverse=True
        )[:count],
        "fp_heavy_negative": sorted(
            negative_rows, key=lambda row: row["false_positive"], reverse=True
        )[:count],
    }
    write_diagnostics(
        model,
        dataset,
        selected,
        output_dir,
        device,
        threshold,
    )

    summary = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "split": args.split,
        "tile_count": len(dataset),
        "positive_tile_count": len(dataset.positive_indices),
        "negative_tile_count": len(dataset.negative_indices),
        "selected_threshold": threshold,
        "selected_metrics": best,
        "positive_tile_subset": positive_summary,
        "negative_tile_subset": negative_summary,
        "diagnostic_examples": {
            category: len(category_rows) for category, category_rows in selected.items()
        },
    }
    (output_dir / f"{output_stem}_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    split_title = "Validation" if args.split == "val" else "Test"
    report = [
        f"# {split_title} analysis",
        "",
        f"- Checkpoint epoch: {summary['checkpoint_epoch']}",
        f"- {split_title} tiles: {summary['tile_count']}",
        f"- Positive {args.split} tiles: {summary['positive_tile_count']}",
        f"- Selected threshold: {threshold:.2f}",
        f"- Foreground Dice: {best['foreground_dice']:.4f}",
        f"- Foreground IoU: {best['foreground_iou']:.4f}",
        f"- Foreground precision: {best['foreground_precision']:.4f}",
        f"- Foreground recall: {best['foreground_recall']:.4f}",
        f"- Positive-tile-only Dice: {positive_summary['foreground_dice']:.4f}",
        f"- Positive-tile-only IoU: {positive_summary['foreground_iou']:.4f}",
        f"- Negative tiles with false positives: {negative_summary['tiles_with_false_positives']}/{negative_summary['tile_count']}",
        "",
        (
            "The threshold was fixed from validation before this test evaluation."
            if args.split == "test"
            else "The threshold was selected only on validation; test data was not read."
        ),
    ]
    (output_dir / f"{output_stem}_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
