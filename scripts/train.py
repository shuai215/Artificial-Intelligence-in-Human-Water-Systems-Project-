"""Train a segmentation baseline from a TOML configuration."""

from __future__ import annotations

import argparse
import math
import random
import sys
import tomllib
from pathlib import Path

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from green_roofs.dataset import GreenRoofDataset  # noqa: E402
from green_roofs.engine import fit  # noqa: E402
from green_roofs.losses import build_loss  # noqa: E402
from green_roofs.models import build_model  # noqa: E402
from green_roofs.sampling import BalancedTileBatchSampler  # noqa: E402
from green_roofs.transforms import SegmentationTransform  # noqa: E402


def resolve_path(config_path: Path, value: str) -> Path:
    return (config_path.parent / value).resolve()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate one forward pass without downloading weights or training.",
    )
    args = parser.parse_args()
    config_path = args.config.resolve()
    with config_path.open("rb") as file:
        config = tomllib.load(file)

    seed = int(config["training"]["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    data_config = config["data"]
    dataset_root = resolve_path(config_path, data_config["dataset_root"])
    processed_root = resolve_path(config_path, data_config["processed_root"])
    image_size = int(data_config["image_size"])
    train_transform = SegmentationTransform(
        training=True,
        augment=bool(data_config["augment_training"]),
        image_size=image_size,
    )
    evaluation_transform = SegmentationTransform(
        training=False, augment=False, image_size=image_size
    )
    train_dataset = GreenRoofDataset(
        dataset_root,
        processed_root,
        processed_root / "manifests" / "train.csv",
        train_transform,
    )
    val_dataset = GreenRoofDataset(
        dataset_root,
        processed_root,
        processed_root / "manifests" / "val.csv",
        evaluation_transform,
    )

    batch_size = int(config["training"]["batch_size"])
    num_workers = int(config["training"]["num_workers"])
    sampling_config = config["sampling"]
    positive_only = bool(sampling_config.get("positive_only", False))
    if positive_only:
        positive_indices = set(train_dataset.positive_indices)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=WeightedRandomSampler(
                [float(index in positive_indices) for index in range(len(train_dataset))],
                num_samples=len(train_dataset),
                replacement=True,
                generator=torch.Generator().manual_seed(seed),
            ),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        effective_samples_per_epoch = len(train_dataset)
        planned_positive_per_epoch = effective_samples_per_epoch
    else:
        positive_fraction = float(sampling_config["positive_tile_fraction"])
        positive_repeats = sampling_config.get("positive_repeats_per_epoch")
        samples_per_epoch = len(train_dataset)
        if positive_repeats is not None:
            positive_draws = math.ceil(
                len(train_dataset.positive_indices) * float(positive_repeats)
            )
            samples_per_epoch = math.ceil(
                positive_draws / positive_fraction / batch_size
            ) * batch_size
        train_batch_sampler = BalancedTileBatchSampler(
            positive_indices=train_dataset.positive_indices,
            negative_indices=train_dataset.negative_indices,
            batch_size=batch_size,
            positive_fraction=positive_fraction,
            seed=seed,
            samples_per_epoch=samples_per_epoch,
        )
        effective_samples_per_epoch = len(train_batch_sampler) * batch_size
        planned_positive_per_epoch = round(
            effective_samples_per_epoch * positive_fraction
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_batch_sampler,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model_config = config["model"]
    model = build_model(
        model_config,
        pretrained=(
            bool(model_config.get("pretrained", True)) and not args.check_only
        ),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = build_loss(config["loss"])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["training"]["learning_rate"])
    )
    output_dir = resolve_path(config_path, config["training"]["output_dir"])
    print(
        f"device={device} model={model_config['name']} train={len(train_dataset)} "
        f"train_positive={len(train_dataset.positive_indices)} "
        f"train_negative={len(train_dataset.negative_indices)} "
        f"val={len(val_dataset)} augmentation={data_config['augment_training']} "
        f"samples_per_epoch={effective_samples_per_epoch} "
        f"planned_positive={planned_positive_per_epoch} "
        f"planned_negative={effective_samples_per_epoch - planned_positive_per_epoch} "
        f"positive_only={positive_only} "
        f"freeze_encoder_bn={model_config.get('freeze_encoder_batch_norm', False)} "
        f"loss={config['loss']['name']} output={output_dir}"
    )
    if args.check_only:
        images, masks, tile_ids = next(iter(train_loader))
        positive_tiles = int((masks.flatten(1).sum(dim=1) > 0).sum().item())
        with torch.no_grad():
            logits = model(images.to(device))
            loss = criterion(logits, masks.to(device))
        print(
            f"check_only=passed batch={len(tile_ids)} positive_tiles={positive_tiles} "
            f"negative_tiles={len(tile_ids) - positive_tiles} "
            f"logits_shape={tuple(logits.shape)} loss={loss.item():.6f}"
        )
        return

    fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        epochs=int(config["training"]["epochs"]),
        output_dir=output_dir,
        early_stopping_patience=config["training"].get("early_stopping_patience"),
        early_stopping_min_delta=float(
            config["training"].get("early_stopping_min_delta", 0.0)
        ),
    )


if __name__ == "__main__":
    main()
