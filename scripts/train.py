"""Train a segmentation baseline from a TOML configuration.

The entry point reads the experiment configuration, prepares the data loaders,
builds the model, loss, and optimizer, and then runs training.
"""

from __future__ import annotations

import argparse
import math
import random
import tomllib
from pathlib import Path

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler


'''dataset.py: Handles loading images and masks
engine.py: Executes the actual training loop
losses.py: Defines loss functions such as BCE and Dice
models.py: Defines the ResNet50-U-Net model
sampling.py: Controls the ratio of positive to negative samples
transforms.py: Handles resizing, augmentation, etc.
'''

from green_roofs.dataset import GreenRoofDataset
from green_roofs.engine import fit
from green_roofs.losses import build_loss
from green_roofs.models import build_model
from green_roofs.sampling import BalancedTileBatchSampler
from green_roofs.transforms import SegmentationTransform


def resolve_path(config_path: Path, value: str) -> Path:
    return (config_path.parent / value).resolve()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

#This function is responsible for creating the DataLoader for the training data.
def build_train_loader(
    dataset: GreenRoofDataset,
    sampling_config: dict[str, object],
    batch_size: int,
    num_workers: int,
    seed: int,
) -> tuple[DataLoader, int, int, bool]:
    positive_only = bool(sampling_config.get("positive_only", False))
    if positive_only:
        positive_indices = set(dataset.positive_indices)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=WeightedRandomSampler(
                [float(index in positive_indices) for index in range(len(dataset))],
                num_samples=len(dataset),
                replacement=True,
                generator=torch.Generator().manual_seed(seed),
            ),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        return loader, len(dataset), len(dataset), True

    positive_fraction = float(sampling_config["positive_tile_fraction"])
    #We want positive samples to be sampled an average of a certain number of times per epoch.
    positive_repeats = sampling_config.get("positive_repeats_per_epoch")
    samples_per_epoch = len(dataset)
    if positive_repeats is not None:
        positive_draws = math.ceil(
            len(dataset.positive_indices) * float(positive_repeats)
        )
        samples_per_epoch = math.ceil(
            positive_draws / positive_fraction / batch_size
        ) * batch_size
    sampler = BalancedTileBatchSampler(
        positive_indices=dataset.positive_indices,
        negative_indices=dataset.negative_indices,
        batch_size=batch_size,
        positive_fraction=positive_fraction,
        seed=seed,
        samples_per_epoch=samples_per_epoch,
    )
    effective_samples = len(sampler) * batch_size
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return loader, effective_samples, round(effective_samples * positive_fraction), False


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
    seed_everything(seed)

    data_config = config["data"]
    dataset_root = resolve_path(config_path, data_config["dataset_root"])
    processed_root = resolve_path(config_path, data_config["processed_root"])
    image_size = int(data_config["image_size"])
    #Create data augmentation
    train_transform = SegmentationTransform(
        training=True,
        augment=bool(data_config["augment_training"]),
        image_size=image_size,
    )
    #No data augmentation is performed on the validation set.
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
    (
        train_loader,
        effective_samples_per_epoch,
        planned_positive_per_epoch,
        positive_only,
    ) = build_train_loader(
        train_dataset,
        config["sampling"],
        batch_size,
        num_workers,
        seed,
    )
    #shuffle=False,The validation set does not use balanced sampling.
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
    #PyTorch uses the default settings for AdamW.weight_decay = 0.01
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
    #Check whether each mask contains at least one positive pixel.
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
