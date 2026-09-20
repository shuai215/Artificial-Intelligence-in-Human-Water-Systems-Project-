"""Reusable training loop shared by U-Net and future SegFormer experiments."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from green_roofs.metrics import BinarySegmentationMetrics


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    metrics = BinarySegmentationMetrics()
    loss_sum = 0.0
    sample_count = 0

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, masks, _ in loader:
            images = images.to(device)
            masks = masks.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, masks)
            if training:
                loss.backward()
                optimizer.step()
            batch_size = images.shape[0]
            loss_sum += float(loss.item()) * batch_size
            sample_count += batch_size
            metrics.update(logits.detach(), masks)

    result = metrics.compute()
    result["loss"] = loss_sum / sample_count
    return result


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epochs: int,
    output_dir: Path,
    early_stopping_patience: int | None = None,
    early_stopping_min_delta: float = 0.0,
) -> list[dict[str, float | int]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float | int]] = []
    best_dice = -1.0
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        train_result = run_epoch(model, train_loader, criterion, device, optimizer)
        val_result = run_epoch(model, val_loader, criterion, device)
        row: dict[str, float | int] = {"epoch": epoch}
        row.update({f"train_{key}": value for key, value in train_result.items()})
        row.update({f"val_{key}": value for key, value in val_result.items()})
        history.append(row)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "validation": val_result,
        }
        torch.save(checkpoint, output_dir / "last.pt")
        improved = (
            val_result["foreground_dice"] > best_dice + early_stopping_min_delta
        )
        if improved:
            best_dice = val_result["foreground_dice"]
            epochs_without_improvement = 0
            torch.save(checkpoint, output_dir / "best.pt")
        else:
            epochs_without_improvement += 1

        print(
            f"epoch={epoch:03d} train_loss={train_result['loss']:.5f} "
            f"val_loss={val_result['loss']:.5f} "
            f"val_dice={val_result['foreground_dice']:.4f} "
            f"val_iou={val_result['foreground_iou']:.4f}"
        )
        if (
            early_stopping_patience is not None
            and epochs_without_improvement >= early_stopping_patience
        ):
            print(
                f"early_stopping epoch={epoch} best_val_dice={best_dice:.4f} "
                f"patience={early_stopping_patience}"
            )
            break

    (output_dir / "history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    return history
