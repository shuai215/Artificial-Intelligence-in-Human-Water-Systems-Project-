"""Foreground-focused binary semantic-segmentation metrics."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


def binary_metrics_from_counts(
    true_positive: int,
    false_positive: int,
    false_negative: int,
) -> dict[str, float]:
    """Compute foreground metrics from accumulated confusion counts."""
    def ratio(numerator: int, denominator: int) -> float:
        return numerator / denominator if denominator else 0.0

    return {
        "foreground_dice": ratio(
            2 * true_positive,
            2 * true_positive + false_positive + false_negative,
        ),
        "foreground_iou": ratio(
            true_positive,
            true_positive + false_positive + false_negative,
        ),
        "foreground_precision": ratio(
            true_positive,
            true_positive + false_positive,
        ),
        "foreground_recall": ratio(
            true_positive,
            true_positive + false_negative,
        ),
    }


@dataclass
class BinarySegmentationMetrics:
    true_positive: int = 0
    false_positive: int = 0
    false_negative: int = 0
    true_negative: int = 0

    def update(self, logits: Tensor, target: Tensor, threshold: float = 0.5) -> None:
        if logits.ndim != 4 or logits.shape[1] != 1:
            raise ValueError(f"Expected logits shaped [B,1,H,W], got {tuple(logits.shape)}")
        predicted_positive = torch.sigmoid(logits[:, 0]) >= threshold
        actual_positive = target == 1
        self.true_positive += int((predicted_positive & actual_positive).sum().item())
        self.false_positive += int((predicted_positive & ~actual_positive).sum().item())
        self.false_negative += int((~predicted_positive & actual_positive).sum().item())
        self.true_negative += int((~predicted_positive & ~actual_positive).sum().item())

    def compute(self) -> dict[str, float]:
        return binary_metrics_from_counts(
            self.true_positive,
            self.false_positive,
            self.false_negative,
        )
