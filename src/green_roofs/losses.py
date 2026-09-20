"""Binary segmentation losses operating on one-channel raw logits."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional


def _prepare(logits: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
    if logits.ndim != 4 or logits.shape[1] != 1:
        raise ValueError(f"Expected logits shaped [B,1,H,W], got {tuple(logits.shape)}")
    logits = logits[:, 0]
    target = target.float()
    if logits.shape != target.shape:
        raise ValueError(
            f"Logit/target shape mismatch: {tuple(logits.shape)} vs {tuple(target.shape)}"
        )
    return logits, target


def soft_dice_loss(logits: Tensor, target: Tensor, smooth: float = 1.0) -> Tensor:
    logits, target = _prepare(logits, target)
    probabilities = torch.sigmoid(logits)
    dimensions = (1, 2)
    intersection = (probabilities * target).sum(dim=dimensions)
    denominator = probabilities.sum(dim=dimensions) + target.sum(dim=dimensions)
    dice = (2.0 * intersection + smooth) / (denominator + smooth)
    return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        prepared_logits, prepared_target = _prepare(logits, target)
        bce = functional.binary_cross_entropy_with_logits(
            prepared_logits, prepared_target
        )
        dice = soft_dice_loss(logits, target)
        return self.bce_weight * bce + self.dice_weight * dice


class FocalBCEDiceLoss(nn.Module):
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        focal_weight: float = 0.5,
        dice_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        prepared_logits, prepared_target = _prepare(logits, target)
        bce = functional.binary_cross_entropy_with_logits(
            prepared_logits, prepared_target, reduction="none"
        )
        probabilities = torch.sigmoid(prepared_logits)
        probability_true_class = (
            probabilities * prepared_target
            + (1.0 - probabilities) * (1.0 - prepared_target)
        )
        alpha_factor = (
            self.alpha * prepared_target
            + (1.0 - self.alpha) * (1.0 - prepared_target)
        )
        focal = (
            alpha_factor * (1.0 - probability_true_class).pow(self.gamma) * bce
        ).mean()
        dice = soft_dice_loss(logits, target)
        return self.focal_weight * focal + self.dice_weight * dice


class TverskyLoss(nn.Module):
    def __init__(self, alpha: float = 0.3, beta: float = 0.7, smooth: float = 1.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        prepared_logits, prepared_target = _prepare(logits, target)
        probabilities = torch.sigmoid(prepared_logits)
        dimensions = (1, 2)
        true_positive = (probabilities * prepared_target).sum(dim=dimensions)
        false_positive = (probabilities * (1.0 - prepared_target)).sum(
            dim=dimensions
        )
        false_negative = ((1.0 - probabilities) * prepared_target).sum(
            dim=dimensions
        )
        score = (true_positive + self.smooth) / (
            true_positive
            + self.alpha * false_positive
            + self.beta * false_negative
            + self.smooth
        )
        return 1.0 - score.mean()


def build_loss(config: dict[str, object]) -> nn.Module:
    name = str(config["name"])
    if name == "bce_dice":
        return BCEDiceLoss(
            bce_weight=float(config.get("bce_weight", 0.5)),
            dice_weight=float(config.get("dice_weight", 0.5)),
        )
    if name == "focal_bce_dice":
        return FocalBCEDiceLoss(
            alpha=float(config.get("alpha", 0.25)),
            gamma=float(config.get("gamma", 2.0)),
            focal_weight=float(config.get("focal_weight", 0.5)),
            dice_weight=float(config.get("dice_weight", 0.5)),
        )
    if name == "tversky":
        return TverskyLoss(
            alpha=float(config.get("alpha", 0.3)),
            beta=float(config.get("beta", 0.7)),
            smooth=float(config.get("smooth", 1.0)),
        )
    raise ValueError(f"Unsupported loss: {name}")
