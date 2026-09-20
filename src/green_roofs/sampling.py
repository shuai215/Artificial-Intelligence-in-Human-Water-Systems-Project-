"""Reproducible tile-level sampling strategies."""

from __future__ import annotations

import math
import random
from collections.abc import Iterator, Sequence

from torch.utils.data import Sampler


class BalancedTileBatchSampler(Sampler[list[int]]):
    """Draw an exact positive/negative tile fraction in every training batch."""

    def __init__(
        self,
        positive_indices: Sequence[int],
        negative_indices: Sequence[int],
        batch_size: int,
        positive_fraction: float = 0.5,
        seed: int = 42,
        samples_per_epoch: int | None = None,
    ) -> None:
        if not positive_indices or not negative_indices:
            raise ValueError("Both positive and negative tile indices are required")
        if batch_size < 2:
            raise ValueError("batch_size must be at least 2")
        if not 0.0 < positive_fraction < 1.0:
            raise ValueError("positive_fraction must be between 0 and 1")
        if samples_per_epoch is not None and samples_per_epoch <= 0:
            raise ValueError("samples_per_epoch must be positive")
        minimum_positive = math.floor(batch_size * positive_fraction)
        maximum_positive = math.ceil(batch_size * positive_fraction)
        if minimum_positive < 1 or maximum_positive >= batch_size:
            raise ValueError(
                "The requested fraction and batch size must allow both tile classes "
                "in every batch"
            )

        self.positive_indices = tuple(positive_indices)
        self.negative_indices = tuple(negative_indices)
        self.batch_size = batch_size
        self.positive_fraction = positive_fraction
        self.seed = seed
        self.samples_per_epoch = (
            samples_per_epoch
            if samples_per_epoch is not None
            else len(positive_indices) + len(negative_indices)
        )
        self.epoch = 0

    def __len__(self) -> int:
        return math.ceil(self.samples_per_epoch / self.batch_size)

    def __iter__(self) -> Iterator[list[int]]:
        generator = random.Random(self.seed + self.epoch)
        self.epoch += 1
        assigned_positive = 0
        for batch_index in range(len(self)):
            target_positive = round(
                (batch_index + 1) * self.batch_size * self.positive_fraction
            )
            positive_per_batch = target_positive - assigned_positive
            positive_per_batch = max(1, min(self.batch_size - 1, positive_per_batch))
            assigned_positive += positive_per_batch
            negative_per_batch = self.batch_size - positive_per_batch
            positive = generator.choices(
                self.positive_indices, k=positive_per_batch
            )
            negative = generator.choices(
                self.negative_indices, k=negative_per_batch
            )
            batch = [*positive, *negative]
            generator.shuffle(batch)
            yield batch
