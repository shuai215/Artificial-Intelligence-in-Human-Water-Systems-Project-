"""Deterministic spatial-block dataset splitting."""

from __future__ import annotations

import random
from collections import defaultdict

from .tiles import TileKey


def assign_spatial_splits(
    tile_stats: dict[TileKey, int],
    block_size: int,
    seed: int,
) -> tuple[dict[TileKey, tuple[str, str]], dict[str, dict[str, int]]]:
    """Assign whole spatial blocks while balancing tiles and positive pixels."""
    min_x = min(key.x for key in tile_stats)
    min_y = min(key.y for key in tile_stats)
    blocks: dict[str, list[TileKey]] = defaultdict(list)
    for key in tile_stats:
        block_x = (key.x - min_x) // block_size
        block_y = (key.y - min_y) // block_size
        blocks[f"b{block_x:02d}_{block_y:02d}"].append(key)

    split_ratios = {"train": 0.70, "val": 0.15, "test": 0.15}
    total_tiles = len(tile_stats)
    total_positive = sum(tile_stats.values())
    targets = {
        split: {
            "tiles": total_tiles * ratio,
            "positive_pixels": total_positive * ratio,
        }
        for split, ratio in split_ratios.items()
    }
    assigned = {
        split: {"blocks": 0, "tiles": 0, "positive_pixels": 0}
        for split in split_ratios
    }

    rng = random.Random(seed)
    block_items = list(blocks.items())
    rng.shuffle(block_items)
    #If two blocks have the same number of positive pixels, their original relative order is preserved.
    #Sort by the number of positive pixels in descending order.
    block_items.sort(
        key=lambda item: sum(tile_stats[key] for key in item[1]), reverse=True
    )

    #Start a block, then divide into blocks.
    block_splits: dict[str, str] = {}
    for block_id, keys in block_items:
        #How many tiles are in the current block?
        block_tiles = len(keys)
        block_positive = sum(tile_stats[key] for key in keys)
        scores: dict[str, float] = {}
        for split in split_ratios:
            tile_deficit = (
                targets[split]["tiles"] - assigned[split]["tiles"]
            ) / max(targets[split]["tiles"], 1)
            positive_deficit = (
                targets[split]["positive_pixels"]
                - assigned[split]["positive_pixels"]
            ) / max(targets[split]["positive_pixels"], 1)
            #Importance of tile count       = 35%
            #Importance of positive pixels = 65%
            scores[split] = 0.35 * tile_deficit + 0.65 * positive_deficit
        #Give it to whoever has the highest score.
        split = max(scores, key=lambda candidate: scores[candidate])
        block_splits[block_id] = split
        assigned[split]["blocks"] += 1
        assigned[split]["tiles"] += block_tiles
        assigned[split]["positive_pixels"] += block_positive

    tile_assignments = {
        key: (block_splits[block_id], block_id)
        for block_id, keys in blocks.items()
        for key in keys
    }
    return tile_assignments, assigned
