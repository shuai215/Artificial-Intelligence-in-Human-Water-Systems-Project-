"""Canonical prepared-dataset manifest schema, serialization, and validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from .tiles import TILE_SIZE, ZOOM, TileKey


SPLITS = ("train", "val", "test")
MANIFEST_FIELDS = (
    "x",
    "y",
    "image_relpath",
    "mask_relpath",
    "block_id",
    "positive_pixels",
)


@dataclass(frozen=True)
class TileRecord:
    key: TileKey
    image_relpath: str
    mask_relpath: str
    positive_pixels: int
    block_id: str
    split: str

    @property
    def positive_fraction(self) -> float:
        return self.positive_pixels / (TILE_SIZE * TILE_SIZE)


def write_manifests(output_root: Path, records: Iterable[TileRecord]) -> None:
    """Write deterministic train/validation/test manifest CSV files."""
    rows = [
        {
            "x": record.key.x,
            "y": record.key.y,
            "image_relpath": record.image_relpath,
            "mask_relpath": record.mask_relpath,
            "block_id": record.block_id,
            "positive_pixels": record.positive_pixels,
            "split": record.split,
        }
        for record in records
    ]
    frame = pd.DataFrame(rows)
    manifest_root = output_root / "manifests"
    manifest_root.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        split_frame = (
            frame.loc[frame["split"] == split, list(MANIFEST_FIELDS)]
            .sort_values(["x", "y"])
            .reset_index(drop=True)
        )
        split_frame.to_csv(
            manifest_root / f"{split}.csv",
            index=False,
            encoding="utf-8",
        )


def load_manifest(path: Path) -> pd.DataFrame:
    """Read source facts and derive convenient manifest fields in memory."""
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"Manifest is empty: {path}")
    missing = set(MANIFEST_FIELDS) - set(frame.columns)
    if missing:
        raise ValueError(f"Manifest lacks required fields {sorted(missing)}: {path}")

    frame = frame.loc[:, list(MANIFEST_FIELDS)].copy()
    for column in ("x", "y", "positive_pixels"):
        numeric = pd.to_numeric(frame[column], errors="raise")
        if not numeric.mod(1).eq(0).all():
            raise ValueError(f"Manifest field {column} must contain integers: {path}")
        frame[column] = numeric.astype("int64")

    for column in ("image_relpath", "mask_relpath", "block_id"):
        if frame[column].isna().any() or frame[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"Manifest field {column} contains empty values: {path}")
        frame[column] = frame[column].astype(str)

    if frame.duplicated(["x", "y"]).any():
        raise ValueError(f"Manifest contains duplicate tile coordinates: {path}")
    if (frame["positive_pixels"] < 0).any() or (
        frame["positive_pixels"] > TILE_SIZE * TILE_SIZE
    ).any():
        raise ValueError(f"Manifest positive_pixels is outside mask bounds: {path}")

    frame.insert(
        0,
        "tile_id",
        str(ZOOM)
        + "_"
        + frame["x"].astype(str)
        + "_"
        + frame["y"].astype(str),
    )
    frame["positive_fraction"] = frame["positive_pixels"] / (TILE_SIZE * TILE_SIZE)
    frame["has_green_roof"] = (frame["positive_pixels"] > 0).astype("int8")
    return frame


def load_split_manifests(processed_root: Path) -> pd.DataFrame:
    """Read and combine all canonical split manifests with a split column."""
    frames = []
    for split in SPLITS:
        frame = load_manifest(processed_root / "manifests" / f"{split}.csv")
        frame.insert(0, "split", split)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)
