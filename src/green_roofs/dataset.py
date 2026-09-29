"""Manifest-based datasets for green-roof semantic segmentation."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from .manifests import load_manifest


JointTransform = Callable[[Image.Image, Image.Image], tuple[Tensor, Tensor]]


class GreenRoofDataset(Dataset[tuple[Tensor, Tensor, str]]):
    """Read image/mask pairs from one frozen spatial-split manifest."""

    def __init__(
        self,
        dataset_root: Path,
        processed_root: Path,
        manifest_path: Path,
        transform: JointTransform,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.processed_root = Path(processed_root)
        self.manifest_path = Path(manifest_path)
        self.transform = transform
        self.records = load_manifest(self.manifest_path).to_dict(orient="records")

        #A tile is considered a positive tile if its mask contains at least one positive pixel.
        self.positive_indices = [
            index
            for index, record in enumerate(self.records)
            if record["positive_pixels"] > 0
        ]
        self.negative_indices = [
            index
            for index, record in enumerate(self.records)
            if record["positive_pixels"] == 0
        ]

    #A PyTorch Dataset needs to implement `__len__()` and `__getitem__()`.
    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, str]:
        record = self.records[index]
        image_path = self.dataset_root / record["image_relpath"]
        mask_path = self.processed_root / record["mask_relpath"]
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        with Image.open(mask_path) as source:
            mask = source.convert("L")
        image_tensor, mask_tensor = self.transform(image, mask)
        return image_tensor, mask_tensor, record["tile_id"]
