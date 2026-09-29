"""Synchronized image/mask transforms with deterministic evaluation behavior."""

from __future__ import annotations

import random

import torch
from PIL import Image
from torch import Tensor
from torchvision.transforms import functional as functional

#Mean and standard deviation of the three RGB channels calculated from the ImageNet training set.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
#The input images are also normalized in the same way as in ImageNet.

def image_to_tensor(image: Image.Image) -> Tensor:
    #Convert one RGB image to the normalization used by every model.
    return functional.normalize(
        functional.to_tensor(image), IMAGENET_MEAN, IMAGENET_STD
    )


class SegmentationTransform:
    #Transform an RGB image and binary mask

    def __init__(self, training: bool, augment: bool, image_size: int = 256) -> None:
        if not training and augment:
            raise ValueError("Random augmentation must not be enabled for evaluation")
        self.training = training
        self.augment = augment
        self.image_size = image_size

    def __call__(self, image: Image.Image, mask: Image.Image) -> tuple[Tensor, Tensor]:
        size = [self.image_size, self.image_size]
        image = functional.resize(image, size, interpolation=functional.InterpolationMode.BILINEAR)
        #The mask only supports the NEAREST interpolation method.
        mask = functional.resize(mask, size, interpolation=functional.InterpolationMode.NEAREST)

        if self.training and self.augment:
            if random.random() < 0.5:
                image = functional.hflip(image)
                mask = functional.hflip(mask)
            if random.random() < 0.5:
                image = functional.vflip(image)
                mask = functional.vflip(mask)
            quarter_turns = random.randrange(4)
            if quarter_turns:
                angle = 90 * quarter_turns
                image = functional.rotate(image, angle)
                mask = functional.rotate(mask, angle, interpolation=functional.InterpolationMode.NEAREST)

        image_tensor = image_to_tensor(image)
        mask_tensor = functional.pil_to_tensor(mask).squeeze(0).long()
        #sanity check
        unique_values = torch.unique(mask_tensor)
        if not all(value.item() in (0, 1) for value in unique_values):
            raise ValueError(f"Mask must contain only 0/1, got {unique_values.tolist()}")
        return image_tensor, mask_tensor
