"""SegFormer-B0 adapter for binary green-roof segmentation."""

from __future__ import annotations

from torch import Tensor, nn
from torch.nn import functional
from transformers import SegformerConfig, SegformerForSemanticSegmentation


class SegFormerB0(nn.Module):
    """Return full-resolution logits from a lightweight MiT-B0 SegFormer."""

    def __init__(
        self,
        num_classes: int = 1,
        pretrained: bool = True,
        checkpoint: str = "nvidia/mit-b0",
    ) -> None:
        super().__init__()
        self.model = (
            SegformerForSemanticSegmentation.from_pretrained(
                checkpoint,
                num_labels=num_classes,
                id2label={0: "green_roof"},
                label2id={"green_roof": 0},
                ignore_mismatched_sizes=True,
            )
            if pretrained
            else SegformerForSemanticSegmentation(
                SegformerConfig(num_labels=num_classes)
            )
        )

    def forward(self, inputs: Tensor) -> Tensor:
        logits = self.model(pixel_values=inputs).logits
        return functional.interpolate(
            logits,
            size=inputs.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
