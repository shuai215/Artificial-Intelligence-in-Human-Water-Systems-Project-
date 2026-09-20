"""Segmentation model implementations."""

from torch import nn

from .exercise_resnet50_unet import ExerciseResNet50UNet
from .segformer import SegFormerB0


def build_model(config: dict[str, object], *, pretrained: bool | None = None) -> nn.Module:
    """Build the configured segmentation model behind one shared interface."""
    name = str(config["name"])
    use_pretrained = (
        bool(config.get("pretrained", True)) if pretrained is None else pretrained
    )
    num_classes = int(config.get("num_classes", 1))
    if name == "exercise_resnet50_unet":
        return ExerciseResNet50UNet(
            num_classes=num_classes,
            pretrained=use_pretrained,
            freeze_encoder_batch_norm=bool(
                config.get("freeze_encoder_batch_norm", False)
            ),
        )
    if name == "segformer_b0":
        return SegFormerB0(
            num_classes=num_classes,
            pretrained=use_pretrained,
            checkpoint=str(config.get("checkpoint", "nvidia/mit-b0")),
        )
    raise ValueError(f"Unsupported model: {name}")


__all__ = ["ExerciseResNet50UNet", "SegFormerB0", "build_model"]
