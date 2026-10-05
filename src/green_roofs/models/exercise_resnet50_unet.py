"""ResNet50 encoder-decoder adapted from the Roofpedia exercise code.

This is an AlbuNet-style U-Net variant, not the original symmetric U-Net.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional
from torchvision.models import ResNet50_Weights, resnet50


class ConvRelu(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.convolution = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=False
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return functional.relu(self.convolution(inputs), inplace=True)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.convolution = ConvRelu(in_channels, out_channels)

    def forward(self, inputs: Tensor) -> Tensor:
        inputs = functional.interpolate(inputs, scale_factor=2, mode="nearest")
        return self.convolution(inputs)


class ExerciseResNet50UNet(nn.Module):
    """Binary AlbuNet-style model used by the teaching exercise."""

    def __init__(
        self,
        num_classes: int = 1,
        num_filters: int = 32,
        pretrained: bool = True,
        freeze_encoder_batch_norm: bool = False,
    ) -> None:
        super().__init__()
        self.freeze_encoder_batch_norm = freeze_encoder_batch_norm
        #Use torchvision's ResNet50 directly as the encoder
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        self.encoder = resnet50(weights=weights)
        self.center = DecoderBlock(2048, num_filters * 8)
        self.decoder0 = DecoderBlock(2048 + num_filters * 8, num_filters * 8)
        self.decoder1 = DecoderBlock(1024 + num_filters * 8, num_filters * 8)
        self.decoder2 = DecoderBlock(512 + num_filters * 8, num_filters * 4)
        self.decoder3 = DecoderBlock(256 + num_filters * 2, num_filters * 2)
        self.decoder4 = DecoderBlock(num_filters * 4, num_filters)
        self.decoder5 = ConvRelu(num_filters, num_filters)
        self.classifier = nn.Conv2d(num_filters, num_classes, kernel_size=1)
        if self.freeze_encoder_batch_norm:
            for module in self.encoder.modules():
                if isinstance(module, nn.BatchNorm2d):
                    for parameter in module.parameters():
                        parameter.requires_grad = False

    def train(self, mode: bool = True) -> "ExerciseResNet50UNet":
        super().train(mode)
        if mode and self.freeze_encoder_batch_norm:
            for module in self.encoder.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
        return self

    def forward(self, inputs: Tensor) -> Tensor:
        height, width = inputs.shape[-2:]
        if height % 32 or width % 32:
            raise ValueError("Input height and width must be divisible by 32")

        encoder0 = self.encoder.maxpool(
            self.encoder.relu(self.encoder.bn1(self.encoder.conv1(inputs)))
        )
        encoder1 = self.encoder.layer1(encoder0)
        encoder2 = self.encoder.layer2(encoder1)
        encoder3 = self.encoder.layer3(encoder2)
        encoder4 = self.encoder.layer4(encoder3)

        center = self.center(functional.max_pool2d(encoder4, kernel_size=2, stride=2))
        decoder0 = self.decoder0(torch.cat([encoder4, center], dim=1))
        decoder1 = self.decoder1(torch.cat([encoder3, decoder0], dim=1))
        decoder2 = self.decoder2(torch.cat([encoder2, decoder1], dim=1))
        decoder3 = self.decoder3(torch.cat([encoder1, decoder2], dim=1))
        decoder4 = self.decoder4(decoder3)
        decoder5 = self.decoder5(decoder4)
        return self.classifier(decoder5)
