from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from green_roofs.dataset import GreenRoofDataset  # noqa: E402
from green_roofs.losses import BCEDiceLoss, FocalBCEDiceLoss, TverskyLoss  # noqa: E402
from green_roofs.manifests import MANIFEST_FIELDS  # noqa: E402
from green_roofs.metrics import BinarySegmentationMetrics  # noqa: E402
from green_roofs.models import ExerciseResNet50UNet, build_model  # noqa: E402
from green_roofs.sampling import BalancedTileBatchSampler  # noqa: E402
from green_roofs.transforms import SegmentationTransform  # noqa: E402


class DatasetTests(unittest.TestCase):
    def test_manifest_pair_is_loaded_and_normalized(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_root = root / "raw"
            processed_root = root / "processed"
            image_path = dataset_root / "images" / "tile.png"
            mask_path = processed_root / "masks" / "tile.png"
            manifest_path = processed_root / "manifests" / "train.csv"
            image_path.parent.mkdir(parents=True)
            mask_path.parent.mkdir(parents=True)
            manifest_path.parent.mkdir(parents=True)
            Image.new("RGB", (256, 256), (120, 140, 160)).save(image_path)
            mask = Image.new("L", (256, 256), 0)
            mask.putpixel((20, 30), 1)
            mask.save(mask_path)
            with manifest_path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=MANIFEST_FIELDS,
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "x": 1,
                        "y": 2,
                        "image_relpath": "images/tile.png",
                        "mask_relpath": "masks/tile.png",
                        "block_id": "b00_00",
                        "positive_pixels": "1",
                    }
                )

            dataset = GreenRoofDataset(
                dataset_root,
                processed_root,
                manifest_path,
                SegmentationTransform(training=False, augment=False),
            )
            image_tensor, mask_tensor, tile_id = dataset[0]
            self.assertEqual(tuple(image_tensor.shape), (3, 256, 256))
            self.assertEqual(tuple(mask_tensor.shape), (256, 256))
            self.assertEqual(mask_tensor.dtype, torch.int64)
            self.assertEqual(mask_tensor[30, 20].item(), 1)
            self.assertEqual(tile_id, "19_1_2")
            self.assertEqual(dataset.positive_indices, [0])
            self.assertEqual(dataset.negative_indices, [])
            self.assertAlmostEqual(
                dataset.records[0]["positive_fraction"], 1 / (256 * 256)
            )
            self.assertEqual(dataset.records[0]["has_green_roof"], 1)

    def test_evaluation_rejects_random_augmentation(self) -> None:
        with self.assertRaises(ValueError):
            SegmentationTransform(training=False, augment=True)


class MetricTests(unittest.TestCase):
    def test_foreground_metrics(self) -> None:
        logits = torch.tensor([[[[-5.0, 5.0], [5.0, -5.0]]]])
        target = torch.tensor([[[0, 1], [0, 0]]])
        metrics = BinarySegmentationMetrics()
        metrics.update(logits, target)
        result = metrics.compute()
        self.assertAlmostEqual(result["foreground_precision"], 0.5)
        self.assertAlmostEqual(result["foreground_recall"], 1.0)
        self.assertAlmostEqual(result["foreground_iou"], 0.5)
        self.assertAlmostEqual(result["foreground_dice"], 2 / 3)


class ModelTests(unittest.TestCase):
    def test_segformer_preserves_spatial_shape(self) -> None:
        model = build_model(
            {"name": "segformer_b0", "num_classes": 1}, pretrained=False
        ).eval()
        with torch.no_grad():
            output = model(torch.zeros(1, 3, 64, 64))
        self.assertEqual(tuple(output.shape), (1, 1, 64, 64))

    def test_exercise_model_preserves_spatial_shape(self) -> None:
        model = ExerciseResNet50UNet(pretrained=False).eval()
        with torch.no_grad():
            output = model(torch.zeros(1, 3, 64, 64))
        self.assertEqual(tuple(output.shape), (1, 1, 64, 64))

    def test_encoder_batch_norm_can_be_frozen(self) -> None:
        model = ExerciseResNet50UNet(
            pretrained=False, freeze_encoder_batch_norm=True
        ).train()
        batch_norm_layers = [
            module
            for module in model.encoder.modules()
            if isinstance(module, torch.nn.BatchNorm2d)
        ]
        self.assertTrue(batch_norm_layers)
        self.assertTrue(all(not module.training for module in batch_norm_layers))
        self.assertTrue(
            all(
                not parameter.requires_grad
                for module in batch_norm_layers
                for parameter in module.parameters()
            )
        )
        self.assertTrue(model.decoder0.training)


class LossTests(unittest.TestCase):
    def test_all_losses_accept_one_channel_logits_and_backpropagate(self) -> None:
        target = torch.tensor([[[0, 1], [1, 0]]])
        for criterion in (BCEDiceLoss(), FocalBCEDiceLoss(), TverskyLoss()):
            logits = torch.zeros(1, 1, 2, 2, requires_grad=True)
            loss = criterion(logits, target)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertIsNotNone(logits.grad)


class SamplingTests(unittest.TestCase):
    def test_samples_per_epoch_must_be_positive(self) -> None:
        with self.assertRaisesRegex(ValueError, "samples_per_epoch must be positive"):
            BalancedTileBatchSampler([0], [1], batch_size=2, samples_per_epoch=0)

    def test_balanced_batches_are_exact_and_reproducible(self) -> None:
        positive = [0, 1, 2]
        negative = [3, 4, 5, 6, 7]
        first = BalancedTileBatchSampler(
            positive, negative, batch_size=4, positive_fraction=0.5, seed=42
        )
        second = BalancedTileBatchSampler(
            positive, negative, batch_size=4, positive_fraction=0.5, seed=42
        )
        first_batches = list(first)
        self.assertEqual(first_batches, list(second))
        for batch in first_batches:
            self.assertEqual(sum(index in positive for index in batch), 2)
            self.assertEqual(sum(index in negative for index in batch), 2)

    def test_non_integer_batch_fraction_is_balanced_over_the_epoch(self) -> None:
        positive = [0, 1, 2]
        negative = [3, 4, 5, 6, 7]
        sampler = BalancedTileBatchSampler(
            positive,
            negative,
            batch_size=4,
            positive_fraction=0.3,
            seed=42,
            samples_per_epoch=100,
        )
        batches = list(sampler)
        positive_count = sum(
            index in positive for batch in batches for index in batch
        )
        sample_count = sum(len(batch) for batch in batches)
        self.assertAlmostEqual(positive_count / sample_count, 0.3, places=2)
        self.assertTrue(
            all(1 <= sum(index in positive for index in batch) <= 2 for batch in batches)
        )


if __name__ == "__main__":
    unittest.main()
