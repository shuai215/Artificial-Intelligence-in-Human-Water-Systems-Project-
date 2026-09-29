# Spatially evaluated green-roof segmentation

This repository provides a reproducible pipeline for mapping green roofs from
Berlin aerial imagery under spatially grouped evaluation. It connects geospatial
preprocessing, reference-mask generation, ResNet50 U-Net and SegFormer-B0
training, validation-based threshold selection, frozen test evaluation, and
aligned inference on 2025 imagery.

The central result is clear: background-only training scenes are essential.
Keeping them increased U-Net test Dice from 0.2632 to 0.3839 and reduced the
number of negative test tiles with false-positive predictions from 85 to 16.

## Main findings

All reported comparisons use the same frozen PLZ 12357 spatial manifests and a
matched training/evaluation protocol.

| Finding | Evidence |
|---|---|
| Negative scenes improve overlap and background rejection | Dice 0.3839 versus 0.2632; false-positive negative tiles 16 versus 85 |
| ResNet50 U-Net is the strongest tested architecture | Dice 0.3839 versus 0.2268 for SegFormer-B0 |
| BCE-Dice provides the best overlap–precision balance | Dice 0.3839 and precision 0.4269 in the all-tile condition |
| Pure Tversky is screened out before test evaluation | The tested formulation produced an all-background validation solution |

Detailed evidence is retained in [`reports/`](reports/).

## Evaluation design

```text
2016 RGB tiles + polygon annotations + PLZ boundary
                         |
                         v
             rasterized reference masks
                         |
                         v
       spatial 8 x 8 block train / validation / test split
                         |
                         v
              U-Net or SegFormer training
                         |
                         v
          validation checkpoint and threshold selection
                         |
                         v
                   frozen test evaluation

2025 JP2 imagery -> alignment to the 2016 tile grid -> frozen-model inference
```

Complete spatial blocks are assigned to a single split, preventing adjacent
tiles from being distributed across training and evaluation data. The test set
is not used for checkpoint or threshold selection.

The supplied green-roof polygons contain boundary ambiguity, local offsets,
and missing visible vegetation. They are therefore used as **reference
annotations**, not error-free physical ground truth. Metrics quantify agreement
with the same reference target across controlled experiments. See
[`DATA_AUDIT.md`](DATA_AUDIT.md) for the full data contract.

## Repository scope

The repository contains code, experiment configurations, tests, and compact
result records. Raw imagery, annotations, model checkpoints, and generated
predictions remain outside Git.

The commands below assume this local layout:

```text
project/
├── code/          # this repository
├── annotation/    # local study-area boundary
├── data/          # local raw and processed imagery
└── runs/          # local checkpoints and evaluation outputs
```

## Setup

Python 3.12 and [`uv`](https://docs.astral.sh/uv/) are required. The committed
`pyproject.toml` and `uv.lock` define the environment; no `requirements.txt` is
used.

From `project/code`:

```powershell
uv sync --frozen
uv run python -m unittest discover -s tests -v
```

## Prepare and validate the dataset

Preprocessing selects tiles whose centres fall inside PLZ 12357, rasterizes the
two polygon layers, assigns spatial blocks, writes manifests, and creates a QA
contact sheet.

```powershell
uv run python scripts/prepare_green_roofs.py `
  ..\data\green_roofs\green_roofs `
  ..\data\processed\green_roofs_2016_plz12357 `
  --study-area ..\annotation\green_roofs_2016\berlin_plz_12357.gpkg `
  --block-size 8 `
  --seed 42

uv run python scripts/validate_green_roofs.py `
  ..\data\green_roofs\green_roofs `
  ..\data\processed\green_roofs_2016_plz12357
```

The frozen dataset contains 1,635 tiles: 1,180 training, 203 validation, and
252 test tiles. A total of 182 tiles contain reference foreground pixels.

Optional evidence-generation commands:

```powershell
uv run python scripts/create_label_quality_review.py `
  ..\data\green_roofs\green_roofs `
  ..\data\processed\green_roofs_2016_plz12357 `
  ..\Pre-processing\label_quality_check_plz12357 `
  --sample-count 50 `
  --seed 42

uv run python scripts/analyze_class_balance.py `
  ..\data\processed\green_roofs_2016_plz12357

uv run python scripts/analyze_spatial_distance.py `
  ..\data\processed\green_roofs_2016_plz12357
```

## Experiments

| Configuration | Role |
|---|---|
| `unet_plz12357_all.toml` | Primary U-Net: positive and negative tiles, BCE-Dice |
| `unet_plz12357_positive_only.toml` | Negative-scene ablation |
| `unet_plz12357_all_focal_dice.toml` | All-tile focal BCE-Dice condition |
| `unet_plz12357_positive_only_focal_dice.toml` | Positive-only focal BCE-Dice condition |
| `unet_plz12357_all_tversky.toml` | Validation-gated pure Tversky screen |
| `segformer_b0_plz12357.toml` | Matched SegFormer-B0 comparison |

The four scene-policy/loss configurations form a controlled 2 x 2 experiment.
The primary all-tile U-Net and SegFormer-B0 isolate the architecture choice.

Check the complete data/model/loss path before training:

```powershell
uv run python scripts/train.py configs/unet_plz12357_all.toml --check-only
uv run python scripts/train.py configs/segformer_b0_plz12357.toml --check-only
```

Run an experiment:

```powershell
uv run python scripts/train.py configs/unet_plz12357_all.toml
```

The shared protocol uses seed 42, 256 x 256 inputs, batch size 4, AdamW at
`1e-4`, 1,180 sampled tiles per epoch, and a 20-epoch limit. Balanced runs draw
590 positive and 590 negative tiles per epoch.

## Select the threshold and evaluate

Select the probability threshold on validation:

```powershell
uv run python scripts/analyze_validation.py `
  configs/unet_plz12357_all.toml `
  ..\runs\unet_plz12357_all\best.pt `
  ..\runs\unet_plz12357_all\validation_analysis
```

Freeze the selected threshold for the test evaluation:

```powershell
uv run python scripts/analyze_validation.py `
  configs/unet_plz12357_all.toml `
  ..\runs\unet_plz12357_all\best.pt `
  ..\runs\unet_plz12357_all\test_analysis `
  --split test `
  --threshold 0.05
```

## Transfer to 2025 imagery

The primary U-Net checkpoint is applied to JP2 bands 1–3 after reprojection onto
the frozen 2016 XYZ grid:

```powershell
uv run python scripts/predict_2025.py `
  configs/unet_plz12357_all.toml `
  ..\runs\unet_plz12357_all\best.pt `
  0.05 unet_plz12357_all `
  ..\data\green_roofs\green_roofs\orthophotos_2025\12357 `
  ..\data\processed\green_roofs_2016_plz12357 `
  ..\data\processed\green_roofs_2025_plz12357
```

The 2025 outputs support temporal-transfer inspection. Without independent 2025
reference annotations, they are prediction observations rather than accuracy
measurements. `create_2025_comparison_grids.py` generates a deterministic,
stratified 50-image U-Net/SegFormer review set.

## Project structure

```text
configs/                 experiment definitions
reports/                 result-to-claim records
scripts/                 runnable preparation, training, evaluation, and inference entry points
src/green_roofs/         reusable geospatial and segmentation implementation
tests/                   preprocessing, spatial-distance, and training-pipeline tests
DATA_AUDIT.md             data and reference-annotation contract
pyproject.toml            package and dependency definition
uv.lock                   reproducible dependency lock
```

Key implementation paths:

- `preprocessing.py` orchestrates tile selection, rasterization, spatial splitting, manifests, and QA;
- `prepared_dataset.py` validates masks, manifests, counts, and block isolation;
- `dataset.py`, `transforms.py`, and `sampling.py` build the training input path;
- `models/` provides ResNet50 U-Net and SegFormer-B0 through one model factory;
- `losses.py`, `metrics.py`, and `engine/trainer.py` implement the shared experiment loop;
- `analyze_validation.py` performs validation threshold selection and frozen test evaluation;
- `predict_2025.py` performs aligned temporal-transfer inference.

## Result records

- [`negative_tile_ablation.md`](reports/negative_tile_ablation.md): negative scenes improve overlap and false-positive control;
- [`focal_loss_ablation.md`](reports/focal_loss_ablation.md): BCE-Dice with all tiles provides the strongest balance;
- [`model_comparison.md`](reports/model_comparison.md): ResNet50 U-Net leads the matched architecture comparison;
- [`tversky_validation_screen.md`](reports/tversky_validation_screen.md): validation screening selects BCE-Dice;
- [`label_quality_audit.md`](reports/label_quality_audit.md): evaluation uses consistent reference annotations.
