# Green-roof semantic segmentation

This repository contains the reproducible code for detecting and segmenting
green roofs in Berlin aerial imagery.

## Current milestone: reproducible U-Net baseline pipeline

The raw dataset is kept outside this Git repository under
`project/data/green_roofs/green_roofs`. It contains:

- 2016 orthophotos as XYZ Web Mercator PNG tiles;
- 2025 orthophotos as JPEG 2000 mosaics;
- green-roof labels as WGS 84 Shapefile polygons;
- 2016 building footprints as GeoJSON.

Install the project dependencies and run the dataset audit from `project/code`:

```powershell
uv sync
.\.venv\Scripts\Activate.ps1
```

```powershell
python scripts/inspect_green_roofs.py `
  ..\data\green_roofs\green_roofs `
  --output reports\green_roofs_inventory.json
```

## Prepare the segmentation dataset

The preprocessing command rasterizes the two green-roof label layers into
binary 0/1 masks. It keeps the original images in place and writes generated
data to a separate directory:

```powershell
python scripts/prepare_green_roofs.py `
  ..\data\green_roofs\green_roofs `
  ..\data\processed\green_roofs_2016_plz12357 `
  --study-area ..\annotation\green_roofs_2016\berlin_plz_12357.gpkg `
  --block-size 8 `
  --seed 42
```

Validate every image/mask reference, mask value, pixel count, and spatial block:

```powershell
python scripts/validate_green_roofs.py `
  ..\data\green_roofs\green_roofs `
  ..\data\processed\green_roofs_2016_plz12357
```

The study-area layer is the current official PLZ 12357 reference boundary in
EPSG:3857; preprocessing keeps a tile when its centre is covered by the polygon.
The prepared directory contains 1,635 masks, spatially grouped train/validation/
test CSV manifests, `dataset_summary.json`, and a QA contact sheet. Manifests
store paths relative to their respective raw and processed roots, so generated
data remains portable. CSV files store only `x`, `y`, image/mask paths,
`block_id`, and `positive_pixels`; `tile_id`, `positive_fraction`, and
`has_green_roof` are derived when a manifest is loaded.

### Preprocessing architecture

`green_roofs.preprocessing` is the stable public entry point and only
orchestrates the pipeline. Focused implementation modules are:

- `tiles.py`: XYZ keys, discovery, and EPSG:3857 bounds;
- `study_area.py`: polygon loading and tile-centre study-area filtering;
- `labels.py`: GeoPandas/Pyogrio loading, CRS normalization, and topology;
- `rasterization.py`: Shapely spatial queries and Rasterio mask generation;
- `spatial_split.py`: deterministic block assignment;
- `manifests.py`: raw-fact CSV schema, serialization, and runtime derivation;
- `prepared_dataset.py`: QA contact sheets and full image/mask validation.

The data flow is `tiles → study-area filter + labels → masks → spatial split →
manifests/QA`.
Existing callers can continue importing `prepare_dataset()` and
`validate_prepared_dataset()` from `green_roofs.preprocessing`.

See [`DATA_AUDIT.md`](DATA_AUDIT.md) for the current findings and modeling
recommendation.

## Prepare the QGIS reannotation workspace

Create a georeferenced VRT from all 3,510 local XYZ tiles and copy the two
source-label Shapefiles into one editable GeoPackage:

```powershell
python scripts/prepare_qgis_annotation.py `
  ..\data\green_roofs\green_roofs `
  ..\data\processed\green_roofs_2016 `
  ..\annotation\green_roofs_2016

..\tools\QGIS\bin\python-qgis-ltr.bat `
  scripts\build_qgis_annotation_project.py `
  ..\annotation\green_roofs_2016 `
  ..\data\green_roofs\green_roofs
```

Open `../annotation/green_roofs_2016/green_roofs_reannotation.qgz`. Only the
`green roofs — editable v2` layer should be edited. The VRT references the raw
PNG files and does not duplicate their pixels. See the README beside the QGIS
project for the review-status policy and digitizing procedure.

The workspace also contains `berlin_plz_12357.gpkg`, downloaded from the
official Berlin postcode WFS and reprojected to EPSG:3857. It is a current
reference boundary rather than a verified historical 2016 boundary.

## Analyze spatial train–prediction distances

Reproduce a local version of the Meyer–Pebesma spatial-distance diagnostic for
the frozen split. The analysis compares each training tile with its nearest
other training tile and each validation/test tile with its nearest training
tile:

```powershell
python scripts/analyze_spatial_distance.py `
  ..\data\processed\green_roofs_2016
```

The command writes PNG/PDF/SVG figures, row-level source distances, and a JSON
summary under `spatial_distance_analysis`. The current dataset covers Berlin
postal code 12357, not the full Berlin administrative area. The diagnostic
therefore evaluates the spatial difficulty of the current frozen split; it is
not yet a city-wide deployment analysis. XYZ tile centres are transformed with
Pyproj to ETRS89 / UTM zone 33N (EPSG:25833), and nearest neighbours are queried
with SciPy's `cKDTree`.

## Create the manual label-review package

Generate a deterministic, coverage-stratified sample of 50 positive tiles. Each
output pairs the original image with a green mask overlay and is accompanied by
a CSV for recording manual review decisions:

```powershell
python scripts/create_label_quality_review.py `
  ..\data\green_roofs\green_roofs `
  ..\data\processed\green_roofs_2016 `
  ..\Pre-processing\label_quality_check `
  --sample-count 50 `
  --seed 42
```

## Train the first U-Net baseline

The baseline reuses the exercise's ResNet50/AlbuNet-style model, but not its
destructive blank-tile removal or random file split. Both models will read the
same frozen spatial manifests. Training and validation transforms are separate;
all loss experiments use the same training-only flips and 90-degree rotations,
while validation remains deterministic and follows the natural class balance.
The model produces one binary logit channel.

Run from `project/code` after completing the manual label review:

```powershell
python scripts/train.py configs/unet_baseline.toml
```

Before formal training, validate the complete input, sampling, model-forward,
and loss pipeline without downloading pretrained weights or updating parameters:

```powershell
python scripts/train.py configs/unet_baseline.toml --check-only
```

Stage-one loss ablations use identical data, model, augmentation, 50/50
positive/negative training batches, and optimizer settings:

- `configs/unet_baseline.toml`: BCE + Dice;
- `configs/unet_focal_dice.toml`: focal BCE + Dice;
- `configs/unet_tversky.toml`: Tversky loss.

The PLZ 12357 experiment keeps one frozen spatial split for both training
policies. Validation and test always retain all positive and negative tiles:

```powershell
python scripts/train.py configs/unet_plz12357_all.toml --check-only
python scripts/train.py configs/unet_plz12357_positive_only.toml --check-only
```

- `unet_plz12357_all.toml`: 50/50 positive/negative training batches;
- `unet_plz12357_positive_only.toml`: only positive training tiles via
  PyTorch `WeightedRandomSampler`, with the same samples/optimizer steps per
  epoch as the all-tile experiment.

Analyze the selected checkpoint on validation data only, sweep probability
thresholds, export per-tile errors, and generate TP/FP/FN diagnostics:

```powershell
python scripts/analyze_validation.py `
  configs/unet_baseline.toml `
  ..\runs\unet_baseline\best.pt `
  ..\runs\unet_baseline\validation_analysis
```

The original baseline showed strong overfitting. A separate, non-overwriting
v2 configuration limits each epoch to two expected draws per positive tile,
freezes ResNet50 encoder BatchNorm, and adds validation-Dice early stopping:

```powershell
python scripts/train.py configs/unet_baseline_v2.toml --check-only
python scripts/train.py configs/unet_baseline_v2.toml
```

The v2 run stopped after only 2,064 optimizer steps and underfit relative to the
12,520-step original baseline. The follow-up v3 keeps the same reduced sampling
and frozen BatchNorm design but allows up to 50 epochs and uses early-stopping
patience 15 (6,450 maximum optimizer steps):

```powershell
python scripts/train.py configs/unet_baseline_v3.toml --check-only
python scripts/train.py configs/unet_baseline_v3.toml
```

The targeted sampler ablation returns to every v1 setting and changes only the
training tile mixture from 50/50 to approximately 30% positive and 70% negative:

```powershell
python scripts/train.py configs/unet_sampler_30_70.toml --check-only
python scripts/train.py configs/unet_sampler_30_70.toml
```

With batch size four, individual batches alternate deterministically between
one and two positive tiles so that the full epoch reaches the requested ratio.
The current 626 full batches contain 751 planned positive and 1,753 planned
negative draws (29.99%/70.01%).

The first run may download ImageNet ResNet50 weights. Configuration is in
`configs/unet_baseline.toml`; outputs are written to
`project/runs/unet_baseline` and include `best.pt`, `last.pt`, and CSV/JSON
training histories. The test manifest is deliberately excluded from model
selection. Final test evaluation uses the threshold frozen on validation; see
`reports/negative_tile_ablation.md` and `reports/model_comparison.md`.
The role of every retained TOML file is listed in `configs/README.md`.

## Train the SegFormer-B0 comparison

SegFormer-B0 reuses the same frozen manifests, 256×256 inputs, augmentation,
balanced sampler, BCE + Dice loss, optimizer settings, and validation metrics
as `unet_baseline_v3.toml`. Its ImageNet-pretrained MiT-B0 encoder and MLP
decoder provide the Transformer comparison; the adapter upsamples its logits
to the original mask size.

```powershell
python scripts/train.py configs/segformer_b0.toml --check-only
python scripts/train.py configs/segformer_b0.toml
```

The first full run downloads `nvidia/mit-b0`. Results are written to
`project/runs/segformer_b0`; use `scripts/analyze_validation.py` with the same
config and `best.pt` to select the validation threshold without reading test
labels. After freezing that threshold, evaluate test once with
`--split test --threshold <validation-threshold>`. The completed comparison is
recorded in
`reports/model_comparison.md`.

Key modules:

- `src/green_roofs/dataset.py`: manifest-based image/mask matching;
- `src/green_roofs/transforms.py`: synchronized transforms and normalization;
- `src/green_roofs/sampling.py`: exact 50/50 training-tile batch sampling;
- `src/green_roofs/losses.py`: the three binary loss alternatives;
- `src/green_roofs/models/`: replaceable model definitions;
- `src/green_roofs/engine/trainer.py`: shared training/validation engine;
- `src/green_roofs/metrics.py`: foreground Dice, IoU, precision, and recall.

## Predict the aligned 2025 imagery

`predict_2025.py` reprojects JP2 bands 1–3 from the RGBI product onto the
frozen 2016 XYZ grid, caches the aligned RGB tiles, and writes binary masks
with a validation-frozen threshold. For example:

```powershell
python scripts/predict_2025.py `
  configs/segformer_b0.toml `
  ..\runs\segformer_b0\best.pt `
  0.45 segformer_b0 `
  ..\data\green_roofs\green_roofs\orthophotos_2025\12357 `
  ..\data\processed\green_roofs_2016_plz12357 `
  ..\data\processed\green_roofs_2025_plz12357
```

These outputs are unlabeled predictions, not accuracy measurements. Each model
directory contains a `manifest.csv` and `summary.json` for later manual review
or evaluation against independently annotated 2025 masks.

Create a deterministic 50-image four-panel review set, stratified across
three-model agreement, pairwise agreement, single-model predictions, and
no-prediction controls:

```powershell
python scripts/create_2025_comparison_grids.py `
  ..\data\processed\green_roofs_2025_plz12357 `
  ..\data\processed\green_roofs_2025_plz12357\comparison_50
```

The stratified review set intentionally over-represents rare disagreement
patterns and therefore must not be used to estimate their overall frequency.

Run all tests:

```powershell
python -m unittest discover -s tests -v
```

## Analyze class balance

Recompute tile-level and pixel-level positive/negative ratios directly from all
three manifests and their binary masks:

```powershell
python scripts/analyze_class_balance.py `
  ..\data\processed\green_roofs_2016
```

Results are written to
`data/processed/green_roofs_2016/class_balance_analysis` as JSON, CSV, and a
human-readable Markdown report. The script also rejects masks containing values
other than 0 and 1.
