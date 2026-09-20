# Development log

## 2026-08-26 — Pitch presentation storyboard

- Added `presentation/PITCH_PLAN.md` as the source-of-truth storyboard for the
  required 12-minute project pitch and the optional 10-minute Q&A appendix.
- Mapped every assignment requirement to a slide and incorporated the final PLZ
  12357 preprocessing statistics, controlled training ablation, calibrated
  validation metrics, subgroup findings, limitations, and untouched-test
  boundary.
- Files: `presentation/PITCH_PLAN.md` and `DEV_LOG.md`.

## 2026-08-26 — Positive/negative validation subset analysis

- Extended `analyze_validation.py` without changing the frozen split or
  threshold-selection rule. It now reports aggregate foreground metrics for
  the 22 positive validation tiles and false-positive incidence for the 181
  negative validation tiles.
- The all-tile checkpoint selected threshold 0.05 and reached full-validation
  Dice 0.4754, positive-tile Dice 0.5653, and false positives on 10/181 negative
  tiles. The positive-only checkpoint selected threshold 0.95 and reached
  full-validation Dice 0.4221, positive-tile Dice 0.5001, and false positives on
  65/181 negative tiles.
- The opposite optimal thresholds reveal strongly different probability
  calibration. Positive-only training did not outperform all-tile training even
  on the positive subset and spread smaller false-positive regions across more
  negative scenes.
- Files: `scripts/analyze_validation.py`, `DEV_LOG.md`, and both
  `project/runs/unet_plz12357_*/validation_analysis/` output directories.
- Validation: compilation and all 22 unit tests passed before both complete
  203-tile validation analyses.

## 2026-08-26 — PLZ 12357 positive-only U-Net run

- Completed the paired 20-epoch positive-only training run on the same
  1,180/203 train/validation manifests and with the same 1,180 optimizer samples
  per epoch as the all-tile baseline. All sampled training tiles were positive;
  the complete validation distribution was unchanged.
- The best checkpoint occurred at epoch 11: validation Dice 0.4159, IoU 0.2625,
  precision 0.3971, recall 0.4366, and loss 0.3484 at threshold 0.5.
- Compared with the all-tile best checkpoint, positive-only training increased
  recall (0.4366 vs 0.3635) but reduced precision (0.3971 vs 0.5591), Dice
  (0.4159 vs 0.4406), and probability/loss quality (0.3484 vs 0.1133). This is
  consistent with removing negative-only training scenes increasing detections
  and false positives.
- Outputs: `project/runs/unet_plz12357_positive_only/{best.pt,last.pt,
  history.csv,history.json}`. Checkpoint inspection confirmed epochs 11 and 20.

## 2026-08-26 — PLZ 12357 all-tile U-Net run

- Completed the 20-epoch `unet_plz12357_all.toml` baseline on the RTX 3060
  Laptop GPU using the frozen 1,180/203 train/validation manifests.
- The 50/50 sampler drew 590 positive and 590 negative tiles per epoch. The
  best validation checkpoint occurred at epoch 10: Dice 0.4406, IoU 0.2825,
  precision 0.5591, recall 0.3635, and loss 0.1133 at the fixed 0.5 threshold.
- Epoch 20 reached training Dice 0.8105 but validation Dice 0.3154, with large
  validation fluctuations across epochs. Use `best.pt`, not `last.pt`, and run
  the validation threshold/error analysis before interpreting the model.
- Outputs: `project/runs/unet_plz12357_all/{best.pt,last.pt,history.csv,
  history.json}`. Checkpoint inspection confirmed epochs 10 and 20 respectively.

## 2026-08-26 — PLZ 12357 study area and positive-only ablation

- Added `study_area.py` to load a polygon layer with GeoPandas/Pyogrio, project
  it to EPSG:3857, and retain XYZ tiles whose centres are covered by the unioned
  study-area geometry.
- Added `--study-area` to preprocessing and recorded its path, centre-selection
  rule, discovered count, and excluded count in `dataset_summary.json`.
- Added paired training configurations over one frozen spatial split: the
  all-tile experiment uses 50/50 batches, while the positive-only experiment
  uses PyTorch `WeightedRandomSampler`. Both draw 1,180 samples per epoch so
  optimizer-step counts remain comparable; validation remains unchanged and
  includes both positive and negative tiles.
- Restored the standard-library CSV import required by the manual label-review
  package writer.
- Files: `src/green_roofs/{study_area,preprocessing,label_qa}.py`,
  `scripts/{prepare_green_roofs,train}.py`,
  `configs/unet_plz12357_{all,positive_only}.toml`,
  `tests/test_preprocessing.py`, `README.md`, `DATA_AUDIT.md`, and `DEV_LOG.md`.
- Validation: all 22 unit tests passed. A new, non-overwriting processed dataset
  contains 1,635 masks (182 positive), 643,038 foreground pixels, and
  1,180/203/252 train/validation/test tiles across 28/5/5 non-overlapping
  spatial blocks. Both training configurations passed CUDA forward/loss checks;
  the shared validation split contains all 203 tiles.

## 2026-08-26 — Raw-fact-only manifests

- Reduced persisted manifest rows to six source-of-truth fields: XYZ indices,
  relative image/mask paths, spatial block ID, and positive-pixel count.
- `tile_id`, `positive_fraction`, and `has_green_roof` are now derived by
  `load_manifest()` in memory. Existing manifests with the former extra columns
  remain readable because the loader selects only the canonical raw fields.
- Removed the derived-field inconsistency test matrix. The existing Dataset
  integration test now writes the minimal schema and verifies all three runtime
  derivations together.
- Files: `src/green_roofs/manifests.py`, `tests/test_training_pipeline.py`,
  `README.md`, and `DEV_LOG.md`.
- Validation: compilation and all 21 focused unit tests passed. A temporary
  full rebuild wrote exactly the six-field CSV header, then passed prepared-data
  validation and class-balance analysis with 3,510 tiles, 184 positive tiles,
  644,525 foreground pixels, and 2,462/560/488 split counts. The temporary
  output was removed; existing processed data was not changed.

## 2026-08-26 — Manifest normalization and redundancy cleanup

- Removed unused per-label `source`/`source_index` fields and stopped reporting
  `invalid_geometries`, which could only be zero after the fail-fast geometry
  validation. Also removed the redundant exterior-ring count while retaining
  polygon-part and interior-ring topology statistics.
- Added `manifests.py` as the single schema/read/write boundary for prepared
  CSV files. Dataset loading, QA, prepared-data validation, QGIS metadata,
  spatial-distance analysis, and class-balance analysis now share it.
- Enforced manifest invariants for unique and coordinate-consistent `tile_id`,
  pixel bounds, `positive_fraction`, and `has_green_roof`, so retained derived
  columns cannot silently drift from `positive_pixels`.
- Reused the project-wide zoom constant, consolidated duplicate mask-overlay
  rendering and binary-metric formulas, and fixed the class-balance CLI's CSV
  writer import exposed by end-to-end testing.
- Files: `src/green_roofs/{labels,preprocessing,manifests,prepared_dataset,
  dataset,label_qa,qgis_annotation,spatial_distance,metrics}.py`,
  `scripts/{analyze_class_balance,analyze_validation}.py`,
  `tests/test_training_pipeline.py`, `README.md`, and `DEV_LOG.md`.
- Validation: compilation and all 22 unit tests passed. A temporary full rebuild,
  validation, and class-balance analysis reproduced 3,510 tiles, 983 labels,
  184 positive tiles, 644,525 foreground pixels, and the 2,462/560/488 split
  counts with no spatial-block leakage. The temporary output was removed and
  the existing processed dataset was not changed.

## 2026-08-26 — Direct label projection to EPSG:3857

- Simplified vector-label loading so each GeoPandas/Pyogrio layer is projected
  directly from its declared source CRS to EPSG:3857 before concatenation.
- Removed the intermediate normalize-to-EPSG:4326 and post-concatenation
  reprojection steps; this matches the Mercantile XYZ bounds and Rasterio mask
  transform used immediately downstream.
- Preserved the public CRS constants for compatibility and updated the vector
  loading regression test to require EPSG:3857 output.
- Files: `src/green_roofs/labels.py`, `tests/test_preprocessing.py`, and
  `DEV_LOG.md`.
- Validation: all 21 unit tests passed; a temporary full rebuild/validation
  reproduced 3,510 tiles, 983 labels, 184 positive tiles, 644,525 foreground
  pixels, and the 2,462/560/488 split counts exactly. Temporary output was
  removed and the existing processed dataset was not changed.

## 2026-08-26 — Preprocessing module decomposition

- Split the 381-line multi-responsibility `preprocessing.py` into focused
  modules: `tiles.py`, `labels.py`, `rasterization.py`, `spatial_split.py`, and
  `prepared_dataset.py`.
- Reduced `preprocessing.py` to a 111-line orchestration/facade module containing
  `prepare_dataset()` plus compatibility re-exports for existing callers.
- Moved QGIS annotation to import tile primitives directly from `tiles.py`, so
  implementation modules no longer depend on the orchestration facade.
- Updated preprocessing tests to import and exercise their owning modules; CLI
  scripts still verify the stable facade imports.
- Files: `src/green_roofs/{preprocessing,tiles,labels,rasterization,
  spatial_split,prepared_dataset,qgis_annotation}.py`,
  `tests/test_preprocessing.py`, `README.md`, and `DEV_LOG.md`.
- Validation: compilation and all 21 unit tests passed after the move. A
  temporary real-data rebuild and validation reproduced 3,510 masks, 983 labels,
  971 intersecting labels, 184 positive tiles, 644,525 foreground pixels, and
  the 2,462/560/488 train/validation/test tile counts exactly. Existing
  processed data remained untouched and the temporary output was removed.

## 2026-08-26 — Unified GeoPandas/Shapely/Rasterio GIS pipeline

- Replaced the incremental PyShp data layer with GeoPandas using the Pyogrio
  engine; source Polygon/MultiPolygon topology and CRS metadata now remain in a
  GeoDataFrame through preprocessing.
- Reprojected labels to EPSG:3857 with Pyproj, selected per-tile candidates with
  the Shapely spatial index, and rasterized all candidates once per tile with a
  Rasterio transform derived from Mercantile XYZ bounds.
- Removed custom global-pixel conversion, signed-ring-area topology inference,
  per-feature Pillow mask unions, and handwritten PNG/SHP/DBF audit parsing.
- Replaced the audit path with Rasterio image inspection, GeoPandas vector
  inspection, Shapely coverage predicates, and Pyproj geodesic resolution.
- Replaced the local spherical distance approximation with Pyproj transformation
  to ETRS89 / UTM zone 33N (EPSG:25833); SciPy `cKDTree` remains the nearest-
  neighbour engine.
- Replaced `ogr2ogr`/temporary-GeoJSON GeoPackage construction with
  GeoPandas/Pyogrio/Shapely and replaced QGIS tile-coordinate formulas with
  Mercantile. The VRT source-window XML remains explicit because the raw XYZ
  PNG files contain no individual georeferencing. Cross-drive VRT references
  now fall back safely to absolute paths.
- Replaced manifest CSV writing and QGIS manifest loading with Pandas. Removed
  PyShp from `requirements.txt`; added GeoPandas, Pyogrio, Shapely, Pyproj,
  Mercantile, and Pandas. Fiona, Xarray/Rioxarray, and direct GDAL Python
  bindings were intentionally not added because this pipeline has no matching
  use case for them.
- Files: `src/green_roofs/{preprocessing,spatial_distance,qgis_annotation}.py`,
  `scripts/{inspect_green_roofs,analyze_spatial_distance,
  prepare_qgis_annotation}.py`, `tests/{test_preprocessing,
  test_spatial_distance}.py`, `requirements.txt`, `README.md`, `DATA_AUDIT.md`,
  and `DEV_LOG.md`.
- Validation: compilation and all 21 unit tests passed in `exercise/.venv`,
  including a GeoPackage layer-write/read regression test.
  The real-data audit passed; a temporary 3,510-mask rebuild and validation
  passed with 983 labels, 971 intersecting labels, 184 positive tiles, 644,525
  foreground pixels, and no spatial-block leakage. EPSG:25833 distance analysis
  completed. A temporary GeoPackage contained 983 MultiPolygon labels and 3,510
  Polygon tile cells in EPSG:3857 and was read back successfully with Pyogrio.
- Risk/TODO: the checked-in processed masks/manifests and prior distance report
  remain the earlier dataset version. Regeneration would change mask counts,
  block assignments, and distance statistics; create a versioned output rather
  than overwriting them, repeat label QA, and do not compare checkpoints across
  mask/split versions.

## 2026-08-26 — Geospatial library refactor

- Replaced the handwritten Shapefile/DBF binary readers with PyShp while
  preserving the existing feature, metadata, and audit interfaces.
- Replaced Pillow ring-by-ring polygon filling with
  `rasterio.features.rasterize`; PyShp's Polygon/MultiPolygon GeoJSON topology
  now supplies exterior/hole grouping before conversion to tile-pixel space.
- Replaced chunked all-pairs nearest-distance matrices with
  `scipy.spatial.cKDTree.query`, retaining chunking and matching-index
  self-exclusion behavior.
- Added PyShp, Rasterio, and SciPy to `requirements.txt`, documented dependency
  installation, and added a temporary-Shapefile regression test.
- Files: `src/green_roofs/{preprocessing,spatial_distance}.py`,
  `scripts/inspect_green_roofs.py`, `tests/test_preprocessing.py`,
  `requirements.txt`, `README.md`, and `DEV_LOG.md`.
- Validation: `compileall` passed; all 21 unit tests passed in
  `exercise/.venv` (PyShp 3.1.6, Rasterio 1.5.1, SciPy 1.18.0); the real-data
  audit passed; a temporary full rebuild and validation passed for all 3,510
  masks, 983 source features, and all spatial blocks.
- Behavior change: Rasterio's pixel-centre inclusion rule produces 184 positive
  tiles and 644,525 foreground pixels, versus 185 and 657,181 in the existing
  Pillow-generated dataset. Existing processed masks were not overwritten;
  regenerate them deliberately before the next training run, then repeat label
  QA and do not compare checkpoints trained against different mask versions.

## 2026-08-25 — Spatial train–prediction distance diagnostic

- Added a reproducible Meyer–Pebesma-style spatial-distance preprocessing step
  for the frozen 2016 tile split in Berlin postal code 12357.
- Defined sample-to-sample as each of 2,502 training tiles to its nearest other
  training tile, and sample-to-prediction as each of 1,008 validation/test tiles
  to its nearest training tile. Distances use tile centres projected to a local
  metre coordinate system.
- Generated a two-panel spatial-layout/density figure plus row-level CSV source
  data and a JSON summary under
  `../data/processed/green_roofs_2016/spatial_distance_analysis`.
- Results: median distance 46.55 m versus 93.11 m; P90 46.55 m versus
  186.22 m; maximum held-out distance 372.42 m.
- Files: `src/green_roofs/spatial_distance.py`,
  `scripts/analyze_spatial_distance.py`, `tests/test_spatial_distance.py`,
  `requirements.txt`, `README.md`, and generated analysis outputs.
- Validation: four spatial-distance unit tests passed; source preflight passed
  20/20 checks with no warnings; PDF text audit found a 5.25 pt minimum glyph
  size; PNG/PDF/SVG/TIFF outputs were visually inspected at final size.
- Risk/TODO: this evaluates the current split within postal code 12357, not
  city-wide Berlin deployment. Re-run against a prepared 2025/full-Berlin
  prediction grid before making city-scale generalization claims.

## 2026-08-18 — Green-roof dataset audit

- Added a dependency-free audit script for image-tile, Shapefile-header, DBF-schema,
  projection, and GeoJSON metadata inspection.
- Documented the raw-data layout and the proposed preprocessing milestone.
- Files: `scripts/inspect_green_roofs.py`, `README.md`, `DEV_LOG.md`.
- Validation: audited all 3,510 PNG headers; Pillow decoding independently
  confirmed that every tile is 256×256 RGBA and fully opaque. The four JP2 files
  open as 10,000×10,000 RGBA. `py_compile` passed. The first script run found a
  Windows GBK console incompatibility with a German filename; stdout is now
  explicitly configured as UTF-8 and the rerun passed.
- Risk/TODO: Shapefile polygons still need topology-aware rasterization; use a
  spatial split to prevent neighboring tiles from leaking across data splits.

## 2026-08-18 — Binary masks and spatial dataset split

- Implemented a dependency-light Polygon Shapefile reader, WGS 84 → XYZ pixel
  conversion, topology-aware binary rasterization, and mask union for the entire-
  roof and partial-roof layers.
- Generated 3,510 uint8 masks under `data/processed/green_roofs_2016`; 185 tiles
  contain positives, totaling 657,181 pixels (0.286%).
- Created deterministic 8×8-tile spatial train/validation/test blocks plus CSV
  manifests, split statistics, and an image/mask/overlay QA contact sheet.
- Added reusable validation for file references, 0/1 mask values, manifest pixel
  counts, unique tile membership, and cross-split block leakage.
- Files: `src/green_roofs/preprocessing.py`, `scripts/prepare_green_roofs.py`,
  `scripts/validate_green_roofs.py`, `tests/test_preprocessing.py`,
  `requirements.txt`, `README.md`, `DATA_AUDIT.md`.
- Commands: `python -m unittest discover -s tests -v` and the prepare/validate
  commands documented in `README.md`.
- QA/fix: visual overlays align with roof boundaries. A centimetre-scale source
  polygon exposed cancellation in the initial signed-area formula; translating
  coordinates to a local origin fixed the classification and the processed data
  was rebuilt and revalidated.
- Risk/TODO: foreground class imbalance remains severe. Inspect additional random
  and boundary QA samples before training, then add a 512×512 dataset loader and
  baseline model.

## 2026-08-24 — Manual label-review package

- Added a reproducible label-QA workflow that selects 50 positive tiles across
  the full foreground-coverage range and creates side-by-side original/green-mask
  overlay images for human inspection.
- Generated the review package at `../Pre-processing/label_quality_check`, with
  50 PNG pairs, an editable `review.csv`, selection metadata, and instructions.
- Selection seed: 42. The sample covers train/validation/test tiles (34/5/11)
  and foreground fractions from 0.00004578 to 0.41004944.
- Files: `src/green_roofs/label_qa.py`,
  `scripts/create_label_quality_review.py`, `tests/test_preprocessing.py`,
  `README.md`, and the generated review-package files.
- Validation: `python -m unittest discover -s tests -v` passed all 6 tests;
  low-, medium-, and high-coverage pair images were also opened and checked.
- Risk/TODO: the package has been generated, but the 50 human judgements are
  still pending. Do not treat the sample as evidence of label accuracy until
  `review.csv` has been completed and summarized.

## 2026-08-24 — Manifest-based U-Net baseline scaffold

- Added a manifest Dataset, separate training/evaluation transforms, ImageNet
  normalization, foreground Dice/IoU/precision/recall, reusable trainer, TOML
  configuration, and a training entry point.
- Adapted the teaching model under the explicit name
  `ExerciseResNet50UNet`; it is a ResNet50/AlbuNet-style variant rather than the
  original symmetric U-Net.
- Preserved all empty tiles and the frozen 8×8 spatial split. Baseline random
  augmentation is disabled, and evaluation transforms reject augmentation.
- Files: `configs/unet_baseline.toml`, `scripts/train.py`,
  `src/green_roofs/{dataset,transforms,metrics}.py`,
  `src/green_roofs/models/**`, `src/green_roofs/engine/**`,
  `tests/test_training_pipeline.py`, `requirements.txt`, and `README.md`.
- Validation: all 10 unit tests passed, including a model forward pass. A real
  train-manifest record loaded as float32 `[3,256,256]` with an int64
  `[256,256]` binary mask. CUDA is available in the selected exercise venv.
- Risk/TODO: no training was started. Complete the manual label audit, decide a
  justified imbalance strategy, and compare the teacher's standard U-Net with
  this exercise variant before the final baseline run.

## 2026-08-24 — Standalone class-balance analysis

- Added `scripts/analyze_class_balance.py` to independently read the frozen
  train/val/test manifests and all referenced binary masks.
- Exported per-split and overall tile-/pixel-level class statistics to
  `data/processed/green_roofs_2016/class_balance_analysis` as JSON, CSV, and
  Markdown.
- Validation: processed all 3,510 masks and rejected non-binary values. Results
  reproduce 185 positive versus 3,325 negative tiles and 657,181 foreground
  versus 229,374,179 background pixels.
- Risk/TODO: these descriptive ratios do not by themselves determine optimal
  loss weights; loss selection still requires validation experiments.
- Follow-up: standardized the generated Markdown report and JSON definitions to
  English and regenerated all outputs; a Unicode scan found no Chinese text.

## 2026-08-24 — Stage-1 loss-ablation pipeline prepared

- Standardized the teaching ResNet50/AlbuNet-style model to one binary logit
  channel and updated metrics to use sigmoid probabilities.
- Added BCE + Dice, focal BCE + Dice, and Tversky loss implementations plus
  three TOML experiment configurations. Their data, model, augmentation,
  sampler, optimizer, seed, batch size, and epoch settings are identical.
- Added an exact 50/50 positive-/negative-tile training batch sampler. Validation
  remains sequential and follows the natural class distribution.
- Enabled the same training-only horizontal/vertical flips and 90-degree
  rotations in all three configurations; evaluation augmentation remains
  prohibited.
- Added `--check-only` to validate a real forward/loss pass without downloading
  pretrained weights, backpropagating, saving checkpoints, or training.
- Files: `src/green_roofs/{dataset,losses,metrics,sampling}.py`,
  `src/green_roofs/models/exercise_resnet50_unet.py`, `scripts/train.py`,
  `configs/unet_*.toml`, `tests/test_training_pipeline.py`, `README.md`, and
  `../REPORT_NOTES.md`.
- Validation: all 12 tests passed. The configuration parity audit passed. A real
  CUDA check used a four-tile batch with exactly two positive and two negative
  tiles, produced `[4,1,256,256]` logits, and evaluated BCE + Dice successfully.
- Risk/TODO: formal training was intentionally not started. Complete and resolve
  the manual label audit first. Building priors and hard-negative mining depend
  on a selected Stage-1 checkpoint and remain deferred.

## 2026-08-24 — Baseline validation diagnosis and v2 preparation

- Analyzed the epoch-15 `best.pt` on all 536 validation tiles without reading
  the test split. Exported a 0.05–0.95 threshold sweep, per-tile errors, and 30
  five-panel TP/FP/FN diagnostics under
  `runs/unet_baseline/validation_analysis`.
- Best validation threshold was 0.05, but Dice improved only from 0.2806 to
  0.2819 (IoU 0.1641, precision 0.2242, recall 0.3795), confirming that threshold
  calibration is not the main limitation.
- Added optional frozen ResNet50 encoder BatchNorm, configurable positive draws
  per epoch, and validation-Dice early stopping. Created the non-overwriting
  `configs/unet_baseline_v2.toml` with 516 samples per epoch, two expected draws
  per positive tile, 40 maximum epochs, and patience six.
- Files: `scripts/analyze_validation.py`, `scripts/train.py`,
  `src/green_roofs/models/exercise_resnet50_unet.py`,
  `src/green_roofs/engine/trainer.py`, `configs/unet_baseline_v2.toml`,
  `tests/test_training_pipeline.py`, `README.md`, and `../REPORT_NOTES.md`.
- Validation: all 13 tests passed; the v2 CUDA check used exactly two positive
  and two negative tiles, produced `[4,1,256,256]` logits, froze encoder
  BatchNorm, and performed no parameter update. No v2 training was started.
- Risk/TODO: complete the label-review record, run v2 separately, and compare
  validation stability before proceeding to focal/Tversky ablations.

## 2026-08-24 — Undertrained v2 reviewed; v3 configured

- The v2 run stopped at epoch 16 after 2,064 optimizer steps. Its best epoch was
  10, with validation Dice 0.2305 and IoU 0.1302, below the v1 Dice of 0.2806.
- The comparison was compute-imbalanced: v1 used approximately 12,520 optimizer
  steps. The v2 result therefore does not isolate the effect of reduced sampling
  or frozen encoder BatchNorm.
- Added `configs/unet_baseline_v3.toml`. It preserves v2 data, sampling, loss,
  augmentation, and BatchNorm settings while increasing the maximum to 50
  epochs and early-stopping patience to 15, for at most 6,450 optimizer steps.
- Risk/TODO: v3 formal training has not been started. Compare v3 with v1 before
  creating focal/Tversky configurations based on the revised schedule.

## 2026-08-24 — Controlled 30/70 sampler ablation prepared

- After v3 peaked at validation Dice 0.2120 with precision 0.1462 and recall
  0.3856, returned to all v1 settings and created
  `configs/unet_sampler_30_70.toml`.
- The only experimental change is the positive-tile fraction from 0.50 to 0.30;
  the output directory is separate. Model, unfrozen BatchNorm, 20 epochs, full
  epoch size, augmentation, BCE + Dice, optimizer, batch size, and seed match v1.
- Updated the sampler to support non-integer per-batch ratios by distributing
  one- and two-positive batches deterministically. The effective 2,504-sample
  epoch contains 751 planned positive and 1,753 planned negative draws.
- Validation: configuration parity passed; all 14 tests passed; CUDA check-only
  produced a valid `[4,1,256,256]` output without training.
- Risk/TODO: formal 30/70 training has not started. Compare its validation
  precision and Dice directly with v1 before adding photometric augmentation.

## 2026-08-24 — 30/70 sampler result validated

- The controlled run peaked at epoch 18 with validation Dice 0.3255, IoU 0.1944,
  precision 0.2426, and recall 0.4941, improving all four metrics over v1.
- Ran validation-only threshold analysis on `best.pt`. Threshold 0.05 produced
  Dice 0.3311, IoU 0.1984, precision 0.2320, and recall 0.5777. The small Dice
  gain shows that sampling, rather than calibration alone, drove the improvement.
- Exported threshold metrics, per-tile errors, and 30 diagnostic images under
  `runs/unet_sampler_30_70/validation_analysis`. The test split was not read.
- Visual QA still found confident predictions on ordinary dark roofs and total
  misses on some visually ambiguous annotated roofs.
- Risk/TODO: the 30/70 result is promising but based on one seed. Complete the
  label-review record and test one controlled photometric-augmentation change
  before spatial cross-validation or hard-negative mining.

## 2026-08-25 — Reproducible QGIS reannotation preparation added

- Added pure XYZ/Web-Mercator mosaic calculations and VRT generation for the
  complete rectangular 54×65 source grid.
- Added a CLI that creates a unified editable label GeoPackage and a tile-index
  layer carrying frozen split, block, and positive-mask metadata.
- Added a PyQGIS builder for a relative-path QGZ project with clear editable and
  reference layers; source Shapefiles and image tiles remain unchanged.
- Validation: all 8 targeted preprocessing tests passed. Full discovery reached
  the training test module but the default `D:\python` environment lacks Torch;
  this unrelated import limitation did not affect the targeted geospatial tests.
- Files: `src/green_roofs/qgis_annotation.py`,
  `scripts/prepare_qgis_annotation.py`,
  `scripts/build_qgis_annotation_project.py`, `tests/test_preprocessing.py`,
  and `README.md`.

## 2026-08-26 — PLZ boundary acquisition made reproducible

- Added official Berlin postcode-WFS acquisition to fresh QGIS workspace
  generation. It filters `plz='12357'`, converts the one MultiPolygon to
  EPSG:3857, and stores it separately from editable green-roof labels.
- Updated the PyQGIS project builder with a transparent cyan boundary style.
- Validation: the downloaded geometry is valid and the regenerated QGZ
  references it by relative path.
- Risk: the live WFS is current and may change independently of the 2016 image
  date; preserve the acquisition date and temporal caveat.

## 2026-08-27 — Separate source-label references added to QGIS project

- Extended the PyQGIS project builder to load the 55-feature entire-roof and
  928-feature partial-roof Shapefiles independently from the editable merged
  GeoPackage.
- Marked both source layers read-only, assigned distinct yellow/magenta styles,
  and disabled them by default.
- Validation: QGZ inspection confirmed both relative data sources and
  `readOnly=1` settings.

## 2026-08-26 — Nature-style pitch deck created and verified

- Created `presentation/green_roof_pitch.pptx` from the approved pitch plan:
  12 main slides plus six Q&A evidence slides and an appendix divider.
- Added editable native charts, GIS workflow graphics, local validation
  diagnostics, spatial-distance evidence, and speaker notes with per-slide
  source blocks.
- Added `presentation/TERMINOLOGY_LEDGER.md`, `asset_manifest.md`,
  `pptx_audit.md`, and `qa_report.md` for terminology, evidence provenance, and
  delivery QA.
- Generated the missing PLZ 12357 spatial-distance analysis using the existing
  script; median held-out-to-training distance is 93 m versus 47 m within
  training.
- Validation: `slides_test.py` passed with no overflow; the Nature PPTX audit
  found high=0 and medium=1 (intentional cover-image crop); Microsoft PowerPoint
  rendering was used to correct title-reflow issues.
- Risk/TODO: all model findings remain validation-only. Freeze the selected
  threshold before reading the untouched test set; full-Berlin and cross-year
  generalization remain untested.

## 2026-08-27 — Spatial-split method insert deck created

- Created `presentation/spatial_split_method_slides.pptx`, a four-slide,
  insert-ready explanation of the current 8 × 8 block split algorithm.
- Clarified that 0.35/0.65 are deficit-score weights rather than dataset split
  ratios, and separated the hard no-leakage constraint from approximate balance.
- Included the observed train/validation/test tile, positive-tile, foreground-
  pixel, and block distributions from the PLZ 12357 preprocessing outputs.
- Added speaker notes with per-slide source blocks and a dedicated QA record at
  `presentation/spatial_split_method_slides.qa.md`.
- Validation: `slides_test.py` passed with no overflow detected; final text
  inspection confirmed all four source-note blocks and the required statistics.
- Risk/TODO: positive-tile balance is still indirect because the current greedy
  score optimizes tile count and foreground-pixel mass only; integer programming
  remains a possible future refinement.

## 2026-09-18 — Ponytail full-mode code audit and simplification

- Reused canonical manifest/tile helpers instead of indirect re-exports and
  duplicate Web-Mercator bounds calculations; removed one unused import and
  hard-coded tile-size literals from VRT generation.
- Simplified the binary prediction expression, resolved a mypy ambiguity in
  spatial split selection, and made invalid `samples_per_epoch <= 0` fail fast.
- Added a sampler boundary test.
- Validation: `compileall` passed; all 23 `unittest` tests passed in the existing
  `exercise/.venv` uv environment; targeted mypy check passed.
- Risk/TODO: `project/code` has no local `.venv`, `pyproject.toml`, or `uv.lock`;
  tests currently rely on the compatible environment at `exercise/.venv`.

## 2026-09-18 — Project-local uv environment

- Added `pyproject.toml` from the existing runtime requirements and retained the
  CUDA 12.6 PyTorch index already used by the workspace.
- Generated `uv.lock`, created the ignored `.venv`, and updated README setup.
- Validation: Python 3.12.14, PyTorch 2.14.0+cu126, torchvision 0.29.0+cu126,
  CUDA available; `compileall` and all 23 `unittest` tests passed.
- Risk/TODO: Rasterio tests emit upstream Affine `PendingDeprecationWarning`s;
  they do not affect current behavior.

## 2026-09-18 — SegFormer-B0 comparison pipeline

- Added a full-resolution SegFormer-B0 adapter using the ImageNet-pretrained
  `nvidia/mit-b0` encoder and a shared model factory used by training and
  validation analysis.
- Added `configs/segformer_b0.toml` with the same data split, sampling, loss,
  optimizer, and early-stopping settings as the U-Net v3 comparison baseline.
- Added `transformers`, documented the experiment, and added a model shape test.
- Validation: all 24 tests passed; the real-data CUDA check produced logits of
  shape `(4, 1, 256, 256)` and a finite loss; pretrained weights loaded and ran
  a CUDA forward pass successfully.
- Risk/TODO: the pretrained checkpoint contains only the MiT-B0 encoder, so the
  segmentation decode head starts randomly initialized as intended.
- Completed the formal run: early stopping selected epoch 5 and stopped at
  epoch 20. Validation threshold 0.45 produced Dice 0.2966 and IoU 0.1741.
- Ran the same threshold analysis for U-Net v3: threshold 0.55, Dice 0.2126,
  IoU 0.1189. Added `reports/model_comparison.md`; test data remains untouched.

## 2026-09-18 — Removed duplicate dependency manifest

- Deleted `requirements.txt`; `pyproject.toml` is now the only dependency
  declaration and `uv.lock` is the reproducible lock file.
- Historical `requirements.txt` references in this log were preserved because
  they accurately describe earlier development steps.
- Validation: `uv lock --check` and all 24 `unittest` tests passed.

## 2026-09-18 — Frozen-threshold final test evaluation

- Extended the existing checkpoint analyzer with `--split test` and required
  fixed `--threshold` input, preventing test-set threshold tuning.
- Negative-tile ablation: all-tile U-Net test Dice 0.3839 versus positive-only
  0.2632; affected negative tiles 16/222 versus 85/222; false-positive pixels
  32,526 versus 58,039.
- Model comparison: U-Net v3 and SegFormer-B0 test Dice were effectively tied
  at 0.2632 and 0.2639. SegFormer raised recall but also false positives.
- Added `reports/negative_tile_ablation.md` and updated
  `reports/model_comparison.md`. All 24 tests passed before evaluation.
- Risk/TODO: these are single-seed results; repeated-seed uncertainty remains
  necessary for a strong architecture-level claim.

## 2026-09-20 — Aligned 2025 cross-temporal predictions

- Added `scripts/predict_2025.py` to align four-band 2025 JP2 imagery to the
  frozen 2016 PLZ 12357 XYZ grid, reuse RGB normalization and checkpoints, and
  export binary masks plus per-tile manifests without treating 2016 labels as
  2025 truth.
- Generated 1,635 aligned RGB tiles and 1,635 masks each for the all-tile U-Net
  (threshold 0.05), U-Net v3 (0.55), and SegFormer-B0 (0.45).
- Validation: all 24 unit tests passed; real JP2/CUDA smoke test passed; full
  output audit confirmed 1,635 unique 256×256 binary masks per model and exact
  manifest pixel counts.
- Risk/TODO: accuracy is unknown until independent 2025 ground-truth masks are
  created; the predictions show visible domain-shift candidates for review.

## 2026-09-20 — Four-panel 2025 manual-review set

- Added `scripts/create_2025_comparison_grids.py` and generated 50 review PNGs
  with 2025 RGB plus magenta overlays from the main U-Net, U-Net v3, and
  SegFormer-B0.
- Stratified selection: 9 triple-overlap, 20 pair-only-overlap, 17 single-model,
  and 4 no-prediction controls; representative intensity levels use the
  10th–90th percentile range rather than only extreme predictions.
- Validation: confirmed 50 unique 512×596 PNGs, expected category counts, and
  a complete CSV index; all 24 unit tests passed.
- Risk/TODO: this is a deliberately balanced QA sample, not a prevalence or
  accuracy sample; independent 2025 annotation is still required.

## 2026-09-20 — Git baseline and readability refactor

- Created Git baseline commit `3ac8885` after excluding the local uv
  environment, PowerPoint build cache, and QGIS style database.
- Split training sampler construction from `scripts/train.py`; split threshold
  accumulation, per-tile statistics, and diagnostics from
  `scripts/analyze_validation.py` without changing experiment behavior.
- Removed all manual `sys.path` injection from scripts and tests, reused the
  shared mask-overlay renderer, and added `configs/README.md` to distinguish
  final, historical, and exploratory configurations without deleting any.
- Validation: `uv lock --check`, compilation, and all 24 tests passed; both
  balanced and positive-only training checks passed on CUDA; refactored
  validation metrics exactly matched the historical JSON; all 50 regenerated
  comparison PNGs were byte-identical.
- Risk/TODO: CUDA seeds do not guarantee bitwise determinism across every
  platform; completed checkpoints and frozen reports remain the source of
  truth for the reported experiments.
