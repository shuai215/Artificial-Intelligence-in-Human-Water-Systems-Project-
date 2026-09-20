# Pitch presentation plan

## Working title

**Spatially Reliable Green-Roof Segmentation from Aerial Imagery**

Subtitle: **A GIS-Aware U-Net Study in Berlin PLZ 12357**

## Assignment contract

The main presentation is designed for a one-person project and should take
approximately 12 minutes. A separate appendix supports up to 10 minutes of
questions.

The deck must cover:

- a brief introduction and definitions;
- the research goal and research questions;
- a state-of-the-art overview and workflow/model choice;
- the proposed preprocessing and model method;
- findings from data exploration, preprocessing, and first model runs;
- expected or assumed results that have not yet been tested;
- challenges and limitations.

## Communication job

By the end, the course audience should understand that credible green-roof
segmentation depends not only on model architecture, but also on GIS-aligned
labels, spatially independent evaluation, and representative negative training
scenes.

## Central takeaway

> For area-wide green-roof mapping, learning where green roofs are absent is as
> important as learning their appearance.

## Research goal

Develop and evaluate a reproducible semantic-segmentation workflow for mapping
green roofs from RGB aerial imagery in Berlin postal-code area 12357.

## Research questions

1. **RQ1:** How effectively can a ResNet50 U-Net segment green roofs from RGB
   aerial imagery under spatially grouped evaluation?
2. **RQ2:** Does removing negative-only training tiles improve green-roof
   segmentation, or does it increase false-positive predictions?

## Research contribution

- A GIS-aware polygon-to-mask preprocessing workflow.
- A postcode-constrained, leakage-resistant spatial train/validation/test split.
- A controlled all-tile versus positive-only training ablation using identical
  validation data and optimizer-step budgets.
- Positive-tile and negative-tile subgroup analysis in addition to aggregate
  segmentation metrics.

## Narrative arc

```text
Urban mapping problem
    ↓
Why a standard random segmentation experiment is not sufficient
    ↓
GIS preprocessing and spatially independent evaluation
    ↓
Severe class imbalance motivates a controlled training ablation
    ↓
First results show that negative training scenes improve reliability
    ↓
What is supported now, what remains uncertain, and what comes next
```

Do not use a separate agenda slide. Each slide should answer a question raised
by the previous slide.

## Main deck: 12 slides / approximately 11:50

| Slide | Takeaway-style title | Main purpose | Target time |
|---:|---|---|---:|
| 1 | Spatially Reliable Green-Roof Segmentation | Minimal title and study context | 0:20 |
| 2 | Green-roof mapping is a rare-object segmentation task | Define the topic and task | 0:50 |
| 3 | The main gap is credible evaluation, not a lack of models | Introduce imbalance, spatial leakage, and label alignment | 1:00 |
| 4 | We test whether negative scenes are necessary | State the goal, RQs, and contribution | 1:00 |
| 5 | A ResNet50 U-Net provides a transparent baseline | Summarize model families and justify the baseline | 0:55 |
| 6 | GIS preprocessing turns roof polygons into aligned masks | Explain polygon-to-mask conversion | 1:20 |
| 7 | Spatial blocks keep neighboring tiles in the same split | Explain postcode filtering and spatial splitting | 1:10 |
| 8 | Only 0.60% of pixels represent green roofs | Present data and imbalance findings | 1:05 |
| 9 | One frozen split isolates the effect of negative training scenes | Present the controlled experiment | 1:05 |
| 10 | All-tile training performs better after threshold calibration | Present aggregate validation results | 1:20 |
| 11 | Positive-only training spreads false positives across more scenes | Present subgroup and diagnostic findings | 1:10 |
| 12 | Current evidence favors negative-aware training, but test remains untouched | Close with limitations, hypotheses, and next steps | 0:35 |

## Slide-by-slide storyboard

### Slide 1 — Spatially Reliable Green-Roof Segmentation

**Narrative job:** Establish the task and study area without technical detail.

**Visible content:**

- Main title.
- Subtitle: `A GIS-Aware U-Net Study in Berlin PLZ 12357`.
- Presenter name, course, and date.

**Visual:** One real aerial tile with a subtle green-roof mask overlay. Keep the
title slide minimal.

### Slide 2 — Green-roof mapping is a rare-object segmentation task

**Narrative job:** Satisfy the brief introduction requirement.

**Visible content:**

- Green roof: a roof surface partially or fully covered by vegetation.
- Semantic segmentation: assign every image pixel to background or green roof.
- Desired output: a spatially explicit mask, not only a tile-level label.

**Visual:** `Original aerial image → binary green-roof mask` using one real
example.

### Slide 3 — The main gap is credible evaluation, not a lack of models

**Narrative job:** Establish the research gap.

**Core issues:**

1. Green-roof pixels are extremely rare.
2. Neighboring aerial tiles are spatially correlated.
3. Source labels are geographic polygons rather than aligned pixel masks.

**Audience-facing claim:**

> Segmentation performance can look overly optimistic when label conversion,
> spatial dependence, and negative scenes are not handled explicitly.

**Sources needed before final deck:** Primary literature on remote-sensing
segmentation, spatial leakage/autocorrelation, and class imbalance.

### Slide 4 — We test whether negative scenes are necessary

**Narrative job:** State the research goal, questions, and contribution.

**Visible content:**

- RQ1 and RQ2 in shortened form.
- Three contributions: GIS masks, spatial split, controlled negative-scene
  ablation.

**Visual:** A simple question-to-method composition; avoid dense text boxes.

### Slide 5 — A ResNet50 U-Net provides a transparent baseline

**Narrative job:** Cover state of the art and model choice.

**State-of-the-art overview:**

- U-Net and encoder-decoder CNNs.
- DeepLabv3+ and multi-scale context.
- Transformer-based segmentation such as SegFormer.

**Why the current baseline:**

- ImageNet-pretrained ResNet50 encoder.
- Skip connections help retain small-roof boundaries.
- Suitable for a limited labeled dataset.
- Interpretable, reproducible starting point before more complex models.

Do not claim that U-Net is the current absolute state of the art. Describe it as
a transparent and reproducible baseline.

### Slide 6 — GIS preprocessing turns roof polygons into aligned masks

**Narrative job:** Explain the core preprocessing method.

```text
XYZ tile z/x/y
    ↓
Mercantile EPSG:3857 bounds
    ↓
Shapely tile polygon and spatial-index intersection
    ↓
Rasterio affine transform
    ↓
256×256 binary mask
```

**Speaker emphasis:**

- The original tile is not reprojected or resampled.
- XYZ indices define the tile's EPSG:3857 extent.
- Green-roof polygons are projected to EPSG:3857.
- Rasterio maps geographic coordinates directly to the 256×256 mask.
- `all_touched=False` uses pixel-centre inclusion.

**Visual:** One real three-stage example: original image, overlay, binary mask.

### Slide 7 — Spatial blocks keep neighboring tiles in the same split

**Narrative job:** Explain study-area filtering and leakage prevention.

```text
3,510 rectangular-grid tiles
    ↓ tile centre covered by PLZ polygon
1,635 study-area tiles
    ↓ 8×8 spatial blocks
1,180 train / 203 validation / 252 test
```

**Split objective:**

- 35% weight on relative tile-count deficit.
- 65% weight on relative foreground-pixel deficit.
- Entire blocks remain in exactly one split.

**Results:** 28/5/5 train/validation/test blocks and no block leakage.

**Limitation footer:** The postcode polygon is the current official boundary,
not a verified historical 2016 boundary.

### Slide 8 — Only 0.60% of pixels represent green roofs

**Narrative job:** Turn data exploration into the experimental motivation.

**Key numbers:**

- 1,635 selected tiles.
- 182 positive and 1,453 negative tiles.
- Positive-tile fraction: 11.13%.
- Foreground pixels: 643,038.
- Total pixels: 107,151,360.
- Foreground-pixel fraction: 0.6001%.

**Main conclusion:**

> Pixel-level imbalance is much more severe than tile-level imbalance.

**Visual:** Two strong percentage comparisons rather than a dense inventory
table.

### Slide 9 — One frozen split isolates the effect of negative training scenes

**Narrative job:** Demonstrate that the ablation is controlled.

**Shared settings:**

- Same ResNet50 U-Net.
- BCE + Dice loss.
- AdamW, learning rate `1e-4`.
- Batch size 4, 20 epochs.
- 1,180 samples per epoch.
- Same augmentation and full validation split.

**Only experimental change:**

| All-tile training | Positive-only training |
|---|---|
| 590 positive draws | 1,180 positive draws |
| 590 negative draws | 0 negative draws |

**Augmentation:** synchronized horizontal flip, vertical flip, and random
0°/90°/180°/270° rotation. Validation and test have no random augmentation.

### Slide 10 — All-tile training performs better after threshold calibration

**Narrative job:** Present the primary first model result.

| Training | Selected threshold | Dice | IoU | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| All tiles | 0.05 | **0.4754** | **0.3119** | **0.4981** | **0.4547** |
| Positive-only | 0.95 | 0.4221 | 0.2675 | 0.4688 | 0.3839 |

**Interpretation:**

- All-tile training is better on every reported aggregate metric.
- Opposite selected thresholds reveal strongly different probability
  calibration.
- Thresholds were selected using the same 203-tile validation set.
- These are validation findings, not final test results.

**Visual:** A compact grouped comparison plus a strong threshold annotation.

### Slide 11 — Positive-only training spreads false positives across more scenes

**Narrative job:** Show why the aggregate difference matters operationally.

**Positive validation subset:**

| Training | Dice | IoU | Precision | Recall |
|---|---:|---:|---:|---:|
| All tiles | **0.5653** | **0.3940** | **0.7468** | **0.4547** |
| Positive-only | 0.5001 | 0.3335 | 0.7174 | 0.3839 |

**Negative validation subset:**

| Training | Negative tiles with false positives | False-positive pixels |
|---|---:|---:|
| All tiles | **10 / 181** | 31,962 |
| Positive-only | 65 / 181 | **29,839** |

**Interpretation:** Positive-only training produces slightly fewer total
false-positive pixels but spreads smaller errors across many more negative
scenes. It also fails to outperform all-tile training on positive tiles.

**Visual:** One representative false-positive diagnostic from each model.

### Slide 12 — Current evidence favors negative-aware training, but test remains untouched

**Narrative job:** Resolve the research question while respecting evidence
limits.

**Supported now:**

- The GIS workflow produces aligned binary masks.
- Spatial blocks prevent direct neighboring-tile leakage.
- Negative training scenes improve validation reliability.
- Positive-only training causes calibration and false-positive problems.

**Challenges and limitations:**

- Only 182 positive tiles and 22 positive validation tiles.
- Potential polygon misalignment, missing labels, and boundary uncertainty.
- One small Berlin postcode area and RGB-only 2016 imagery.
- Validation is used for checkpoint and threshold selection.
- 2025 temporal transfer has not been tested.

**Assumed result / next hypothesis:** The all-tile model is expected to remain
better on the untouched test set, but this must not be stated as a result until
the test is evaluated once after model and threshold choices are frozen.

**Closing line:**

> For area-wide green-roof mapping, learning where green roofs are absent is as
> important as learning their appearance.

## Assignment-requirement mapping

| Required topic | Covered by |
|---|---|
| Brief topic introduction | Slide 2 |
| Research goal and questions | Slides 3–4 |
| State-of-the-art overview | Slide 5 |
| Why this model/workflow | Slides 5 and 9 |
| Proposed preprocessing | Slides 6–7 |
| Suitable model(s) | Slide 5 |
| Data-exploration and preprocessing findings | Slide 8 |
| First model results | Slides 10–11 |
| Assumed results | Slide 12 |
| Challenges and limitations | Slide 12 |

## Q&A appendix

Appendix slides are not included in the 12-minute main timing.

1. Why EPSG:3857 is used for XYZ alignment and EPSG:25833 for distance.
2. How `from_bounds()` maps geographic coordinates to 256×256 pixels.
3. Why tile-centre postcode selection was chosen over intersection or full
   containment.
4. How the 8×8 spatial block and 35/65 allocation score work.
5. Exact augmentation and sampling behavior.
6. Training curves and validation threshold curves.
7. Additional best-positive, false-negative-heavy, and false-positive-heavy
   examples.

## Existing evidence and local assets

- Dataset summary:
  `../../data/processed/green_roofs_2016_plz12357/dataset_summary.json`
- Class-balance report:
  `../../data/processed/green_roofs_2016_plz12357/class_balance_analysis/`
- QA contact sheet:
  `../../data/processed/green_roofs_2016_plz12357/qa/top_positive_tiles.png`
- All-tile history:
  `../../runs/unet_plz12357_all/history.csv`
- Positive-only history:
  `../../runs/unet_plz12357_positive_only/history.csv`
- All-tile validation analysis:
  `../../runs/unet_plz12357_all/validation_analysis/`
- Positive-only validation analysis:
  `../../runs/unet_plz12357_positive_only/validation_analysis/`
- Postcode boundary:
  `../../annotation/green_roofs_2016/berlin_plz_12357.gpkg`

## Assets still needed before building the PPTX

- Generate spatial-distance analysis for the new PLZ 12357 split.
- Export a clean postcode-boundary and selected-tile map from QGIS.
- Select one clear image/overlay/mask example from the QA outputs.
- Select representative diagnostics from both validation-analysis folders.
- Build training-history and threshold-comparison charts from the CSV outputs.
- Search and verify primary literature for green-roof mapping, U-Net/DeepLab/
  SegFormer, spatial leakage, and remote-sensing class imbalance.
- Record every external claim and asset in slide speaker-note source blocks.

## Visual and writing rules

- Main slide language: English.
- Speaker notes may contain an English script plus concise Chinese reminders.
- Use one main claim and one primary visual per slide.
- Prefer takeaway titles over topic labels such as `Method` or `Results`.
- Avoid code screenshots and long module/file lists in the main deck.
- Keep technical implementation details in the appendix.
- Do not mix fixed-threshold and threshold-calibrated results without labeling
  them explicitly.
- Never present validation results as test performance.
- Suggested minimum typography without a supplied template: 50 pt title,
  35 pt slide titles, 24 pt subheads, and 16 pt body text.

## Final production checklist

1. Verify all local statistics against their JSON/CSV source files.
2. Verify state-of-the-art claims using primary literature.
3. Select a presentation template or explicit visual direction.
4. Build the 12 main slides and appendix.
5. Add speaker notes and source blocks.
6. Render every slide and inspect it at full size.
7. Correct clipping, overlap, unreadable labels, and unexpected wrapping.
8. Rehearse to approximately 11:30–12:00.
9. Freeze the selected model and threshold before any final test evaluation.
