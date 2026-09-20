# Presentation asset manifest

| Asset | Source | Intended slide | Preparation | QA |
|---|---|---:|---|---|
| raw tile `19/281806/172161.png` | 2016 orthophoto source, selected by the PLZ-filtered `manifests/train.csv` | 1, 2, 6 | Embedded without editing | Pass: tile is present in the filtered manifest; full 256 x 256 image preserved |
| mask `19/281806/172161.png` | Rasterized label output | 6 | Embedded without editing | Pass: complete binary mask preserved |
| `spatial_distance_distribution.png` | Existing spatial-distance analysis script output | 7 | Embedded as full-width evidence | Pass: axes, legends, panel labels and caption preserved |
| `fp_heavy_negative_01_19_281814_172170.png` | All-tile validation diagnostics | 11 | Embedded in full with contain fit | Pass: all five panels and threshold label preserved |
| `fp_heavy_negative_01_19_281773_172128.png` | Positive-only validation diagnostics | 11 | Embedded in full with contain fit | Pass: all five panels and threshold label preserved |

No decorative stock imagery is used. All visual evidence comes from the project outputs.
