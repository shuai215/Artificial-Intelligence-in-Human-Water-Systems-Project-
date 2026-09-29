# Negative scenes improve overlap and false-positive control

Retaining negative-only training tiles produced the strongest U-Net result and
substantially reduced false-positive predictions. The controlled comparison
holds the PLZ 12357 spatial split, model, augmentation, optimizer, loss, batch
size, seed, and sampled tiles per epoch fixed; the training-scene policy is the
only changed factor. Thresholds were selected on validation and frozen before
test evaluation.

| Training data | Frozen threshold | Test Dice | Positive-tile Dice | FP negative tiles | FP pixels |
|---|---:|---:|---:|---:|---:|
| Positive and negative tiles | 0.05 | **0.3839** | **0.4697** | **16/222** | **32,526** |
| Positive tiles only | 0.95 | 0.2632 | 0.3787 | 85/222 | 58,039 |

Training with positive and negative scenes increased test Dice by 0.1207
absolute (45.9% relative to positive-only training), reduced affected negative
test tiles from 85 to 16, and reduced false-positive pixels by 44.0%. Negative
scenes therefore provide essential background supervision: they improve
foreground overlap while preventing green-roof predictions from spreading into
background-only imagery.

Sources:

- `../../runs/unet_plz12357_all/test_analysis/test_summary.json`
- `../../runs/unet_plz12357_positive_only/test_analysis/test_summary.json`
