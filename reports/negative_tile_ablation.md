# Negative-only training tile ablation

Both U-Net runs use the same PLZ 12357 spatial split, model, augmentation,
optimizer, loss, batch size, seed, and number of sampled tiles per epoch. The
only experimental change is whether negative-only training tiles are retained.
Thresholds were selected on validation and frozen before test evaluation.

| Training data | Frozen threshold | Test Dice | Positive-tile Dice | FP negative tiles | FP pixels |
|---|---:|---:|---:|---:|---:|
| Positive and negative tiles | 0.05 | **0.3839** | **0.4697** | **16/222** | **32,526** |
| Positive tiles only | 0.95 | 0.2632 | 0.3787 | 85/222 | 58,039 |

Removing negative-only training tiles reduced test Dice by 0.1207 absolute
(31.4% relative), affected over five times as many negative test tiles with
false positives, and increased false-positive pixels by 78.4%. The experiment
therefore supports retaining negative-only training scenes.

Sources:

- `../../runs/unet_plz12357_all/test_analysis/test_summary.json`
- `../../runs/unet_plz12357_positive_only/test_analysis/test_summary.json`
