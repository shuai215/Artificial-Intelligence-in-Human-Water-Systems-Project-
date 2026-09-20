# U-Net v3 vs. SegFormer-B0 comparison

Both models use the same frozen spatial manifests, 256×256 inputs, training
augmentation, 50/50 tile sampler, BCE + Dice loss, AdamW settings, and
validation metrics. Thresholds were selected independently on validation and
then frozen before the single final test evaluation.

## Validation

| Model | Parameters | Best epoch | Stop epoch | Threshold | Dice | IoU | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ResNet50 U-Net v3 | 39,390,281 | 15 | 30 | 0.55 | 0.2126 | 0.1189 | 0.1573 | 0.3278 |
| SegFormer-B0 | 3,714,401 | 5 | 20 | 0.45 | 0.2966 | 0.1741 | 0.2172 | 0.4676 |

SegFormer-B0 improves validation Dice by 0.0840 absolute (39.5% relative) and
uses about 10.6× fewer parameters. On the 26 positive validation tiles, its
foreground Dice is 0.5967 versus 0.3986 for U-Net v3. It also spreads false
positives across more negative tiles (95 versus 31), although the total
false-positive pixel count is only moderately higher (159,432 versus 144,679).

## Test

| Model | Frozen threshold | Dice | IoU | Precision | Recall | FP negative tiles | FP pixels |
|---|---:|---:|---:|---:|---:|---:|---:|
| ResNet50 U-Net v3 | 0.55 | 0.2632 | 0.1516 | **0.4511** | 0.1859 | **14/442** | **16,739** |
| SegFormer-B0 | 0.45 | **0.2639** | **0.1520** | 0.3142 | **0.2275** | 33/442 | 28,653 |

The validation advantage did not transfer to the held-out spatial test blocks:
test Dice differs by only 0.0006 (0.24% relative). SegFormer has higher recall,
but U-Net has substantially higher precision and fewer false positives. With
one seed, the defensible conclusion is that their test overlap is effectively
tied, not that either architecture wins.

Sources:

- `../../runs/unet_baseline_v3/validation_analysis/validation_summary.json`
- `../../runs/segformer_b0/validation_analysis/validation_summary.json`
- `../../runs/unet_baseline_v3/test_analysis/test_summary.json`
- `../../runs/segformer_b0/test_analysis/test_summary.json`
