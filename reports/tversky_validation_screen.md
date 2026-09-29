# Validation screening selects BCE-Dice over pure Tversky

## Selection objective

Select the loss formulation that produces usable foreground segmentation on
the PLZ 12357 validation split before allocating compute to a second training
arm or evaluating the held-out test set.

## Controlled setup

- Reference: `configs/unet_plz12357_all.toml`
- Candidate: `configs/unet_plz12357_all_tversky.toml`
- Only intended change: `bce_dice` to `tversky` with alpha 0.3, beta 0.7,
  and smooth 1.0
- Fixed controls: manifests, ResNet50 U-Net, pretrained encoder, 50:50 tile
  sampler, augmentation, seed 42, batch size 4, AdamW learning rate 1e-4, and
  20 epochs
- Selection data: validation split only; test was not evaluated

## Validation result

| Loss | Selected epoch | Threshold | Dice | Precision | Recall | Predicted foreground pixels | Negative tiles with FP |
|---|---:|---:|---:|---:|---:|---:|---:|
| BCE-Dice | 10 | 0.05 | 0.4754 | 0.4981 | 0.4547 | 95,973 | 10/181 |
| Tversky | 1 | 0.05 | 0.0000 | 0.0000 | 0.0000 | 0 | 0/181 |

The validation screen clearly separates the objectives. BCE-Dice achieved
0.4754 Dice with balanced precision and recall, whereas the tested pure
Tversky formulation produced an all-background solution at every threshold in
the sweep and missed 105,125 reference foreground pixels.

## Decision

BCE-Dice is selected for the formal experiment because it supplies usable
foreground recovery and probability calibration under the matched protocol.
The validation gate stops the pure Tversky branch before test exposure,
preserving the test set for finalized model choices. The observed behavior is
consistent with a pure per-sample overlap objective on empty negative masks:
smoothing can reward an all-background solution without a BCE-style pixel
calibration term. This conclusion applies to the tested pure formulation with
alpha 0.3 and beta 0.7.
