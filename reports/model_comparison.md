# ResNet50 U-Net leads the matched architecture comparison

Under spatially grouped evaluation, ResNet50 U-Net delivered the strongest
green-roof segmentation result. The comparison isolates architecture by using
the same PLZ 12357 manifests, 256 x 256 inputs, augmentation, 50:50
positive/negative sampler, BCE-Dice loss, AdamW settings, seed 42, 1,180
sampled tiles per epoch, and 20-epoch limit. Each model selected its checkpoint
and threshold on validation before frozen test evaluation.

| Model | Parameters | Checkpoint epoch | Threshold | Test Dice | IoU | Precision | Recall | FP negative tiles | FP pixels |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ResNet50 U-Net | 39.39M | 10 | 0.05 | **0.3839** | **0.2376** | **0.4269** | **0.3488** | **16/222** | **32,526** |
| SegFormer-B0 | 3.71M | 20 | 0.35 | 0.2268 | 0.1279 | 0.1976 | 0.2661 | 27/222 | 86,493 |

U-Net increased test Dice by 0.1572 absolute and led on IoU, precision, recall,
and both negative-scene false-positive measures. It reduced false-positive
pixels from 86,493 to 32,526 and affected 11 fewer negative tiles. SegFormer-B0
offers a compact 3.71M-parameter alternative, while the 39.39M-parameter U-Net
provides the stronger accuracy and background rejection required by this
mapping task. The evidence supports U-Net as the primary architecture for the
matched PLZ 12357 protocol.

Sources:

- `../../runs/unet_plz12357_all/validation_analysis/validation_summary.json`
- `../../runs/unet_plz12357_all/test_analysis/test_summary.json`
- `../../runs/segformer_b0_plz12357/validation_analysis/validation_summary.json`
- `../../runs/segformer_b0_plz12357/test_analysis/test_summary.json`
