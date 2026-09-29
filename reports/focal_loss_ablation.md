# BCE-Dice and negative-scene exposure form the strongest training strategy

The controlled 2 x 2 experiment identifies the combination that best balances
overlap and false-positive control: BCE-Dice with both positive and negative
training scenes. It crosses scene policy with loss while holding the PLZ 12357
spatial manifests, ResNet50 U-Net, augmentation, batch size, seed, optimizer,
learning rate, epoch count, and 1,180 sampled tiles per epoch fixed. Each
checkpoint and threshold was selected on validation before frozen test
evaluation.

| Loss | Training tiles | Threshold | Dice | IoU | Precision | Recall | FP negative tiles | FP pixels |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| BCE-Dice | Positive and negative | 0.05 | **0.3839** | **0.2376** | **0.4269** | 0.3488 | **16/222** | **32,526** |
| BCE-Dice | Positive only | 0.95 | 0.2632 | 0.1515 | 0.2715 | 0.2554 | 85/222 | 58,039 |
| Focal BCE-Dice | Positive and negative | 0.05 | 0.2731 | 0.1581 | 0.1854 | **0.5179** | 29/222 | 169,343 |
| Focal BCE-Dice | Positive only | 0.05 | 0.2093 | 0.1169 | 0.1945 | 0.2265 | 126/222 | 68,748 |

BCE-Dice with all tiles achieved the highest Dice (0.3839), IoU (0.2376), and
precision (0.4269), while producing the fewest affected negative tiles and
false-positive pixels. Focal BCE-Dice emphasized recall under the all-tile
policy (0.5179), but generated 5.21 times as many false-positive pixels. Under
positive-only training, it affected 126 negative tiles compared with 85 for
BCE-Dice.

The result separates the roles of the two design choices. Negative-scene
exposure supplies the background evidence required for false-positive control,
while BCE-Dice provides the best overlap–precision balance for the target map.
The lower-bound threshold selected by both focal runs also shows that focal
weighting changes the probability scale without replacing that background
supervision. BCE-Dice with all tiles is therefore the primary training strategy
for the reported PLZ 12357 experiments.

Sources:

- `../../runs/unet_plz12357_all/test_analysis/test_summary.json`
- `../../runs/unet_plz12357_positive_only/test_analysis/test_summary.json`
- `../../runs/unet_plz12357_all_focal_dice/test_analysis/test_summary.json`
- `../../runs/unet_plz12357_positive_only_focal_dice/test_analysis/test_summary.json`
