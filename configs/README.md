# Experiment configurations

Every TOML file is retained as part of the experiment history. The files are
independent on purpose: a completed run can be reproduced without resolving a
configuration inheritance chain.

## Configurations supporting the final research questions

| Configuration | Role |
|---|---|
| `unet_plz12357_all.toml` | Negative-tile ablation: retain positive and negative training tiles |
| `unet_plz12357_positive_only.toml` | Negative-tile ablation: remove negative-only training tiles |
| `unet_baseline_v3.toml` | Frozen spatial-evaluation U-Net used in the architecture comparison |
| `segformer_b0.toml` | Frozen spatial-evaluation Transformer comparison |

## Development history

| Configuration | Role |
|---|---|
| `unet_baseline.toml` | Initial U-Net baseline |
| `unet_baseline_v2.toml` | Frozen-encoder-BatchNorm and early-stopping iteration |

## Exploratory configurations

| Configuration | Role |
|---|---|
| `unet_focal_dice.toml` | Focal BCE + Dice loss trial |
| `unet_tversky.toml` | Tversky loss trial |
| `unet_sampler_30_70.toml` | 30/70 positive/negative tile sampler trial |

Only the first group supports the final reported claims. Do not infer a result
from a configuration file alone; the corresponding checkpoint, frozen
threshold, and report in `../reports/` define the completed experiment.
