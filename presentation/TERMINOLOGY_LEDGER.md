# Terminology ledger

| Canonical term | Meaning in this deck | Avoid / clarify |
|---|---|---|
| green-roof segmentation | Pixel-wise prediction of green-roof area from RGB aerial imagery | Do not call it tile classification |
| positive tile | A tile whose reference mask contains at least one positive pixel | Distinguish from a correctly predicted tile |
| negative-only tile | A tile whose reference mask contains zero positive pixels | Not an invalid or discarded sample |
| all-tile training | Balanced sampling from positive and negative-only training tiles | It is not the natural 11.1% prevalence distribution |
| positive-only training | Training batches sampled only from positive tiles, with replacement | Validation remains the same full set |
| spatial block split | Assignment of 8 x 8 XYZ-tile blocks to train, validation, or test | Not random per-tile splitting |
| deficit score | Weighted sum of the normalized tile-count deficit and foreground-pixel deficit for one candidate split | The 0.35/0.65 values are score weights, not split ratios |
| foreground-pixel balance | Approximate allocation of labelled green-roof pixels across train, validation, and test | Distinguish from the number of positive tiles |
| greedy block assignment | Sequential assignment of whole blocks to the currently highest-scoring split | Not a globally optimal or hard-balanced solution |
| PLZ 12357 study area | Tiles whose centres fall inside the current official Berlin postcode boundary 12357 | Boundary is current, not verified as historical 2016 geometry |
| fixed-threshold metric | Metric evaluated with probability threshold 0.5 | Keep separate from calibrated validation metrics |
| calibrated validation metric | Metric evaluated at a threshold selected on validation data | Never present as test performance |
| held-out prediction tile | Frozen validation or test tile used only in the spatial-distance diagnostic | Test labels have not been used for model scoring |
| Dice / IoU | Pixel-level overlap metrics | Report values as proportions or percentages consistently |
| false-positive scene | A negative-only tile containing at least one predicted positive pixel | Distinguish scene count from false-positive pixel count |
