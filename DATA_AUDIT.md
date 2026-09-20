# Green-roof data audit

Audit date: 2026-08-18

## Dataset inventory

| Source | Count | Format / CRS | Intended role |
|---|---:|---|---|
| 2016 orthophotos | 3,510 tiles | 256×256 RGBA PNG, XYZ zoom 19 / Web Mercator | supervised-training images |
| 2025 orthophotos | 4 mosaics | 10,000×10,000 RGBA JP2 | later inference and temporal comparison |
| Entire green roofs | 55 polygons | Shapefile / WGS 84 | positive mask polygons |
| Partial green-roof surfaces | 928 polygons | Shapefile / WGS 84 | positive mask polygons |
| Building footprints | 6,512 polygons | GeoJSON / EPSG:25833 | optional roof ROI and post-processing |

The 2016 tiles form a complete 54×65 rectangular grid. Their calculated WGS 84
extent is `[13.469925, 52.416241, 13.507004, 52.443455]`, at approximately
0.182 m/pixel near the dataset centre. All 3,510 PNG files decode successfully,
have identical dimensions/mode, and are fully opaque.

The partial-roof labels contain 492 `intensiv` and 436 `extensiv` polygons.
The `area` attribute must be treated cautiously until its unit and zero-valued
records have been validated against geometry-derived areas.

## Recommended segmentation target

Start with **binary semantic segmentation**:

- class 0: non-green-roof/background;
- class 1: green-roof surface;
- positive geometry: union of the 55 entire-roof polygons and 928 partial-roof
  polygons.

Do not use the 6,512 city/building polygons as positive green-roof labels. They
are useful later as a roof-region constraint. A three-class experiment
(`background`, `extensiv`, `intensiv`) can follow, but the entire-roof layer has
no matching intensity category and therefore needs a labeling policy first.

## Data issues that preprocessing must handle

1. **Vector-to-raster conversion:** labels are geographic polygons, not masks.
   Reproject WGS 84 vertices into the XYZ tile pixel coordinate system and
   rasterize them with topology preserved.
2. **Extent mismatch:** 2 entire-roof and 10 partial-roof features are outside
   the supplied 2016 image grid. They must be excluded from supervised samples;
   971 labeled features are fully covered.
3. **CRS differences:** Shapefile labels use WGS 84, XYZ imagery is implicitly
   Web Mercator, and building GeoJSON declares EPSG:25833. Never combine raw
   coordinates without transformation.
4. **Spatial leakage:** neighboring tiles show contiguous ground. Split by
   spatial blocks before extracting train/validation/test patches; do not use a
   random per-tile split.
5. **Class imbalance:** green-roof pixels occupy a small fraction of the scene.
   Retain representative negative tiles and begin with BCE + Dice loss (or a
   weighted equivalent), reporting foreground IoU/Dice and precision/recall.
6. **Temporal domain shift:** train labels belong to 2016. Treat the 2025 JP2
   imagery as unlabeled inference data until its georeferencing and alignment
   are verified; it is not a random test split for the 2016 model.

## MVP preprocessing output

Generate data without modifying the raw directory:

```text
processed/green_roofs_2016/
  images/          # RGB tiles or larger stitched patches
  masks/           # uint8 binary masks: 0 background, 1 green roof
  manifests/
    train.csv
    val.csv
    test.csv
  qa/              # image/mask overlays and summary statistics
```

Recommended first baseline: 512×512 patches assembled from 2×2 source tiles,
spatial block split, pretrained U-Net or SegFormer, and binary foreground Dice +
IoU as the primary validation metrics.

## Preprocessing status

The binary-mask preprocessing milestone was completed on 2026-08-18. All 3,510
tiles now have a mask and belong to exactly one 8×8-tile spatial block. The
result contains 185 positive tiles and 657,181 positive pixels (0.286% of all
pixels). See `../data/processed/green_roofs_2016/dataset_summary.json` for the
split statistics and `qa/top_positive_tiles.png` for visual alignment checks.

On 2026-08-26, the preprocessing implementation was migrated to GeoPandas /
Pyogrio vector I/O, Shapely spatial indexing, Pyproj/Mercantile coordinates,
and Rasterio rasterization. A validated temporary rebuild produced 184 positive
tiles and 644,525 positive pixels because Rasterio uses pixel-centre inclusion.
The existing processed directory above remains the original Pillow-derived
version and was not overwritten. Treat a future regenerated dataset as a new
mask/split version and repeat label QA before training.

The current PLZ 12357 experiment uses the official current postcode polygon as
a study-area reference and retains tiles whose centres fall inside it. This
reduces the rectangular 3,510-tile source grid to 1,635 tiles: 182 positive and
1,453 negative. Rasterio generated 643,038 foreground pixels (0.6001%). The
spatial split contains 1,180/203/252 train/validation/test tiles with no block
leakage. Because the boundary is current rather than a verified historical 2016
boundary, this remains a documented study-area assumption.
