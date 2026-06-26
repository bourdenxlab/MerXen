# Stage 4 — Mask Image Quantification

Quantifies image-channel intensity over the final Cellpose masks produced by
`SEGMENT`. This stage uses `cellpose_masks_tiled.npy` directly, so the measured
pixels are the exact Cellpose label pixels rather than ProSeg, vendor, or
polygon-rasterized masks.

## What it does

1. Read the enriched `latest_spatialdata.zarr`.
2. Load the final nonzero labels from `cellpose_masks_tiled.npy`.
3. Iterate every SpatialData image element and every image channel.
4. Fail if an image's native `(y, x)` shape differs from the mask shape.
5. Compute exact `min`, `median`, `mean`, `max`, and `iqr` for each
   cellpose label/channel.
6. Write `table_MOSAIK_cellpose_image_quantification` into the SpatialData
   zarr and export sidecar files.

## Nextflow process

[`MASK_IMAGE_QUANTIFICATION`](../../workflows/modules/mask_image_quantification.nf)
— one instance per dataset.

- **Input:** enriched zarr plus the Cellpose mask from `SEGMENT`.
- **CLI:** `merxen mask-image-quantification --config mask_image_quantification_config.json`.
- **Output:** `tuple(key, pair_id, platform, latest_input.zarr,
  mask_image_quantification_out/)`.
- **publishDir:** `${outdir}/${pair_id}/${platform}/mask_image_quantification/`
  (symlink mode).

When the stage is active, downstream QC/analysis receives the quantified zarr.
When it is skipped by a stage range, downstream stages continue from the
enriched zarr as before.

## Outputs

| File | Contents |
|------|----------|
| `latest/latest_spatialdata.zarr` | Same durable zarr, updated in place with `table_MOSAIK_cellpose_image_quantification`. |
| `mask_image_quantification_out/*_mask_image_quantification.parquet` | Wide cell × image-channel-stat matrix. |
| `mask_image_quantification_out/*_mask_image_quantification_features.csv` | Feature metadata with `image_key`, `channel`, and `statistic`. |
| `mask_image_quantification_out/*_mask_image_quantification_summary.json` | Dataset, image, feature, and output summary. |

Rows are named `cellpose_<label_id>`. Feature names use
`{image_key}__{channel}__{stat}`.
