# Stage 4 — Mask Image Quantification

Quantifies image-channel intensity over the final Cellpose masks produced by
`SEGMENT`. This stage uses `cellpose_masks_tiled.npy` directly, so the measured
pixels are the exact Cellpose label pixels rather than ProSeg, vendor, or
polygon-rasterized masks.

## What it does

1. Read the enriched `latest_spatialdata.zarr`.
2. Load the final nonzero labels from `cellpose_masks_tiled.npy`.
3. Iterate every SpatialData source image element and every image channel.
   Viewer-cache pyramids and `*_aligned_nonrigid` images, which alignment
   resamples onto the counterpart platform's grid, are skipped.
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

Rows from a Cellpose mask are named `cellpose_<label_id>`. Feature names use
`{image_key}__{channel}__{stat}`.

Images registered by [`REGISTER_IMAGES`](register-images.md), such as
post-Xenium IF, are ordinary source images on the mask grid and are quantified
the same way. An existing quantification table is reused only when its
`image_key` set matches the zarr's current source images; registering or
removing an image triggers re-quantification.

## Native rows (ATERA)

A native row has no Cellpose mask. Its config sets `mask_path: null` and
`mask_label_key: cell_labels`, and the stage quantifies the store's own vendor
cell labels. The mask slot of the task holds an empty placeholder file, which
the stage never reads.

- **Lazy reads.** The level-0 label array is read lazily, tile by tile. The
  pixel counts use a tiled `bincount`, and the full mask is never loaded or
  copied. `tile_size` is rounded up to whole label chunks (4096 px for
  Atera), so no chunk is decoded more than once per pass; the summary records
  the tile used as `tile_size_px`. Memory is set by the per-cell pixel value
  lists of the second pass, as on the Cellpose path.
- **Rows.** There is one row per vendor cell, named by its vendor
  `source_cell_id` from `atera_cell_boundaries`.
- **Outputs.** The table is `table_atera_cell_boundaries_image_quantification`,
  with region `atera_cell_boundaries`. The table's `uns` and the summary
  record `mask_label_key`.
- **Hybrid join.** There is no hybrid table, so the hybrid join is skipped.

The vendor labels share the morphology image's pixel grid, so the image-shape
check applies unchanged.

A `native_proseg` row (AT-9) also quantifies its stitched ProSeg cells in the
same task (one writer per store): the config lists them in `extra_layers`
(`mask_label_key: proseg_vendor_cell_labels`, `shape_key: MOSAIK_proseg_vendor`,
`table_key: table_MOSAIK_proseg_vendor_image_quantification`). An ATERA
`cellpose_proseg` row (AT-7) lists its ProSeg cells (`proseg_cell_labels`,
`MOSAIK_proseg`) and its Cellpose cells (`cellpose_cell_labels`,
`MOSAIK_cellpose`); ATERA layers are always quantified from label elements.
Each extra layer gets its own table (region: its shapes) and its own sidecar
files, and is skipped on a re-run like the main layer. `extra_layers` is empty
for every other row.

On the public full section (91,431 cells, 4-channel 42,548 × 61,369 image) the
task takes about 3 min at 9.6 GB peak memory (AT-6 exit run), far under the
160 GB reservation.

When `table_MOSAIK_proseg_hybrid` exists, the stage joins these Cellpose-mask
measurements onto hybrid cells by their shared `instance_id`. It does not
re-quantify expanded hybrid polygons. The hybrid expression matrix remains
unchanged; joined features are available as `cellpose_image__*` `obs` columns
and in `.obsm["cellpose_image_quantification"]`.
