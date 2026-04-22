# Stage 3 — Enrichment

Takes the segmented zarr produced by stage 2 and adds explicit shape layers,
platform-specific source boundaries and images, and per-shape expression
tables that downstream QC and comparison consume.

## What it does

1. Add a canonical ProSeg shape layer under a standard key.
2. Convert the Cellpose mask array to polygon geometries and add it as a
   second shape layer.
3. Copy the original platform shapes (MERSCOPE cell boundaries or Xenium
   cell/nucleus boundaries), source images, and vendor-provided cell table
   into the enriched zarr.
4. For every shape layer, bin transcripts into cells and write a gene ×
   cell counts table (`table_<shape_name>`). Uses the `cell_id` column on
   each shape to avoid re-binning from scratch where possible.

The result is a single self-contained "latest" SpatialData zarr that
carries **all** shape variants side by side so QC and comparison can compare
them head-to-head.

## Nextflow process

[`ENRICH`](../../workflows/modules/enrichment.nf) — one instance per dataset.

- **Input:** `tuple(key, pair_id, platform, enrich_config_json, latest_zarr,
  mask_path)`.
  - The `SEGMENT` output (`proseg_base_latest.zarr`) is symlinked in as
    `latest_input.zarr`.
  - The Cellpose mask is symlinked in as `enrich_input_mask.npy`.
- **CLI:** `merxen enrich --config enrich_config.json`.
- **Output:** `tuple(key, pair_id, platform, latest_input.zarr, enrich_out/)`.
- **publishDir:** `${outdir}/${pair_id}/${platform}/enrichment/` (symlink mode).

The durable zarr written by this stage lives at
`${outdir}/${pair_id}/${platform}/latest/latest_spatialdata.zarr`. The
work-dir `latest_input.zarr` is just the staged handle that Nextflow passes
forward.

## Python entry points

| Function | File |
|----------|------|
| CLI `enrich_command` | [cli/run_enrichment.py](../../src/merxen/cli/run_enrichment.py) |
| `enrich_single_latest` | [enrichment/enrich.py:531](../../src/merxen/enrichment/enrich.py#L531) |
| `run_per_shape_assignment_for_dataset` | [enrichment/assignment.py:300](../../src/merxen/enrichment/assignment.py#L300) |
| `compute_table_from_points_for_shape` | [enrichment/assignment.py:132](../../src/merxen/enrichment/assignment.py#L132) |
| `build_gene_list_from_base_table` | [enrichment/assignment.py:88](../../src/merxen/enrichment/assignment.py#L88) |

## Config schema

`EnrichmentConfig` — [config.py:157](../../src/merxen/config.py#L157).

| Field | Description |
|-------|-------------|
| `dataset_name` | Identifier, e.g. `EXAMPLE01_MERSCOPE`. |
| `platform` | `"MERSCOPE"` or `"XENIUM"` — controls which vendor layers are copied. |
| `latest_zarr_path` | Input zarr (copied in as `latest_input.zarr`). |
| `mask_path` | The Cellpose mask produced by `SEGMENT` (`enrich_input_mask.npy` in the work dir). |
| `original_data_path` | Path back to the stage-1 `source_spatialdata.zarr`. Used to copy vendor shapes and images. |
| `output_dir` | Where to write `enrich_out/` summaries. |
| `persistent_output_path` | Durable "latest" zarr path under `results/.../latest/`. |
| `transform_path` | Optional transform override, same semantics as in segmentation. |

## Walkthrough

1. **Skip if already enriched.** `_is_already_enriched` short-circuits the
   whole stage when the expected layers already exist and `force_rerun` is
   false.
2. **ProSeg shape layer.** Pick the best existing shape key
   (`cell_boundaries`, `cell_boundaries_refined`, `shapes`, ...) and
   mirror it to a canonical `MOSAIK_proseg` key. Downstream code always
   looks at this stable key.
3. **Cellpose shape layer.** Load `cellpose_masks_tiled.npy`, convert to
   polygons, and store as `MOSAIK_cellpose`.
4. **Copy vendor layers.** For MERSCOPE: the original cell boundaries are
   cloned to `merscope_old_shapes`, the source z-plane images are copied into
   the enriched zarr, and a lazy max-projection image is added under
   `MERSCOPE_z_projection`. For Xenium: cell and nucleus boundaries and
   morphology images are copied.
5. **Per-shape tables.** `run_per_shape_assignment_for_dataset` iterates over
   each shape layer and builds a gene × cell counts table by streaming
   transcript points in chunks of `chunk_rows=750_000`. When a vendor table
   already exists (Xenium's cell_table, for example) it is cloned to the
   right region name rather than recomputed.
6. **Atomic replace.** The enriched zarr is written to a temp sibling path,
   then atomically moved over the durable `latest_spatialdata.zarr`. That is
   the point where the non-enriched intermediate disappears.

## Outputs

| File | Contents |
|------|----------|
| `latest/latest_spatialdata.zarr` | Durable enriched SpatialData object used by all downstream analysis. |
| `enrichment/latest_input.zarr` | Staged symlink to the durable latest zarr for workflow chaining. |
| `enrich_out/` | Assignment summary CSVs per shape (transcripts assigned, gene totals). |

## What "per-shape" means

Each shape layer (ProSeg polygons, Cellpose polygons, vendor boundaries) gets
its own `table_<shape_name>`. That way downstream comparisons can be done
**within the same dataset** across segmentation methods, not just across
platforms. Gene list is taken from the base table at the start of
enrichment so all tables share a column order.

## Common failures

- **`No shapes found in latest zarr`** — stage 2 did not produce any cell
  polygons. Rerun `SEGMENT`.
- **`No points found`** — the transcripts table is missing from the
  SpatialData zarr; this can happen if the stage-1 build excluded
  transcripts.
- **Conflicts on re-runs.** When re-running with edits, delete the
  `table_*` elements first or rely on `--force-rerun` so the stage can
  overwrite.
