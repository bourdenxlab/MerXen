# Stage 3 — Enrichment

Takes the segmented zarr produced by stage 2 and adds explicit shape layers,
platform-specific source boundaries and images, and per-shape expression
tables that downstream QC and comparison consume.

## What it does

1. Add a canonical ProSeg shape layer under a standard key.
2. Convert the cell and nuclei Cellpose mask arrays to polygon geometries as
   `MOSAIK_cellpose` and `cellpose_nuclei`.
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
  mask_path, nuclei_mask_path)`.
  - The `SEGMENT` output (`proseg_base_latest.zarr`) is symlinked in as
    `latest_input.zarr`.
  - The Cellpose mask is symlinked in as `enrich_input_mask.npy`.
  - The nuclei mask is symlinked in as `enrich_input_nuclei_mask.npy`.
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
| `native_label_check`, `stream_label_lookup` (native rows) | [enrichment/label_lookup.py](../../src/merxen/enrichment/label_lookup.py) |
| `build_gene_list_from_base_table` | [enrichment/assignment.py:88](../../src/merxen/enrichment/assignment.py#L88) |

## Config schema

`EnrichmentConfig` — [config.py:157](../../src/merxen/config.py#L157).

| Field | Description |
|-------|-------------|
| `dataset_name` | Identifier, e.g. `EXAMPLE01_MERSCOPE`. |
| `platform` | `"MERSCOPE"`, `"XENIUM"` or `"ATERA"` — controls which vendor layers are copied. |
| `latest_zarr_path` | Input zarr (copied in as `latest_input.zarr`). |
| `mask_path` | The Cellpose mask produced by `SEGMENT` (`enrich_input_mask.npy` in the work dir); `null` for native rows. |
| `nuclei_mask_path` | The DAPI-only nuclei mask produced by the `segment_nuclei` stage; `null` for native rows. |
| `original_data_path` | Path back to the stage-1 `source_spatialdata.zarr`. Used to copy vendor shapes and images. |
| `output_dir` | Where to write `enrich_out/` summaries. |
| `persistent_output_path` | Durable "latest" zarr path under `results/.../latest/`. |
| `transform_path` | Optional transform override, same semantics as in segmentation. |
| `segmentation_source` | `cellpose_proseg` (default) or `native` (ATERA vendor segmentation; see below). |
| `native_label_check` | Run the native label check (default `true`; native rows only). |

## Walkthrough

1. **Skip if already enriched.** `_is_already_enriched` short-circuits the
   whole stage when the expected layers already exist and `force_rerun` is
   false. If a previous run left only a partial enrichment, the stage rebuilds
   those partial artifacts and clears derived per-shape tables before
   assignment runs again.
2. **ProSeg shape layer.** Pick the best existing shape key
   (`cell_boundaries`, `cell_boundaries_refined`, `shapes`, ...) and
   mirror it to a canonical `MOSAIK_proseg` key. Downstream code always
   looks at this stable key.
3. **Cellpose shape layer.** Load `cellpose_masks_tiled.npy`, convert to
   polygons, and store as `MOSAIK_cellpose`.
4. **Nuclei shape layer.** Convert `cellpose_nuclei_masks_tiled.npy` with the
   same pixel-to-micron transform and store it as `cellpose_nuclei`.
5. **Copy vendor layers.** For MERSCOPE: the original cell boundaries are
   cloned to `merscope_old_shapes`, and only the max-projection image is stored
   under `MERSCOPE_z_projection`. Legacy source zarrs with one image per z
   plane are projected during enrichment rather than copied plane-by-plane. For
   Xenium: cell and nucleus boundaries and morphology focus images are copied.
6. **Per-shape tables.** `run_per_shape_assignment_for_dataset` iterates over
   each shape layer and builds a gene × cell counts table by streaming
   transcript points in chunks of `chunk_rows=750_000`. When a vendor table
   already exists (Xenium's cell_table, for example) it is cloned to the
   right region name rather than recomputed.
7. **In-place element writes.** Enrichment writes each added shape, image, and
   table into the backed `latest_spatialdata.zarr` with SpatialData's
   element-level writer. The stage no longer creates a full temp copy of the
   zarr just to add these additive layers.

## Outputs

| File | Contents |
|------|----------|
| `latest/latest_spatialdata.zarr` | Durable enriched SpatialData object used by all downstream analysis. |
| `enrichment/latest_input.zarr` | Staged symlink to the durable latest zarr for workflow chaining. |
| `enrich_out/` | Assignment summary CSVs per shape (transcripts assigned, gene totals). |

## Native rows (ATERA, `segmentation_source=native`)

A native row runs no Cellpose or ProSeg. Its latest store comes from
[`NATIVE_LATEST`](segmentation.md): the vendor cell and nucleus shapes are
already renamed to `atera_cell_boundaries` and `atera_nucleus`, the vendor
table is `table_original`, and the registry holds `original` and
`original_nucleus`. `ENRICH` receives empty placeholder files in its two mask
slots, and its config carries `segmentation_source: native` with both mask
paths `null`.

Native enrichment (`_enrich_native_in_place`):

1. **Checks the elements.** It needs the vendor shapes, `table_original`, the
   native image and the vendor label elements `cell_labels` and
   `nucleus_labels`. Missing shapes or `table_original` are re-created from the
   source store (`native.ensure_native_elements`). Missing images, labels or
   points are shared payloads that only `NATIVE_LATEST` can restore, so the
   stage stops and says so.
2. **Runs the label check.** This is one streaming pass over the points,
   partition by partition, in a thread pool bounded by `DASK_NUM_WORKERS`. It
   reads only the label blocks each partition touches, so it never loads a
   whole mask. The partitions run in their natural (tile) order, and decoded
   blocks are kept in a 3 GiB LRU shared by the partitions in flight, so each
   block is decoded about once instead of once per partition. On the public
   full section the pass takes about 1.5–2 min with 12 workers and 12.5 GB
   peak memory. It uses the vendor's corner convention in the vendor's float32
   arithmetic: a transcript at `x` µm is on pixel
   `floor(float32(x) * float32(1 / ps))`. Float64 `x / ps` disagrees with the
   vendor on about 1.4e-5 of transcripts, all within 0.0012 px of a pixel
   edge.
   - **Cell labels.** The `cell_labels` label of every transcript is compared
     with the vendor's `original_assignment`. More than 1e-6 of rows
     disagreeing is an error. That usually means a wrong pixel size or
     transform.
   - **Nucleus labels.** The `nucleus_labels` label is mapped to its owning
     cell through `atera_nucleus.cell_instance_id`, because nucleus ids are not
     cell ids. A cell with two nuclei gets the transcripts of both.
   - **Vendor flag.** Agreement with the vendor's `overlaps_nucleus` flag is
     reported.
   - **Pass rule.** The check passes only if every entry of `checks` holds:
     - `cell_assignment`: at most 1e-6 of rows disagree with the vendor;
     - `cell_labels_in_table`: no cell label lies beyond the table's ids;
     - `nucleus_labels_known`: every nucleus label has a nucleus row with an
       owning cell;
     - `nuclei_inside_own_cell`: every nuclear transcript lies on its owning
       cell's label, so a broken `cell_instance_id` join fails here;
     - `nucleus_overlaps`: at most 1e-6 of rows disagree with
       `overlaps_nucleus`;
     - `points_scale`: the pixel size used is the one the store's points
       transform records.

     A failed check stops the stage before anything is written to the table.
     `failures` in the summary names the failing checks.
3. **Adds per-cell counts to `table_original`.** These are the only change to
   the table:
   - `merxen_cell_transcript_counts` and `merxen_nucleus_transcript_counts`
     count all features;
   - `merxen_cell_gene_counts` and `merxen_nucleus_gene_counts` count only
     `is_gene` rows.
4. **Records provenance.** It sets `merxen_schema.segmentation_source=native`
   and writes the vendor provenance under `merxen_schema.native_segmentation`:
   the analysis software version, the segmentation stain and the per-cell
   segmentation methods. It writes the completion record
   `merxen_native_enrichment`. No `MOSAIK_*` layer, Cellpose shape or Cellpose
   model identity is written.

A second run on an enriched store is a skip, which re-writes the label-check
summary from the store. `native_label_check: false` turns the pass off.

Per-shape assignment on a native row does no clone and no spatial join. It
registers `atera_cell_boundaries` with `table_original` as it is. Nucleus
shapes keyed by nucleus row (`atera_nucleus`) never get a per-shape table, on
any route.

The partial-enrichment cleanup of a native store removes only the derived
`table_*` tables. The images, labels and points are payloads hardlinked from
the source store, and the cleanup never touches them. An unreadable raster in
a native store (the `KeyError('ome')` of an interrupted write) is therefore not
cleaned up: the stage stops and asks for a rebuild with `NATIVE_LATEST`.

Native outputs in `enrich_out/`:

| File | Contents |
|------|----------|
| `<dataset>_native_label_check.json` | `passed`, `checks` and `failures`; mismatch counts and tolerance, nucleus attribution and `overlaps_nucleus` agreement, the pixel size and arithmetic used (`pixel_precision`) and whether the pixel size agrees with the store's points transform. |
| `<dataset>_per_shape_assignment_summary.csv` | One `native_table_original` row. |

## What "per-shape" means

Each shape layer (ProSeg polygons, Cellpose polygons, vendor boundaries) gets
its own `table_<shape_name>`. That way downstream comparisons can be done
**within the same dataset** across segmentation methods, not just across
platforms. Gene list is taken from the base table at the start of
enrichment so all tables share a column order.

## Common failures

- **`No shapes found in latest zarr`** — stage 2 did not produce any cell
  polygons. Rerun `SEGMENT`.
- **`Native label check failed`** (native rows) — the cell labels disagree
  with the vendor assignment, so the pixel size or transform does not match
  the points. Check `transform_path` and the store's `atera/experiment*.spatial`
  sidecars; `<dataset>_native_label_check.json` has the counts.
- **`Native latest store lacks payload elements`** — the native latest store
  lost an image, label or points element. Rebuild it with the segment stage
  (`NATIVE_LATEST`).
- **`No points found`** — the transcripts table is missing from the
  SpatialData zarr; this can happen if the stage-1 build excluded
  transcripts.
- **`... was computed from an image with bead removal ...` or `... predates the
  correction`.** The latest zarr's `segmentation_image_preprocessing.json` does
  not match the source SpatialData's bead-removal record (or is missing next to
  a bead-corrected source), so its cells came from another image. Move
  `latest/` aside or run segmentation with `--force_proseg_rerun true`; see
  [segmentation](segmentation.md).
- **Conflicts on re-runs.** When re-running with edits, rely on
  `--force-rerun` so additive layers can be replaced element by element.
