# Stage 4 — QC

Computes per-cell geometry and transcript-assignment metrics on one analysis
segmentation branch of an enriched zarr. Runs independently for each active
platform and selected branch; results feed into cross-platform comparison in
paired mode and visualization in all modes.

## What it does

For the branch-specific shape/table pair selected by Nextflow:

- **Geometry metrics per cell** — area, perimeter, convex area, circularity,
  solidity, eccentricity, aspect ratio, log10 area.
- **Transcript metrics per cell** — transcripts per cell and unique genes
  per cell, derived from the selected AnnData table when `table_key` is set.
- **Summary statistics** — n cells, n transcripts total/assigned, percent
  assigned, medians for area, eccentricity, transcripts/cell, genes/cell.
- **Hybrid diagnostics** — for `proseg_hybrid`, Cellpose/ProSeg count deltas,
  area growth, rejected external transcripts, fallback rates, assignment-source
  counts, per-gene count changes, and spatial diagnostic maps.
- **Registration check** — whether the branch's cells sit on their
  transcripts (see below).

## Registration check

`segmentation_registration_check`
([qc/registration.py](../../src/merxen/qc/registration.py)) catches a
segmentation written to the wrong coordinates, such as the 2026-09-08 VZG2
run whose Cellpose / ProSeg cells were offset by (+21.9, +111.8) µm. It runs
for every branch and uses no labels:

1. Eight 400 µm windows are centred on random cells of the branch.
2. **Density:** mean transcripts within 4 µm of each centroid, divided by
   the same count at 1000 uniform random points per window. On the stored
   P1212 and P7513 (both platforms) and VZG2 `original_seg` layers it is
   1.7–3.1; the misregistered VZG2 branches score 1.04–1.28.
3. **Offset:** the centroid density map is cross-correlated with that of the
   platform's own cells (`merscope_cell_boundaries` / `xenium_cell_boundaries`;
   skipped for `original_seg` itself). The reported shift is platform minus
   branch, median over windows with a clear peak (z ≥ 4), on a 2 µm grid.
   It is 0 for every registered layer above and (-22, -112) µm for the VZG2
   branches.

The status is `warn` when the density ratio is below 1.5 or the shift is
larger than 5 µm, and `skipped` when fewer than 50 centroids or no transcripts
fall inside the windows. Coordinates are compared in each element's intrinsic
(micron) frame, not a SpatialData coordinate system.

A warning is logged and written to the outputs; the task still succeeds.
With `--qc_registration_strict true` a `warn` makes `merxen qc` exit non-zero
after writing its outputs, so (with the default `errorStrategy = "ignore"`)
downstream analysis of that branch is skipped. Nextflow does not publish a
failed task's outputs, and `publishDir` keeps whatever an earlier run left
there, so read a failed check from the task's work dir. From the launch
directory, `nextflow log <run_name> -f name,exit,workdir | grep '^QC'` lists
the QC tasks; the result is `<workdir>/qc_out/*_registration_qc.json`.

## Nextflow process

[`VALIDATE_ANALYSIS_LAYER`](../../workflows/modules/qc.nf) first checks that the
selected table and shape share the same cell identifiers. For
`proseg_hybrid`, it also requires the registered assignment, background, and
provenance columns before the QC/analysis fan-out is launched.

[`QC`](../../workflows/modules/qc.nf) — one instance per dataset.

- **Input:** `tuple(key, pair_id, platform, segmentation, latest_zarr, table_key, shape_key)`.
- **CLI:** `merxen qc --config qc_config.json`. The config JSON is built
  inline by the process itself.
- **Output:** `tuple(key, pair_id, platform, segmentation, latest_zarr, qc_out/, table_key, shape_key)`.
- **publishDir:** `${outdir}/${pair_id}/${platform}/${segmentation}/qc/` (symlink mode).

## Python entry points

| Function | File |
|----------|------|
| CLI `qc_command` | [cli/run_qc.py](../../src/merxen/cli/run_qc.py) |
| `compute_dataset_qc` | [qc/metrics.py:111](../../src/merxen/qc/metrics.py#L111) |
| `save_dataset_qc` | [qc/metrics.py:205](../../src/merxen/qc/metrics.py#L205) |

## Config schema

`QCConfig` — [config.py:169](../../src/merxen/config.py#L169).

| Field | Description |
|-------|-------------|
| `dataset_name` | Used as the output filename stem and the `dataset` column in every metric DataFrame. |
| `latest_zarr_path` | Enriched zarr from stage 3. |
| `output_dir` | Where `qc_out/` is populated. |
| `table_key` | Optional AnnData table used for transcript/cell and gene/cell metrics. |
| `shape_key` | Optional shape layer used for geometry metrics. |
| `registration_check` | Run the registration check (default `true`). |
| `registration_reference_shape_key` | The platform's own cell shapes for the offset estimate. Nextflow passes `merscope_cell_boundaries` or `xenium_cell_boundaries`. |
| `registration_strict` | Exit non-zero when the check warns. Nextflow passes `params.qc_registration_strict` (default `false`). |

## Walkthrough

1. `compute_dataset_qc` opens the enriched zarr and resolves the requested
   `shape_key` and `table_key` when provided.
2. Build a per-cell geometry DataFrame using shapely: area, perimeter,
   convex hull area, circularity
   (`4π·area / perimeter²`), solidity (`area / convex_area`), and
   eccentricity / aspect ratio from a fitted ellipse.
3. When a table is provided, compute transcripts-per-cell and unique genes
   per cell directly from its expression matrix. Without a table key, fall
   back to grouping the points table on `assignment` / `cell` / `cell_id`.
4. Package the metrics into a dict with `summary`, `geometry_metrics`, and
   `cell_metrics`, and free the loaded zarr.
5. `save_dataset_qc` writes CSVs and a pickle (see outputs below).

## Outputs

Written under `qc_out/` (published to
`${outdir}/${pair_id}/${platform}/${segmentation}/qc/`):

| File | Contents |
|------|----------|
| `<dataset>_qc_summary.csv` | Single-row summary table with the headline numbers. |
| `<dataset>_geometry_metrics.csv` | One row per cell — all geometry columns. |
| `<dataset>_cell_metrics.csv` | One row per cell — transcripts_per_cell, genes_per_cell, `dataset`. |
| `<dataset>_qc.pkl` | Pickle with `summary`, `geometry_metrics`, `cell_metrics` for fast reload. |
| `<dataset>_registration_qc.json` | Registration check: status, reasons, density ratio, shift, thresholds and per-window diagnostics. |
| `<dataset>_hybrid_cell_diagnostics.csv` | Per-hybrid-cell construction diagnostics and Cellpose/ProSeg count deltas. |
| `<dataset>_hybrid_assignment_sources.csv` | Counts and percentages for every hybrid assignment provenance class. |
| `<dataset>_hybrid_fallback_reasons.csv` | Cellpose-fallback reason counts and percentages of hybrid cells. |
| `<dataset>_hybrid_gene_count_changes.csv` | Per-gene hybrid totals and changes relative to Cellpose and ProSeg. |
| `<dataset>_hybrid_area_growth_map.png` | Spatial map of hybrid area growth relative to the Cellpose prior. |
| `<dataset>_hybrid_rejected_transcripts_map.png` | Spatial map of capped or unsupported external transcripts. |

The `<dataset>` stem is lowercased, e.g. `example01_merscope_qc_summary.csv`.
Hybrid-only files are emitted only for the `proseg_hybrid` branch.

## Interpreting the summary

| Field | What to watch for |
|-------|-------------------|
| `n_cells` | Much lower than expected → segmentation under-segmenting; check Cellpose thresholds. |
| `pct_assigned` | Low → Cellpose masks don't cover transcript-dense regions; revisit diameter / thresholds. |
| `median_eccentricity` | Close to 1 → cells look elongated/fragmented; usually a segmentation artefact. |
| `median_area` | Wildly different between platforms → mosaic/pixel transform or `voxel_size` mismatch. |
| `median_transcripts_per_cell` | Sudden drops between runs → transcript QV filter or panel mismatch. |
| `registration_status` | `warn` → cells are not on their transcripts; check the mask-to-micron transform (`merscope_transform_path`, VZG2 manifest origin) before using the branch. |
| `registration_density_ratio` | Near 1 → centroids sit on random tissue rather than cell bodies. |
| `registration_shift_um` | Above 5 µm → systematic offset against the platform's cells; the per-window `shift_x_um` / `shift_y_um` give its direction. |

## Failure modes

- **`No shapes found in <zarr>`** — enrichment failed or wrote an empty
  shape layer. Inspect the upstream enrichment log.
- **`No assignment column found`** — no `table_key` was supplied and the points
  table lacks `assignment` / `cell` / `cell_id`.
- **`No gene column found`** — same as above, for the gene label column.
