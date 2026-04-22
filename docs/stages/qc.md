# Stage 4 — QC

Computes per-cell geometry and transcript-assignment metrics on the enriched
zarr of a single dataset. Runs independently for MERSCOPE and Xenium; results
feed into the cross-platform comparison and the visualization stage.

## What it does

For the primary shape layer of the enriched zarr:

- **Geometry metrics per cell** — area, perimeter, convex area, circularity,
  solidity, eccentricity, aspect ratio, log10 area.
- **Transcript metrics per cell** — transcripts per cell and unique genes
  per cell, derived by grouping the points table on the assignment column.
- **Summary statistics** — n cells, n transcripts total/assigned, percent
  assigned, medians for area, eccentricity, transcripts/cell, genes/cell.

## Nextflow process

[`QC`](../../workflows/modules/qc.nf) — one instance per dataset.

- **Input:** `tuple(key, pair_id, platform, latest_zarr)`.
- **CLI:** `merxen qc --config qc_config.json`. The config JSON is built
  inline by the process itself (no per-dataset parameters beyond the zarr
  path and output dir).
- **Output:** `tuple(key, pair_id, platform, latest_zarr, qc_out/)`.
- **publishDir:** `${outdir}/${pair_id}/${platform}/qc/` (symlink mode).

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

## Walkthrough

1. `compute_dataset_qc` opens the enriched zarr and picks the first shape
   layer (`sdata.shapes[0]`) and the first points layer.
2. Build a per-cell geometry DataFrame using shapely: area, perimeter,
   convex hull area, circularity
   (`4π·area / perimeter²`), solidity (`area / convex_area`), and
   eccentricity / aspect ratio from a fitted ellipse.
3. Resolve assignment (`assignment` / `cell` / `cell_id`) and gene
   (`feature_name` / `gene` / `target`) columns on the points table
   using `first_existing_col`. Raise if neither shape is found.
4. `_compute_cell_metrics_from_points` groups assigned points and computes
   transcripts-per-cell and unique-genes-per-cell.
5. Package the metrics into a dict with `summary`, `geometry_metrics`, and
   `cell_metrics`, and free the loaded zarr.
6. `save_dataset_qc` writes CSVs and a pickle (see outputs below).

## Outputs

Written under `qc_out/` (published to `${outdir}/${pair_id}/${platform}/qc/`):

| File | Contents |
|------|----------|
| `<dataset>_qc_summary.csv` | Single-row summary table with the headline numbers. |
| `<dataset>_geometry_metrics.csv` | One row per cell — all geometry columns. |
| `<dataset>_cell_metrics.csv` | One row per cell — transcripts_per_cell, genes_per_cell, `dataset`. |
| `<dataset>_qc.pkl` | Pickle with `summary`, `geometry_metrics`, `cell_metrics` for fast reload. |

The `<dataset>` stem is lowercased, e.g. `example01_merscope_qc_summary.csv`.

## Interpreting the summary

| Field | What to watch for |
|-------|-------------------|
| `n_cells` | Much lower than expected → segmentation under-segmenting; check Cellpose thresholds. |
| `pct_assigned` | Low → Cellpose masks don't cover transcript-dense regions; revisit diameter / thresholds. |
| `median_eccentricity` | Close to 1 → cells look elongated/fragmented; usually a segmentation artefact. |
| `median_area` | Wildly different between platforms → mosaic/pixel transform or `voxel_size` mismatch. |
| `median_transcripts_per_cell` | Sudden drops between runs → transcript QV filter or panel mismatch. |

## Failure modes

- **`No shapes found in <zarr>`** — enrichment failed or wrote an empty
  shape layer. Inspect the upstream enrichment log.
- **`No assignment column found`** — the points table in the zarr lacks
  `assignment` / `cell` / `cell_id`. Usually means ProSeg wasn't the last
  writer.
- **`No gene column found`** — same as above, for the gene label column.
