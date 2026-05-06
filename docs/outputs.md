# Outputs

This page documents every directory and file the pipeline writes under
`${outdir}` (the Nextflow `--outdir` parameter, default `./results`).

## Top-level layout

```
${outdir}/
├── nextflow/
│   ├── report.html
│   ├── timeline.html
│   └── trace.tsv
├── <pair_id_1>/
│   ├── merscope/
│   │   ├── spatialdata/
│   │   ├── segmentation/
│   │   ├── enrichment/
│   │   └── qc/
│   ├── xenium/
│   │   ├── spatialdata/
│   │   ├── segmentation/
│   │   ├── enrichment/
│   │   └── qc/
│   ├── alignment/
│   ├── alignment_qc/
│   ├── comparison/
│   ├── visualization/
│   └── clustering_squidpy/
├── <pair_id_2>/
│   └── ...
└── ...
```

`<pair_id>` comes straight from the `pair_id` column of the samplesheet.

Nextflow also keeps its own working directory at `./work/` (next to the
`workflows/` folder by default). That's cache state, not output — safe to
delete between full runs, but required for `-resume`.

## Per-stage artifacts

### SpatialData build

Path: `${outdir}/<pair_id>/<platform>/spatialdata/`

| File | Contents |
|------|----------|
| `source_spatialdata.zarr` | Platform-specific SpatialData zarr. Either freshly built from raw data or symlinked from a samplesheet-provided cache. |

Published with `mode: "symlink"` — the target of the symlink is the Nextflow
work directory or the cached path. See
[Caching and reuse](pipeline.md#caching-and-reuse).

### Latest SpatialData

Path: `${outdir}/<pair_id>/<platform>/latest/`

| File | Contents |
|------|----------|
| `latest_spatialdata.zarr` | Durable current SpatialData artifact. Segmentation writes the refined ProSeg result here, then enrichment atomically replaces it with the fully enriched object. This is the primary downstream input. |

### Segmentation

Path: `${outdir}/<pair_id>/<platform>/segmentation/`

| File | Contents |
|------|----------|
| `proseg_base_latest.zarr` | Staged symlink to `../latest/latest_spatialdata.zarr`. |
| `cellpose_masks_tiled.npy` | Global-pixel uint32 mask from tiled Cellpose. Fed into enrichment. |
| `transcripts_for_proseg.csv` | ProSeg input: per-transcript rows with seeded `cell_id`. Retained for debugging. |

### Enrichment

Path: `${outdir}/<pair_id>/<platform>/enrichment/`

| File | Contents |
|------|----------|
| `latest_input.zarr` | Staged symlink to `../latest/latest_spatialdata.zarr`. |
| `enrich_out/` | Assignment summary CSVs per shape (transcripts assigned, gene totals). |

### QC

Path: `${outdir}/<pair_id>/<platform>/qc/`

| File | Contents |
|------|----------|
| `qc_out/<dataset>_qc_summary.csv` | Single-row headline stats. |
| `qc_out/<dataset>_geometry_metrics.csv` | Per-cell geometry (area, perimeter, eccentricity, ...). |
| `qc_out/<dataset>_cell_metrics.csv` | Per-cell transcripts_per_cell, genes_per_cell. |
| `qc_out/<dataset>_qc.pkl` | Pickle with summary + DataFrames for fast reload. |

`<dataset>` is lowercased, e.g. `example01_merscope`.

### Alignment

Path: `${outdir}/<pair_id>/alignment/`

Only present when `--enable_alignment true`.

| File | Contents |
|------|----------|
| `align_out/alignment_transform.json` | Spateo parameters, affine matrix, serialized RBF metadata, and displacement summary. |
| `align_out/alignment_coords/*.csv` | Raw, rigid, and non-rigid alignment centroid tables. |

`ALIGN` updates the existing MERSCOPE latest zarr in place: raw vector elements
remain untouched, rigid affine transforms are saved to `merxen_xenium`, and new
`*_aligned_nonrigid` vector elements store materialized non-rigid coordinates.
Xenium is not copied; downstream stages keep using the original Xenium latest
zarr as the fixed reference.

### Alignment QC

Path: `${outdir}/<pair_id>/alignment_qc/`

Only present when `--enable_alignment true`.

| File | Contents |
|------|----------|
| `alignment_qc_out/<pair_id>_alignment_qc.json` | SABench-style grid metrics and centroid distance summary. |
| `alignment_qc_out/<pair_id>_alignment_qc_metrics.csv` | Single-row CSV with the same metrics. |
| `alignment_qc_out/<pair_id>_alignment_overlay.png` | Xenium/MERSCOPE centroid overlay after alignment. |

### Comparison

Path: `${outdir}/<pair_id>/comparison/`

| File | Contents |
|------|----------|
| `compare_out/<pair_id>_total_counts_compare.csv` | Gene × platform total counts. |
| `compare_out/<pair_id>_assigned_counts_compare.csv` | Gene × platform counts from the primary cell table. |
| `compare_out/<pair_id>_total_normalized_compare.csv` | CP10K-normalized total counts. |
| `compare_out/<pair_id>_assigned_normalized_compare.csv` | CP10K-normalized assigned counts. |
| `compare_out/<pair_id>_comparison_metrics.json` | Platform totals + log-log linear-fit metrics. |

### Visualization

Path: `${outdir}/<pair_id>/visualization/`

| File | Contents |
|------|----------|
| `visualize_out/<pair_id>_gene_scatter_total_normalized.png` | MERSCOPE vs Xenium log-log scatter, all transcripts. |
| `visualize_out/<pair_id>_gene_scatter_assigned_normalized.png` | MERSCOPE vs Xenium log-log scatter, assigned transcripts only. |
| `visualize_out/<pair_id>_geometry_hist.png` | Overlaid Xenium/MERSCOPE step histograms of cell area, eccentricity, etc. |
| `visualize_out/<pair_id>_cell_violin.png` | Side-by-side platform violins for transcripts-per-cell and genes-per-cell. |
| `visualize_out/<pair_id>_transcript_overview.png` | 3x2 density, full scatter, and fixed crop transcript overview. |
| `visualize_out/<pair_id>_sanity_overlay.png` | Paired 250 um image crops with all shape contours and transcript assignment status. |
| `visualize_out/<pair_id>_assignment_rate_bar.png` | Bar chart comparing `pct_assigned` across platforms. |

### Squidpy clustering

Path: `${outdir}/<pair_id>/clustering_squidpy/`

| File | Contents |
|------|----------|
| `clustering_squidpy_out/<platform>/<pair_id>_<platform>_qc_histograms.png` | Histograms for transcripts/cell, genes/cell, cell area, nucleus ratio, and control/blank counts. |
| `clustering_squidpy_out/<platform>/<pair_id>_<platform>_qc_metrics.csv` | Per-cell QC metrics used for the histogram panel. |
| `clustering_squidpy_out/<platform>/<pair_id>_<platform>_umap.png` | Scanpy UMAP colored by total counts, genes by counts, and Leiden cluster. |
| `clustering_squidpy_out/<platform>/<pair_id>_<platform>_spatial_scatter_leiden.png` | Squidpy spatial scatter colored by Leiden cluster. |
| `clustering_squidpy_out/<platform>/<pair_id>_<platform>_clustered.h5ad` | Filtered, normalized, log-transformed, clustered AnnData object with raw counts in `layers["counts"]`. |

## Nextflow reports

Path: `${outdir}/nextflow/`

| File | What it shows |
|------|---------------|
| `report.html` | HTML summary of each process: status, duration, CPU, memory. |
| `timeline.html` | Per-task Gantt chart. |
| `trace.tsv` | Tab-separated per-task metrics incl. peak RSS, peak VMEM, realtime, workdir. |

All three are configured in
[workflows/nextflow.config:75-96](../workflows/nextflow.config#L75-L96) and
are overwritten on each run.

## Nextflow working directory

`./work/` (relative to where `nextflow` was invoked). Contains one directory
per task with the full execution context: config JSON, stdout, stderr,
symlinks to inputs, the process's working files. Cached by hash so `-resume`
can short-circuit successful stages. Safe to delete when you no longer need
to resume.

## Log files

`.nextflow.log` (most recent run) plus a rolling history
(`.nextflow.log.1`, `.nextflow.log.2`, ...). Useful for debugging failed
runs — tail `.nextflow.log` while a pipeline runs to watch progress.
