# Stage 6 — Visualization

Produces PNG and PDF plots for either a paired dataset or a single-platform
dataset. In paired mode it writes gene-level scatter plots, combined QC plots,
a paired transcript overview, and a paired sanity-check image overlay. In
single-platform mode it writes one-dataset alternatives for the paired plots.

## What it does

1. In paired mode, rerun the gene comparison internally (re-opens the enriched zarrs) and
   plot log-log scatter plots of MERSCOPE vs Xenium normalized counts.
   In single-platform mode, plot top gene abundance for total and assigned
   normalized counts. Both modes drop control features with
   `apply_dataset_filter` before normalising
   ([comparison.md](comparison.md#walkthrough)). Xenium plots written before
   that filter used the shared control rules are stale when the transcripts
   hold the features it now also drops:
   `<pair_id>_gene_scatter_total_normalized.png` and
   `<sample_id>_gene_abundance_total_normalized.png`, plus the assigned
   plots when the table was built from the `cell_feature_matrix` files.
2. Recompute per-dataset QC metrics and plot either combined paired QC plots
   or one-platform geometry/cell-metric plots.
3. Plot a paired 3x2 or single-platform 3x1 transcript overview with density
   heatmaps, full-field scatter subsamples, and a micron crop.
4. Plot paired or single 250 um sanity crops with image backgrounds, all shape
   contours, and ProSeg assigned/unassigned transcripts.
5. Plot an assignment-rate bar chart for the available platform(s).

ATERA (whole-transcriptome) rows draw steps 3 and 4 as density rasters over
every transcript instead; see [ATERA figures](#atera-figures-density-over-every-transcript).

## Nextflow process

[`VISUALIZE`](../../workflows/modules/visualization.nf) — one instance per
`pair_id`. In paired mode it waits for `COMPARE`; in single-platform mode it
runs after QC or after restored published zarrs.

- **Input:** `tuple(pair_id, samples_json)`, where `samples_json` contains one
  or two `{sample_id, platform, zarr_path}` records.
- **CLI:** `merxen visualize --config visualize_config.json`.
- **Output:** `tuple(pair_id, visualize_out/)`.
- **publishDir:** `${outdir}/${pair_id}/visualization/` (copy mode).

## Python entry points

| Function | File |
|----------|------|
| CLI `visualize_command` | [cli/run_visualization.py](../../src/merxen/cli/run_visualization.py) |
| `plot_gene_scatter` | [visualization/gene_scatter.py:14](../../src/merxen/visualization/gene_scatter.py#L14) |
| `plot_gene_abundance` | [visualization/gene_scatter.py](../../src/merxen/visualization/gene_scatter.py) |
| `plot_geometry_histograms_comparison` | [visualization/qc_plots.py](../../src/merxen/visualization/qc_plots.py) |
| `plot_cell_metrics_violin_comparison` | [visualization/qc_plots.py](../../src/merxen/visualization/qc_plots.py) |
| `plot_assignment_bar` | [visualization/qc_plots.py](../../src/merxen/visualization/qc_plots.py) |
| `plot_transcript_overview` | [visualization/density_overview.py](../../src/merxen/visualization/density_overview.py) |
| `plot_single_transcript_overview` | [visualization/density_overview.py](../../src/merxen/visualization/density_overview.py) |
| `plot_pair_sanity_crops` | [visualization/sanity_plots.py](../../src/merxen/visualization/sanity_plots.py) |
| `plot_single_sanity_crop` | [visualization/sanity_plots.py](../../src/merxen/visualization/sanity_plots.py) |
| `plot_rasterised_transcript_overview` | [visualization/density_overview.py](../../src/merxen/visualization/density_overview.py) |
| `stream_hist2d`, `composite_rgba` | [visualization/raster_density.py](../../src/merxen/visualization/raster_density.py) |

## Config schema

`VisualizationConfig` — [config.py:246](../../src/merxen/config.py#L246).

| Field | Description |
|-------|-------------|
| `output_dir` | Where `visualize_out/` is populated. |
| `pair_id` | Prefix for output filenames. |
| `samples` | One or two sample configs with `sample_id`, `platform`, and `zarr_path`. |
| `merscope_zarr_path` / `xenium_zarr_path` | Legacy paired fields still accepted by the Python config. |

## Outputs

Written under `visualize_out/`:

Each listed `.png` plot is also written as a same-stem `.pdf`.

| Kind | File | Contents |
|------|------|----------|
| Gene scatter | `<pair_id>_gene_scatter_total_normalized.png` | MERSCOPE vs Xenium, all transcripts (normalized). |
| Gene scatter | `<pair_id>_gene_scatter_assigned_normalized.png` | MERSCOPE vs Xenium, transcripts assigned to cells. |
| Gene abundance | `<sample_id>_gene_abundance_total_normalized.png` | Single-platform top gene abundance, all transcripts. |
| Gene abundance | `<sample_id>_gene_abundance_assigned_normalized.png` | Single-platform top gene abundance, assigned transcripts. |
| Geometry | `<pair_id>_geometry_hist.png` | Overlaid step histograms of area, eccentricity, etc. |
| Geometry | `<sample_id>_geometry_hist.png` | Single-platform geometry histograms. |
| Cell metrics | `<pair_id>_cell_violin.png` | Platform violins for transcripts/cell and genes/cell on log y axes. |
| Cell metrics | `<sample_id>_cell_violin.png` | Single-platform per-cell metric violins. |
| Transcript overview | `<pair_id>_transcript_overview.png` | 3x2 density, full scatter, and fixed crop transcript overview. |
| Transcript overview | `<sample_id>_transcript_overview.png` | 3x1 single-platform density, full scatter, and crop transcript overview. |
| Sanity overlay | `<pair_id>_sanity_overlay.png` | Paired 250 um image crops with shape contours and assignment status. |
| Sanity overlay | `<sample_id>_sanity_overlay.png` | Single-platform 250 um image crop with shape contours and assignment status. |
| Sanity crop helper | `<pair_id>_sanity_overlay_crop_location.png` | MERSCOPE raw, MERSCOPE aligned, and Xenium crop locations used for the sanity overlay. |
| Sanity crop helper | `<sample_id>_sanity_overlay_crop_location.png` | Single-platform crop location helper. |
| Assignment rate | `<pair_id>_assignment_rate_bar.png` | Bar chart of `pct_assigned` per platform. |

## Notes

- The visualization stage does **not** read the CSVs produced by the
  comparison stage; it recomputes them. This keeps stages independent but
  means large zarrs are opened twice per run.
- The sanity overlay prefers `MERSCOPE_z_projection` and `morphology_focus`
  image layers and draws ProSeg, ProSeg hybrid, Cellpose-SAM, and the platform's
  original segmentation. The active branch supplies the assignment layer.
  Hybrid views use stored `hybrid_assignment` values and distinguish
  `single_mask`, `proseg_overlap`, `ambiguous_overlap`, and `outside`
  provenance. When MERSCOPE aligned vectors are available, the crop is selected
  in aligned Xenium space and then rendered in raw MERSCOPE image space so the
  image, transcripts, and boundaries stay registered.
- Points coordinate columns are resolved with `first_existing_col` across
  `x`, `x_micron`, `x_location`, `global_x`, `x_global_px`, `observed_x`
  (and the corresponding `y_*`) for transcript plotting.

## ATERA figures: density over every transcript

A whole-transcriptome section holds more than a billion transcripts. MerXen
neither caps nor subsamples them: platforms whose registry entry sets
`rasterised_transcript_plots` (ATERA) draw their transcripts as density rasters
binned over all of them. XENIUM and MERSCOPE figures are unchanged.

- **Streaming binner.** `raster_density.stream_hist2d` reads the points
  element one parquet file at a time, in record batches of the needed columns
  only, and adds each batch to `uint64` accumulators with `np.bincount`.
  Memory is bounded by one batch plus the grid. Bins follow `np.histogram2d`:
  each holds `[edge, next edge)` and the last also holds its right edge.
- **Pruning.** When the build's points index still matches the parquet
  listing and footers (`points_aggregates.fingerprint_matches`), only the
  files whose indexed bounding box meets the window are read; otherwise every
  file is. Section bounds come from the build aggregates, else from the
  parquet column statistics.
- **Transcript overview** (`<sample_id>_transcript_overview.png`), from one
  pass over the section and one pruned pass over the crop:
  1. density over the section in about `heatmap_bins` (400) square bins per
     axis, summed from the 10 um raster of panel 2 (the bin edge is the
     multiple of 10 um closest to `span / 400`), colour limit at the 99.5th
     percentile of the non-empty bins (not the fixed 1,700 used by the
     XENIUM/MERSCOPE panels);
  2. 10 um bins over the section, log colour scale (replaces the 250,000-point
     scatter subsample);
  3. 1 um bins over the 2 mm centre crop, log colour scale (replaces the
     200,000-point crop subsample).
  Titles say "all transcripts" and give the binned count.
- **Sanity overlay** (`<sample_id>_sanity_overlay.png`): the transcripts of the
  250 um crop are a two-layer 0.5 um raster, assigned (yellow, on top) and
  unassigned (red), with per-bin opacity `log1p(count) / log1p(p99.5)`. The
  split reads the registered assignment column of the layer
  (`original_assignment` for the native `original_seg` layer, from the
  `merxen_schema` registry), not a polygon join. The legend gives the exact
  counts. The panel has a 100 um scale bar (all 10x-family platforms).
- **Image placement.** The background image is placed with the pixel size the
  build recorded (`atera/experiment.spatial` in the store; never a fallback)
  and the SpatialData corner convention: pixel `i` covers
  `[i * ps, (i + 1) * ps)` microns.
- **Crop location** (`<sample_id>_sanity_overlay_crop_location.png`): the crop
  box is drawn over the build's transcript density grid (or, with a stale
  index, a 50 um raster of every transcript), not over sampled cell outlines.
