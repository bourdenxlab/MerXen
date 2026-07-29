# Stage 6 — Visualization

Produces PNG and PDF plots for either a paired dataset or a single-platform
dataset. In paired mode it writes gene-level scatter plots, combined QC plots,
a paired transcript overview, and a paired sanity-check image overlay. In
single-platform mode it writes one-dataset alternatives for the paired plots.

## What it does

1. In paired mode, rerun the gene comparison internally (re-opens the enriched zarrs) and
   plot log-log scatter plots of MERSCOPE vs Xenium normalized counts.
   In single-platform mode, plot top gene abundance for total and assigned
   normalized counts.
2. Recompute per-dataset QC metrics and plot either combined paired QC plots
   or one-platform geometry/cell-metric plots.
3. Plot a paired 3x2 or single-platform 3x1 transcript overview with density
   heatmaps, full-field scatter subsamples, and a micron crop.
4. Plot paired or single 250 um sanity crops with image backgrounds, all shape
   contours, and ProSeg assigned/unassigned transcripts.
5. Plot an assignment-rate bar chart for the available platform(s).

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
