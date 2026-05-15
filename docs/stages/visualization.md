# Stage 6 — Visualization

Produces a fixed set of PNG and PDF plots for a pair: gene-level scatter plots,
combined QC plots, a paired transcript overview, and a paired sanity-check
image overlay. Runs once per pair, after comparison has completed.

## What it does

1. Rerun the gene comparison internally (re-opens the enriched zarrs) and
   plot log-log scatter plots of MERSCOPE vs Xenium normalized counts.
2. Recompute per-dataset QC metrics and plot combined geometry histograms and
   cell metric violins across both platforms.
3. Plot a paired 3x2 transcript overview with density heatmaps, full-field
   scatter subsamples, and a fixed micron crop.
4. Plot paired 250 um sanity crops with image backgrounds, all shape contours,
   and ProSeg assigned/unassigned transcripts.
5. Plot an assignment-rate bar chart comparing the percentage of transcripts
   assigned across platforms.

## Nextflow process

[`VISUALIZE`](../../workflows/modules/visualization.nf) — one instance per
`pair_id`, downstream of `COMPARE`.

- **Input:** `tuple(pair_id, merscope_zarr, xenium_zarr)`.
- **CLI:** `merxen visualize --config visualize_config.json`.
- **Output:** `tuple(pair_id, visualize_out/)`.
- **publishDir:** `${outdir}/${pair_id}/visualization/` (copy mode).

## Python entry points

| Function | File |
|----------|------|
| CLI `visualize_command` | [cli/run_visualization.py](../../src/merxen/cli/run_visualization.py) |
| `plot_gene_scatter` | [visualization/gene_scatter.py:14](../../src/merxen/visualization/gene_scatter.py#L14) |
| `plot_geometry_histograms_comparison` | [visualization/qc_plots.py](../../src/merxen/visualization/qc_plots.py) |
| `plot_cell_metrics_violin_comparison` | [visualization/qc_plots.py](../../src/merxen/visualization/qc_plots.py) |
| `plot_assignment_bar` | [visualization/qc_plots.py](../../src/merxen/visualization/qc_plots.py) |
| `plot_transcript_overview` | [visualization/density_overview.py](../../src/merxen/visualization/density_overview.py) |
| `plot_pair_sanity_crops` | [visualization/sanity_plots.py](../../src/merxen/visualization/sanity_plots.py) |

## Config schema

`VisualizationConfig` — [config.py:246](../../src/merxen/config.py#L246).

| Field | Description |
|-------|-------------|
| `merscope_zarr_path` | Enriched MERSCOPE zarr. |
| `xenium_zarr_path` | Enriched Xenium zarr. |
| `output_dir` | Where `visualize_out/` is populated. |
| `pair_id` | Prefix for output filenames. |

## Outputs

Written under `visualize_out/`:

Each listed `.png` plot is also written as a same-stem `.pdf`.

| Kind | File | Contents |
|------|------|----------|
| Gene scatter | `<pair_id>_gene_scatter_total_normalized.png` | MERSCOPE vs Xenium, all transcripts (normalized). |
| Gene scatter | `<pair_id>_gene_scatter_assigned_normalized.png` | MERSCOPE vs Xenium, transcripts assigned to cells. |
| Geometry | `<pair_id>_geometry_hist.png` | Overlaid step histograms of area, eccentricity, etc. |
| Cell metrics | `<pair_id>_cell_violin.png` | Platform violins for transcripts/cell and genes/cell on log y axes. |
| Transcript overview | `<pair_id>_transcript_overview.png` | 3x2 density, full scatter, and fixed crop transcript overview. |
| Sanity overlay | `<pair_id>_sanity_overlay.png` | Paired 250 um image crops with shape contours and assignment status. |
| Sanity crop helper | `<pair_id>_sanity_overlay_crop_location.png` | MERSCOPE raw, MERSCOPE aligned, and Xenium crop locations used for the sanity overlay. |
| Assignment rate | `<pair_id>_assignment_rate_bar.png` | Bar chart of `pct_assigned` per platform. |

## Notes

- The visualization stage does **not** read the CSVs produced by the
  comparison stage; it recomputes them. This keeps stages independent but
  means large zarrs are opened twice per run.
- The sanity overlay prefers `MERSCOPE_z_projection` and `morphology_focus`
  image layers, uses `MOSAIK_proseg` as the assignment shape layer, and draws
  ProSeg, Cellpose-SAM, and the platform's original segmentation. When MERSCOPE
  aligned vectors are available, the crop is selected in aligned Xenium space
  and then rendered in raw MERSCOPE image space so the image, transcripts, and
  boundaries stay registered.
- Points coordinate columns are resolved with `first_existing_col` across
  `x`, `x_micron`, `x_location`, `global_x`, `x_global_px`, `observed_x`
  (and the corresponding `y_*`) for transcript plotting.
