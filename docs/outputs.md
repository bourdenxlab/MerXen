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
├── mecr_reference/
│   └── mecr_reference_out/
├── <pair_id_1>/
│   ├── merscope/
│   │   ├── spatialdata/
│   │   ├── segmentation/
│   │   ├── enrichment/
│   │   ├── compute_cortical_depth/
│   │   ├── distance_from_object/
│   │   ├── reseg/
│   │   │   └── qc/
│   │   └── original_seg/
│   │       └── qc/
│   ├── xenium/
│   │   ├── spatialdata/
│   │   ├── segmentation/
│   │   ├── enrichment/
│   │   ├── compute_cortical_depth/
│   │   ├── distance_from_object/
│   │   ├── reseg/
│   │   │   └── qc/
│   │   └── original_seg/
│   │       └── qc/
│   ├── alignment/
│   ├── alignment_qc/
│   ├── reseg/
│   │   ├── mecr/
│   │   ├── comparison/
│   │   ├── visualization/
│   │   ├── spatial_gene_analysis/
│   │   ├── clustering_squidpy/
│   │   └── mapmycells/
│   └── original_seg/
│       ├── mecr/
│       ├── comparison/
│       ├── visualization/
│       ├── spatial_gene_analysis/
│       ├── clustering_squidpy/
│       └── mapmycells/
├── <pair_id_2>/
│   └── ...
├── distance_from_object/
│   └── cohort/
│       ├── merscope/
│       └── xenium/
└── ...
```

`<pair_id>` comes straight from the `pair_id` column of the samplesheet. In
single-platform mode, only the selected `<platform>/` directory is present and
paired-only `alignment/`, `alignment_qc/`, and `comparison/` directories are
not written.
`reseg/` and `original_seg/` are controlled by `--analysis_segmentation`;
the default `both` writes both branches. Upstream build, segmentation,
enrichment, and latest SpatialData artifacts are shared.
Every `.png` plot listed below is also written as a same-stem `.pdf`.

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
| `latest_spatialdata.zarr` | Durable current SpatialData artifact. Segmentation writes the refined ProSeg result here, then enrichment updates it in place with additive shapes, images, and tables. This is the primary downstream input. |

### Segmentation

Path: `${outdir}/<pair_id>/<platform>/segmentation/`

| File | Contents |
|------|----------|
| `proseg_base_latest.zarr` | Staged symlink to `../latest/latest_spatialdata.zarr`. |
| `cellpose_masks_tiled.npy` | Cleaned global-pixel uint32 mask from tiled Cellpose. Fed into ProSeg and enrichment. |
| `cellpose_stitching_stats.json` | Diagnostics for object-level tile stitching, including accepted labels, duplicate skips, edge-touching labels, and conflict pixels. |
| `cellpose_nuclei_masks_tiled.npy` | DAPI-only Cellpose `nuclei` labels, retained independently of transcript assignment. |
| `cellpose_nuclei_stitching_stats.json` | Equivalent stitching diagnostics for the nuclei mask. |
| `transcripts_for_proseg.csv` | ProSeg input: per-transcript rows with seeded `cell_id`. Retained for debugging. |

### Enrichment

Path: `${outdir}/<pair_id>/<platform>/enrichment/`

| File | Contents |
|------|----------|
| `latest_input.zarr` | Staged symlink to `../latest/latest_spatialdata.zarr`. |
| `enrich_out/` | Assignment summary CSVs per shape (transcripts assigned, gene totals). |

### Mask Image Quantification

Path: `${outdir}/<pair_id>/<platform>/mask_image_quantification/`

| File | Contents |
|------|----------|
| `latest_input.zarr` | Staged symlink to `../latest/latest_spatialdata.zarr`, updated in place with `table_MOSAIK_cellpose_image_quantification`. |
| `mask_image_quantification_out/*_mask_image_quantification.parquet` | Wide Cellpose cell × image-channel-stat matrix. |
| `mask_image_quantification_out/*_mask_image_quantification_features.csv` | Feature metadata for image key, channel, and statistic. |
| `mask_image_quantification_out/*_mask_image_quantification_summary.json` | Summary of quantified images, cells, features, and sidecar paths. |

### Cortical Depth

Path: `${outdir}/<pair_id>/<platform>/compute_cortical_depth/`

Only present when `--cortical_depth_enabled true`.

| File | Contents |
|------|----------|
| `latest_input.zarr` | Staged symlink to `../latest/latest_spatialdata.zarr`, updated in place with cortical-depth columns unless disabled. |
| `compute_cortical_depth_out/cortical_ribbon_mask.tif` | Rasterized cortical ribbon mask. |
| `compute_cortical_depth_out/streamlines.geojson` | Pial-to-WM streamlines as GeoJSON LineStrings. |
| `compute_cortical_depth_out/streamlines.parquet` | Point-level streamline table with tangential position, thickness, and QC flags. |
| `compute_cortical_depth_out/depth_contours.geojson` | Laplace depth contours, usually 10%-90%. |
| `compute_cortical_depth_out/equivolumetric_depth_contours.geojson` | Equal-area/equivolumetric depth contours. |
| `compute_cortical_depth_out/<segmentation>/*_cells_with_cortical_depth.parquet` | Per-cell sidecar table with depth columns for each selected segmentation branch. |
| `compute_cortical_depth_out/*_cortical_depth_overlay.png` | QC overlay with pial, optional WM, ribbon, contours, and streamlines. PDF copy is also written. |
| `compute_cortical_depth_out/*_laplace_equivolumetric_difference.png` | Raster difference plot of `laplace_depth - equivolumetric_depth`. PDF copy is also written. |
| `compute_cortical_depth_out/<segmentation>/*_cells_laplace_depth.png` | Cells colored by `laplace_depth`. PDF copy is also written. |
| `compute_cortical_depth_out/<segmentation>/*_cells_equivolumetric_depth.png` | Cells colored by `equivolumetric_depth`. PDF copy is also written. |
| `compute_cortical_depth_out/<segmentation>/*_cells_tissue_annotation.png` | All cells colored as `grey_matter`, `white_matter`, `excluded`, or `outside_brain`. PDF copy is also written. |
| `compute_cortical_depth_out/cortical_depth_qc_summary.json` | Cell inside/outside counts, assigned counts, streamline thickness stats, failed/flagged streamlines, warnings. |

The updated AnnData `obs` columns include `inside_cortical_ribbon`,
`cortical_depth_annotation`, `laplace_depth`, `equivolumetric_depth`,
`distance_to_pia_um`, `distance_to_wm_um`, `streamline_thickness_um`,
`tangential_position_um`, `nearest_streamline_id`, `column_id`, and
`cortical_depth_qc_flag`.

### Distance from object

Per-block path:
`${outdir}/<pair_id>/<platform>/distance_from_object/`

Only present when `--distance_from_object_enabled true`.

| File | Contents |
|------|----------|
| `latest_input.zarr` | Durable zarr updated in place with nearest-object distance metadata on each selected table. |
| `distance_from_object_out/registered_object_annotations.geojson` | Validated, normalized copy of the already registered polygon input. |
| `distance_from_object_out/<segmentation>/cells_with_object_distance.parquet` | Per-cell coordinates, nearest object ID/type, unsigned/signed edge distance, inside flag, proximity bin, tissue annotation, and QC flag. |
| `distance_from_object_out/<segmentation>/pseudobulk_counts.h5ad` | Raw near/far gene sums for this `pair_id`, restricted to grey matter. |
| `distance_from_object_out/<segmentation>/pseudobulk_samples.csv` | Pseudobulk group and eligible-cell counts. |
| `distance_from_object_out/<segmentation>/cells_object_distance.png` | Spatial distance QC plot; PDF is also written. |
| `distance_from_object_out/<segmentation>/cells_object_proximity_counts.png` | Tissue-by-proximity counts; PDF is also written. |
| `distance_from_object_out/distance_from_object_summary.json` | Object, tissue, proximity, and pseudobulk counts for all branches. |

Cohort path:
`${outdir}/distance_from_object/cohort/<platform>/`

| File | Contents |
|------|----------|
| `distance_from_object_cohort_out/<segmentation>/paired_pseudobulk_counts.h5ad` | Complete near/far tissue-block pairs sharing a gene panel. |
| `distance_from_object_cohort_out/<segmentation>/paired_pseudobulk_samples.csv` | PyDESeq2 sample metadata. |
| `distance_from_object_cohort_out/<segmentation>/near_vs_far_differential_expression.csv` | Paired `~ pair_id + proximity` PyDESeq2 results. A Parquet copy is also written. |
| `distance_from_object_cohort_out/<segmentation>/near_vs_far_volcano.png` | Near-versus-far volcano plot; PDF is also written. |
| `distance_from_object_cohort_out/distance_from_object_cohort_summary.json` | Completion/skip status and complete block IDs for every branch. |

See [Distance from object](stages/distance-from-object.md) for the exact bins,
grey-matter filter, and segmentation mapping.

### QC

Path: `${outdir}/<pair_id>/<platform>/<analysis_segmentation>/qc/`

| File | Contents |
|------|----------|
| `qc_out/<dataset>_qc_summary.csv` | Single-row headline stats. |
| `qc_out/<dataset>_geometry_metrics.csv` | Per-cell geometry (area, perimeter, eccentricity, ...). |
| `qc_out/<dataset>_cell_metrics.csv` | Per-cell transcripts_per_cell, genes_per_cell. |
| `qc_out/<dataset>_qc.pkl` | Pickle with summary + DataFrames for fast reload. |

`<dataset>` is lowercased, e.g. `example01_merscope`.

### MECR

Shared reference path: `${outdir}/mecr_reference/mecr_reference_out/`

| File | Contents |
|------|----------|
| `mecr_reference_panel_genes.csv` | Union of spatial-panel genes requested from the complete WHB reference. |
| `mecr_reference_gene_statistics.csv` | Per-gene, per-broad-class detection fractions and Python Wilcoxon statistics, including threshold and uniqueness flags. |
| `mecr_reference_markers.csv` | Unique broad-class markers that pass the paper's strict detection thresholds. |
| `mecr_reference_pairs.csv` | Reference intersection, union, and MECR for every eligible cross-class marker pair. |
| `mecr_reference_distribution.png` | WHB reference pair-MECR histogram with descriptive mean and median lines; neither line is used for pair selection. |
| `mecr_reference_manifest.json` | Reference paths, normalization, thresholds, class/cell counts, Wilcoxon settings, and output paths. |

Per-branch path: `${outdir}/<pair_id>/<analysis_segmentation>/mecr/mecr_out/`

| File | Contents |
|------|----------|
| `<platform>/<sample_id>_mecr_pairs.csv` | Every eligible cross-class gene pair with single-gene detection counts, intersection, union, and MECR. |
| `<platform>/<sample_id>_mecr_summary.json` | Sample MECR, median pair rate, cell/panel counts, scored-pair count, and zero-union count. |
| `<pair_id>_mecr_pairs.csv` | Combined long pair table for all active platforms. |
| `<pair_id>_mecr_summary.csv` | One summary row per active platform for direct platform comparison. |
| `<pair_id>_mecr_distribution.png` | Pair-level MECR distributions by platform; a PDF copy is also written. |
| `<pair_id>_mecr_platform_comparison.png` | MERSCOPE-versus-Xenium scatter for exactly shared eligible pairs, with an identity line and the largest differences labelled. |
| `<pair_id>_mecr_class_pair_heatmap.png` | Per-platform heatmaps of median MECR for each broad-class pairing. |
| `<pair_id>_mecr_barnyard_pairs.csv` | Deterministic barnyard pair selection and its canonical, highest-MECR, or most-detected reason. |
| `plots/barnyard/<gene_1>--<gene_2>.png` | Side-by-side platform cell-count scatterplots for a selected pair. Coordinates use natural raw-count space by default; titles report exact all-cell MECR and intersection/union counts. |
| `<pair_id>_mecr_manifest.json` | Formula, aggregation and zero-union policies, reference marker path, sample summaries, and output paths. |

The sample-level metric is the unweighted mean of finite pair rates. Undefined
zero-union pairs remain in the pair table as NaN and are excluded from that
mean. See [Mutually exclusive co-expression rate](stages/mecr.md).

### Alignment

Path: `${outdir}/<pair_id>/alignment/`

Only present for paired rows whose effective `enable_alignment` value is `true`.

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

Only present for paired rows whose effective `enable_alignment` value is `true`.

| File | Contents |
|------|----------|
| `alignment_qc_out/<pair_id>_alignment_qc.json` | SABench-style grid metrics and centroid distance summary. |
| `alignment_qc_out/<pair_id>_alignment_qc_metrics.csv` | Single-row CSV with the same metrics. |
| `alignment_qc_out/<pair_id>_alignment_overlay.png` | Xenium/MERSCOPE centroid overlay after alignment. |

### Comparison

Path: `${outdir}/<pair_id>/<analysis_segmentation>/comparison/`

Only present in `--analysis_mode paired`.

| File | Contents |
|------|----------|
| `compare_out/<pair_id>_total_counts_compare.csv` | Gene × platform total counts. |
| `compare_out/<pair_id>_assigned_counts_compare.csv` | Gene × platform counts from the primary cell table. |
| `compare_out/<pair_id>_total_normalized_compare.csv` | CP10K-normalized total counts. |
| `compare_out/<pair_id>_assigned_normalized_compare.csv` | CP10K-normalized assigned counts. |
| `compare_out/<pair_id>_comparison_metrics.json` | Platform totals + log-log linear-fit metrics. |

### Visualization

Path: `${outdir}/<pair_id>/<analysis_segmentation>/visualization/`

| File | Contents |
|------|----------|
| `visualize_out/<pair_id>_gene_scatter_total_normalized.png` | MERSCOPE vs Xenium log-log scatter, all transcripts. |
| `visualize_out/<pair_id>_gene_scatter_assigned_normalized.png` | MERSCOPE vs Xenium log-log scatter, assigned transcripts only. |
| `visualize_out/<pair_id>_geometry_hist.png` | Overlaid Xenium/MERSCOPE step histograms of cell area, eccentricity, etc. |
| `visualize_out/<pair_id>_cell_violin.png` | Side-by-side platform violins for transcripts-per-cell and genes-per-cell. |
| `visualize_out/<pair_id>_transcript_overview.png` | 3x2 density, full scatter, and fixed crop transcript overview. |
| `visualize_out/<pair_id>_sanity_overlay.png` | Paired 250 um image crops with all shape contours and transcript assignment status. |
| `visualize_out/<pair_id>_sanity_overlay_crop_location.png` | Helper plot showing the MERSCOPE raw, MERSCOPE aligned, and Xenium crop locations used for the sanity overlay. |
| `visualize_out/<pair_id>_assignment_rate_bar.png` | Bar chart comparing `pct_assigned` across platforms. |

Single-platform runs write the available-platform equivalents with
`<sample_id>` prefixes, where `<sample_id>` is `<pair_id>_MERSCOPE` or
`<pair_id>_XENIUM`:

| File | Contents |
|------|----------|
| `visualize_out/<sample_id>_gene_abundance_total_normalized.png` | Top gene abundance for all transcripts. |
| `visualize_out/<sample_id>_gene_abundance_assigned_normalized.png` | Top gene abundance for assigned transcripts only. |
| `visualize_out/<sample_id>_geometry_hist.png` | Single-platform geometry histograms. |
| `visualize_out/<sample_id>_cell_violin.png` | Single-platform transcripts/cell and genes/cell violins. |
| `visualize_out/<sample_id>_transcript_overview.png` | 3x1 single-platform transcript density, full scatter, and crop overview. |
| `visualize_out/<sample_id>_sanity_overlay.png` | Single-platform 250 um sanity crop. |
| `visualize_out/<sample_id>_sanity_overlay_crop_location.png` | Crop-location helper for the single-platform sanity crop. |
| `visualize_out/<sample_id>_assignment_rate_bar.png` | Assignment-rate bar for the selected platform. |

### Spatial gene analysis

Path: `${outdir}/<pair_id>/<analysis_segmentation>/spatial_gene_analysis/`

| File | Contents |
|------|----------|
| `spatial_gene_analysis_out/<platform>/<pair_id>_<platform>_spatial_gene_autocorrelation.csv` | Full per-gene Moran's I and Geary's C table, with available normal and FDR-adjusted p-values. |
| `spatial_gene_analysis_out/<platform>/<pair_id>_<platform>_spatial_gene_autocorrelation_rankings.csv` | Top and bottom `spatial_gene_analysis_top_n` genes for Moran's I and Geary's C. |
| `spatial_gene_analysis_out/<platform>/plots/distributions/<pair_id>_<platform>_spatial_autocorrelation_distribution.png` | Histograms of Moran's I and Geary's C values across genes. |
| `spatial_gene_analysis_out/<platform>/plots/spatial_genes/<metric>/<top_or_bottom>/*.png` | Individual spatial expression plots for ranked genes. |
| `spatial_gene_analysis_out/<platform>/<pair_id>_<platform>_transcript_spatial_patterns.csv` | Per-gene nuclear/cytoplasmic/extracellular enrichment, signed-distance summaries, pair-band summaries, and pattern labels computed from transcript coordinates. |
| `spatial_gene_analysis_out/<platform>/<pair_id>_<platform>_transcript_signed_distance.parquet` | Long gene × boundary × signed-distance-bin enrichment table. |
| `spatial_gene_analysis_out/<platform>/<pair_id>_<platform>_transcript_pair_correlation.parquet` | Long gene × nested-null × distance-band table with thinning and empirical/FDR statistics. |
| `spatial_gene_analysis_out/<platform>/<pair_id>_<platform>_transcript_spatial_pattern_rankings.csv` | Separate rankings for compartment, proximity, distance-bin, and pair-pattern metrics. |
| `spatial_gene_analysis_out/<platform>/plots/transcript_patterns/*.png` | Tissue/local compartment maps with cell/nuclear outlines and distance/pair profiles. |
| `spatial_gene_analysis_out/<platform>/<pair_id>_<platform>_spatial_gene_analysis_manifest.json` | Parameters, retained cell/gene counts, and output path manifest. |

### Squidpy clustering

Path: `${outdir}/<pair_id>/<analysis_segmentation>/clustering_squidpy/`

| File | Contents |
|------|----------|
| `clustering_squidpy_out/<platform>/plots/qc/<pair_id>_<platform>_qc_histograms.png` | Histograms for transcripts/cell, genes/cell, cell area, nucleus ratio, and control/blank counts. |
| `clustering_squidpy_out/<platform>/<pair_id>_<platform>_qc_metrics.csv` | Per-cell QC metrics used for the histogram panel. |
| `clustering_squidpy_out/<platform>/plots/umap/<pair_id>_<platform>_umap.png` | Scanpy UMAP colored by total counts, genes by counts, and Leiden cluster. |
| `clustering_squidpy_out/<platform>/plots/spatial/<pair_id>_<platform>_spatial_scatter_leiden.png` | Squidpy spatial scatter colored by Leiden cluster, with clean axes and a 200 um scale bar. |
| `clustering_squidpy_out/<platform>/plots/spatial_grid/<pair_id>_<platform>_spatial_scatter_leiden_grid.png` | Small-multiple spatial grid with each de novo Leiden cluster highlighted in red against all other cells in grey. |
| `clustering_squidpy_out/<platform>/<pair_id>_<platform>_clustered.h5ad` | Control-feature-filtered, cell/gene-filtered, normalized, log-transformed, clustered AnnData object with raw non-control counts in `layers["counts"]`. |
| `clustering_squidpy_out/gpu_vram/<pair_id>_<analysis_segmentation>_summary.json` | Peak task-matched and total device VRAM sampled during the `CLUSTERING_SQUIDPY` task. |
| `clustering_squidpy_out/gpu_vram/<pair_id>_<analysis_segmentation>_samples.tsv` | Raw `nvidia-smi` GPU memory samples, including compute-app PID matches. |

By default, the same `<sample_id>_clustered.h5ad` path is still written and
remains the downstream MapMyCells input. In hierarchical mode, the H5AD also
includes `leiden_broad`,
`broad_atlas_label`, `broad_class`, `neuron_split_label`,
`subcluster_label`, and `hierarchical_cluster` in `obs`.

The stage also mutates each platform's
`${outdir}/<pair_id>/<platform>/latest/latest_spatialdata.zarr` by default,
adding or replacing the final clustered table for the active segmentation:
`table_MOSAIK_proseg_clustering_squidpy` for `reseg` and
`table_original_clustering_squidpy` for `original_seg`. Set
`--clustering_squidpy_write_spatialdata_table false` for H5AD-only output.

Additional QC artifacts are written under
`clustering_squidpy_out/<platform>/<sample_id>_hierarchical/`:

| File | Contents |
|------|----------|
| `<sample_id>_hierarchical_manifest.json` | Branch settings, output paths, and clustering status. |
| `<sample_id>_broad_cluster_annotation.csv` | Broad cluster atlas assignment, score, runner-up, margin, and marker count. |
| `<sample_id>_broad_annotation_scores.csv` | Cluster-by-atlas score table. |
| `<sample_id>_broad_resolved_markers.csv` | Panel-overlapping markers selected for each atlas label. |
| `plots/annotation/<sample_id>_broad_annotation_score_heatmap.png` | Broad annotation score heatmap. |
| `branch_<class>/...` | Per-branch H5AD plus UMAP, spatial scatter, spatial grid, and panel-gene dotplot in `plots/` subfolders for non-neuron broad classes and extra atlas classes. |
| `branch_<class>/tables/dotplot/...` | Mean-expression and fraction-expressing summaries used for branch panel-gene dotplots. |
| `branch_neurons/<sample_id>_neurons_split_*` | Neuron Excitatory/Inhibitory/Other annotation tables, heatmap, plots, and split H5AD. |
| `branch_neurons/split_<label>/...` | Per-neuron-split subtype H5AD plus UMAP, spatial scatter, spatial grid, panel-gene dotplot, and dotplot summary tables. |

### MapMyCells

Path: `${outdir}/<pair_id>/<analysis_segmentation>/mapmycells/`

| File | Contents |
|------|----------|
| `mapmycells_out/<platform>/<pair_id>_<platform>_mapmycells_query.h5ad` | Local mapper query AnnData with selected counts copied into `X`. |
| `mapmycells_out/<platform>/<pair_id>_<platform>_mapmycells.csv` | Per-cell MapMyCells assignments and confidence columns. |
| `mapmycells_out/<platform>/<pair_id>_<platform>_mapmycells_extended.json` | Full MapMyCells JSON result, including config, log, marker genes, and taxonomy tree. |
| `mapmycells_out/<platform>/<pair_id>_<platform>_mapmycells.log` | MapMyCells run log. |
| `mapmycells_out/<platform>/<pair_id>_<platform>_mapmycells_stdout.log` | Captured stdout from the local mapper process. |
| `mapmycells_out/<platform>/<pair_id>_<platform>_mapmycells_stderr.log` | Captured stderr from the local mapper process, including startup/import errors. |
| `mapmycells_out/<platform>/<pair_id>_<platform>_mapmycells_command.json` | Exact command invoked by the stage. |
| `mapmycells_out/<platform>/<pair_id>_<platform>_mapmycells_umap.png` | Existing Squidpy/Scanpy UMAP coordinates colored by MapMyCells assignment. |
| `mapmycells_out/<platform>/<pair_id>_<platform>_mapmycells_umap_cluster_by_supercluster/supercluster_<name>.png` | Per-supercluster UMAPs with cells outside the supercluster in grey and member cells colored by MapMyCells cluster. |
| `mapmycells_out/<platform>/<pair_id>_<platform>_mapmycells_spatial.png` | Spatial coordinates colored by MapMyCells assignment. |
| `mapmycells_out/<platform>/<pair_id>_<platform>_mapmycells_quality_scatter.png` | Extended-JSON QC panels for supercluster and cluster assignment quality. |
| `mapmycells_out/<platform>/<pair_id>_<platform>_mapmycells_supercluster_assignment_qc.png` | Supercluster cell counts, confidence summaries, and low-confidence fractions. |
| `mapmycells_out/<platform>/<pair_id>_<platform>_mapmycells_cluster_assignment_qc.png` | Cluster cell counts, confidence summaries, and low-confidence fractions. |
| `mapmycells_out/<platform>/<pair_id>_<platform>_mapmycells_spatial_supercluster_grid.png` | Small-multiple spatial grid with each supercluster highlighted in red against all other cells in grey. |
| `mapmycells_out/<platform>/<pair_id>_<platform>_mapmycells_annotated.h5ad` | Clustered AnnData with assignment columns added to `obs` using the `mapmycells_` prefix and mapper metadata in `uns["merxen_mapmycells"]`; plot paths are recorded, but plot images are separate PNGs. |
| `mapmycells_out/region_<region_name>/<platform>/<pair_id>_<platform>_mapmycells_*` | Region-specific MapMyCells outputs when `mapmycells_reference_mode` includes `region`; annotated H5AD columns use `mapmycells_region_<region_name>_`. |
| `mapmycells_out/<pair_id>_mapmycells_manifest.json` | Per-pair manifest summarizing selected reference mode, whole-brain and region references, ROI labels, filtering counts, bootstrap settings, and output paths. |

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
