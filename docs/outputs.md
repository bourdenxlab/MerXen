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
│   ├── alignment_materialization/
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
paired-only `alignment/`, `alignment_materialization/`, `alignment_qc/`, and
`comparison/` directories are not written.
`reseg/`, `original_seg/`, `proseg_mask/`, and `proseg_hybrid/` are controlled
by `--analysis_segmentation`; the default `all` writes all four branches, while
an explicit `both` writes only `reseg` and `original_seg`.
Upstream build, segmentation,
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

#### SpatialData identifier contract

MerXen writes a versioned `merxen_schema` root attribute. Its
`segmentations` registry explicitly pairs each branch's points element and
assignment column with its shape, table, instance key, coordinate variant, and
ID namespace. Consumers must use this registry rather than selecting the first
points element or inferring pairings from element names.

Operational cell IDs use the positive `uint64` field `instance_id` everywhere:
the point assignment, shape column and index, and table instance key have the
same value and dtype. Raster label value `0` is reserved for background and a
nullable point assignment uses null for unassigned. Source/vendor IDs remain in
`source_cell_id`; ProSeg's zero-based ID remains in `proseg_internal_id`.
`transcript_id` is a unique positive `uint64`, while a reader-provided ID is
retained separately as `source_transcript_id`.

The primary assignments are:

| Registry branch | Point assignment | Shape | Table |
|-----------------|------------------|-------|-------|
| `original` | `original_assignment` when supplied by the instrument | vendor cell-boundary layer | `table` / `table_original` |
| `proseg` | `assignment` | `MOSAIK_proseg` | `table_MOSAIK_proseg` |
| `cellpose` | geometric table only | `MOSAIK_cellpose` | `table_MOSAIK_cellpose` |
| `proseg_geometry_assignment` | `proseg_geometry_assignment` when present | `MOSAIK_proseg` | `table_MOSAIK_proseg_geometry_assignment` |
| `proseg_hybrid` | `hybrid_assignment` | `MOSAIK_proseg_hybrid` | `table_MOSAIK_proseg_hybrid` |

Aligned non-rigid variants receive their own registry entries. They refer to
the aligned points and shapes explicitly and do not silently reuse a native
coordinate-space pairing.

SpatialData 0.8 stores point payloads as Parquet inside a Zarr metadata group.
A generic Zarr traversal can therefore emit
`Object at points.parquet is not recognized as a component of a Zarr
hierarchy`. Use `spatialdata.read_zarr()` to read the store; the Parquet child
is expected and is not, by itself, evidence of corruption.

### Segmentation

Path: `${outdir}/<pair_id>/<platform>/segmentation/`

| File | Contents |
|------|----------|
| `proseg_base_latest.zarr` | Staged symlink to `../latest/latest_spatialdata.zarr`. |
| `cellpose_masks_tiled.npy` | Cleaned global-pixel uint32 mask from tiled Cellpose. Fed into ProSeg and enrichment. |
| `cellpose_cellprobs_tiled.npy` | Float32 Cellpose probability logits aligned with accepted mask pixels and fed into ProSeg. |
| `cellpose_stitching_stats.json` | Diagnostics for object-level tile stitching, including accepted labels, duplicate skips, edge-touching labels, and conflict pixels. |
| `cellpose_nuclei_masks_tiled.npy` | DAPI-only Cellpose `nuclei` labels, retained independently of transcript assignment. |
| `cellpose_nuclei_stitching_stats.json` | Equivalent stitching diagnostics for the nuclei mask. |
| `transcripts_for_proseg.csv` | ProSeg input: per-transcript rows with seeded `cell_id`. Retained for debugging. |
| `transcripts_for_proseg.transforms.json` | Pixel-to-micron affine that seeded `transcripts_for_proseg.csv`. A re-run refuses to reuse the CSV when it differs from the current transform. |

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
| `mask_image_quantification_out/*_mask_image_quantification_summary.json` | Summary of quantified images, cells, features, sidecar paths, and the identifier-based hybrid-cell join. |

When the hybrid table exists, Cellpose measurements are joined to
`table_MOSAIK_proseg_hybrid` by the preserved Cellpose `instance_id`. The
hybrid expression matrix is unchanged; image features are stored in prefixed
`obs` columns and `.obsm["cellpose_image_quantification"]`.

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
| `compute_cortical_depth_out/cortical_depth_qc_summary.json` | Cell inside/outside counts, assigned counts, streamline thickness stats, failed/flagged streamlines, warnings (including `frame_mismatch_suspected:<segmentation>`), and per table the boundary frame, the cell coordinate source and the frame check `edge_to_nearest_cell_median_um` / `ribbon_bin_occupancy` ([Coordinate frame](stages/cortical-depth.md#coordinate-frame)). |

The updated AnnData `obs` columns include `inside_cortical_ribbon`,
`cortical_depth_annotation`, `laplace_depth`, `equivolumetric_depth`,
`distance_to_pia_um`, `distance_to_wm_um`, `streamline_thickness_um`,
`tangential_position_um`, `nearest_streamline_id`, `column_id`, and
`cortical_depth_qc_flag`; the table's `uns['cortical_depth']` records the
boundary frame and cell coordinate source they were computed from.

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
| `qc_out/<dataset>_registration_qc.json` | Segmentation-to-transcript registration check (status, density ratio, offset against the platform's cells). See [stages/qc.md](stages/qc.md#registration-check). |
| `qc_out/<dataset>_hybrid_cell_diagnostics.csv` | Hybrid construction diagnostics plus per-cell Cellpose/ProSeg count and area changes. Hybrid branch only. |
| `qc_out/<dataset>_hybrid_assignment_sources.csv` | Hybrid transcript assignment-provenance counts and percentages. Hybrid branch only. |
| `qc_out/<dataset>_hybrid_fallback_reasons.csv` | Cellpose-fallback reason counts and percentages. Hybrid branch only. |
| `qc_out/<dataset>_hybrid_gene_count_changes.csv` | Per-gene hybrid totals and changes relative to Cellpose and ProSeg. Hybrid branch only. |
| `qc_out/<dataset>_hybrid_area_growth_map.png` | Spatial hybrid area-growth diagnostics. Hybrid branch only. |
| `qc_out/<dataset>_hybrid_rejected_transcripts_map.png` | Spatial rejected-external-transcript diagnostics. Hybrid branch only. |

For `proseg_hybrid`, QC additionally writes per-cell Cellpose/ProSeg count and
area comparisons, assignment-provenance counts, gene-count changes, and spatial
maps of area growth and rejected external transcripts.

`<dataset>` is lowercased, e.g. `example01_merscope`.

### MECR

Shared reference path: `${outdir}/mecr_reference/mecr_reference_out/`

| File | Contents |
|------|----------|
| `mecr_reference_panel_genes.csv` | Union of spatial-panel genes requested from the complete species-matched whole-brain reference. |
| `mecr_reference_gene_statistics.csv` | Per-gene, per-broad-class detection fractions and Python Wilcoxon statistics, including threshold and uniqueness flags. |
| `mecr_reference_markers.csv` | Unique broad-class markers that pass the paper's strict detection thresholds. |
| `mecr_reference_pairs.csv` | Reference intersection, union, and MECR for every eligible cross-class marker pair. |
| `mecr_reference_distribution.png` | Whole-brain reference pair-MECR histogram with descriptive mean and median lines; neither line is used for pair selection. |
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
| `align_out/alignment_transform.json` | Selected backend and mode, physical-coordinate transform, parameters, QC, and dependency versions. |
| `align_out/transform_chain.json` | Explicit moving-physical → image → registration → fixed-physical matrix chain. |
| `align_out/forward_displacement_field.npz` / `backward_displacement_field.npz` | Reloadable sampled non-rigid fields, when available. |
| `align_out/registration_summary.json` / `.csv` | DAPI QC, selection status, thresholds, and attempt summaries. |
| `align_out/resume_manifest.json` | Input paths, platform roles, and parameters used to validate direct-run resume. |
| `align_out/shared_tissue_mask.npy` / `.tif` | Intersection of fixed and registered-moving tissue on the original fixed DAPI grid. |
| `align_out/shared_tissue_mask_registration.npy` / `.tif` | Equivalent mask on the padded registration grid used for coordinate-domain annotation. |
| `align_out/registration_inputs/` / `registration_images/` | Halo-suppressed DAPI images, acquired-support/tissue/validity masks, support-boundary metrics, the locked shared-tissue mask and feather, and stable VALIS inputs. |
| `align_out/valis/` / `qc/` | Locked non-rigid VALIS artifacts plus partial-overlap candidates, seed-labelled physical objective plots and local score slice, overlays, checkerboards, feature, mask, displacement, and deformation diagnostics. |
| `align_out/alignment_coords/` | Legacy coordinate diagnostics; retained as an empty contract directory for VALIS. |

`alignment_materialization/materialization_summary.json` reports `complete` for
a rebuild or `current` for a validated idempotent reuse. `ALIGN` only computes
and embeds the transform; `MATERIALIZE_ALIGNMENT` updates the existing
MERSCOPE latest Zarr in place. Native elements remain untouched. Derived points,
shapes, and tables use `*_aligned_nonrigid`; labels use
`<aligned_shape>_labels`; and the full multichannel image is
`MERSCOPE_z_projection_aligned_nonrigid`. Pyramids and outlines use the normal
viewer-cache naming convention in `merxen_xenium`.

The MERSCOPE root `merxen_alignment` attribute is a version-2 manifest with
fixed-grid geometry, transform/native fingerprints, mappings, dtype/channel
metadata, artifact QC, and completion status. `_merxen_alignment/` holds a
portable transform copy. The Xenium root contains a matching
`merxen_alignment_pair_reference`; consumers should reject pair or transform
fingerprint mismatches.

### Alignment QC

Path: `${outdir}/<pair_id>/alignment_qc/`

Only present for paired rows whose effective `enable_alignment` value is `true`.

| File | Contents |
|------|----------|
| `alignment_qc_out/<pair_id>_alignment_qc.json` | Collated DAPI overlap, similarity, feature, affine, and deformation QC. |
| `alignment_qc_out/<pair_id>_alignment_qc_metrics.csv` | Single-row CSV with the same metrics. |
| `alignment_qc_out/<pair_id>_alignment_overlay.png` | Selected Xenium/MERSCOPE DAPI registration overlay. |

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
A `map_first` run of a species that has not flipped (mouse, until M9)
writes the same key with the suffix `_mapfirst` (e.g.
`table_MOSAIK_proseg_clustering_squidpy_mapfirst`;
`clustering_squidpy_table_key_suffix`) and never the legacy table; a human
`map_first` run writes the unsuffixed key since the human flip (M8) unless a
suffix is given; its
clustered H5AD records the suffix, the MENDER policy and the pair's
cross-platform scope in `uns["merxen_hierarchical_clustering"]`
([Map-first clustering runs](stages/annotation.md#map-first-clustering-runs-m5)).

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

### MENDER spatial domains

Path: `${outdir}/<pair_id>/<segmentation>/mender/mender_out/<platform>/`

| File | Contents |
|------|----------|
| `<sample_id>_mender_annotated.h5ad` | Original clustered H5AD plus categorical `obs["mender_domain"]` and `uns["merxen_mender"]`. |
| `<sample_id>_mender_context.h5ad` | MENDER context representation and context embedding; this potentially large matrix is not imported into SpatialData. |
| `<sample_id>_mender_cells.parquet` | Immutable cell IDs, native coordinates, input state, and MENDER domain. |
| `input/input_manifest.json` | Source H5AD/table/native shape, state counts, coordinate range, and exact MENDER settings. |
| `tables/domain_sizes.tsv` | Domain sizes, state entropy, dominant composition, and number of states present. |
| `tables/state_by_domain.tsv` | Long cell-state-by-domain counts and within-domain fractions. |
| `tables/scale_neighbour_summary.tsv` | Minimum, median, mean, and maximum neighbour counts at every scale. |
| `plots/spatial_domains.{png,pdf}` | Native-coordinate domain map. |
| `plots/context_umap.{png,pdf}` | MENDER context embedding colored by domain. |
| `plots/state_domain_heatmap.{png,pdf}` | Cell-state composition heatmap by domain. |
| `mender_manifest.json` | Standalone artifact and provenance manifest. |
| `spatialdata_import_manifest.json` | Shared-lock SpatialData import record and domain counts. |

Only `mender_domain` and `uns["merxen_mender"]` are added to the derived
clustered SpatialData table. Existing clustering, GASTON, cortical-depth, and
MapMyCells annotations are preserved.

### MapMyCells

Path: `${outdir}/<pair_id>/<analysis_segmentation>/mapmycells/`

| File | Contents |
|------|----------|
| `mapmycells_out/<platform>/<pair_id>_<platform>_mapmycells_query.h5ad` | Local mapper query AnnData with selected counts copied into `X`. |
| `mapmycells_out/<platform>/<pair_id>_<platform>_mapmycells_query_gene_ids.json` | Gene-ID resolution report: IDs recovered by the cached reference lookup or `annotation_gene_id_fallback_csv`, and the features left unresolved with their reasons. |
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
| `mapmycells_out/<pair_id>_mapmycells_manifest.json` | Per-pair manifest summarizing selected reference mode, whole-brain and region references, ROI labels, filtering counts, bootstrap settings, output paths, the gene-ID fallback table used (`gene_id_fallback_csv`, or `gene_id_fallback_ignored` when it has no IDs of the query species), and the per-sample `gene_id_resolution` summaries. |

### Annotation reference bundles (in development)

Written only by `--annotation_prepare_only` runs (panel and bundles) and by
`map_first` rows (human by default since M8; mouse with
`--clustering_squidpy_mode map_first` or the row column); a run whose rows
are all legacy writes none of these directories. See
[Reference-based annotation](stages/annotation.md#pipeline-processes).

| Path | Contents |
|------|----------|
| `<pair_id>/<segmentation>/annotation_panel/annotation_panel_out/panel_report.json` | Gene-ID resolution by source, unresolved features, controls removed, panel mode, set c and panel families (`ANNOTATE_PANEL`). |
| `<pair_id>/<segmentation>/annotation_panel/annotation_panel_out/panel_genes*.json` | One declared annotation panel per file: sorted Ensembl IDs, symbols, platforms, `panel_hash`. |
| `<pair_id>/<segmentation>/annotation_panel/annotation_panel_out/required_bundles.json` | The (reference, panel) bundles this pair × segmentation needs, with `n_required`. |
| `<pair_id>/<segmentation>/annotation_panel/annotation_panel_out/annotation_config.json` | The annotation config both commands read, as written from the pipeline params. |
| `annotation_reference_prep/<reference_id>/<panel_hash or panel_independent>/bundle_ref.json` | The bundle `ANNOTATE_REFERENCE_PREP` got or built: `build_hash`, bundle `path` and store root (identical bytes on every re-run of an unchanged bundle; the task log says whether it was built or reused). |
| `<pair_id>/<segmentation>/annotation_map/annotation_map_out/<platform>/<sid>_mmc_<run_id>.parquet` | `CLUSTERING_SQUIDPY_ANNOTATE_MAP` (`map_first` only): one row per mapped table cell × taxonomy level of one MapMyCells run (assignment, name, bootstrap and aggregate probability, correlation, five runner-ups); `run_id` is the reference id, `+_setc` for set c, `+_xpanel` for a `per_platform` pair's intersection run. Run metadata (`build_hash`, query fingerprint, engine parameters, ctm version, tidy schema version) in the parquet schema. |
| `<pair_id>/<segmentation>/annotation_map/annotation_map_out/<platform>/<sid>_ct_provisional.parquet` | One row per segmented object: identity, `total_counts`, `n_genes`, `in_table`, **provisional** raw-threshold `ct_<level>_*` and `ct_final_*` labels, and the raw engine columns `mmc_<reference>_<level>_*`. Marked provisional in the name and the parquet metadata; never feeds the clustered H5AD (RESOLVE, M4, replaces it). |
| `<pair_id>/<segmentation>/annotation_map/annotation_map_out/<platform>/<sid>_mouse_regions.parquet` | Mouse only: the region step's per-cell columns, one row per table cell (`tile_i`, `tile_j`, `tile_region`, `inferred_region`, `confident_neuron`, `region_dropped_level`, `region_pruned_changed`; [Mouse region step](stages/annotation.md#mouse-region-step-m6)). |
| `<pair_id>/<segmentation>/annotation_map/annotation_map_out/<platform>/<sid>_mmc_wmb_panel_pruned.parquet` | Mouse only, when nodes were dropped: the tidy table of the re-mapped cells (`--nodes_to_drop`); the unpruned run keeps `<sid>_mmc_wmb_panel.parquet`, and RESOLVE merges the two. |
| `<pair_id>/<segmentation>/annotation_map/annotation_map_out/<platform>/<sid>_mmc_<run_id>.extended.json.gz` | The extended MapMyCells JSON, only with `annotation_keep_extended_json`. |
| `<pair_id>/<segmentation>/annotation_map/annotation_map_out/map_manifest.json` | Per sample and run: input identity, table cells, controls removed, query fingerprint, bundle `build_hash` and lookup digests, engine parameters, ctm version and commit, wall time, peak RSS, reuse provenance (`reused`, `reused_from`, `same_mapping_as`); `panel_status` (`refused` with `panel_reasons` and no runs). The reuse source of the next run (`annotation_reuse_published`). |
| `<pair_id>/<segmentation>/annotation_map/annotation_map_out/logs/`, `annotation_config.json` | Mapper stdout / stderr / ctm logs per run; the annotation config the task read. |
| `<pair_id>/<segmentation>/annotation_resolve/annotation_resolve_out/<platform>/<sid>_celltype_labels.parquet` | `CLUSTERING_SQUIDPY_ANNOTATE_RESOLVE` (`map_first` only): the label table of every segmented object (identity, counts, `in_table`, per-level names, raw scores, runner-ups, margins and statuses, final label and tier, consensus, flags with their continuous companions, soft broad vectors, the raw engine columns); the provenance JSON in the parquet schema (`merxen_annotation`). |
| `<pair_id>/<segmentation>/annotation_resolve/annotation_resolve_out/<platform>/<sid>_annotation_manifest.json` | The sample's `AnnotationProvenance`: panel and trust (human: with `real_data_qc`, the real-data QC outcome per check, the downgrades applied and whether the checks only warned), references and bundle hashes, engine, resolvability (emitted bins, extrapolated bins, resolvable share), thresholds and floor sources, gate, consensus, flags and composition. |
| `<pair_id>/<segmentation>/annotation_resolve/annotation_resolve_out/<pair_id>_resolve_summary.json`, `annotation_config.json` | Per sample: trust, degraded mode, gate level and warning, the segmented objects the warning is over (`n_segmented`, with `n_segmented_source` `given` or `objects`), confident share per level of table cells (`confident_share_table`) and of segmented objects (`confident_share_segmented`), resolvable share per level, consensus tiers, realised flag rates, the restricted-lookup record and `resolvability_inherited`, the sample's own-panel composition (`composition_run`), and for a resolvability version-7 bundle `resolvability_v7` (the emission members, the applied decisions' bins by route, the class-depth prediction per level and class at the sample's own per-class depth, and the `flag_nonneuronal_high_depth` counts), and for a human sample `real_qc` (the downgrade-only real-data QC of plan §8.8: `enabled`, the outcome per check `per_check` — `pass`, `warn`, `fail`, `not_applicable` or `not_evaluable` —, every outcome with its numbers, the effects and downgrades, `seeded` and `warn_only`, the flag-rate readings and the per-check tables; with `real_qc.enabled` false only `enabled`); per pair: the composition JSD with block-bootstrap CIs, `cross_platform` (which run fed it, `jsd_run`, and the statistics level; capped at `broad_only` with reason `real_qc:paired_concordance` when paired concordance withholds the supercluster-level statistics) and, for a human pair, `real_qc_config`. `schema_version` 3 (2 before the real-data QC). A `per_platform` pair's JSD and compositions come from the WHB runs on the intersection panel (`_xpanel`; kinds soft, soft ≥ 30 counts and argmax: the own-panel confident labels rest on different gene sets), and are `broad_only` with `flag` and `reasons` when the intersection has fewer than 100 genes or is broad-only by resolvability (plan §8.5); its per-sample compositions and `soft_broad_*` stay on the own panel and are not compared across platforms. Deterministic for given inputs (no clock, wall time or absolute input path), so a deep-cached reader re-runs only when a label changes. The annotation config the task resolved with. |
| `<pair_id>/<segmentation>/annotation_report/annotation_report_out/` | `ANNOTATION_REPORT` (map_first runs, after FINALIZE, cortical depth and MENDER when they run) / `merxen annotation-report` (M7): `report.html` (static), `figures/<item>_<name>.{png,pdf,csv}` (one CSV per figure), `tables/<item>__<name>.csv`, `<pair_id>_platform_gene_factors.csv`, `acceptance_metrics.json` (every measured metric with its §14 criterion and the provenance footer; deterministic) and `report_run.json`. See [the annotation report](stages/annotation.md#annotation-report-merxen-annotation-report-m7). |
| `<pair_id>/<segmentation>/annotation_resolve/annotation_resolve_run.json` | The RESOLVE task's run record: `created_at`, `wall_time_s`, the absolute MAP manifest and output paths, the summary's sha256. Published beside `annotation_resolve_out`, never staged by a downstream task (`merxen annotate-resolve --run-record`; the standalone default is `<out>/<pair>_resolve_run.json`). |

The bundles themselves live in the reference store
(`annotation_reference_store`, default `${outdir}/annotation_references`),
not under these directories, and are never deleted by the pipeline.

#### Gate-P outputs (M13)

Gate P is a standalone acceptance programme, not a pipeline stage:
`merxen annotation-panel-simulate --gate-p` or
`scripts/acceptance/new_panel.py`
([Gate P](stages/annotation.md#gate-p-simulation-based-validation-of-a-panel-family-m13)).
It writes the base simulation's files into `--out-dir`
(`simulate_report.json`, with a `gate_p` block that records a refusal or
failure, `SIMULATE_REPORT.txt` and the rest of the
[panel simulation](cli.md#merxen-annotation-panel-simulate) outputs), and its own
files into `<out-dir>/gate_p/`. Its bundles go to the separate gate-P store it
is pointed at, never to the production store.

| Path (under `<out-dir>/gate_p/`) | Contents |
|------|----------|
| `gate_p_run.json` | The run record: `status` (`scored`, or `stopped` before any leave-one-donor-out build, with `stop_reasons`: `pool_sizes` and / or `np2_unaccepted_parents`), NP2's weak and collapsed parents against the accepted entries (`np2_parents`: each parent's key and name, the accepted and unaccepted ones, the entries matching none), the code that ran (`code_commit`, `code_commit_source`: as `simulate_report.json` records them), `family_id`, `passes`, `reasons`, `validated_max_level`, the dry-run verdict (`dry_run`, with `--gate-p-dry-run`), the times (`measured_seconds`, `prep_seconds`, `gate_p_seconds`, `leave_one_donor_out_seconds`, `replicate_seconds`), `simulated_cells` (with `simulated_cells_this_run` and `simulated_cells_prep_rows`: NP9's version-7 scaling basis, M13 D10 (a)), the time reference and its basis, the process-tree peak memory, each replicate's record, the identity re-runs (`replicate_identity`, `prep_identity`), NP5's depth source, the NP6 factor report, the readings the run took (`open_readings`) and the written files. |
| `gate_p_pool_sizes.csv` | Per extra donor and judged class: its frontal test cells, its own and every donor's admissible other-region cells, the cells the D1 drop removes, and the default test set's count (D2; written before any leave-one-donor-out build). |
| `gate_p_report.json`, `GATE_P_REPORT.txt` | The NP1–NP9 report: C_P and the excluded classes per level, the per-(level, class) records, `validated_max_level`, the family checks (NP1, NP2, NP8, NP9) and the headline verdict. |
| `gate_p_class_records.csv` | One row per (level, class): `in_class_set`, status (`validated`, `failed:NP<k>` or `not_evaluable`), every failing criterion and member, `validated_min_depth` and `tested_max_depth` (version 7: with each member's values). |
| `gate_p_class_sets.csv`, `gate_p_level_walk.csv` | C_P per level with each class's test cells, share and reference share; the coarse-to-fine level walk (`share_ok`, the classes validated and not, `complete`) that gives `validated_max_level`. |
| `<table>__<member>.csv` | Per emission member (`r1_contam_ho_seed0`, …): `tested_sets`, `np3_verdicts`, `np3_depths`, `np4_stats`, `np4_sets`, `np4_seed`, `np5_agreement`, `np5_thresholds`, `np5_spread`, `np5_extrapolated`, `np5_class`, `np6_verdicts`, `np7_wrong_node` and `np7_excluded` (human), and with `--gate-p-x1-factors` `np6_factors`. Also `np5_ensemble_agreement.csv` (version 7, report only) and `pool_sizes.csv`. |
| `replicates/<donor>/seed<k>/<member>.parquet` | Each replicate's gate-P rows, written as soon as it is mapped (base members at every seed, stress members and the clean upper bound at seed 0). A rerun into a directory that holds replicates is refused. |

A family that passes is promoted only by its gate-P PR, which adds rows to
the packaged tables in `src/merxen/assets/annotation/`
(`diagnostics.write_simulation_family`): one `validated_panels.csv` row with
`validation_basis = simulation`, the family's `validated_panel_genes.csv`
rows, and its `validated_panel_levels.csv` rows (`family_id`, `panel_hash`,
`level`, `class`, `in_class_set`, `status`, `validated_min_depth`,
`tested_max_depth`, `evidence`). RESOLVE then records the family's validation
basis and table digests in each sample's provenance (`panel`).

#### Label table schema (`<sid>_celltype_labels.parquet`)

One row per segmented object of the sample, in the order of its prepared
H5AD; `merxen.annotation.schema` holds the contract (`column_specs`,
`validate_label_table`) and every reader uses it. The provenance JSON
(`AnnotationProvenance`, the same as `<sid>_annotation_manifest.json`) is in
the parquet schema metadata under `merxen_annotation`.
`merxen.annotation.pipeline.read_label_table` returns both.

| Columns | Type | Meaning |
|---|---|---|
| `cell_id`, `instance_id` | string, int64 | Object id (the H5AD `obs_names`) and segmentation instance id. |
| `pair_id`, `sample_id`, `platform`, `segmentation`, `species`, `anatomical_region`, `panel_hash` | category | Identity; `panel_hash` is the declared panel the sample was mapped on. |
| `total_counts`, `n_genes`, `genes_per_count` | int32, int32, float32 | Counts after control removal, detected genes, and their ratio. |
| `in_table` | bool | Whether the object is a table cell (`total_counts >= min_counts` of the clustering run, or a cell of a published clustered table). Objects outside the table are `low_counts` at every level. |
| `depth_bin`, `resolvability_extrapolated` | nullable int32, bool | The cell's depth-grid bin of the resolvability tables, and whether its emission decision was carried from a pooled or deeper bin (§8.3). |
| `n_missing_panel_genes` | int32 | Panel genes the sample lacks. |
| `ct_<level>_name` | category | The assigned name at the level (null when the status is `low_counts`, `not_applicable` or `not_attempted_gate`). Levels: human `lineage`, `broad`, `nt`, `supercluster`, `seaad_subclass` (`cluster` only with `annotation_allow_fine_levels`); mouse `broad`, `class`, `nt`, `subclass`. |
| `ct_<level>_raw`, `ct_<level>_conf` | float32 | The bootstrap probability (lineage, broad, NT: summed over the class's superclusters; SEA-AD subclass: `aggregate_probability`) and the confidence, equal to raw in v1 (raw thresholds; calibrated probabilities are v1.1). NaN where the level is `low_counts` or `not_applicable`. |
| `ct_<level>_corr`, `ct_<level>_runner_up`, `ct_<level>_margin` | float32, category, float32 | MapMyCells `avg_correlation`, best runner-up and the margin of the raw probability to it (SEA-AD subclass: on the `aggregate_probability` scale, the runner-up's aggregate being its bootstrap probability times the parent class's aggregate; never above raw). |
| `ct_<level>_status` | category | One `CellStatus` value (below). |
| `ct_<level>_validated` | bool | Confident and inside the panel family's validated region (level, class, depth); always `false` in a provisional family, whose labels carry the banner. |
| `ct_final_level`, `ct_final_name` | category | The deepest confident level of the chain and its name (`none` / `Mixed/Unknown` when no level is confident). |
| `ct_consensus_tier` | int8 | Agreement of the informative methods' 7-class calls (a call is informative when it meets its threshold): the size of the largest agreeing group, 1 when only one method is informative, **0 only for a confident disagreement** (two or more informative, all different), **-1 when no method is informative** (and outside the table, or without the primary); capped by the degraded mode (v1 `whb_sea`: 2; `whb_only`: 1). Read tier 0 as disagreement and -1 as "no evidence"; agreement, not accuracy. |
| `ct_branch`, `ct_leaf`, `ct_mender_state` | category | The map-first hierarchy keys: branch (human: `Neurons/Excitatory`, `Neurons/Inhibitory`, `Neurons/unresolved`, the glial and vascular classes, `Oligodendrocyte lineage/unresolved`, `Mixed/Unknown`), leaf (the confident supercluster or `unresolved`) and the MENDER state. |
| `flag_low_counts`, `flag_below_floor`, `flag_method_disagree`, `flag_implausible`, `flag_cop_suppressed` | bool | Status-mirroring flags: outside the table; the deepest attempted level of the chain is `below_floor`, `method_disagree` or `implausible` (H2 counts `flag_implausible` over table cells). `flag_cop_suppressed` (plan §4.3): WHB called COP, the lineage is confident and the COP rule failed (fewer than 120 counts or supercluster raw below 0.69, and no confident SEA-AD OPC call), whichever broad check decides the status: the broad floor, resolvability and the threshold come first, and the COP rule itself gives `below_floor` under 120 counts and `low_confidence` otherwise. Report only; statuses and labels do not depend on it. |
| `contamination_score`, `neg_counts`, `flag_contaminated` | float32, nullable int32, nullable bool | Share and count of the cell's counts on its assigned class's negative genes; the flag against the dataset's beta-binomial null (null where the class × platform stratum has no null or is uninformative). |
| `expected_genes_q95`, `flag_diffuse_profile` | float32, nullable bool | q95 of the distinct genes of 200 multinomial draws from the class profile at the cell's depth; flag when the cell detects more (null in an uninformative stratum). |
| `ood_z`, `flag_ood` | float32, nullable bool | Robust z of the supercluster `avg_correlation` within class × platform × depth bin; flag below -3. |
| `microglia_stat`, `microglia_weight`, `flag_microglial_spillover` | float32, nullable bool | Mouse spill-over flag (§7.4, M6): E3's specific-gene likelihood-ratio statistic, the fitted spill weight and the flag (statistic >= 10, weight >= 0.05) on the panel's derived microglia genes; null with fewer than 3 genes or a failed astrocyte false-positive check, and for human (off, OD-C5). Spill-over prevalence, never a microglia count. |
| `flag_region_incoherent`, `region_coherence`, `flag_astro_lowcount` | nullable bool, float32, bool | Mouse only (M6): F1 (a region-restricted class with kNN30 class coherence < 0.1 and class bootstrap < 0.8; null without coordinates), the coherence, and an Astro-Epen call below 100 counts. |
| `mmc_wmb_unpruned_{class,subclass}_{name,bp}`, `region_pruned_changed`, `inferred_region` | category, float32, bool, category | Mouse region step (§7.2): the unpruned call, whether pruning changed the cell's class or subclass, and its tile's present division. |
| `flag_nonneuronal_high_depth` | nullable bool | Resolvability version 7 only (plan §8.3 v7.9; report-only): a non-neuronal cell with at least `real_qc.nonneuronal_high_depth_counts` (1,000) counts whose (level, class, depth bin) was emitted on its own ensemble verdict and is marked `nonneuronal_high_depth` (real merged-mask glia lose confidence there). Null outside the table and for every cell of a version-6 bundle or a run without resolvability tables. An optional contract column: tables written before it validate. Never changes a status. |
| `exclude_hard`, `discovery_caution` | bool | Excluded from downstream use: outside the table, implausible at every attempted level, or a `failed` gate (a refused panel fails the gate); any report-only flag set (contaminated, diffuse, OOD, method disagreement, spill-over, astro low count; null counts as unset). The report-only flags never change a status. |
| `soft_broad_<class>`, `soft_broad_unallocated` | float32 | Human soft broad vector of each table cell (sums to 1; NaN outside the table): the assigned WHB supercluster's and its runner-ups' bootstrap probabilities aggregated to the seven broad classes, sinks and the residual unallocated (§5.5). Mouse: `soft_class_<class>`. |
| `mmc_<reference>_<level>_*` | mixed | The raw engine columns of every MAP run (`mmc_whb_*`, `mmc_whb_setc_*`, `mmc_seaad_*`): label, name, bootstrap and aggregate probability, correlation and, at the leaf level, five runner-ups. |

`CellStatus` values (the first failing check wins, in this order, at every
level including the SEA-AD subclass, whose SEA-AD agreement check is its
`method_disagree`):

| Status | Meaning |
|---|---|
| `low_counts` | Outside the table (`in_table` false). |
| `not_attempted_gate` | The dataset gate or a refused panel stopped the level (e.g. supercluster on a `broad_only` dataset). |
| `not_applicable` | The level does not apply to the call (NT outside neurons). |
| `implausible` | The WHB node is a sink or not plausible for the region (lineage keeps it when SEA-AD agrees at lineage). |
| `parent_unresolved` | The parent level is not confident. |
| `below_floor` | Counts below the class × platform floor. |
| `not_resolvable` | Resolvability does not emit the (level, class, depth bin) for this dataset's composition. |
| `low_confidence` | The raw (or local) threshold is not met. |
| `method_disagree` | SEA-AD does not agree below 60 counts, or confidently disagrees from 60. |
| `single_method` | Below 60 counts with no second method (degraded mode `whb_only`, unless `annotation_allow_single_method`). |
| `confident` | Every check passed. |

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
