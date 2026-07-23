# Python API overview

The installable `merxen` package lives under [src/merxen/](../src/merxen/). It
is organised as one subpackage per pipeline concern. This page is a
navigational map of the key public functions — for per-stage details see
[Pipeline stages](index.md#pipeline-stages). Docstrings in the source are the
authoritative reference.

## Package layout

```
merxen/
├── config.py            # Pydantic configs (pipeline contract)
├── memory.py            # RAM monitoring helpers
├── _typing.py           # shared small types
├── cli/                 # Click entry points (one per stage)
├── io/                  # I/O: samplesheet, zarr, transcripts, images
│   └── builders/        # per-platform SpatialData builders
├── segmentation/        # Cellpose tiling, ProSeg subprocess, mask utilities
├── enrichment/          # shape layers + per-shape gene tables
├── cortical_depth/      # Laplace/equal-area cortical-depth coordinates
├── distance_from_object/ # polygon-edge distance + paired pseudobulk DE
├── qc/                  # per-dataset and cross-platform metrics
├── visualization/       # plotting
├── analysis/            # Scanpy/Squidpy downstream analysis
└── alignment/           # optional Spateo cross-section registration
```

The subpackage structure mirrors the Nextflow stage graph:
`build → segment-nuclei → segment → enrich → mask-image-quantification → qc → align →
alignment-qc → compare → visualize → spatial-gene-analysis →
clustering-squidpy → compute-cortical-depth → distance-from-object →
mapmycells`. Cortical depth is skipped unless `--cortical_depth_enabled true`
is set. Alignment is skipped unless
`--enable_alignment true` is set, object distance is skipped unless
`--distance_from_object_enabled true` is set, and MapMyCells is opt-in because
it requires local reference files.

## `merxen.config`

All pipeline parameters as Pydantic v2 models. CLI commands validate JSON
configs against these.

- Top-level per-stage: `SpatialDataBuildConfig`, `SegmentationConfig`,
  `EnrichmentConfig`, `MaskImageQuantificationConfig`, `CorticalDepthConfig`,
  `DistanceFromObjectConfig`, `DistanceFromObjectCohortConfig`,
  `QCConfig`, `AlignmentConfig`, `AlignmentQCConfig`, `ComparisonConfig`,
  `VisualizationConfig`, `SpatialGeneAnalysisConfig`,
  `ClusteringSquidpyConfig`.
- Sub-models: `CellposeConfig`, `TilingConfig`, `MaskFilterConfig`,
  `ProsegConfig`, `ProsegHybridConfig`, `MemoryConfig`, `DatasetConfig`,
  `MerscopeBuildConfig`, `XeniumBuildConfig`.
- `PipelineConfig(BaseSettings)` — loads `MERXEN_*` env vars.
- `load_config_from_json(path, cls)` — helper every CLI uses.

Reference: [Configuration → Pydantic config models](configuration.md#pydantic-config-models).

## `merxen.io`

### `io.samplesheet` — [samplesheet.py](../src/merxen/io/samplesheet.py)
- `SamplePair` dataclass — one samplesheet row with optional MERSCOPE and
  Xenium inputs plus optional row-level analysis/stage overrides.
- `parse_samplesheet(csv_path) -> list[SamplePair]`.
- `validate_samplesheet(pairs, analysis_mode="paired")`.
- `required_platforms_for_mode(analysis_mode)`.

Used by unit tests and scripts; the Nextflow workflow parses the CSV itself
with Groovy.

### `io.builders` — [builders/](../src/merxen/io/builders/)
- `build_spatialdata_artifact(config, *, force_rerun)` — orchestrator.
- `write_merscope_spatialdata(...)` / `write_xenium_spatialdata(...)`.

### `io.spatialdata_io` — [spatialdata_io.py](../src/merxen/io/spatialdata_io.py)
- `write_spatialdata_zarr(sdata, path, ...)`.
- `convert_to_latest_zarr(raw_path, latest_path, quality_column_alias=...)` —
  schema-migrates ProSeg's raw output and can restore a platform-native
  quality-column name.
- `upgrade_spatialdata_contract(path, ...)` — atomically migrates a reusable
  pre-schema latest store.
- `prepare_source_spatialdata_contract(sdata, platform=...)` — normalizes
  source-reader IDs while preserving vendor provenance.

### `io.spatialdata_schema` — [spatialdata_schema.py](../src/merxen/io/spatialdata_schema.py)
- `stamp_merxen_schema(...)` / `register_segmentation_branch(...)` — publish
  the versioned points/assignment/shape/table registry.
- `with_stable_transcript_ids(...)` — assign unique positive transcript row IDs.
- `validate_merxen_schema(sdata, deep=False)` — check identifier and
  referential integrity.

### `io.transcript_io` — [transcript_io.py](../src/merxen/io/transcript_io.py)
- `to_pandas(df_like)` — best-effort dask / pandas / pyarrow → pandas.
- `resolve_col(obj, names, required=True)` — pick the first matching column.
- `iter_points_chunks(points_obj, ...)` — chunked iterator over a points
  table, with memory checks.
- `write_proseg_csv_from_points(...)` — seed transcripts with cell IDs and
  emit a ProSeg-friendly CSV.

### `io.image_source` — [image_source.py](../src/merxen/io/image_source.py)
- `build_image_source(image, as_float32)` — lazy image reader.
- `fetch_tile(source, y0, y1, x0, x1)` — crop a tile for Cellpose.
- `prepare_merscope_plane_sources(...)` / `fetch_merscope_projected_tile(...)`.
- `prepare_cellpose_input(...)` — per-channel stacking for Cellpose.

## `merxen.segmentation`

### `segmentation.pipeline` — [pipeline.py](../src/merxen/segmentation/pipeline.py)
- `run_segmentation_pipeline(config, *, force_rerun)` — full stage entry point.
- `run_cellpose_nuclei_segmentation(config, *, force_rerun)` — reusable
  DAPI-only nuclei entry point.

### `segmentation.cellpose` — [cellpose.py](../src/merxen/segmentation/cellpose.py)
- `build_cellpose_model(config)`.
- `run_tiled_cellpose(...)` — global-pixel mask from core-owned object stitching.
- `build_cellpose_affine_to_microns(matrix, ...)`.
- `assign_labels_from_masks(...)`.

### `segmentation.proseg` — [proseg.py](../src/merxen/segmentation/proseg.py)
- `run_proseg_refinement(...)` — subprocess-driven wrapper around the
  external ProSeg binary.

### `segmentation.proseg_hybrid` — [proseg_hybrid.py](../src/merxen/segmentation/proseg_hybrid.py)

- `select_bulk_transcripts(...)` — robust dominant transcript-component
  selection with Cellpose-aware tie breaking.
- `build_local_convex_expansion_geometry(...)` — capped local convex wedges
  driven by supported ProSeg-foreground transcript chains.
- `smooth_growth_only_geometry(...)` — fixed-micron boundary closing and
  outward rounding that cannot remove existing mask area or exceed the cap.
- `build_hybrid_cell_geometry(...)` — compose local-convex expansion and
  growth-only smoothing, with Cellpose as the geometric lower bound.
- `assign_transcripts_to_hybrid_masks(...)` — single-mask geometric and
  overlap-only ProSeg assignment.
- `run_proseg_hybrid_refinement(...)` — persist hybrid SpatialData polygons,
  augmented points, and counts.

### `segmentation.mask_filter` / `segmentation.mask_geometry`
- `filter_cell_by_regionprops(mask, config)`.
- `masks_to_polygons(mask, ...)`.

## `merxen.enrichment`

### `enrichment.enrich` — [enrich.py](../src/merxen/enrichment/enrich.py)
- `enrich_single_latest(config, *, force_rerun, ...)` — adds shapes, images,
  and vendor tables.

### `enrichment.assignment` — [assignment.py](../src/merxen/enrichment/assignment.py)
- `run_per_shape_assignment_for_dataset(...)` — per-shape gene counts.
- `compute_table_from_points_for_shape(...)`.
- `sanitize_table_key`, `resolve_points_cols`, `ensure_shape_has_cell_id`,
  `build_gene_list_from_base_table`, `clone_table_for_region`.

## `merxen.mask_image_quantification`

### `mask_image_quantification` — [mask_image_quantification.py](../src/merxen/mask_image_quantification.py)
- `build_mask_image_quantification_table(...)` — exact image-channel stats
  over nonzero Cellpose mask labels.
- `run_mask_image_quantification(config, *, force_rerun)` — stage entry point
  that writes the SpatialData table and sidecar outputs.

## `merxen.cortical_depth`

### `cortical_depth.pipeline` — [pipeline.py](../src/merxen/cortical_depth/pipeline.py)
- `run_cortical_depth(config)` — full stage entry point.

### Internal modules
- `boundaries.py` — read role-labelled GeoJSON boundaries and masks.
- `ribbon.py` — construct/rasterize the cortical ribbon.
- `laplace.py` — sparse 2D Laplace solve and bilinear interpolation.
- `streamlines.py` — normalized-gradient streamlines.
- `equivolumetric.py` — nearest-streamline equal-area depth approximation.
- `assign_cells.py` — per-cell depth, thickness, tangential coordinate, and QC flags.
- `plotting.py` — GeoJSON contours and QC overlays.

## `merxen.distance_from_object`

- `load_object_annotations(...)` — validate registered Polygon/MultiPolygon
  GeoJSON and normalize IDs/types.
- `assign_distances_to_objects(...)` — assign centroids to nearest polygon
  edges and label near/middle/far/beyond proximity.
- `build_pair_pseudobulk(...)` — sum raw near/far counts for eligible grey
  matter cells within one tissue block.
- `run_paired_differential_expression(...)` — paired PyDESeq2 with
  `~ pair_id + proximity`.
- `run_distance_from_object(config)` and
  `run_distance_from_object_cohort(config)` — per-platform and cohort stage
  entry points.

## `merxen.qc`

### `qc.metrics` — [metrics.py](../src/merxen/qc/metrics.py)
- `compute_dataset_qc(latest_zarr_path, dataset_name)`.
- `save_dataset_qc(qc_result, output_dir, dataset_name)`.

### `qc.gene_comparison` — [gene_comparison.py](../src/merxen/qc/gene_comparison.py)
- `compute_gene_comparison(xenium_sdata, merscope_sdata)`.
- `compute_gene_comparison_from_paths(xenium_zarr_path, merscope_zarr_path)`.
- `compute_gene_summary(sdata_obj, dataset_name)`.
- `compute_gene_summary_from_path(zarr_path, dataset_name)`.
- `gene_totals_from_points`, `gene_totals_from_table`,
  `apply_dataset_filter`, `normalize_counts`, `compare_df`, `fit_linear`.

## `merxen.visualization`

One module per plot family:

| Module | Public functions |
|--------|------------------|
| `gene_scatter` | `plot_gene_scatter`, `plot_gene_abundance` |
| `qc_plots` | `plot_geometry_histograms`, `plot_geometry_histograms_comparison`, `plot_cell_metrics_violin`, `plot_cell_metrics_violin_comparison`, `plot_assignment_bar` |
| `density_overview` | `density_hist2d`, `plot_density_overview`, `plot_transcript_overview`, `plot_single_transcript_overview` |
| `sanity_plots` | `plot_sanity_overlay`, `plot_pair_sanity_crops`, `plot_single_sanity_crop` |

Most plotting functions take DataFrames / arrays. The paired transcript and
sanity plots take already-opened SpatialData objects so they can work lazily
with large point tables and image pyramids.

## `merxen.alignment`

- `TransformResult` dataclass.
- `register_pair(merscope_sdata, xenium_sdata, config)` — builds paired
  AnnData objects, runs Spateo alignment, and fits affine/RBF transforms.
- `run_alignment_pipeline(config)` — CLI/Nextflow entry point for `ALIGN`.
- `run_alignment_qc(config)` — CLI/Nextflow entry point for `ALIGN_QC`.
- `fit_affine_matrix`, `fit_nonrigid_transform` — reusable transform helpers.

See [Section alignment](stages/alignment.md).

## `merxen.analysis`

### `analysis.clustering_squidpy`
- `load_spatialdata_adata(...)` — read a SpatialData zarr and return an
  AnnData object with `.obsm["spatial"]` populated for Squidpy.
- `add_qc_metrics(adata)` — compute Scanpy QC metrics plus
  blank/control/negative probe summaries when present.
- `remove_control_features(adata)` — drop blank/control/negative variables
  before clustering while recording the removed names in `.uns`.
- `run_scanpy_clustering(adata, ...)` — filter, normalize, log-transform,
  PCA, neighbors, UMAP, and Leiden clustering. It accepts `key_added` and
  `input_layer` for branch reclustering from raw counts.
- `load_atlas_marker_sets`, `score_clusters_by_atlas_markers` — WHB marker
  parsing and cluster-level atlas marker scoring used by hierarchical mode.
- `plot_qc_histograms`, `plot_umap`, `plot_spatial_scatter` — PNG writers.
- `save_qc_metrics`, `save_clustered_adata` — CSV and `.h5ad` outputs.
- `write_clustered_spatialdata_table` — add or replace the final clustered
  AnnData as a derived table in a SpatialData zarr.
- `run_hierarchical_scanpy_clustering(adata, config, ...)` — default broad
  annotation, branch subclustering, neuron split, and hierarchical QC artifact
  writer when `hierarchical_enabled` is true.
- `run_clustering_squidpy(config)` — full stage entry point for
  direct single-environment CLI use.
- `prepare_clustering_squidpy`, `compute_clustering_squidpy`,
  `finalize_clustering_squidpy` — H5AD process boundary used by Nextflow so
  RAPIDS compute never imports or writes SpatialData.

See [Squidpy clustering](stages/clustering-squidpy.md).

### `analysis.spatial_gene_analysis`
- `run_spatial_gene_analysis(config)` — full stage entry point for
  `SPATIAL_GENE_ANALYSIS`.
- `prepare_spatial_autocorr_adata(adata, ...)` — remove controls, filter,
  normalize, and log-transform input expression.
- `add_spatial_neighbors(adata, ...)` — build the Squidpy generic-coordinate
  spatial neighbor graph.
- `compute_spatial_autocorrelation(adata)` — calculate per-gene Moran's I and
  Geary's C.
- `ranked_spatial_autocorr_genes(metrics, ...)` — extract top and bottom genes
  per metric for reporting and plotting.
- `plot_autocorr_distributions` / `plot_spatial_gene_expression` — PNG writers.

See [Spatial gene analysis](stages/spatial-gene-analysis.md).

### `analysis.mecr`

- `run_mecr_reference(config)` — stream the complete WHB neuron and non-neuron
  references, discover panel-overlapping mutually exclusive broad-class
  markers, and write audit tables.
- `load_whb_panel_reference(config, panel_genes=...)` — join WHB taxonomy
  metadata and build a sparse, panel-restricted normalized reference.
- `discover_reference_markers(reference, ...)` — run Scanpy's Python Wilcoxon
  and apply the paper's strict detection and marker-uniqueness rules.
- `compute_mecr_pair_metrics(spatial_adata, markers)` — calculate binary
  intersection-over-union rates for every cross-class marker pair.
- `summarize_mecr_pair_metrics(pair_metrics, ...)` — calculate the unweighted
  sample mean while recording undefined zero-union pairs.
- `select_barnyard_pairs(pair_metrics, ...)` — deterministically choose
  canonical, high-MECR, and highly detected eligible pairs.
- `plot_reference_mecr_histogram`, `plot_platform_mecr_comparison`,
  `plot_class_pair_mecr_heatmaps`, and `plot_barnyard_pairs` — reference and
  spatial MECR diagnostic plot writers.
- `run_mecr(config)` — score all active platforms for one pair and
  segmentation and write combined tables and plots.

See [Mutually exclusive co-expression rate](stages/mecr.md).

### `analysis.mapmycells`

- `prepare_mapmycells_query(input_h5ad, output_h5ad, ...)` — copy the selected
  AnnData layer, normally `counts`, into `X` for MapMyCells.
- `build_mapmycells_command(config, ...)` — construct the local
  `cell_type_mapper.cli.from_specified_markers` invocation.
- `annotate_h5ad_with_mapmycells(input_h5ad, csv_path, output_h5ad)` — attach
  CSV assignment columns to `obs`.
- `prepare_region_mapmycells_reference(config)` — build or reuse cached Allen
  WHB ROI-specific precomputed stats and marker lookup files.
- `run_mapmycells(config)` — full stage entry point for `MAPMYCELLS`.

See [MapMyCells](stages/mapmycells.md).

## `merxen.memory`

Process-wide RAM monitoring. Not user-facing but useful when writing
long-running stages.

- `memory_snapshot_gb()` — RSS / available / CUDA reservation snapshot.
- `log_status(msg)` — logs with the current snapshot.
- `enforce_memory_limit(max_gb, note)` — raises if RSS exceeds limit.
- `force_release(note)` — GC + CUDA cache release.
- `clear_cuda_cache()`.

## Adding a new stage

A short checklist derived from the existing pattern:

1. Add a Pydantic config model to [config.py](../src/merxen/config.py).
2. Implement the stage function as a pure Python entry point that takes the
   config instance.
3. Add a Click command in [cli/](../src/merxen/cli/) that calls
   `load_config_from_json` and then the stage function. Register it in
   [cli/__init__.py](../src/merxen/cli/__init__.py).
4. Add a Nextflow module in [workflows/modules/](../workflows/modules/) that
   writes a heredoc JSON config and invokes the new CLI command.
5. Wire the channel into [workflows/main.nf](../workflows/main.nf).
6. Add tests in [tests/](../tests/) mirroring the new subpackage.
7. Document the stage under `docs/stages/` and link it from
   [docs/index.md](index.md) and [docs/pipeline.md](pipeline.md).
