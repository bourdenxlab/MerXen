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
├── qc/                  # per-dataset and cross-platform metrics
├── visualization/       # plotting
├── analysis/            # Scanpy/Squidpy downstream analysis
└── alignment/           # optional Spateo cross-section registration
```

The subpackage structure mirrors the Nextflow stage graph:
`build → segment → enrich → qc → align → alignment-qc → compare → visualize
→ clustering-squidpy`. Alignment is skipped unless `--enable_alignment true`
is set.

## `merxen.config`

All pipeline parameters as Pydantic v2 models. CLI commands validate JSON
configs against these.

- Top-level per-stage: `SpatialDataBuildConfig`, `SegmentationConfig`,
  `EnrichmentConfig`, `QCConfig`, `AlignmentConfig`, `AlignmentQCConfig`,
  `ComparisonConfig`, `VisualizationConfig`, `ClusteringSquidpyConfig`.
- Sub-models: `CellposeConfig`, `TilingConfig`, `MaskFilterConfig`,
  `ProsegConfig`, `MemoryConfig`, `DatasetConfig`,
  `MerscopeBuildConfig`, `XeniumBuildConfig`.
- `PipelineConfig(BaseSettings)` — loads `MERXEN_*` env vars.
- `load_config_from_json(path, cls)` — helper every CLI uses.

Reference: [Configuration → Pydantic config models](configuration.md#pydantic-config-models).

## `merxen.io`

### `io.samplesheet` — [samplesheet.py](../src/merxen/io/samplesheet.py)
- `SamplePair` dataclass — one pair of platforms.
- `parse_samplesheet(csv_path) -> list[SamplePair]`.
- `validate_samplesheet(pairs)`.

Used by unit tests and scripts; the Nextflow workflow parses the CSV itself
with Groovy.

### `io.builders` — [builders/](../src/merxen/io/builders/)
- `build_spatialdata_artifact(config, *, force_rerun)` — orchestrator.
- `write_merscope_spatialdata(...)` / `write_xenium_spatialdata(...)`.

### `io.spatialdata_io` — [spatialdata_io.py](../src/merxen/io/spatialdata_io.py)
- `write_spatialdata_zarr(sdata, path, ...)`.
- `convert_to_latest_zarr(raw_path, latest_path)` — schema-migrates ProSeg's
  raw output to the SpatialData version the rest of the code reads.

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

### `segmentation.cellpose` — [cellpose.py](../src/merxen/segmentation/cellpose.py)
- `build_cellpose_model(config)`.
- `run_tiled_cellpose(...)` — global-pixel mask from overlapping tiles.
- `build_cellpose_affine_to_microns(matrix, ...)`.
- `assign_labels_from_masks(...)`.

### `segmentation.proseg` — [proseg.py](../src/merxen/segmentation/proseg.py)
- `run_proseg_refinement(...)` — subprocess-driven wrapper around the
  external ProSeg binary.

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

## `merxen.qc`

### `qc.metrics` — [metrics.py](../src/merxen/qc/metrics.py)
- `compute_dataset_qc(latest_zarr_path, dataset_name)`.
- `save_dataset_qc(qc_result, output_dir, dataset_name)`.

### `qc.gene_comparison` — [gene_comparison.py](../src/merxen/qc/gene_comparison.py)
- `compute_gene_comparison(xenium_sdata, merscope_sdata)`.
- `compute_gene_comparison_from_paths(xenium_zarr_path, merscope_zarr_path)`.
- `gene_totals_from_points`, `gene_totals_from_table`,
  `apply_dataset_filter`, `normalize_counts`, `compare_df`, `fit_linear`.

## `merxen.visualization`

One module per plot family:

| Module | Public functions |
|--------|------------------|
| `gene_scatter` | `plot_gene_scatter` |
| `qc_plots` | `plot_geometry_histograms`, `plot_geometry_histograms_comparison`, `plot_cell_metrics_violin`, `plot_cell_metrics_violin_comparison`, `plot_assignment_bar` |
| `density_overview` | `density_hist2d`, `plot_density_overview`, `plot_transcript_overview` |
| `sanity_plots` | `plot_sanity_overlay`, `plot_pair_sanity_crops` |

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
  PCA, neighbors, UMAP, and Leiden clustering.
- `plot_qc_histograms`, `plot_umap`, `plot_spatial_scatter` — PNG writers.
- `save_qc_metrics`, `save_clustered_adata` — CSV and `.h5ad` outputs.
- `run_clustering_squidpy(config)` — full stage entry point for
  `CLUSTERING_SQUIDPY`.

See [Squidpy clustering](stages/clustering-squidpy.md).

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
