# Configuration

MerXen configuration lives in three places:

1. **Shell environment** — a `.env` file for absolute paths and machine limits.
2. **Nextflow parameters** — defaults in
   [workflows/nextflow.config](../workflows/nextflow.config), overridable per
   run with `--<name>`.
3. **Pydantic models** — [src/merxen/config.py](../src/merxen/config.py) — the
   authoritative schema every CLI command validates its JSON config against.

## Environment variables

Copy the template and fill in values:

```bash
cp .env.example .env
```

Variables, from [.env.example](../.env.example):

| Variable | Default | Used by |
|----------|---------|---------|
| `MERXEN_OUTPUT_ROOT` | `./results` | `PipelineConfig.output_root`. Not consumed directly by the pipeline today — Nextflow's `--outdir` is authoritative — but available to Python code that imports `PipelineConfig()`. |
| `MERXEN_PROSEG_INSTALL_PATH` | `/usr/bin/proseg` | Optional Python-side default for `PipelineConfig.proseg_install_path`. Nextflow uses `proseg_install_path` from `workflows/nextflow.config`. |
| `MERXEN_MAX_RAM_GB` | `600.0` | `PipelineConfig.max_ram_gb`. Mirrored into the Nextflow `max_ram_gb` param. |

The `MERXEN_` prefix is wired up via `model_config = {"env_prefix": "MERXEN_"}`
in [config.py:202](../src/merxen/config.py#L202).

## Nextflow parameters

Defaults in [workflows/nextflow.config](../workflows/nextflow.config). Override
any of them with `--<name>` on the command line.

### Required

| Param | Description |
|-------|-------------|
| `samplesheet` | Path to the samplesheet CSV. |

### Stage-specific

| Param | Default | Description |
|-------|---------|-------------|
| `proseg_search_paths` | `/usr/bin/proseg`, `/usr/local/bin/proseg` | Ordered paths checked by `ENSURE_PROSEG` before segmentation. Entries may be executable paths or directories containing `proseg`; `command -v proseg` is checked after this list. |
| `proseg_install_path` | `/usr/bin/proseg` | Destination used when ProSeg is missing and automatic install is enabled. If the directory is not writable, the bootstrap step requests `sudo`. |
| `proseg_auto_install` | `true` | Install ProSeg automatically with Cargo when no configured search path contains an executable binary. |
| `proseg_cargo_package` | `proseg` | Cargo package name installed by the bootstrap step. |

### General

| Param | Default | Description |
|-------|---------|-------------|
| `outdir` | `./results` | Output root. |
| `analysis_mode` | `paired` | Fallback row mode: `paired`, `merscope`, or `xenium`. A non-empty samplesheet `analysis_mode` value overrides this per row. |
| `enable_alignment` | `false` | Fallback row alignment switch. A non-empty samplesheet `enable_alignment` value overrides this per row; alignment only applies to paired rows. |
| `analysis_segmentation` | `both` | Fallback downstream analysis branches after enrichment. Valid values: `both`, `reseg`, `original_seg`; comma-separated combinations are accepted. A non-empty samplesheet `analysis_segmentation` value overrides this per row. |
| `mask_image_quantification_enabled` | `true` | Insert the Cellpose-mask image quantification stage between enrichment and QC. A non-empty samplesheet `mask_image_quantification_enabled` value overrides this per row. |
| `cortical_depth_enabled` | `false` | Insert the cortical-depth stage before QC. Requires per-sample pial/tissue-edge annotations, with optional gray/white boundaries for depth pieces. A non-empty samplesheet `cortical_depth_enabled` value overrides this per row. |
| `force_spatialdata_build` | `false` | Rebuild SpatialData zarrs even if cached. |
| `start_stage` | `build_spatialdata` | Fallback first stage. Skipped upstream stages are read from published outputs. A samplesheet `start_stage` value overrides this per row. |
| `stop_stage` | `clustering_squidpy` | Fallback last stage. This includes `spatial_gene_analysis`, which runs between visualization and clustering. MapMyCells is available after clustering but opt-in because it requires reference files. A samplesheet `stop_stage` value overrides this per row. |
| `only_stage` | `null` | Fallback single-stage selector. A row-level `only_stage` overrides row start/stop values; row start/stop values suppress the global `only_stage` fallback for that row. |
| `gpu_process_lock_enabled` | `true` | Serialize local GPU-heavy processes with a file lock so `SEGMENT`, GPU `ALIGN`, and GPU `CLUSTERING_SQUIDPY` do not compete for one workstation GPU. |
| `gpu_process_lock_file` | `${projectDir}/.merxen_gpu.lock` | File used for the local GPU lock. Override only when coordinating multiple runs from the same machine. |

Stage names accepted by `start_stage`, `stop_stage`, and `only_stage` are:
`build_spatialdata`, `segment`, `enrich`, `mask_image_quantification`,
`qc`, `align`, `align_qc`, `compare`, `visualize`,
`spatial_gene_analysis`, `clustering_squidpy`, `compute_cortical_depth`, and
`mapmycells`.
`mask_image_quantification` is
available only when the effective `mask_image_quantification_enabled` value is
`true`. `compute_cortical_depth` is available only when the effective
`cortical_depth_enabled` value is `true`. `align` and `align_qc` are available
only for rows whose effective `enable_alignment` value is `true`.
`align`, `align_qc`, and `compare` are available only when
`analysis_mode = paired`.

### Cellpose

| Param | Default | Description |
|-------|---------|-------------|
| `cellpose_model_type` | `cyto3` | Cellpose model preset. |
| `cellpose_gpu` | `true` | Use GPU for inference. |
| `cellpose_diameter` | `null` | Cell diameter (px). `null` → Cellpose auto-estimates. |
| `cellpose_flow_threshold` | `0.8` | Cellpose flow threshold. |
| `cellpose_cellprob` | `-5.0` | Cellpose cell probability threshold. |
| `cellpose_tile_overlap` | `0.15` | Cellpose model's internal fractional tile overlap. |
| `cellpose_bsize` | `256` | Cellpose internal batch block size. |
| `cellpose_tile_size_candidates` | `[6144, 4096, 3072, 2048]` | Candidate halo tile sizes probed from largest to smallest. |
| `cellpose_min_tile_size` | `1024` | Smallest allowed Cellpose halo tile size. |
| `cellpose_stitch_overlap_px` | `256` | Halo overlap, in pixels, used for MerXen object-level stitching. |
| `cellpose_stitch_status_every_tiles` | `10` | Progress/status interval for Cellpose tile stitching. |
| `cellpose_filter_per_tile` | `true` | Apply regionprops filtering to each tile before stitching. |
| `cellpose_duplicate_iou_threshold` | `0.25` | Skip an owned tile object as a duplicate when overlap IoU with an existing label is at least this value. |
| `cellpose_duplicate_overlap_fraction` | `0.5` | Skip an owned tile object as a duplicate when at least this fraction of its pixels overlap one existing label. |
| `cellpose_min_remaining_fraction` | `0.05` | Skip a non-duplicate object if too little of it remains after preserving existing labels. |
| `cellpose_edge_touch_policy` | `keep` | Keep or skip labels touching an artificial tile edge; `keep` records them in stitching stats. |
| `cellpose_write_stitching_stats` | `true` | Write `cellpose_stitching_stats.json` beside the stitched mask. |
| `cellpose_final_min_area_um2` | `5.0` | Drop final Cellpose masks smaller than this area before ProSeg. |
| `cellpose_final_max_area_um2` | `400.0` | Drop final Cellpose masks larger than this area before ProSeg. |
| `cellpose_final_filter_chunk_mb` | `256` | Approximate row-chunk size for streaming the final mask filter. |

### Mask image quantification

| Param | Default | Description |
|-------|---------|-------------|
| `mask_image_quantification_enabled` | `true` | Run Cellpose-mask image quantification after enrichment by default. |
| `mask_image_quantification_max_forks` | `2` | Maximum concurrent quantification processes. |

### Cortical depth

| Param | Default | Description |
|-------|---------|-------------|
| `cortical_depth_enabled` | `false` | Run `COMPUTE_CORTICAL_DEPTH` as a terminal stage after `CLUSTERING_SQUIDPY`. |
| `cortical_depth_coordinate_unit_um` | `1.0` | Microns per coordinate unit in annotation/cell coordinates. Use the image pixel size if annotations are in pixel coordinates; keep `1.0` when coordinates are already microns. |
| `cortical_depth_raster_resolution_um` | `5.0` | Finite-difference raster spacing. Smaller values improve geometry fidelity and increase memory/time. |
| `cortical_depth_raster_padding_um` | `null` | Optional padding around the ribbon bounds. `null` uses a small automatic padding. |
| `cortical_depth_boundary_band_um` | `null` | Width of the rasterized Dirichlet boundary band. `null` uses about 1.5 raster pixels. |
| `cortical_depth_boundary_smoothing_window` | `0` | Optional moving-average smoothing window over boundary vertices. |
| `cortical_depth_streamline_spacing_um` | `50.0` | Approximate pial arc-length spacing between streamline seeds. |
| `cortical_depth_streamline_step_um` | `null` | Integration step length. `null` uses about half the raster resolution. |
| `cortical_depth_streamline_max_steps` | `4000` | Maximum integration steps per streamline. |
| `cortical_depth_streamline_resample_points` | `101` | Number of points stored per streamline. |
| `cortical_depth_side_boundary_distance_um` | `25.0` | Distance from artificial side boundaries used to flag cells/streamlines. |
| `cortical_depth_contour_levels` | `0.1..0.9` | Depth contours written to GeoJSON/QC overlays. |
| `cortical_depth_write_spatialdata_table` | `true` | Replace selected SpatialData tables with cortical-depth columns added to `obs`. |
| `cortical_depth_max_forks` | `2` | Maximum concurrent cortical-depth processes. |

### ProSeg

Before `SEGMENT` runs, the `ENSURE_PROSEG` process checks
`proseg_search_paths`, then `command -v proseg`. If no executable is found and
`proseg_auto_install=true`, it runs `cargo install proseg` into a temporary
root and copies the resulting binary to `proseg_install_path`. System-owned
install paths trigger a `sudo` prompt.

| Param | Default | Description |
|-------|---------|-------------|
| `proseg_samples` | `1200` | MCMC samples. |
| `proseg_voxel_size` | `0.5` | Voxel size (µm). |
| `proseg_burnin_voxel_size` | `1.0` | Burn-in voxel size (µm). |
| `proseg_nuclear_reassignment_prob` | `0.25` | Nuclear reassignment probability. |
| `proseg_diffusion_probability` | `0.25` | Diffusion probability. |
| `proseg_cell_compactness` | `0.04` | Cell compactness prior. |
| `proseg_num_threads` | `32` | ProSeg thread count. |
| `default_merscope_voxel_layers` | `7` | Fallback when samplesheet column is empty. |
| `default_xenium_voxel_layers` | `2` | Fallback when samplesheet column is empty. |

### Platform-specific

| Param | Default | Description |
|-------|---------|-------------|
| `xenium_min_qv` | `20.0` | Minimum transcript QV to retain. |

### Alignment

Alignment is optional because it requires Spateo and its heavier dependencies.
When a paired row's effective `enable_alignment` value is `true`, Nextflow runs
`ALIGN` in `environment.alignment.yml`. The process checks whether MerXen's
shimmed Spateo import works; if it does not, it installs pinned Spateo/Dynamo
Git refs inside the alignment env and then restores modern AnnData for
SpatialData compatibility. Non-alignment stages keep using `environment.yml`.

| Param | Default | Description |
|-------|---------|-------------|
| `enable_alignment` | `false` | Run `ALIGN` and `ALIGN_QC` between QC and comparison by default. A samplesheet `enable_alignment` value can override this per paired row. |
| `alignment_conda` | `environment.alignment.yml` | Conda env file or existing env path used only for `ALIGN`. |
| `alignment_bootstrap_dependencies` | `true` | Install pinned Spateo/Dynamo requirements inside the `ALIGN` env when `merxen check-alignment-deps` fails. |
| `alignment_dynamo_requirement` | Git pin for `dynamo-release` v1.5.3 | Requirement installed by the alignment bootstrap. |
| `alignment_spateo_requirement` | Git pin for Spateo main commit `1bd8a35...` | Requirement installed by the alignment bootstrap; resolves to `spateo-release` 1.1.1. |
| `alignment_anndata_requirement` | `anndata>=0.12.10` | Requirement installed after Spateo/Dynamo to restore SpatialData-compatible AnnData. |
| `alignment_device` | `auto` | Spateo device; `auto` uses CUDA when available. |
| `alignment_dtype` | `float32` | Spateo tensor precision; lower memory than float64. |
| `alignment_selected_mode` | `nonrigid` | Coordinate set used by downstream alignment transforms. |
| `alignment_spateo_mode` | `SN-S` | Spateo morpho-align mode. |
| `alignment_max_iter` | `360` | Spateo optimization iterations. |
| `alignment_nonrigid_start_iter` | `220` | Iteration where non-rigid refinement starts. |
| `alignment_beta` | `0.005` | Spateo non-rigid kernel width. |
| `alignment_lambda_vf` | `3000.0` | Spateo vector-field regularization. |
| `alignment_k` | `15` | Spateo control-point count. |
| `alignment_partial_robust_level` | `100` | Robustness level for partial overlap. |
| `alignment_allow_flip` | `true` | Allow Spateo's coarse initialization to test a mirrored orientation. |
| `alignment_svi_mode` | `false` | Use full pairwise matching on the sampled cells instead of SVI mini-batches. |
| `alignment_n_sampling` | `1000` | SVI batch size. |
| `alignment_sparse_top_k` | `512` | Sparse matching top-k used by Spateo. |
| `alignment_chunk_capacity` | `1` | Spateo chunk capacity. |
| `alignment_use_hvg` | `false` | Select highly variable genes before alignment. `false` uses the shared panel. |
| `alignment_n_top_genes` | `100` | Number of HVGs used for alignment. |
| `alignment_use_pca` | `true` | Run joint PCA on shared expression features before Spateo. |
| `alignment_n_pcs` | `50` | Number of joint PCA components used for Spateo matching. |
| `alignment_max_alignment_cells` | `35000` | Deterministic per-platform cell subsample used for Spateo optimization. |
| `alignment_seed` | `21` | Seed for deterministic alignment subsampling. |
| `alignment_max_nonrigid_anchors` | `5000` | Maximum RBF anchors for full-data transform application. |
| `alignment_pytorch_cuda_alloc_conf` | `expandable_segments:True,max_split_size_mb:256` | PyTorch allocator setting exported by `ALIGN`. |
| `alignment_max_forks` | `1` | Maximum concurrent `ALIGN` tasks. Raise only when multiple GPUs or sufficient VRAM are available. |
| `alignment_qc_grid_rows` / `alignment_qc_grid_cols` | `10` / `10` | SABench-style QC grid dimensions. |

### Squidpy clustering

| Param | Default | Description |
|-------|---------|-------------|
| `clustering_squidpy_drop_control_features` | `true` | Remove blank/negative/control-like features before cell/gene filtering and clustering. |
| `clustering_squidpy_min_counts` | `10` | Minimum counts per cell passed to `scanpy.pp.filter_cells`. |
| `clustering_squidpy_min_cells` | `5` | Minimum cells per gene passed to `scanpy.pp.filter_genes`. |
| `clustering_squidpy_normalize_target_sum` | `null` | Optional target sum for `scanpy.pp.normalize_total`; `null` uses Scanpy's default. |
| `clustering_squidpy_normalize_exclude_highly_expressed` | `false` | Exclude highly expressed genes from Scanpy size-factor calculation. |
| `clustering_squidpy_normalize_max_fraction` | `0.05` | Fraction threshold used when excluding highly expressed genes. |
| `clustering_squidpy_n_pcs` | `60` | Maximum PCs for `scanpy.pp.pca`. |
| `clustering_squidpy_n_neighbors` | `30` | Neighbor count for `scanpy.pp.neighbors`. |
| `clustering_squidpy_leiden_resolution` | `0.5` | Leiden clustering resolution. |
| `clustering_squidpy_umap_min_dist` | `0.3` | Minimum distance parameter for `scanpy.tl.umap`. |
| `clustering_squidpy_umap_spread` | `1.0` | Spread parameter for `scanpy.tl.umap`. |
| `clustering_squidpy_random_seed` | `0` | Seed for PCA/UMAP/Leiden. |
| `clustering_squidpy_spatial_point_size` | `0.5` | Highlight point size for spatial cluster grid plots. |
| `clustering_squidpy_spatial_scatter_point_size` | `2.0` | Point size for regular spatial scatter plots. |
| `clustering_squidpy_figure_dpi` | `180` | DPI for PNG plots. |
| `clustering_squidpy_use_gpu` | `true` | Use RAPIDS single-cell acceleration when available. |
| `clustering_squidpy_gpu_conda` | `environment.clustering-gpu.yml` | Dedicated RAPIDS environment used only by `CLUSTERING_SQUIDPY_COMPUTE`. |
| `clustering_squidpy_gpu_container` | Site GPU image path | Dedicated RAPIDS image used only by `CLUSTERING_SQUIDPY_COMPUTE` with Apptainer. Build it from `Dockerfile.clustering-gpu` or override this path. |
| `clustering_squidpy_max_forks` | `4` | Maximum concurrent Squidpy clustering tasks. GPU-backed tasks still share the local GPU lock when enabled. |
| `clustering_squidpy_gpu_vram_monitor` | `true` | Run a lightweight `nvidia-smi` sampler around each `CLUSTERING_SQUIDPY_COMPUTE` task. |
| `clustering_squidpy_gpu_vram_monitor_interval_seconds` | `2` | Sampling interval for the clustering GPU VRAM monitor. |
| `clustering_squidpy_write_spatialdata_table` | `true` | Add or replace a final clustered AnnData table in each source `latest_spatialdata.zarr`. |
| `clustering_squidpy_hierarchical_enabled` | `true` | Run broad atlas-guided annotation and per-branch subclustering. Set to `false` for the legacy one-shot Leiden workflow. |
| `clustering_squidpy_broad_leiden_resolution` | `0.2` | Low-resolution Leiden round used for broad atlas annotation. |
| `clustering_squidpy_subcluster_leiden_resolution` | `0.5` | Default Leiden resolution for non-neuron broad-class branches. |
| `clustering_squidpy_subcluster_resolution_overrides` | `[:]` | Optional Nextflow map from broad class or neuron split label to a custom branch Leiden resolution. |
| `clustering_squidpy_neuron_split_leiden_resolution` | `0.15` | Coarse neuron round used before Excitatory/Inhibitory/Other annotation. |
| `clustering_squidpy_neuron_subcluster_leiden_resolution` | `0.5` | Default Leiden resolution for neuron subtype branches. |
| `clustering_squidpy_min_branch_cells` | `50` | Smallest branch/split size that will be reclustered. Smaller groups keep labels but skip PCA/UMAP/Leiden. |
| `clustering_squidpy_broad_marker_lookup_path` | WHB marker JSON path | MapMyCells query marker lookup used for atlas-guided cluster annotation. |
| `clustering_squidpy_broad_taxonomy_metadata_path` | WHB taxonomy CSV path | Allen `cluster_annotation_term.csv` used to map marker lookup IDs to atlas labels. |
| `clustering_squidpy_broad_cluster_membership_path` | WHB membership CSV path | Allen membership metadata used for neuron neurotransmitter split labels. |
| `clustering_squidpy_broad_reference_cache_dir` | `/media/mathieubo/SSD1/MerXen/mapmycells` | Cache searched for WHB taxonomy metadata and reference H5AD gene-symbol metadata. |
| `clustering_squidpy_broad_marker_level` | `CCN202210140_SUPC` | Atlas taxonomy level scored for broad annotations. |
| `clustering_squidpy_broad_min_marker_overlap` | `3` | Minimum query-panel marker overlap required to score an atlas label. |
| `clustering_squidpy_broad_max_markers_per_label` | `80` | Maximum resolved markers used per atlas label. |
| `clustering_squidpy_broad_score_margin_threshold` | `0.0` | Minimum difference between best and runner-up atlas scores; lower margins become `Mixed/Unknown`. |
| `clustering_squidpy_broad_unknown_label` | `Mixed/Unknown` | Label used when no atlas marker set scores confidently. |

### Spatial gene analysis

| Param | Default | Description |
|-------|---------|-------------|
| `spatial_gene_analysis_drop_control_features` | `true` | Remove blank/negative/control-like genes before autocorrelation. |
| `spatial_gene_analysis_min_counts` | `0` | Optional minimum total counts per cell before analysis. |
| `spatial_gene_analysis_min_cells` | `5` | Minimum cells with a gene detected before calculating metrics. |
| `spatial_gene_analysis_normalize_target_sum` | `null` | Optional target sum for `scanpy.pp.normalize_total`; `null` uses Scanpy's default. |
| `spatial_gene_analysis_normalize_exclude_highly_expressed` | `false` | Exclude highly expressed genes from Scanpy size-factor calculation. |
| `spatial_gene_analysis_normalize_max_fraction` | `0.05` | Fraction threshold used when excluding highly expressed genes. |
| `spatial_gene_analysis_n_neighbors` | `6` | Spatial nearest-neighbor count used by Squidpy's generic-coordinate neighbor graph. |
| `spatial_gene_analysis_top_n` | `10` | Number of highest and lowest genes retained for each metric ranking. |
| `spatial_gene_analysis_spatial_point_size` | `2.0` | Point size for individual spatial gene expression plots. |
| `spatial_gene_analysis_figure_dpi` | `180` | PNG output DPI. |
| `spatial_gene_analysis_max_forks` | `4` | Maximum concurrent spatial gene analysis tasks. |

### MapMyCells

| Param | Default | Description |
|-------|---------|-------------|
| `mapmycells_reference_mode` | `both` | Which references to run: `whole_brain`, `region`, or `both`. |
| `mapmycells_marker_lookup_path` | WHB marker JSON path | JSON marker lookup file for the whole-brain reference. Required when `reference_mode` includes `whole_brain`. |
| `mapmycells_precomputed_stats_path` | WHB stats H5 path | HDF5 precomputed stats file for the whole-brain reference. Required when `reference_mode` includes `whole_brain`. |
| `mapmycells_region_name` | `frontal_a44_a45_a46_a32_acc` | Short safe name used in region output directories and annotation prefixes. |
| `mapmycells_region_labels` | `["Human A44-A45", "Human A46", "Human A32", "Human ACC"]` | Allen WHB `region_of_interest_label` values used to build the strict region reference. May be a Nextflow list, JSON list, or comma-separated string. |
| `mapmycells_region_cache_dir` | `/media/mathieubo/SSD1/MerXen/mapmycells` | Durable cache for Allen WHB downloads and generated region reference files. |
| `mapmycells_region_min_cells_per_leaf` | `10` | Drop region taxonomy leaf aliases with fewer cells than this before precomputing stats. |
| `mapmycells_region_force_rebuild` | `false` | Rebuild the generated region reference even if matching cached files exist. |
| `mapmycells_region_query_markers_n_per_utility` | `10` | Marker count target passed to Allen's `QueryMarkerRunner` for the region reference. |
| `mapmycells_drop_level` | `null` | Optional taxonomy level to drop before mapping. |
| `mapmycells_normalization` | `raw` | Query normalization passed to MapMyCells. |
| `mapmycells_bootstrap_factor` | `0.9` | Marker downsampling factor for bootstrapping; default keeps the historical spatial-data setting. |
| `mapmycells_bootstrap_iteration` | `100` | Number of bootstrap iterations. |
| `mapmycells_n_processors` | `8` | Number of worker processes passed to MapMyCells. |
| `mapmycells_chunk_size` | `null` | Optional cells-per-worker chunk size. |
| `mapmycells_rng_seed` | `null` | Optional mapper random seed. |
| `mapmycells_max_gb` | `null` | Optional memory budget for H5AD conversion. |
| `mapmycells_tmp_dir` | `null` | Optional fast temporary directory for mapper scratch data. |
| `mapmycells_cloud_safe` | `false` | Passed to MapMyCells `cloud_safe`. |
| `mapmycells_flatten` | `false` | Flatten taxonomy and map directly to leaf nodes. |
| `mapmycells_verbose_csv` | `false` | Include verbose confidence columns when supported by the mapper. |
| `mapmycells_plots_only` | `false` | Reuse existing mapper CSV/extended JSON outputs in published `mapmycells_out/` and regenerate only the annotated H5AD and plots. |
| `mapmycells_query_layer` | `counts` | AnnData layer copied into `X` before mapping. Use `null` to keep current `X`. |
| `mapmycells_gene_id_column` | `null` | Optional `var` column used as gene identifiers for the query H5AD. |
| `mapmycells_obs_id_column` | `null` | Optional `obs` column used as cell identifiers for the query H5AD. |

### Resource limits

| Param | Default | Description |
|-------|---------|-------------|
| `max_ram_gb` | `640` | System memory limit passed to `MemoryConfig`. |
| `warn_ram_gb` | `600` | RAM warning threshold. |
| `transcript_chunk_rows` | `1_000_000` | Points chunk size when streaming transcripts. |

The same values are enforced at the executor level:

```groovy
executor {
    name = "local"
    cpus = 72
    memory = "640 GB"
}
```

Per-process CPU/memory requests and default concurrency guards
([nextflow.config](../workflows/nextflow.config)):

| Process | CPUs | Memory | Max forks |
|---------|-----:|-------:|-----------|
| `BUILD_SPATIALDATA` | 8 | 80 GB | `build_spatialdata_max_forks` = 3 |
| `SEGMENT` | 32 | 220 GB | `segment_max_forks` = 1 |
| `ENRICH` | 8 | 300 GB | unbounded |
| `QC` | 4 | 24 GB | unbounded |
| `ALIGN` | 12 | 100 GB | `alignment_max_forks` = 1 |
| `ALIGN_QC` | 4 | 32 GB | unbounded |
| `COMPARE` | 4 | 32 GB | unbounded |
| `VISUALIZE` | 4 | 32 GB | unbounded |
| `CLUSTERING_SQUIDPY` | 8 | 32 GB | `clustering_squidpy_max_forks` = 4 |
| `MAPMYCELLS` | 8 | 160 GB | unbounded |

On local single-GPU runs, `SEGMENT`, `ALIGN` when `alignment_device != "cpu"`,
and `CLUSTERING_SQUIDPY` when `clustering_squidpy_use_gpu=true` also share
`gpu_process_lock_file`. The lock is held for the full process shell, then
released automatically when the task exits.

All processes use `errorStrategy = "ignore"` with
`workflow.failOnIgnore = true`. A failed task therefore stops only branches that
depend on its missing outputs, while unrelated samples continue. The overall
Nextflow run still exits non-zero if any task failure was ignored.

## Pydantic config models

Every CLI subcommand takes `--config <path>.json` and validates the JSON
against a Pydantic model. Adding, removing, or renaming fields in these
models is the ground truth for how stages are configured.

| Model | Stage | File |
|-------|-------|------|
| `SpatialDataBuildConfig` | `build-spatialdata` | [config.py:112](../src/merxen/config.py#L112) |
| `SegmentationConfig` | `segment` | [config.py:146](../src/merxen/config.py#L146) |
| `EnrichmentConfig` | `enrich` | [config.py:157](../src/merxen/config.py#L157) |
| `QCConfig` | `qc` | [config.py:169](../src/merxen/config.py#L169) |
| `AlignmentConfig` | `align` | [config.py](../src/merxen/config.py) |
| `AlignmentQCConfig` | `alignment-qc` | [config.py](../src/merxen/config.py) |
| `ComparisonConfig` | `compare` | [config.py](../src/merxen/config.py) |
| `VisualizationConfig` | `visualize` | [config.py](../src/merxen/config.py) |
| `ClusteringSquidpyConfig` | `clustering-squidpy` | [config.py](../src/merxen/config.py) |
| `MapMyCellsConfig` | `mapmycells` | [config.py](../src/merxen/config.py) |

Nested sub-models:

- `CellposeConfig`, `TilingConfig`, `MaskFilterConfig` — Cellpose behaviour.
- `ProsegConfig` — ProSeg parameters, including `binary_path`.
- `MemoryConfig` — memory limits and chunk sizes.
- `DatasetConfig` — one dataset (one half of a pair) within a `SegmentationConfig`.
- `MerscopeBuildConfig` / `XeniumBuildConfig` — platform-specific build options
  nested under `SpatialDataBuildConfig`.
- `PipelineConfig(BaseSettings)` — top-level, loaded from `MERXEN_*` env vars.

`load_config_from_json(path, cls)` in
[config.py:205](../src/merxen/config.py#L205) is the helper every CLI entry
point uses to parse and validate.

## Precedence

1. CLI flags passed to `nextflow run` override `nextflow.config` defaults.
2. `nextflow.config` defaults populate the JSON config written into the work
   directory.
3. The Python stage loads that JSON, validated through the Pydantic model.
4. `MERXEN_*` environment variables only affect code that instantiates
   `PipelineConfig()` directly — they do **not** back-propagate into
   `nextflow.config`. Set them explicitly via `--<name>` if you need them in
   Nextflow.
