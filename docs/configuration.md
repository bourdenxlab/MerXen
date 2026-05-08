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
| `PROSEG_BINARY` | — | Passed in to the Nextflow `--proseg_binary` flag; referenced by `ProsegConfig.binary_path`. |
| `MERXEN_OUTPUT_ROOT` | `./results` | `PipelineConfig.output_root`. Not consumed directly by the pipeline today — Nextflow's `--outdir` is authoritative — but available to Python code that imports `PipelineConfig()`. |
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
| `proseg_binary` | `null` | Path to the ProSeg binary. Required only when `SEGMENT` runs. |

### General

| Param | Default | Description |
|-------|---------|-------------|
| `outdir` | `./results` | Output root. |
| `force_spatialdata_build` | `false` | Rebuild SpatialData zarrs even if cached. |
| `start_stage` | `build_spatialdata` | First stage to run. Skipped upstream stages are read from published outputs. |
| `stop_stage` | `clustering_squidpy` | Last stage to run. MapMyCells is available after this but opt-in because it requires reference files. |
| `only_stage` | `null` | Run exactly one stage; overrides `start_stage` and `stop_stage` when set. |

Stage names accepted by `start_stage`, `stop_stage`, and `only_stage` are:
`build_spatialdata`, `segment`, `enrich`, `qc`, `align`, `align_qc`,
`compare`, `visualize`, `clustering_squidpy`, and `mapmycells`. `align` and
`align_qc` are available only with `enable_alignment = true`.

### Cellpose

| Param | Default | Description |
|-------|---------|-------------|
| `cellpose_model_type` | `cyto3` | Cellpose model preset. |
| `cellpose_gpu` | `true` | Use GPU for inference. |
| `cellpose_diameter` | `null` | Cell diameter (px). `null` → Cellpose auto-estimates. |
| `cellpose_flow_threshold` | `0.8` | Cellpose flow threshold. |
| `cellpose_cellprob` | `-5.0` | Cellpose cell probability threshold. |
| `cellpose_tile_overlap` | `0.15` | Fractional overlap between core tiles. |
| `cellpose_bsize` | `256` | Cellpose internal batch block size. |

### ProSeg

| Param | Default | Description |
|-------|---------|-------------|
| `proseg_samples` | `1200` | MCMC samples. |
| `proseg_voxel_size` | `0.5` | Voxel size (µm). |
| `proseg_burnin_voxel_size` | `1.0` | Burn-in voxel size (µm). |
| `proseg_nuclear_reassignment_prob` | `0.25` | Nuclear reassignment probability. |
| `proseg_diffusion_probability` | `0.25` | Diffusion probability. |
| `proseg_cell_compactness` | `0.04` | Cell compactness prior. |
| `proseg_num_threads` | `75` | ProSeg thread count. |
| `default_merscope_voxel_layers` | `7` | Fallback when samplesheet column is empty. |
| `default_xenium_voxel_layers` | `2` | Fallback when samplesheet column is empty. |

### Platform-specific

| Param | Default | Description |
|-------|---------|-------------|
| `xenium_min_qv` | `20.0` | Minimum transcript QV to retain. |

### Alignment

Alignment is optional because it requires Spateo and its heavier dependencies.
Install Spateo, restore modern AnnData for SpatialData compatibility, then pass
`--enable_alignment true` to Nextflow:

```bash
pip install spateo-release==1.1.1
pip install "anndata>=0.12.10"
```

| Param | Default | Description |
|-------|---------|-------------|
| `enable_alignment` | `false` | Run `ALIGN` and `ALIGN_QC` between QC and comparison. |
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
| `clustering_squidpy_spatial_point_size` | `0.5` | Point size for Squidpy spatial scatter plots. |
| `clustering_squidpy_figure_dpi` | `180` | DPI for PNG plots. |
| `clustering_squidpy_use_gpu` | `true` | Use RAPIDS single-cell acceleration when available. |

### MapMyCells

| Param | Default | Description |
|-------|---------|-------------|
| `mapmycells_marker_lookup_path` | `null` | JSON marker lookup file for `cell_type_mapper.cli.from_specified_markers`. Required when running `mapmycells`. |
| `mapmycells_precomputed_stats_path` | `null` | HDF5 precomputed reference stats file. Required when running `mapmycells`. |
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
| `mapmycells_query_layer` | `counts` | AnnData layer copied into `X` before mapping. Use `null` to keep current `X`. |
| `mapmycells_gene_id_column` | `null` | Optional `var` column used as gene identifiers for the query H5AD. |
| `mapmycells_obs_id_column` | `null` | Optional `obs` column used as cell identifiers for the query H5AD. |

### Resource limits

| Param | Default | Description |
|-------|---------|-------------|
| `max_ram_gb` | `600` | System memory limit passed to `MemoryConfig`. |
| `warn_ram_gb` | `560` | RAM warning threshold. |
| `transcript_chunk_rows` | `1_000_000` | Points chunk size when streaming transcripts. |

The same values are enforced at the executor level:

```groovy
executor {
    name = "local"
    cpus = 75
    memory = "600 GB"
}
```

Per-process CPU/memory requests ([nextflow.config:42-69](../workflows/nextflow.config#L42-L69)):

| Process | CPUs | Memory |
|---------|-----:|-------:|
| `BUILD_SPATIALDATA` | 8 | 200 GB |
| `SEGMENT` | 75 | 500 GB |
| `ENRICH` | 16 | 200 GB |
| `QC` | 8 | 100 GB |
| `ALIGN` | 16 | 300 GB |
| `ALIGN_QC` | 8 | 150 GB |
| `COMPARE` | 8 | 200 GB |
| `VISUALIZE` | 8 | 200 GB |
| `CLUSTERING_SQUIDPY` | 8 | 200 GB |
| `MAPMYCELLS` | 8 | 200 GB |

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
