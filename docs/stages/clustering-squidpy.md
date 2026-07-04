# Squidpy Clustering

Runs a first-pass per-platform Scanpy/Squidpy analysis on the enriched
SpatialData zarrs. In a full pipeline run this stage is downstream of
`SPATIAL_GENE_ANALYSIS`, so the usual alignment QC, visualisation, and per-gene
spatial autocorrelation artifacts have already been written.

## What it does

For each active platform in the run:

1. Open the latest SpatialData zarr and copy the selected AnnData table.
2. Populate `.obsm["spatial"]` from the matching shape centroids when needed.
   If MERSCOPE aligned non-rigid shapes exist, those centroids are preferred.
3. Compute Scanpy QC metrics plus blank/control/negative probe summaries from
   available `obs`, `var`, and `obsm` fields.
4. Plot histograms for transcripts per cell, genes per cell, cell area,
   nucleus ratio, and control/blank counts. Missing MERSCOPE nucleus metrics
   are kept as `NaN` and plotted as unavailable.
5. Run the gentle workflow:

   ```python
   sc.pp.filter_cells(adata, min_counts=10)
   sc.pp.filter_genes(adata, min_cells=5)
   sc.pp.normalize_total(adata, inplace=True)
   sc.pp.log1p(adata)
   sc.pp.pca(adata)
   sc.pp.neighbors(adata)
   sc.tl.umap(adata)
   sc.tl.leiden(adata)
   ```

6. Save a Scanpy UMAP, Squidpy spatial Leiden scatter, per-cell QC CSV, and the
   clustered `.h5ad`.
7. By default, add or replace the final clustered AnnData as a derived table in
   the originating `latest_spatialdata.zarr`. `reseg` writes
   `table_MOSAIK_proseg_clustering_squidpy`; `original_seg` writes
   `table_original_clustering_squidpy`.

By default, the one-shot output path still writes `<sample_id>_clustered.h5ad`,
but the clustering inside that file comes from an atlas-guided hierarchy:

1. Run a broad low-resolution Leiden round into `obs["leiden_broad"]`.
2. Score broad clusters against Allen WHB/MapMyCells marker sets, resolving
   markers by Ensembl ID first and gene symbol second. When the query has only
   symbols, the stage auto-loads WHB reference H5AD `gene_symbol` metadata from
   `reference_cache_dir` to bridge Ensembl marker IDs to panel symbols.
3. Write `obs["broad_atlas_label"]` and the collapsed `obs["broad_class"]`.
   The built-in broad classes are oligodendrocytes, OPCs, astrocytes, neurons,
   microglia, fibroblasts, and vascular cells. Confident extra WHB labels such
   as Ependymal or Choroid plexus are retained.
4. Recluster each broad class from `layers["counts"]`, so every branch is
   renormalized and recomputes PCA, neighbors, UMAP, and Leiden.
5. Split neurons into `Excitatory`, `Inhibitory`, and `Other` with WHB
   neuronal marker groups, then subtype-cluster each split.
6. Write per-round UMAP/spatial plots, annotation heatmaps, marker tables,
   branch H5ADs, and a manifest under `<sample_id>_hierarchical/`.

## Nextflow process

[`CLUSTERING_SQUIDPY`](../../workflows/modules/clustering_squidpy.nf) — one
instance per `pair_id`.

- **Input:** `tuple(pair_id, samples_json)`, where `samples_json` has one or
  two `{sample_id, platform, zarr_path}` records.
- **CLI:** `merxen clustering-squidpy --config clustering_squidpy_config.json`.
- **Output:** `tuple(pair_id, samples_json, clustering_squidpy_out/)`.
- **publishDir:** `${outdir}/${pair_id}/clustering_squidpy/` (copy mode).

When `SPATIAL_GENE_ANALYSIS` is active, the workflow joins on its completion
channel so clustering starts after per-gene spatial analysis. Otherwise, when
only `VISUALIZE` is active, clustering waits for visualisation.
`MAPMYCELLS`, when selected, consumes the clustered H5ADs written here.
`--only_stage clustering_squidpy` reads the published latest zarrs directly.

## Python entry points

| Function | File |
|----------|------|
| CLI `clustering_squidpy_command` | [cli/run_clustering_squidpy.py](../../src/merxen/cli/run_clustering_squidpy.py) |
| `run_clustering_squidpy` | [analysis/clustering_squidpy.py](../../src/merxen/analysis/clustering_squidpy.py) |
| `load_spatialdata_adata` | [analysis/clustering_squidpy.py](../../src/merxen/analysis/clustering_squidpy.py) |
| `run_scanpy_clustering` | [analysis/clustering_squidpy.py](../../src/merxen/analysis/clustering_squidpy.py) |
| `plot_qc_histograms` | [analysis/clustering_squidpy.py](../../src/merxen/analysis/clustering_squidpy.py) |
| `plot_umap` / `plot_spatial_scatter` | [analysis/clustering_squidpy.py](../../src/merxen/analysis/clustering_squidpy.py) |

## Config schema

`ClusteringSquidpyConfig` — [config.py](../../src/merxen/config.py).

| Field | Description |
|-------|-------------|
| `pair_id` | Pair identifier used in output paths. |
| `output_dir` | Where `clustering_squidpy_out/` is populated. |
| `samples` | One or two sample configs: `sample_id`, `platform`, `zarr_path`, optional `table_key`, optional `shape_key`. |
| `drop_control_features` | Remove blank/negative/control-like variables before clustering. |
| `min_counts` / `min_cells` | Gentle cell and gene filtering thresholds. |
| `normalize_target_sum` | Optional `scanpy.pp.normalize_total` target sum. `null` uses Scanpy's default. |
| `normalize_exclude_highly_expressed` / `normalize_max_fraction` | Optional Scanpy size-factor controls for very highly expressed genes. |
| `n_pcs` / `n_neighbors` | PCA and neighbor graph settings. |
| `leiden_resolution` | Leiden clustering resolution. |
| `umap_min_dist` / `umap_spread` | UMAP layout controls for compact vs dispersed embeddings. |
| `random_seed` | Seed for PCA/UMAP/Leiden. |
| `spatial_point_size` | Highlight point size for spatial cluster grid plots. |
| `spatial_scatter_point_size` | Point size for regular spatial scatter plots. |
| `figure_dpi` | PNG output DPI. |
| `use_gpu` | Use RAPIDS single-cell acceleration when available. |
| `clustering_squidpy_max_forks` | Nextflow-side concurrency guard. Defaults to `4`; GPU-backed tasks still share the local GPU lock when enabled. |
| `clustering_squidpy_gpu_vram_monitor` | Nextflow wrapper flag that records `nvidia-smi` VRAM samples for each clustering task. Defaults to `true`. |
| `clustering_squidpy_gpu_vram_monitor_interval_seconds` | Sampling interval for the VRAM monitor. Defaults to `2`. |
| `write_spatialdata_table` | Add or replace the final clustered table in each sample's source `latest_spatialdata.zarr`. Defaults to `true`; set `--clustering_squidpy_write_spatialdata_table false` for H5AD-only output. |
| `hierarchical_enabled` | Run broad atlas annotation plus branch subclustering. Defaults to `true`; set `false` for the legacy one-shot Leiden workflow. |
| `broad_round` | Round-specific broad clustering settings. Default Leiden resolution `0.2`; unspecified fields inherit top-level filtering/PCA/UMAP settings. |
| `subcluster_round` | Default non-neuron branch settings. Default Leiden resolution `0.5`. |
| `subcluster_resolution_overrides` | Optional map from broad class or neuron split label to a Leiden resolution override. |
| `neuron_split_round` | Coarse neuron Exc/Inh/Other split settings. Default Leiden resolution `0.15`. |
| `neuron_subcluster_round` | Neuron subtype settings after the split. Default Leiden resolution `0.5`. |
| `min_branch_cells` | Branches smaller than this are labeled but not reclustered. Default `50`. |
| `broad_annotation` | Marker lookup, Allen taxonomy metadata/cache path, marker overlap limits, and ambiguity thresholds for atlas-guided cluster labels. |

When this stage is selected and hierarchical mode is enabled, workflow preflight
requires the broad marker lookup and taxonomy metadata to exist before any tasks
start.

## Outputs

Written under `clustering_squidpy_out/<platform>/`:

Each listed `.png` plot is also written as a same-stem `.pdf`.

| Kind | File | Contents |
|------|------|----------|
| QC plot | `plots/qc/<sample_id>_qc_histograms.png` | Histograms for count, gene, geometry, nucleus, and control metrics. |
| QC table | `<sample_id>_qc_metrics.csv` | Per-cell QC metrics before filtering. |
| UMAP | `plots/umap/<sample_id>_umap.png` | UMAP colored by total counts, genes by counts, and Leiden. |
| Spatial | `plots/spatial/<sample_id>_spatial_scatter_leiden.png` | Squidpy spatial scatter colored by Leiden with clean axes and a 200 um scale bar. |
| Spatial Leiden grid | `plots/spatial_grid/<sample_id>_spatial_scatter_leiden_grid.png` | Small-multiple spatial grid with each de novo Leiden cluster highlighted in red against all other cells in grey. |
| AnnData | `<sample_id>_clustered.h5ad` | Filtered clustered object, with raw counts in `layers["counts"]`. |

When `write_spatialdata_table` is true, rerunning this stage mutates
`${outdir}/<pair_id>/<platform>/latest/latest_spatialdata.zarr` by adding or
replacing the derived clustered table for the active analysis segmentation. The
table contains the same final filtered cells, UMAP/spatial coordinates, counts
layer, and clustering/cell-type columns as `<sample_id>_clustered.h5ad`.

When `clustering_squidpy_gpu_vram_monitor` is enabled, the Nextflow wrapper also
writes task-level GPU telemetry under `clustering_squidpy_out/gpu_vram/`:

| Kind | File | Contents |
|------|------|----------|
| GPU VRAM summary | `<pair_id>_<analysis_segmentation>_summary.json` | Peak task-matched and total device VRAM from the `nvidia-smi` sampler. |
| GPU VRAM samples | `<pair_id>_<analysis_segmentation>_samples.tsv` | Per-sample GPU memory rows, including all compute apps and task-descendant PID matches. |

With hierarchical mode enabled, additional artifacts are written under
`clustering_squidpy_out/<platform>/<sample_id>_hierarchical/`:

| Kind | File | Contents |
|------|------|----------|
| Manifest | `<sample_id>_hierarchical_manifest.json` | Branch settings, output paths, and clustering status for every broad class/split. |
| Broad annotation | `<sample_id>_broad_cluster_annotation.csv` | Broad Leiden cluster to atlas label/class assignment with score, margin, and marker count. |
| Broad scores | `<sample_id>_broad_annotation_scores.csv` | Cluster-by-atlas marker z-score table. |
| Broad markers | `<sample_id>_broad_resolved_markers.csv` | Reference markers that overlapped the query panel. |
| Broad heatmap | `plots/annotation/<sample_id>_broad_annotation_score_heatmap.png` | QC heatmap of broad cluster marker scores. |
| Branch plots/H5ADs | `branch_<class>/...` | Per-class `.h5ad`, UMAP under `plots/umap/`, spatial plot under `plots/spatial/`, spatial grid under `plots/spatial_grid/`, and panel-gene dotplot under `plots/dotplot/`. |
| Branch dotplot tables | `branch_<class>/tables/dotplot/...` | Per-subcluster mean expression and fraction-expressing summaries for panel genes. |
| Neuron split | `branch_neurons/<sample_id>_neurons_split_*` | Excitatory/Inhibitory/Other annotation tables, split `.h5ad`, heatmap under `plots/annotation/`, and plots under `plots/*/`. |
| Neuron subclusters | `branch_neurons/split_<label>/...` | Per-split neuron subtype `.h5ad`, UMAP, spatial plot, spatial grid, panel-gene dotplot, and dotplot tables. |

## Notebook

The companion notebook
[notebooks/clustering_squidpy_parameter_tuning.ipynb](../../notebooks/clustering_squidpy_parameter_tuning.ipynb)
calls the same functions as the pipeline stage. Use it to tweak thresholds and
plotting parameters interactively before promoting those settings to
Nextflow parameters.
