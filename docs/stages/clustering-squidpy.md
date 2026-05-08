# Squidpy Clustering

Runs a first-pass per-platform Scanpy/Squidpy analysis on the enriched
SpatialData zarrs. In a full pipeline run this stage is downstream of
`VISUALIZE`, so the usual alignment QC and visualisation artifacts have already
been written.

## What it does

For each platform in a pair:

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

## Nextflow process

[`CLUSTERING_SQUIDPY`](../../workflows/modules/clustering_squidpy.nf) — one
instance per `pair_id`.

- **Input:** `tuple(pair_id, merscope_zarr, xenium_zarr)`.
- **CLI:** `merxen clustering-squidpy --config clustering_squidpy_config.json`.
- **Output:** `tuple(pair_id, clustering_squidpy_out/)`.
- **publishDir:** `${outdir}/${pair_id}/clustering_squidpy/` (copy mode).

When `VISUALIZE` is also active, the workflow joins on its completion channel
so clustering starts after visualisation. `--only_stage clustering_squidpy`
reads the published latest zarrs directly.

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
| `samples` | MERSCOPE and Xenium sample configs: `sample_id`, `platform`, `zarr_path`, optional `table_key`, optional `shape_key`. |
| `drop_control_features` | Remove blank/negative/control-like variables before clustering. |
| `min_counts` / `min_cells` | Gentle cell and gene filtering thresholds. |
| `normalize_target_sum` | Optional `scanpy.pp.normalize_total` target sum. `null` uses Scanpy's default. |
| `normalize_exclude_highly_expressed` / `normalize_max_fraction` | Optional Scanpy size-factor controls for very highly expressed genes. |
| `n_pcs` / `n_neighbors` | PCA and neighbor graph settings. |
| `leiden_resolution` | Leiden clustering resolution. |
| `umap_min_dist` / `umap_spread` | UMAP layout controls for compact vs dispersed embeddings. |
| `random_seed` | Seed for PCA/UMAP/Leiden. |
| `spatial_point_size` | Squidpy spatial scatter point size. |
| `figure_dpi` | PNG output DPI. |
| `use_gpu` | Use RAPIDS single-cell acceleration when available. |

## Outputs

Written under `clustering_squidpy_out/<platform>/`:

| Kind | File | Contents |
|------|------|----------|
| QC plot | `<sample_id>_qc_histograms.png` | Histograms for count, gene, geometry, nucleus, and control metrics. |
| QC table | `<sample_id>_qc_metrics.csv` | Per-cell QC metrics before filtering. |
| UMAP | `<sample_id>_umap.png` | UMAP colored by total counts, genes by counts, and Leiden. |
| Spatial | `<sample_id>_spatial_scatter_leiden.png` | Squidpy spatial scatter colored by Leiden. |
| AnnData | `<sample_id>_clustered.h5ad` | Filtered clustered object, with raw counts in `layers["counts"]`. |

## Notebook

The companion notebook
[notebooks/clustering_squidpy_parameter_tuning.ipynb](../../notebooks/clustering_squidpy_parameter_tuning.ipynb)
calls the same functions as the pipeline stage. Use it to tweak thresholds and
plotting parameters interactively before promoting those settings to
Nextflow parameters.
