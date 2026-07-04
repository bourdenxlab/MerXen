# Spatial Gene Analysis

Runs per-gene spatial autocorrelation on the selected analysis table after
visualization and before Squidpy clustering.

## What it does

For each active platform in the run:

1. Open the latest SpatialData zarr and copy the selected AnnData table.
2. Populate `.obsm["spatial"]` from the matching shape centroids, using aligned
   MERSCOPE coordinates when alignment is enabled.
3. Optionally remove blank/negative/control-like genes.
4. Filter low-support genes, normalize cell totals, and log-transform
   expression.
5. Build a Squidpy spatial nearest-neighbor graph from generic xy coordinates.
6. Compute Moran's I and Geary's C for every retained gene.
7. Write a full metrics table, top/bottom ranking table, a distribution plot,
   and individual spatial expression plots for ranked genes.

## Nextflow process

[`SPATIAL_GENE_ANALYSIS`](../../workflows/modules/spatial_gene_analysis.nf) —
one instance per `pair_id` and analysis segmentation branch.

- **Input:** `tuple(pair_id, segmentation, samples_json)`, where
  `samples_json` has one or two `{sample_id, platform, zarr_path, table_key,
  shape_key}` records.
- **CLI:** `merxen spatial-gene-analysis --config spatial_gene_analysis_config.json`.
- **Output:** `tuple(pair_id, segmentation, spatial_gene_analysis_out/)`.
- **publishDir:** `${outdir}/${pair_id}/${segmentation}/spatial_gene_analysis/`
  (copy mode).

When `VISUALIZE` is active, this stage waits for visualization completion. In a
default run through `clustering_squidpy`, clustering waits for
`SPATIAL_GENE_ANALYSIS`.

## Python entry points

| Function | File |
|----------|------|
| CLI `spatial_gene_analysis_command` | [cli/run_spatial_gene_analysis.py](../../src/merxen/cli/run_spatial_gene_analysis.py) |
| `run_spatial_gene_analysis` | [analysis/spatial_gene_analysis.py](../../src/merxen/analysis/spatial_gene_analysis.py) |
| `compute_spatial_autocorrelation` | [analysis/spatial_gene_analysis.py](../../src/merxen/analysis/spatial_gene_analysis.py) |
| `ranked_spatial_autocorr_genes` | [analysis/spatial_gene_analysis.py](../../src/merxen/analysis/spatial_gene_analysis.py) |

## Config schema

`SpatialGeneAnalysisConfig` — [config.py](../../src/merxen/config.py).

| Field | Description |
|-------|-------------|
| `pair_id` | Pair identifier used in output paths. |
| `output_dir` | Where `spatial_gene_analysis_out/` is populated. |
| `samples` | One or two sample configs: `sample_id`, `platform`, `zarr_path`, optional `table_key`, optional `shape_key`. |
| `drop_control_features` | Remove blank/negative/control-like genes before analysis. |
| `min_counts` / `min_cells` | Optional cell and gene filters before normalization. |
| `normalize_target_sum` | Optional `scanpy.pp.normalize_total` target sum. `null` uses Scanpy's default. |
| `normalize_exclude_highly_expressed` / `normalize_max_fraction` | Optional Scanpy size-factor controls for very highly expressed genes. |
| `n_neighbors` | Number of spatial neighbors used by Squidpy. |
| `top_n` | Number of top and bottom genes retained per metric. |
| `spatial_point_size` | Point size for ranked gene spatial expression plots. |
| `figure_dpi` | PNG output DPI. |

## Outputs

Written under `spatial_gene_analysis_out/<platform>/`:

Each listed `.png` plot is also written as a same-stem `.pdf`.

| Kind | File | Contents |
|------|------|----------|
| Metrics table | `<sample_id>_spatial_gene_autocorrelation.csv` | One row per retained gene with Moran's I, Geary's C, and available p-values. |
| Ranking table | `<sample_id>_spatial_gene_autocorrelation_rankings.csv` | Top and bottom `top_n` genes for both Moran's I and Geary's C. |
| Distribution plot | `plots/distributions/<sample_id>_spatial_autocorrelation_distribution.png` | Side-by-side histograms for Moran's I and Geary's C. |
| Spatial gene plots | `plots/spatial_genes/<metric>/<top_or_bottom>/*.png` | One plot per ranked gene, colored by log-normalized expression. |
| Manifest | `<sample_id>_spatial_gene_analysis_manifest.json` | Parameters, retained counts, and output paths. |
