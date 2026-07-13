# MapMyCells

Runs local Allen Institute MapMyCells annotation on the per-platform AnnData
objects produced by the Squidpy clustering stage. It supports the Allen Whole
Human Brain (WHB) taxonomy and the Yao et al. 2023 Whole Mouse Brain (WMB)
taxonomy, including human-to-mouse ortholog mapping. WHB remains the default.

## What it does

For each platform in a pair:

1. Read `<sample_id>_clustered.h5ad` from `clustering_squidpy_out/<platform>/`.
2. Write a MapMyCells query H5AD with raw counts from `layers["counts"]` copied
   into `X`. MapMyCells expects the query cell-by-gene matrix in `X`.
3. Run MapMyCells locally through `python -m merxen.analysis.mapmycells_entrypoint`
   for the configured reference mode: `whole_brain`, `region`, or `both`.
   The whole-brain path uses configured files or downloads Allen's published
   marker lookup and precomputed-stat assets. The region path builds or reuses
   a strict atlas/ROI-specific reference in the durable MapMyCells cache first.
4. Save the extended JSON, CSV, mapper log, stdout/stderr logs, command
   manifest, query H5AD, standalone UMAP/spatial PNG/PDF plots, and a clustered H5AD
   annotated with MapMyCells assignments in `obs` columns prefixed with
   `mapmycells_`. The H5AD also records MapMyCells metadata in
   `uns["merxen_mapmycells"]`, including the paths to the separate PNGs; the
   plot images themselves are not embedded in the H5AD.

Set `--mapmycells_plots_only true` to regenerate the annotated H5AD and plots
from an existing published `mapmycells_out/` directory without preparing a new
query H5AD, rebuilding a region reference, or rerunning MapMyCells. This is
useful after changing plot code. Use it with `--only_stage mapmycells` and the
same `--outdir`, `--mapmycells_reference_mode`, and `--mapmycells_region_name`
used for the original run.

The default `mapmycells_bootstrap_factor` is `0.9` because these data are
spatial transcriptomics panels where the newer single-cell-oriented lower
defaults can be less stable.

For region mode, `mapmycells_region_labels` contains Allen WHB
`region_of_interest_label` values. The default is
`["Human A44-A45", "Human A46", "Human A32", "Human ACC"]`, but the
implementation accepts a list, a JSON list, or a comma-separated string so this
can be adjusted to different frontal region sets later.

## Whole Mouse Brain compatibility

Set `mapmycells_reference_atlas=wmb` to use the Yao et al. 2023 WMB taxonomy.
For human MerXen queries, also keep `mapmycells_query_species=human`. MerXen then
downloads and caches Allen's full-WMB precomputed stats, mouse marker lookup,
and the `mmc_gene_mapper` ortholog database, and passes the database through
`gene_mapping.db_path`. It also drops `CCN20230722_SUPT` by default, matching
Allen's human-to-WMB example. `cell_type_mapper` 1.7.2 or newer is required.

```bash
nextflow run workflows/main.nf \
    --samplesheet workflows/samplesheet.csv \
    --outdir ./results \
    --stop_stage mapmycells \
    --mapmycells_reference_mode whole_brain \
    --mapmycells_reference_atlas wmb \
    --mapmycells_query_species human \
    --mapmycells_region_cache_dir /durable/mapmycells-cache
```

The automatic full-WMB downloads are approximately 1.4 GB for stats, 14 MB for
markers, and 16.2 GB for the cross-species gene database. Explicit
`mapmycells_marker_lookup_path`, `mapmycells_precomputed_stats_path`, and
`mapmycells_gene_mapping_db_path` values override the downloads.

For a mouse-region-specific WMB reference, select mouse
`region_of_interest_acronym` values such as `MOp` or `VIS`:

```bash
nextflow run workflows/main.nf \
    --samplesheet workflows/samplesheet.csv \
    --outdir ./results \
    --stop_stage mapmycells \
    --mapmycells_reference_mode region \
    --mapmycells_reference_atlas wmb \
    --mapmycells_query_species human \
    --mapmycells_region_name motor \
    --mapmycells_region_labels MOp \
    --mapmycells_region_cache_dir /durable/mapmycells-cache
```

Region generation downloads WMB metadata and only the raw expression-matrix
shards named by the selected cells' `feature_matrix_label` values. Individual
shards can be several GB, so the cache must have substantial free space.

## Nextflow process

[`MAPMYCELLS`](../../workflows/modules/mapmycells.nf) — one instance per
`pair_id`.

- **Input:** `tuple(pair_id, clustering_squidpy_out/)`.
- **CLI:** `merxen mapmycells --config mapmycells_config.json`.
- **Output:** `tuple(pair_id, mapmycells_out/)`.
- **publishDir:** `${outdir}/${pair_id}/mapmycells/` (copy mode).

This stage is opt-in. The default `stop_stage` remains `clustering_squidpy` so
existing runs do not require reference files. Run it with:

```bash
nextflow run workflows/main.nf \
    --samplesheet workflows/samplesheet.csv \
    --outdir ./results \
    --stop_stage mapmycells \
    --mapmycells_reference_mode both
```

Use `--only_stage mapmycells` to reuse an existing
`${outdir}/<pair_id>/clustering_squidpy/clustering_squidpy_out/` directory.

## Config Schema

`MapMyCellsConfig` — [config.py](../../src/merxen/config.py).

| Field | Description |
|-------|-------------|
| `pair_id` | Pair identifier used in output paths. |
| `output_dir` | Where `mapmycells_out/` is populated. |
| `samples` | One or two sample configs: `sample_id`, `platform`, `anndata_path`, optional `query_layer`, optional `gene_id_column`, optional `obs_id_column`. |
| `reference_mode` | `whole_brain`, `region`, or `both`; default is `both`. |
| `reference_atlas` | `whb` or `wmb`; default is `whb`. |
| `query_species` | `human` or `mouse`; controls whether WMB mapping needs cross-species gene mapping. |
| `auto_download_references` | Download missing Allen stats, markers, and the WMB gene mapper into the durable cache. |
| `marker_lookup_path` | Optional explicit whole-brain JSON marker lookup; otherwise downloaded when enabled. |
| `precomputed_stats_path` | Optional explicit whole-brain HDF5 stats file; otherwise downloaded when enabled. |
| `gene_mapping_db_path` | Optional `mmc_gene_mapper` SQLite database; required for human-to-WMB mapping when automatic downloads are disabled. |
| `region_name` / `region_labels` | Short output name and WHB ROI labels or WMB ROI acronyms used to build the strict region reference. |
| `region_cache_dir` | Durable cache for Allen WHB/WMB downloads and generated region stats/marker files. |
| `region_min_cells_per_leaf` | Minimum ROI cells required for a leaf `cluster_alias` to stay in the region taxonomy. |
| `region_force_rebuild` | Rebuild generated region reference files even if the cache manifest matches. |
| `region_query_markers_n_per_utility` | Marker count target for region `QueryMarkerRunner`. |
| `drop_level` | Optional taxonomy level to drop before mapping, such as the Whole Mouse Brain supertype level. |
| `normalization` | Passed to `type_assignment.normalization`; `raw` means MapMyCells converts query counts internally. |
| `bootstrap_factor` | Marker downsampling factor per bootstrap iteration. Defaults to `0.9` for spatial data. |
| `bootstrap_iteration` | Number of bootstrapping iterations. |
| `n_processors` / `chunk_size` / `rng_seed` | MapMyCells parallelism and reproducibility controls. |
| `max_gb` / `tmp_dir` | Optional mapper memory and temporary storage controls. |
| `cloud_safe` / `flatten` / `verbose_csv` | Direct MapMyCells CLI options. |
| `plots_only` | Reuse existing mapper CSV/extended JSON outputs and regenerate only annotated H5AD + plots. |

When explicit reference paths are configured, workflow preflight validates them
before any tasks start. Automatically downloaded files are checked against the
sizes in Allen's manifest and partial downloads can resume.

## Outputs

Written under `mapmycells_out/<platform>/`:

| Kind | File | Contents |
|------|------|----------|
| Query AnnData | `<sample_id>_mapmycells_query.h5ad` | Mapper input with query counts in `X`. |
| CSV | `<sample_id>_mapmycells.csv` | Per-cell taxonomy assignments and probabilities. |
| Extended JSON | `<sample_id>_mapmycells_extended.json` | Full MapMyCells result, config, logs, marker genes, and taxonomy tree. |
| Log | `<sample_id>_mapmycells.log` | Mapper log output. |
| Stdout log | `<sample_id>_mapmycells_stdout.log` | Captured process stdout, including the exact command line. |
| Stderr log | `<sample_id>_mapmycells_stderr.log` | Captured process stderr for startup/import errors and mapper tracebacks. |
| Command manifest | `<sample_id>_mapmycells_command.json` | Exact command used for the local mapper call. |
| UMAP plot | `<sample_id>_mapmycells_umap.png` | Existing Squidpy/Scanpy UMAP coordinates colored by MapMyCells assignment. |
| UMAP cluster-by-supercluster plots | `<sample_id>_mapmycells_umap_cluster_by_supercluster/supercluster_<name>.png` | Per-supercluster UMAPs with cells outside the supercluster in grey and member cells colored by MapMyCells cluster. |
| Spatial plot | `<sample_id>_mapmycells_spatial.png` | Spatial coordinates colored by MapMyCells assignment. |
| Quality scatter | `<sample_id>_mapmycells_quality_scatter.png` | Extended-JSON QC panels for supercluster and cluster assignments: cell complexity vs average correlation/bootstrap probability, correlation vs bootstrap probability, aggregate probability, and runner-up margin. |
| Supercluster QC | `<sample_id>_mapmycells_supercluster_assignment_qc.png` | Supercluster cell counts, confidence summaries, and low-confidence fractions. |
| Cluster QC | `<sample_id>_mapmycells_cluster_assignment_qc.png` | Cluster cell counts, confidence summaries, and low-confidence fractions. |
| Supercluster spatial grid | `<sample_id>_mapmycells_spatial_supercluster_grid.png` | Small-multiple spatial grid with each supercluster highlighted in red against all other cells in grey. |
| Annotated AnnData | `<sample_id>_mapmycells_annotated.h5ad` | Clustered AnnData with MapMyCells assignments added to `obs` and mapper metadata in `uns["merxen_mapmycells"]`. |

Each listed `.png` plot is also written as a same-stem `.pdf`.

Region-specific outputs use the same file names under
`mapmycells_out/region_<mapmycells_region_name>/<platform>/`. Their annotated
H5AD columns use the prefix `mapmycells_region_<region_name>_`, and metadata is
stored in `uns["merxen_mapmycells_region_<region_name>"]`.

Full-WMB outputs are written under `mapmycells_out/wmb/<platform>/` with the
`mapmycells_wmb_` column prefix. WMB region outputs are written under
`mapmycells_out/wmb_region_<region_name>/<platform>/` with the
`mapmycells_wmb_region_<region_name>_` prefix, keeping them distinct from WHB
annotations.

The stage also writes `<pair_id>_mapmycells_manifest.json` at the top of
`mapmycells_out/`, including whole-brain and region reference paths, ROI labels,
filtering counts, and per-sample outputs.
