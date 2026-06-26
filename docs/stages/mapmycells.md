# MapMyCells

Runs local Allen Institute MapMyCells annotation on the per-platform AnnData
objects produced by the Squidpy clustering stage. By default it maps against
both the Allen Whole Human Brain reference and a configurable strict WHB region
reference.

## What it does

For each platform in a pair:

1. Read `<sample_id>_clustered.h5ad` from `clustering_squidpy_out/<platform>/`.
2. Write a MapMyCells query H5AD with raw counts from `layers["counts"]` copied
   into `X`. MapMyCells expects the query cell-by-gene matrix in `X`.
3. Run MapMyCells locally through `python -m merxen.analysis.mapmycells_entrypoint`
   for the configured reference mode: `whole_brain`, `region`, or `both`.
   The whole-brain path uses the configured marker lookup JSON and precomputed
   stats HDF5 files. The region path builds or reuses a strict WHB ROI-specific
   reference in the durable MapMyCells cache first.
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
| `marker_lookup_path` | Whole-brain JSON marker lookup file. Required only when `reference_mode` includes `whole_brain`. |
| `precomputed_stats_path` | Whole-brain HDF5 precomputed stats file. Required only when `reference_mode` includes `whole_brain`. |
| `region_name` / `region_labels` | Short region output name and one or more Allen WHB ROI labels used to build the strict region reference. |
| `region_cache_dir` | Durable cache for Allen WHB downloads and generated region stats/marker files. |
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

When this stage is selected, workflow preflight validates whole-brain marker and
precomputed-stat paths before any tasks start, unless `mapmycells_plots_only` is
enabled or `reference_mode` excludes `whole_brain`.

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

The stage also writes `<pair_id>_mapmycells_manifest.json` at the top of
`mapmycells_out/`, including whole-brain and region reference paths, ROI labels,
filtering counts, and per-sample outputs.
